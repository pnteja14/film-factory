# %% [markdown]
# # Film Factory — FLUX.1-dev Image Server
# Run this notebook on Kaggle to serve high-quality image generation to your local pipeline.
# After running all cells, copy the ngrok URL printed at the bottom into your .env file as:
#   KAGGLE_IMAGE_URL=https://xxxx.ngrok-free.app
#   IMAGE_SOURCE=kaggle

# %% [code] — Install dependencies (run once)
# %%capture
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "diffusers>=0.30.0",
    "transformers>=4.44.0",
    "accelerate>=0.33.0",
    "bitsandbytes>=0.43.0",
    "flask>=3.0.0",
    "pyngrok>=7.0.0",
    "huggingface_hub>=0.24.0",
    "sentencepiece",
    "protobuf",
], check=True)
print("Dependencies installed.")

# %% [code] — Authenticate with HuggingFace (FLUX.1-dev is a gated model)
import os

# Read secrets via Kaggle's secret manager (Add-ons → Secrets in the notebook sidebar)
# Must have accepted the FLUX.1-dev license at https://huggingface.co/black-forest-labs/FLUX.1-dev
try:
    from kaggle_secrets import UserSecretsClient
    _s = UserSecretsClient()
    os.environ["HF_TOKEN"]    = _s.get_secret("HF_TOKEN")
    os.environ["NGROK_TOKEN"] = _s.get_secret("NGROK_TOKEN")
    print("Secrets loaded from Kaggle secret manager.")
except Exception as _e:
    print(f"kaggle_secrets unavailable ({_e}) — falling back to environment variables.")

from huggingface_hub import login

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found.\n"
        "Go to Add-ons → Secrets → add a secret named HF_TOKEN with your HuggingFace token.\n"
        "Get your token at https://huggingface.co/settings/tokens"
    )

login(token=HF_TOKEN, add_to_git_credential=False)
print("HuggingFace authentication successful.")

# %% [code] — Load FLUX.1-dev (transformer + T5 both quantized to NF4 — proven T4 approach)
# Downloads to /kaggle/temp/ (37 GB free). Peak VRAM ~8.5 GB. Works on single T4.
import torch, gc
from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
from transformers import BitsAndBytesConfig as TransformersBnbConfig, T5EncoderModel

HF_MODEL_ID = "black-forest-labs/FLUX.1-dev"
os.environ["HF_HOME"] = "/kaggle/temp/hf_cache"   # 37 GB free — fits all components

def _flush():
    gc.collect()
    torch.cuda.empty_cache()

dtype = torch.float16   # T4 has limited bfloat16 support

n_gpus = torch.cuda.device_count()
for i in range(n_gpus):
    vram = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"GPU {i}: {vram:.1f} GB VRAM")

# Step 1: T5 text encoder — 11 GB bf16 → ~2.5 GB NF4
# MUST quantize T5 too — left in bf16 it alone OOMs the T4 during text encoding
print("Loading T5 text encoder in 4-bit NF4 (~2.5 GB)...")
t5_nf4 = TransformersBnbConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
    bnb_4bit_use_double_quant=False,
)
text_encoder_2 = T5EncoderModel.from_pretrained(
    HF_MODEL_ID,
    subfolder="text_encoder_2",
    quantization_config=t5_nf4,
    torch_dtype=dtype,
    token=HF_TOKEN,
)
_flush()

# Step 2: Transformer — 23 GB bf16 → ~5.5 GB NF4
print("Loading FLUX transformer in 4-bit NF4 (~5.5 GB)...")
flux_nf4 = DiffusersBnbConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype,
    bnb_4bit_use_double_quant=False,
)
transformer = FluxTransformer2DModel.from_pretrained(
    HF_MODEL_ID,
    subfolder="transformer",
    quantization_config=flux_nf4,
    torch_dtype=dtype,
    token=HF_TOKEN,
)
_flush()

# Step 3: Full pipeline — VAE + CLIP + tokenizers (~1 GB, tiny)
print("Loading pipeline (VAE + CLIP + tokenizers)...")
pipe = FluxPipeline.from_pretrained(
    HF_MODEL_ID,
    transformer=transformer,
    text_encoder_2=text_encoder_2,
    torch_dtype=dtype,
    token=HF_TOKEN,
)
pipe.enable_sequential_cpu_offload()   # layer-by-layer offload — safest VRAM option on T4
pipe.enable_attention_slicing()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
_flush()
print("FLUX.1-dev ready — transformer + T5 in NF4, sequential CPU offload.")

# %% [code] — Flask server
import io, base64, threading, time, traceback, wave
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
_lock = threading.Lock()   # one generation at a time


def generate_image(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1280,
    height: int = 720,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    seed: int = -1,
) -> str:
    """Generate image and return as base64 PNG string."""
    import gc
    generator = None
    if seed >= 0:
        generator = torch.Generator().manual_seed(seed)

    with _lock:
        gc.collect()
        torch.cuda.empty_cache()
        result = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
        image = result.images[0]
        del result
        gc.collect()
        torch.cuda.empty_cache()

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "FLUX.1-dev", "device": DEVICE})


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(force=True)
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' field"}), 400

        prompt              = data["prompt"]
        negative_prompt     = data.get("negative_prompt", "")
        width               = int(data.get("width",  1280))
        height              = int(data.get("height",  720))
        num_inference_steps = int(data.get("num_inference_steps", 28))
        guidance_scale      = float(data.get("guidance_scale", 3.5))
        seed                = int(data.get("seed", -1))

        # Clamp to FLUX safe dimensions (multiples of 16)
        width  = max(512, min(2048, (width  // 16) * 16))
        height = max(512, min(2048, (height // 16) * 16))

        img_b64 = generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        return jsonify({
            "image_base64": img_b64,
            "format": "png",
            "width": width,
            "height": height,
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# %% [code] — Wan2.1-I2V video generation (lazy loaded)
# Called AFTER all FLUX images are generated — unloads FLUX first to free VRAM.
# Model: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers (fits on T4 once FLUX is unloaded)

wan_pipe = None

def load_wan_pipe():
    global pipe, wan_pipe
    if wan_pipe is not None:
        return wan_pipe

    # Unload FLUX to free VRAM before loading Wan
    if pipe is not None:
        print("Unloading FLUX to free VRAM for Wan2.1-I2V...")
        del pipe
        torch.cuda.empty_cache()
        pipe = None

    print("Loading Wan2.1-I2V-14B-480P...")
    from diffusers import WanImageToVideoPipeline
    from diffusers.utils import export_to_video

    wan_pipe = WanImageToVideoPipeline.from_pretrained(
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        torch_dtype=torch.bfloat16,
    )
    wan_pipe.enable_model_cpu_offload()
    print("Wan2.1-I2V ready.")
    return wan_pipe


@app.route("/generate_video", methods=["POST"])
def generate_video():
    """
    Generate a short video clip from a source image using Wan2.1-I2V.

    Request JSON:
      image_base64   : str   — base64-encoded source image (PNG/JPG)
      prompt         : str   — motion description prompt
      negative_prompt: str   — optional negative prompt
      num_frames     : int   — frames to generate (default 81 = ~5s at 16fps)
      guidance_scale : float — default 5.0

    Response JSON:
      video_base64   : str   — base64-encoded MP4
      num_frames     : int
      fps            : int
    """
    import tempfile
    try:
        data = request.get_json(force=True)
        if not data or "image_base64" not in data or "prompt" not in data:
            return jsonify({"error": "Missing image_base64 or prompt"}), 400

        from PIL import Image

        # Decode source image
        img_bytes = base64.b64decode(data["image_base64"])
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        prompt          = data["prompt"]
        negative_prompt = data.get("negative_prompt", (
            "blurry, low quality, distorted, static still image, no motion, "
            "camera shake, fast motion, duplicate frames, "
            "morphing, warping, melting, flickering, temporal artifacts, "
            "glitching, color shift, identity change, object distortion, "
            "dissolving, smearing, streaking, excessive noise, face distortion"
        ))
        num_frames      = int(data.get("num_frames", 81))
        guidance_scale  = float(data.get("guidance_scale", 5.0))
        fps             = 16

        with _lock:
            vp = load_wan_pipe()
            output = vp(
                image=pil_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=480,
                width=832,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
            ).frames[0]

        # Export to MP4 in a temp file, read back as base64
        from diffusers.utils import export_to_video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        export_to_video(output, tmp_path, fps=fps)
        with open(tmp_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
        os.unlink(tmp_path)

        return jsonify({
            "video_base64": video_b64,
            "num_frames":   num_frames,
            "fps":          fps,
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# %% [code] — MusicGen-Large music generation (lazy loaded)
# Called AFTER all FLUX images are done — unloads FLUX first to free VRAM.
# Model: facebook/musicgen-large (3.3B, MIT license, no gating required)
# Generates up to 30s of scene-specific cinematic music per call.
# The local pipeline sends arc_position + location + scene description as a text
# prompt and receives a base64-encoded WAV in return.

music_model_cache = None   # {"processor": ..., "model": ...} when loaded

def load_music_model():
    global pipe, wan_pipe, music_model_cache
    if music_model_cache is not None:
        return music_model_cache

    # Unload FLUX and Wan to free VRAM
    if pipe is not None:
        print("Unloading FLUX to free VRAM for MusicGen-Large...")
        del pipe
        torch.cuda.empty_cache()
        pipe = None
    if wan_pipe is not None:
        print("Unloading Wan to free VRAM for MusicGen-Large...")
        del wan_pipe
        torch.cuda.empty_cache()
        wan_pipe = None

    print("Loading MusicGen-Large (~3.3B, float16)...")
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
    model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-large",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model.enable_model_cpu_offload()

    music_model_cache = {"processor": processor, "model": model}
    sample_rate = model.config.audio_encoder.sampling_rate
    print(f"MusicGen-Large ready. Sample rate: {sample_rate} Hz")
    return music_model_cache


@app.route("/generate_music", methods=["POST"])
def generate_music():
    """
    Generate scene-specific music using MusicGen-Large.

    Request JSON:
      prompt           : str   — scene-specific music description
                                 e.g. "dark methodical detective score, Victorian manor study,
                                       slow heartbeat pulse, no vocals, cinematic thriller"
      duration_seconds : float — clip length in seconds (1–30, default 20)
      guidance_scale   : float — prompt adherence; 4–6 recommended (default 4.5)
      seed             : int   — -1 for random, >=0 for reproducible

    Response JSON:
      audio_base64  : str  — base64-encoded WAV (mono, 32 kHz)
      sample_rate   : int  — 32000
      duration      : float — actual generated duration in seconds
      model         : str  — "musicgen-large"
    """
    try:
        data = request.get_json(force=True)
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' field"}), 400

        prompt         = data["prompt"]
        duration_s     = float(data.get("duration_seconds", 20))
        guidance_scale = float(data.get("guidance_scale", 4.5))
        seed           = int(data.get("seed", -1))

        # MusicGen-Large safe ceiling: 30s avoids OOM on T4
        duration_s = max(1.0, min(30.0, duration_s))

        with _lock:
            md        = load_music_model()
            processor = md["processor"]
            model     = md["model"]
            sample_rate = model.config.audio_encoder.sampling_rate  # 32000

            # ~51.2 tokens per second of audio at 32 kHz
            max_new_tokens = int(duration_s * 51.2)

            generator = None
            if seed >= 0:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                generator = torch.Generator(device=device).manual_seed(seed)

            inputs = processor(text=[prompt], padding=True, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                audio_values = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )

        audio_np = audio_values[0, 0].cpu().float().numpy()

        # Normalise to -1..1 → 85% headroom
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val * 0.85

        # Encode as 16-bit mono WAV
        audio_int16 = (audio_np * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        return jsonify({
            "audio_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "sample_rate":  sample_rate,
            "duration":     len(audio_np) / sample_rate,
            "model":        "musicgen-large",
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# %% [code] — Start server + ngrok tunnel
from pyngrok import ngrok, conf

# If you have a free ngrok account, paste your authtoken here for stable URLs
# Get it from https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_TOKEN = os.environ.get("NGROK_TOKEN", "")   # set as Kaggle secret named NGROK_TOKEN
if NGROK_TOKEN:
    conf.get_default().auth_token = NGROK_TOKEN

PORT = 5000

server_thread = threading.Thread(
    target=lambda: app.run(host="0.0.0.0", port=PORT, use_reloader=False, debug=False),
    daemon=True,
)
server_thread.start()
time.sleep(2)   # let Flask start

public_url = ngrok.connect(PORT).public_url
print("\n" + "="*60)
print("  FLUX.1-dev + Wan2.1-I2V + MusicGen-Large SERVER IS RUNNING")
print("="*60)
print(f"\n  Copy these into your .env file:\n")
print(f"    KAGGLE_IMAGE_URL={public_url}")
print(f"    KAGGLE_VIDEO_URL={public_url}")
print(f"    KAGGLE_MUSIC_URL={public_url}")
print(f"    IMAGE_SOURCE=kaggle")
print(f"    VIDEO_SOURCE=kaggle")
print(f"    MUSIC_SOURCE=kaggle")
print("\n  Endpoints:")
print(f"    GET  {public_url}/health           — server status")
print(f"    POST {public_url}/generate         — FLUX.1-dev image generation")
print(f"    POST {public_url}/generate_video   — Wan2.1-I2V video clip (unloads FLUX first)")
print(f"    POST {public_url}/generate_music   — MusicGen-Large scene music (unloads FLUX first)")
print("\n  Workflow order:")
print("    1. Generate all images  → /generate")
print("    2. Generate all music   → /generate_music  (FLUX unloads automatically)")
print("    3. Generate video clips → /generate_video  (music model unloads automatically)")
print("\n  Keep this notebook running while the pipeline generates content.")
print("="*60 + "\n")

# Keep the notebook alive
while True:
    time.sleep(60)
