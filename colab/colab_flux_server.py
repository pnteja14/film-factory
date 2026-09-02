# %% [markdown]
# # Film Factory — FLUX.1-dev Image Server (Colab Pro+)
# Requires: Colab Pro+ with A100 GPU (40 GB VRAM)
# - First run: downloads FLUX.1-dev (~25 GB) to Google Drive — takes ~5 min
# - All future runs: loads directly from Drive in ~2 min, no download
#
# After running all cells, copy the ngrok URL into your local .env:
#   KAGGLE_IMAGE_URL=https://xxxx.ngrok-free.app   ← same variable, same API
#   IMAGE_SOURCE=kaggle

# %% [code] — Install dependencies
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "diffusers>=0.30.0",
    "transformers>=4.44.0",
    "accelerate>=0.33.0",
    "flask>=3.0.0",
    "pyngrok>=7.0.0",
    "huggingface_hub>=0.24.0",
    "sentencepiece",
    "protobuf",
    "ftfy",
], check=True)
print("Dependencies installed.")

# %% [code] — Mount Google Drive + authenticate
import os
from google.colab import drive, userdata

drive.mount("/content/drive")

HF_TOKEN    = userdata.get("HF_TOKEN")
NGROK_TOKEN = userdata.get("NGROK_TOKEN")
if not HF_TOKEN:
    raise ValueError("Add HF_TOKEN to Colab Secrets (key icon in sidebar)")

os.environ["HF_TOKEN"]    = HF_TOKEN
os.environ["NGROK_TOKEN"] = NGROK_TOKEN or ""

from huggingface_hub import login
login(token=HF_TOKEN, add_to_git_credential=False)
print("HuggingFace authenticated.")

# %% [code] — Copy model from Drive to local SSD (fast load every session)
import shutil, time

DRIVE_CACHE = "/content/drive/MyDrive/film_factory/models/hf_cache"
LOCAL_CACHE = "/content/hf_cache"
MODEL_MARKER = os.path.join(LOCAL_CACHE, "hub", "models--black-forest-labs--FLUX.1-dev")

os.makedirs(DRIVE_CACHE, exist_ok=True)

if os.path.isdir(MODEL_MARKER):
    print("Local SSD cache found — skipping copy (model loads in ~20 sec).")
else:
    drive_model = os.path.join(DRIVE_CACHE, "hub", "models--black-forest-labs--FLUX.1-dev")
    if os.path.isdir(drive_model):
        print("Copying FLUX.1-dev from Drive → local SSD (~3-5 min, one time per session)...")
        t0 = time.time()
        shutil.copytree(DRIVE_CACHE, LOCAL_CACHE, dirs_exist_ok=True)
        print(f"Copy done in {(time.time()-t0)/60:.1f} min — model will now load in ~20 sec.")
    else:
        print("First ever run — downloading FLUX.1-dev to Drive (~25 GB, ~5 min)...")
        os.makedirs(LOCAL_CACHE, exist_ok=True)

os.environ["HF_HOME"] = LOCAL_CACHE
print(f"HF_HOME → {LOCAL_CACHE}")

# %% [code] — Load FLUX.1-dev (full bf16 on A100 — no quantization, no compromises)
import torch, gc, io, base64

MODEL_ID = "black-forest-labs/FLUX.1-dev"

gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu_name} — {vram_gb:.1f} GB VRAM")

if vram_gb < 35:
    print(f"WARNING: {vram_gb:.1f} GB VRAM detected. FLUX.1-dev needs ~34 GB.")
    print("This notebook requires Colab Pro+ with A100. Switch runtime → GPU → A100.")

from diffusers import FluxPipeline

if os.path.isdir(MODEL_MARKER):
    print("Loading FLUX.1-dev from local SSD cache...")
else:
    print("Downloading FLUX.1-dev (~25 GB) — will cache to Drive for future sessions...")

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

gc.collect()
torch.cuda.empty_cache()
used_gb = torch.cuda.memory_allocated() / 1e9
print(f"FLUX.1-dev ready — full bf16, no quantization.")
print(f"VRAM used: {used_gb:.1f} GB / {vram_gb:.1f} GB")

# After a first-time download, sync local cache back to Drive for future sessions
drive_model = os.path.join(DRIVE_CACHE, "hub", "models--black-forest-labs--FLUX.1-dev")
if not os.path.isdir(drive_model):
    print("Syncing downloaded model to Drive (one time only)...")
    shutil.copytree(LOCAL_CACHE, DRIVE_CACHE, dirs_exist_ok=True)
    print("Sync to Drive complete — future sessions copy from Drive in ~3-5 min.")

# %% [code] — Flask server
import threading, time, traceback, wave
import numpy as np
from flask import Flask, request, jsonify

app   = Flask(__name__)
_lock = threading.Lock()


def generate_image(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1280,
    height: int = 720,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    seed: int = -1,
) -> str:
    generator = None
    if seed >= 0:
        generator = torch.Generator("cuda").manual_seed(seed)

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
    return jsonify({"status": "ok", "model": "FLUX.1-dev", "gpu": torch.cuda.get_device_name(0)})


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
        num_inference_steps = int(data.get("num_inference_steps", 40))
        guidance_scale      = float(data.get("guidance_scale", 4.0))
        seed                = int(data.get("seed", -1))

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
        return jsonify({"image_base64": img_b64, "format": "png", "width": width, "height": height})

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# %% [code] — MusicGen-Large (lazy loaded — unloads FLUX first)
music_model_cache = None

def load_music_model():
    global pipe, music_model_cache
    if music_model_cache is not None:
        return music_model_cache

    if pipe is not None:
        print("Unloading FLUX to free VRAM for MusicGen...")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        pipe = None

    print("Loading MusicGen-Large (~3.3B)...")
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
    model     = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-large",
        torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()
    music_model_cache = {"processor": processor, "model": model}
    print(f"MusicGen-Large ready. Sample rate: {model.config.audio_encoder.sampling_rate} Hz")
    return music_model_cache


@app.route("/generate_music", methods=["POST"])
def generate_music():
    try:
        data = request.get_json(force=True)
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' field"}), 400

        prompt         = data["prompt"]
        duration_s     = min(30.0, max(1.0, float(data.get("duration_seconds", 20))))
        guidance_scale = float(data.get("guidance_scale", 4.5))
        seed           = int(data.get("seed", -1))
        max_new_tokens = int(duration_s * 51.2)

        with _lock:
            md          = load_music_model()
            processor   = md["processor"]
            model       = md["model"]
            sample_rate = model.config.audio_encoder.sampling_rate

            generator = None
            if seed >= 0:
                generator = torch.Generator("cuda").manual_seed(seed)

            inputs = processor(text=[prompt], padding=True, return_tensors="pt")
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
        max_val  = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val * 0.85

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


# %% [code] — Start Flask + ngrok tunnel
from pyngrok import ngrok, conf

if NGROK_TOKEN:
    conf.get_default().auth_token = NGROK_TOKEN

PORT = 5000

server_thread = threading.Thread(
    target=lambda: app.run(host="0.0.0.0", port=PORT, use_reloader=False, debug=False),
    daemon=True,
)
server_thread.start()
time.sleep(2)

public_url = ngrok.connect(PORT).public_url
print("\n" + "="*60)
print("  FLUX.1-dev + MusicGen SERVER RUNNING (Colab Pro+ / A100)")
print("="*60)
print(f"\n  Paste into your local .env:\n")
print(f"    KAGGLE_IMAGE_URL={public_url}")
print(f"    KAGGLE_MUSIC_URL={public_url}")
print(f"    IMAGE_SOURCE=kaggle")
print(f"    MUSIC_SOURCE=kaggle")
print(f"\n  Endpoints:")
print(f"    GET  {public_url}/health")
print(f"    POST {public_url}/generate         — FLUX.1-dev full bf16")
print(f"    POST {public_url}/generate_music   — MusicGen-Large (unloads FLUX first)")
print("\n  Keep this notebook running (kernel stays alive automatically).")
print("="*60 + "\n")
