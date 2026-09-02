# %% [markdown]
# # Film Factory — Colab Runner (No ngrok)
#
# Runs the full pipeline ON Colab — orchestrator + Flask server on the same A100.
# No ngrok tunnel needed. Local HTTP has zero connection timeouts.
#
# ## Before running
# 1. Upload **film_factory_code.zip** to Drive at:
#    `My Drive/film_factory/film_factory_code.zip`
# 2. Upload **film_factory_outputs.zip** to Drive at:
#    `My Drive/film_factory/film_factory_outputs.zip`
#    (run `scripts/zip_for_colab.ps1` locally to create both)
# 3. Add these to Colab Secrets (key icon in sidebar):
#    - `HF_TOKEN`  — HuggingFace token (for Wan + MusicGen model download)
#    - `ANTHROPIC_API_KEY` — required for Edit Planner (Agent 6a calls Claude API)
#
# ## What this does
# - Starts Wan2.1-I2V + MusicGen-Large on localhost:5000
# - Runs orchestrator `--resume-from images` on the SAME machine
# - Wan video clips + music generated locally — no tunnel, no 5-min kill
# - Final episode synced back to Drive when done

# %% [code] — Install system packages + Python deps
import subprocess, sys, os
subprocess.run(["apt-get", "install", "-y", "-q", "ffmpeg",
                "fonts-liberation", "fonts-dejavu-core"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "diffusers>=0.34.0,<0.37.0",
    "transformers>=4.47.0",
    "accelerate>=0.33.0",
    "flask>=3.0.0",
    "ftfy",
    "imageio[ffmpeg]",
    "scipy",
    "python-dotenv",
    "anthropic",
    "requests",
    "Pillow",
    "huggingface_hub>=0.24.0",
    "sentencepiece",
    "protobuf",
    "numpy",
    "openai-whisper",
    "whisperx",
    "librosa",
    "json-repair",
    "pydantic>=2.0",
    "basicsr",          # Real-ESRGAN backbone (RRDB architecture)
    "realesrgan",       # 4x super-resolution for Wan clip upscaling
    "opencv-python-headless",
], check=True)
print("All dependencies installed.")

# %% [code] — Mount Drive + authenticate
from google.colab import drive, userdata

drive.mount("/content/drive")

HF_TOKEN          = userdata.get("HF_TOKEN")
ANTHROPIC_API_KEY = userdata.get("ANTHROPIC_API_KEY") or "dummy"

if not HF_TOKEN:
    raise ValueError("Add HF_TOKEN to Colab Secrets (key icon in sidebar)")

os.environ["HF_TOKEN"]          = HF_TOKEN
os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

from huggingface_hub import login
login(token=HF_TOKEN, add_to_git_credential=False)
print("HuggingFace authenticated.")

# %% [code] — Copy project from Drive + patch paths
import shutil, json, time
from pathlib import Path

# ── CONFIGURE IF YOUR PATHS DIFFER ──────────────────────────────────────────
CODE_ZIP    = "/content/drive/MyDrive/film_factory/film_factory/film_factory_code.zip"
OUTPUTS_ZIP = "/content/drive/MyDrive/film_factory/film_factory/film_factory_outputs.zip"

# Your local Windows project root — exactly as stored in context.json paths
WINDOWS_ROOT = "C:\\Users\\tejap\\OneDrive\\Desktop\\film_factory\\"
COLAB_ROOT   = "/content/film_factory/"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs("/content/film_factory", exist_ok=True)

print("Extracting code...")
shutil.unpack_archive(CODE_ZIP, "/content/film_factory")

print("Extracting outputs (images, audio, context)...")
shutil.unpack_archive(OUTPUTS_ZIP, "/content/film_factory")

# Patch Windows absolute paths → Linux paths in context.json
# (agent_5a re-discovers images by filename, but video generator needs correct paths)
ctx_path = Path("/content/film_factory/outputs/context.json")
if ctx_path.exists():
    with open(ctx_path, encoding="utf-8") as f:
        ctx = json.load(f)

    def _patch(obj):
        if isinstance(obj, str):
            # Windows backslash form
            v = obj.replace(WINDOWS_ROOT, COLAB_ROOT)
            # Windows forward-slash form
            v = v.replace(WINDOWS_ROOT.replace("\\", "/"), COLAB_ROOT)
            # Normalize remaining backslashes to forward slashes
            if v.startswith(COLAB_ROOT):
                v = v.replace("\\", "/")
            return v
        elif isinstance(obj, dict):
            return {k: _patch(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_patch(v) for v in obj]
        return obj

    ctx = _patch(ctx)
    with open(ctx_path, "w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=2)
    print("context.json: Windows paths → Linux paths patched")
else:
    print("WARNING: context.json not found — will generate from scratch")

print(f"\nProject ready at /content/film_factory")

# %% [code] — Copy Wan model from Drive SSD cache
import torch, gc

DRIVE_CACHE = "/content/drive/MyDrive/film_factory/models/hf_cache"
LOCAL_CACHE = "/content/hf_cache"

# Switch model ID here to try Wan2.2 when available: "Wan-AI/Wan2.2-I2V-14B-480P"
WAN_ID      = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
WAN_MARKER  = f"{LOCAL_CACHE}/hub/models--" + WAN_ID.replace("/", "--").replace(".", "-")

os.makedirs(LOCAL_CACHE, exist_ok=True)
os.environ["HF_HOME"] = LOCAL_CACHE

gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu_name} — {vram_gb:.1f} GB VRAM")

if os.path.isdir(WAN_MARKER):
    print("Wan model: local SSD cache hit — will load in ~30s")
else:
    drive_wan = f"{DRIVE_CACHE}/hub/models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers"
    if os.path.isdir(drive_wan):
        print("Copying Wan (90 GB) from Drive → SSD. This takes ~15 min once per session...")
        t0 = time.time()
        shutil.copytree(DRIVE_CACHE, LOCAL_CACHE, dirs_exist_ok=True)
        print(f"Copy done in {(time.time()-t0)/60:.1f} min")
    else:
        print("Wan not on Drive — will download from HuggingFace (~90 GB, ~30 min).")
        print("It will be saved to Drive after loading for future sessions.")

# %% [code] — Flask server (Wan async + MusicGen, localhost only)
import io, base64, threading, traceback, uuid, wave
import numpy as np
from flask import Flask, request, jsonify

app   = Flask(__name__)
_lock = threading.Lock()   # serialises GPU operations

wan_pipe          = None
music_model_cache = None


def load_wan():
    global wan_pipe
    if wan_pipe is not None:
        return wan_pipe

    print(f"Loading {WAN_ID}...", flush=True)
    from diffusers import WanImageToVideoPipeline

    try:
        # Newer diffusers (>=0.34) handle CLIPVisionModel internally
        wan_pipe = WanImageToVideoPipeline.from_pretrained(
            WAN_ID, torch_dtype=torch.bfloat16,
        ).to("cuda")
    except Exception as direct_err:
        print(f"Direct load failed ({direct_err}), trying explicit CLIPVisionModel...", flush=True)
        from transformers import CLIPVisionModel
        image_encoder = CLIPVisionModel.from_pretrained(
            WAN_ID, subfolder="image_encoder", torch_dtype=torch.bfloat16,
        )
        wan_pipe = WanImageToVideoPipeline.from_pretrained(
            WAN_ID, image_encoder=image_encoder, torch_dtype=torch.bfloat16,
        ).to("cuda")

    # Attention slicing processes heads sequentially — reduces peak VRAM during
    # inference from ~70GB to ~45GB on an A100-80GB, preventing OOM on 33-frame clips.
    wan_pipe.enable_attention_slicing("max")
    try:
        wan_pipe.enable_vae_slicing()
    except AttributeError:
        pass

    # ftfy must be in the pipeline module's namespace (Wan2.1 requirement)
    try:
        import ftfy, diffusers.pipelines.wan.pipeline_wan_i2v as _wan_mod
        _wan_mod.ftfy = ftfy
    except Exception:
        pass

    print(f"Wan ready. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    # Save to Drive on first download
    drive_wan = f"{DRIVE_CACHE}/hub/models--" + WAN_ID.replace("/", "--").replace(".", "-")
    if not os.path.isdir(drive_wan) and os.path.isdir(WAN_MARKER):
        print("Syncing Wan model to Drive (one time only)...", flush=True)
        shutil.copytree(LOCAL_CACHE, DRIVE_CACHE, dirs_exist_ok=True)
        print("Wan synced to Drive.", flush=True)

    return wan_pipe


def load_music():
    global music_model_cache, wan_pipe
    if music_model_cache is not None:
        return music_model_cache

    # Unload Wan to reclaim VRAM for MusicGen
    if wan_pipe is not None:
        print("Unloading Wan to free VRAM for MusicGen...")
        del wan_pipe; wan_pipe = None
        gc.collect(); torch.cuda.empty_cache()

    print("Loading MusicGen-Large (~3.3B)...")
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
    model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-large", torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()
    music_model_cache = {"processor": processor, "model": model,
                         "sample_rate": model.config.audio_encoder.sampling_rate}
    print(f"MusicGen-Large ready. SR={music_model_cache['sample_rate']}")
    return music_model_cache


# ── Async video job state ─────────────────────────────────────────────────────
_jobs      = {}
_jobs_lock = threading.Lock()


def _run_video_job(job_id, pil_image, prompt, negative_prompt,
                   num_frames, guidance_scale, flow_shift, num_inference_steps):
    import tempfile
    from diffusers.utils import export_to_video
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
        with _lock:
            vp = load_wan()
            # flow_shift is a scheduler config, not a __call__ kwarg in diffusers <=0.36.
            # Set it on the scheduler before calling, restore after.
            _orig_shift = getattr(vp.scheduler, "shift", None)
            try:
                vp.scheduler.shift = flow_shift
            except Exception:
                pass  # scheduler doesn't support shift — proceed without it
            frames = vp(
                image=pil_image, prompt=prompt, negative_prompt=negative_prompt,
                height=480, width=832, num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            ).frames[0]
            try:
                if _orig_shift is not None:
                    vp.scheduler.shift = _orig_shift
            except Exception:
                pass
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        export_to_video(frames, tmp_path, fps=16)
        with open(tmp_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        os.unlink(tmp_path)
        with _jobs_lock:
            _jobs[job_id] = {"status": "done", "video_base64": video_b64,
                             "num_frames": num_frames, "fps": 16}
    except Exception:
        tb = traceback.format_exc()
        print(f"\n[WAN ERROR] job {job_id}:\n{tb}", flush=True)
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "error": tb}


@app.route("/generate_video_async", methods=["POST"])
def generate_video_async():
    try:
        from PIL import Image
        data = request.get_json(force=True)
        pil_image  = Image.open(io.BytesIO(base64.b64decode(data["image_base64"]))).convert("RGB")
        prompt              = data["prompt"]
        negative_prompt     = data.get("negative_prompt",
            "bright tones, overexposed, still picture, static, blurred details, "
            "low quality, worst quality, JPEG artifacts, watermark, text, "
            "fast motion, camera shake, morphing, warping, flickering, temporal artifacts, "
            "person, human, face, figure")
        num_frames          = int(data.get("num_frames", 81))     # 480P model may hardcode 81 (Issue #73)
        guidance_scale      = float(data.get("guidance_scale", 5.0))  # official I2V default
        flow_shift          = float(data.get("flow_shift", 4.0))   # >3.0 breaks static-lock-in
        num_inference_steps = int(data.get("num_inference_steps", 40))  # official I2V rec (not 50)

        job_id = str(uuid.uuid4())
        with _jobs_lock:
            _jobs[job_id] = {"status": "queued"}
        threading.Thread(
            target=_run_video_job,
            args=(job_id, pil_image, prompt, negative_prompt,
                  num_frames, guidance_scale, flow_shift, num_inference_steps),
            daemon=True,
        ).start()
        return jsonify({"job_id": job_id}), 202
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/generate_video_status/<job_id>", methods=["GET"])
def generate_video_status(job_id):
    with _jobs_lock:
        job = dict(_jobs.get(job_id, {"status": "not_found"}))
    if job.get("status") == "done":
        with _jobs_lock:
            _jobs.pop(job_id, None)
    return jsonify(job)


@app.route("/generate_video", methods=["POST"])
def generate_video():
    """Synchronous endpoint — no timeout risk on localhost."""
    import tempfile
    from PIL import Image
    from diffusers.utils import export_to_video
    try:
        data = request.get_json(force=True)
        pil_image  = Image.open(io.BytesIO(base64.b64decode(data["image_base64"]))).convert("RGB")
        prompt     = data["prompt"]
        neg_prompt = data.get("negative_prompt",
            "bright tones, overexposed, still picture, static, blurred details, "
            "low quality, worst quality, fast motion, camera shake, morphing, "
            "warping, flickering, temporal artifacts, person, human, face, figure, "
            "text, watermark")
        num_frames          = int(data.get("num_frames", 81))
        guidance            = float(data.get("guidance_scale", 5.0))
        flow_shift          = float(data.get("flow_shift", 4.0))
        num_inference_steps = int(data.get("num_inference_steps", 40))

        with _lock:
            vp = load_wan()
            _orig_shift = getattr(vp.scheduler, "shift", None)
            try:
                vp.scheduler.shift = flow_shift
            except Exception:
                pass
            frames = vp(
                image=pil_image, prompt=prompt, negative_prompt=neg_prompt,
                height=480, width=832, num_frames=num_frames, guidance_scale=guidance,
                num_inference_steps=num_inference_steps,
            ).frames[0]
            try:
                if _orig_shift is not None:
                    vp.scheduler.shift = _orig_shift
            except Exception:
                pass

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        export_to_video(frames, tmp_path, fps=16)
        with open(tmp_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        os.unlink(tmp_path)
        return jsonify({"video_base64": video_b64, "num_frames": num_frames, "fps": 16})
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/generate_music", methods=["POST"])
def generate_music():
    try:
        data           = request.get_json(force=True)
        prompt         = data["prompt"]
        duration_s     = min(30.0, max(1.0, float(data.get("duration_seconds", 20))))
        guidance_scale = float(data.get("guidance_scale", 4.5))
        seed           = int(data.get("seed", -1))
        max_new_tokens = int(duration_s * 51.2)

        with _lock:
            md          = load_music()
            processor   = md["processor"]
            model       = md["model"]
            sample_rate = md["sample_rate"]
            generator   = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None
            inputs      = processor(text=[prompt], padding=True, return_tensors="pt")
            inputs      = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                audio_values = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=True, guidance_scale=guidance_scale, generator=generator,
                )

        audio_np = audio_values[0, 0].cpu().float().numpy()
        max_val  = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val * 0.85
        audio_int16 = (audio_np * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(sample_rate); wf.writeframes(audio_int16.tobytes())
        return jsonify({
            "audio_base64": base64.b64encode(buf.getvalue()).decode(),
            "sample_rate": sample_rate,
            "duration": len(audio_np) / sample_rate,
            "model": "musicgen-large",
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "gpu": torch.cuda.get_device_name(0)})

# %% [code] — Start server + run orchestrator
import requests as _req

PORT = 5000
threading.Thread(
    target=lambda: app.run(host="0.0.0.0", port=PORT, use_reloader=False, debug=False, threaded=True),
    daemon=True,
).start()
time.sleep(2)

# Verify server is up
try:
    resp = _req.get(f"http://127.0.0.1:{PORT}/health", timeout=5)
    print(f"Server: {resp.json()}")
except Exception as e:
    print(f"WARNING: Server not responding: {e}")

# Set env for orchestrator subprocess — all generation points to localhost
env = {
    **os.environ,
    "IMAGE_SOURCE":           "kaggle",      # images are cached; agent_5a will skip them
    "KAGGLE_IMAGE_URL":       f"http://127.0.0.1:{PORT}",
    "VIDEO_SOURCE":           "kaggle",
    "KAGGLE_VIDEO_URL":       f"http://127.0.0.1:{PORT}",
    "MUSIC_SOURCE":           "kaggle",
    "KAGGLE_MUSIC_URL":       f"http://127.0.0.1:{PORT}",
    "VOICE_SOURCE":           "colab",       # tells orchestrator to patch audio path
    "HF_HOME":                LOCAL_CACHE,
    "HF_TOKEN":               HF_TOKEN,
    "ANTHROPIC_API_KEY":      ANTHROPIC_API_KEY,
    "HF_API_TOKEN":           "dummy-not-used-images-are-cached",
    "REUSE_EXISTING_IMAGES":  "true",        # skip deletion+regen; images are in outputs zip
}

print("\nStarting pipeline (resume from images: video → music → compose)...")
print("=" * 65)

result = subprocess.run(
    [sys.executable, "-u", "orchestrator.py",
     "--resume", "outputs/context.json",
     "--resume-from", "images"],
    cwd="/content/film_factory",
    env=env,
)

print("=" * 65)
print(f"Orchestrator exit code: {result.returncode}")

# %% [code] — Sync outputs to Drive
# Must match the double-nested path where the code/output zips live.
DRIVE_OUTPUTS = "/content/drive/MyDrive/film_factory/film_factory/outputs"
print(f"Syncing /content/film_factory/outputs → {DRIVE_OUTPUTS} ...")
os.makedirs(DRIVE_OUTPUTS, exist_ok=True)
shutil.copytree("/content/film_factory/outputs", DRIVE_OUTPUTS, dirs_exist_ok=True)
print("Done.")
print(f"  Final video: {DRIVE_OUTPUTS}/final/episode_1.mp4")
print(f"  context.json: {DRIVE_OUTPUTS}/context.json")
