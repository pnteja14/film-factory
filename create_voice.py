import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

VOICE_PROMPT = (
    "A male voice in his late 40s. Extremely deep bass with maximum chest resonance. "
    "Slow, deliberate pace with natural pauses between sentences. Dark, magnetic, and authoritative. "
    "Tobacco-worn texture. Occasionally a dry, mirthless chuckle — not humor, but the quiet amusement "
    "of someone who already knows how the story ends. No emotional brightness. Suitable for true crime "
    "documentary narration — cold, certain, and unsettling. Like a detective who has seen too much and "
    "feels nothing anymore. Speaks as if every word costs something."
)

PREVIEW_TEXT = (
    "Three patients. Three organs. All on his shift. "
    "I told myself it was coincidence. "
    "I was wrong."
)

headers = {
    "Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY')}",
    "Content-Type": "application/json"
}

payload = {
    "model": "qwen-voice-design",
    "input": {
        "action": "create",
        "target_model": "qwen3-tts-vd-2026-01-26",
        "preferred_name": "narrator02",
        "voice_prompt": VOICE_PROMPT,
        "preview_text": PREVIEW_TEXT,
        "language": "en"
    },
    "parameters": {
        "sample_rate": 24000,
        "response_format": "wav"
    }
}

response = requests.post(
    "https://dashscope-intl.aliyuncs.com/api/v1/services/audio/tts/customization",
    headers=headers,
    json=payload,
    timeout=60
)

print(f"Status: {response.status_code}")
result = response.json()
print(f"Response: {result}")

if response.status_code == 200:
    voice_id = result.get("output", {}).get("voice")
    print(f"Voice ID: {voice_id}")
    
    # Save preview audio
    audio_url = result.get("output", {}).get("audio", {}).get("url")
    if audio_url:
        import requests as r
        audio = r.get(audio_url)
        with open("outputs/narrator_v2_preview.wav", "wb") as f:
            f.write(audio.content)
        print("Preview saved: outputs/narrator_v2_preview.wav")
    
    # Update voice_id.json keeping both voices
    voice_path = "memory/voice_id.json"
    if os.path.exists(voice_path):
        with open(voice_path) as f:
            existing = json.load(f)
    else:
        existing = {}
    
    # Migrate old format if needed
    if "voice_id" in existing:
        existing["narrator_v1"] = existing.pop("voice_id")
    
    existing["narrator_v2"] = voice_id
    existing["active"] = "narrator_v2"
    
    with open(voice_path, "w") as f:
        json.dump(existing, f, indent=2)
    
    print("Voice ID saved to memory/voice_id.json")
    print(f"Active voice set to: narrator_v2")