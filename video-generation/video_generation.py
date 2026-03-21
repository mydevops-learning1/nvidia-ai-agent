import os
from pathlib import Path

import requests
import xai_sdk

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
OUTPUT_FILE = BASE_DIR / "generated_video.mp4"


def _load_env_file():
    if not ENV_FILE.exists():
        return

    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()
api_key = os.getenv("XAI_API_KEY")
if not api_key:
    raise ValueError(
        "Missing XAI_API_KEY. Add it to video-generation/.env or set it as an "
        "environment variable before running the script."
    )

client = xai_sdk.Client(api_key=api_key)


def _download_video(video_url, output_file):
    response = requests.get(video_url, stream=True, timeout=300)
    response.raise_for_status()

    with output_file.open("wb") as file_handle:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file_handle.write(chunk)

response = client.video.generate(
    prompt="A glowing crystal-powered rocket launching from the red dunes of Mars, ancient alien ruins lighting up in the background as it soars into a sky full of unfamiliar constellations",
    model="grok-imagine-video",
    duration=10,
    aspect_ratio="16:9",
    resolution="720p",
)

_download_video(response.url, OUTPUT_FILE)

print(response.url)
print(OUTPUT_FILE)
