import base64
import os
import traceback
from datetime import datetime
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

INVOKE_URL = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "generated_images"
ENV_FILE = BASE_DIR / ".env"
LAST_OUTPUT_FILE = BASE_DIR / "last_generated_path.txt"
RUNTIME_DEBUG_FILE = BASE_DIR / "runtime_debug.txt"
LATEST_OUTPUT_FILE = BASE_DIR / "latest_generated.png"


def _debug(message):
    with RUNTIME_DEBUG_FILE.open("a", encoding="utf-8") as debug_file:
        debug_file.write(f"{message}\n")


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


def _build_headers():
    _load_env_file()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing NVIDIA_API_KEY. Add it to ai-image-agent/.env or set it "
            "as an environment variable before running the image agent."
        )

    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }


def generate_image(
    prompt,
    negative_prompt="",
    aspect_ratio="16:9",
    cfg_scale=5,
    steps=50,
    seed=0,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    _debug(f"generate_image_called_from={Path(__file__).resolve()}")
    _debug(f"base_dir={BASE_DIR}")
    _debug(f"default_output_dir={DEFAULT_OUTPUT_DIR}")

    try:
        payload = {
            "prompt": prompt.strip(),
            "cfg_scale": cfg_scale,
            "aspect_ratio": aspect_ratio,
            "seed": seed,
            "steps": steps,
            "negative_prompt": negative_prompt,
        }
        _debug("payload_built=true")

        headers = _build_headers()
        _debug("headers_built=true")

        response = requests.post(
            INVOKE_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )
        _debug(f"http_status={response.status_code}")

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            _debug(f"http_error_body={response.text[:500]}")
            raise RuntimeError(
                f"Image generation failed with status {response.status_code}: {response.text}"
            ) from exc

        data = response.json()
        _debug("response_json_parsed=true")

        # Try different response formats
        image_base64 = None
        
        # Format 1: Direct 'image' field (base64 string)
        if isinstance(data, dict) and "image" in data:
            image_base64 = data.get("image")
        # Format 2: 'artifacts' array
        elif isinstance(data, dict) and "artifacts" in data:
            artifacts = data.get("artifacts")
            if artifacts and len(artifacts) > 0:
                image_base64 = artifacts[0].get("base64")
        
        if not image_base64:
            _debug(f"unexpected_response={str(data)[:500]}")
            raise RuntimeError(f"Image data not found in API response: {data}")

        _debug(f"base64_length={len(image_base64)}")
        image_bytes = base64.b64decode(image_base64)
        _debug(f"image_bytes_length={len(image_bytes)}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _debug(f"output_dir_created={output_dir}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = (output_dir / f"generated_{timestamp}.png").resolve()
        _debug(f"timestamped_target={file_path}")
        _debug(f"latest_target={LATEST_OUTPUT_FILE}")

        # Save the image
        image = Image.open(BytesIO(image_bytes))
        image.save(file_path, "PNG")
        image.save(LATEST_OUTPUT_FILE, "PNG")
        LAST_OUTPUT_FILE.write_text(str(file_path), encoding="utf-8")

        _debug(f"saved_timestamped={file_path}")
        _debug(f"saved_latest={LATEST_OUTPUT_FILE}")
        _debug(f"timestamped_exists={file_path.exists()}")
        _debug(f"latest_exists={LATEST_OUTPUT_FILE.exists()}")
        _debug(f"timestamped_size={file_path.stat().st_size if file_path.exists() else 0}")
        _debug(f"latest_size={LATEST_OUTPUT_FILE.stat().st_size if LATEST_OUTPUT_FILE.exists() else 0}")

        if not file_path.exists() or not LATEST_OUTPUT_FILE.exists():
            raise RuntimeError(f"Image generation appeared to succeed, but no file was saved at {file_path}")

        return str(file_path)
    except Exception as exc:
        _debug(f"exception_type={type(exc).__name__}")
        _debug(f"exception_message={exc}")
        _debug("traceback_start")
        _debug(traceback.format_exc())
        _debug("traceback_end")
        raise
