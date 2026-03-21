from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO
import os
from pathlib import Path
from datetime import datetime

import requests

# ==============================
# 🔐 CONFIG
# ==============================
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
IMAGES_DIR = BASE_DIR / "images"
IMAGE_INVOKE_URL = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium"


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
API_KEY = os.getenv("API_KEY") or os.getenv("api_key")

if not API_KEY:
    raise ValueError(
        "Missing API key. Add API_KEY or api_key to mini-chatgpt/.env or set it "
        "as an environment variable before running the script."
    )

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY
)

# Create images folder
IMAGES_DIR.mkdir(exist_ok=True)

# ==============================
# 🧠 LLM (ChatGPT-like)
# ==============================
def chat_with_llm(messages):
    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=messages
    )
    return response.choices[0].message.content


# ==============================
# 🎨 Image Generator (Midjourney-like)
# ==============================
def generate_image(prompt):
    response = requests.post(
        IMAGE_INVOKE_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "application/json",
        },
        json={
            "prompt": prompt,
            "cfg_scale": 5,
            "aspect_ratio": "1:1",
            "seed": 0,
            "steps": 50,
            "negative_prompt": "",
        },
        timeout=120,
    )

    response.raise_for_status()
    data = response.json()

    image_base64 = None
    if isinstance(data, dict) and "image" in data:
        image_base64 = data.get("image")
    elif isinstance(data, dict) and "artifacts" in data:
        artifacts = data.get("artifacts")
        if artifacts:
            image_base64 = artifacts[0].get("base64")

    if not image_base64:
        raise RuntimeError(f"Image data not found in API response: {data}")

    image_bytes = base64.b64decode(image_base64)

    image = Image.open(BytesIO(image_bytes))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = IMAGES_DIR / f"img_{timestamp}.png"
    image.save(file_path)

    return str(file_path)


# ==============================
# ✨ Prompt Enhancer
# ==============================
def enhance_prompt(prompt):
    messages = [
        {"role": "system", "content": "Make this prompt highly detailed for image generation"},
        {"role": "user", "content": prompt}
    ]
    return chat_with_llm(messages)


# ==============================
# 🎭 Style Engine
# ==============================
def apply_style(prompt, style):
    styles = {
        "anime": "anime style, vibrant colors",
        "realistic": "photorealistic, ultra detailed",
        "cinematic": "cinematic lighting, dramatic shadows"
    }
    return prompt + ", " + styles.get(style, "")


# ==============================
# 🤖 Agent Decision (Brain)
# ==============================
def agent_decision(user_input):
    system_prompt = """
    You are an AI agent.

    Decide:
    - If user asks to create/generate/draw image → return:
      IMAGE: <enhanced prompt>

    - Otherwise → return:
      TEXT: <normal answer>

    Only respond in this exact format.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    return chat_with_llm(messages)


# ==============================
# 🔁 Agent Controller
# ==============================
def run_agent(user_input):

    decision = agent_decision(user_input)

    print("\n🧠 Agent Decision:", decision)

    if decision.startswith("IMAGE:"):
        prompt = decision.replace("IMAGE:", "").strip()

        # Enhance prompt
        enhanced = enhance_prompt(prompt)

        # Optional: apply style
        styled_prompt = apply_style(enhanced, "cinematic")

        print("🎨 Final Prompt:", styled_prompt)

        image_path = generate_image(styled_prompt)

        return {
            "mode": "image",
            "text": f"Image generated for: {prompt}",
            "image_path": image_path,
            "final_prompt": styled_prompt,
        }

    elif decision.startswith("TEXT:"):
        return {
            "mode": "text",
            "text": decision.replace("TEXT:", "").strip(),
            "image_path": None,
            "final_prompt": None,
        }

    else:
        return {
            "mode": "unknown",
            "text": "Agent couldn't understand.",
            "image_path": None,
            "final_prompt": None,
        }


# ==============================
# 🚀 MAIN LOOP
# ==============================
if __name__ == "__main__":

    print("🤖 AI Agent Started (ChatGPT + Midjourney style)")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        response = run_agent(user_input)
        print("AI:", response["text"])
        if response["image_path"]:
            print("Image:", response["image_path"])
