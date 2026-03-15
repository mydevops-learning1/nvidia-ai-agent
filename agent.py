from pathlib import Path
import traceback

import image_generator
from image_generator import generate_image

DEBUG_FILE = Path(__file__).resolve().parent / "runtime_debug.txt"

class ImageAgent:

    def __init__(self):
        print("Image Agent Ready")
        agent_path = Path(__file__).resolve()
        generator_path = Path(image_generator.__file__).resolve()
        debug_text = (
            f"agent.py={agent_path}\n"
            f"image_generator.py={generator_path}\n"
        )
        DEBUG_FILE.write_text(debug_text, encoding="utf-8")
        print("Agent file:", agent_path)
        print("Generator file:", generator_path)
        print("Debug file:", DEBUG_FILE)

    def run(self, user_prompt):
        print("User Prompt:", user_prompt)
        image_path = generate_image(user_prompt)
        print("Image created:", image_path)
        print("Saved to:", image_path)
        return image_path


if __name__ == "__main__":
    agent = ImageAgent()
    user_input = input("Describe the image you want: ").strip()

    if not user_input:
        print("Please enter a valid prompt.")
    else:
        try:
            agent.run(user_input)
        except Exception as error:
            with DEBUG_FILE.open("a", encoding="utf-8") as debug_file:
                debug_file.write(f"agent_exception_type={type(error).__name__}\n")
                debug_file.write(f"agent_exception_message={error}\n")
                debug_file.write(traceback.format_exc())
                debug_file.write("\n")
            print("Failed to generate image:", error)
