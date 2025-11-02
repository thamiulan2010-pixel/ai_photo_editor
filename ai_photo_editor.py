from io import BytesIO
from flask import Flask, render_template, request, send_file
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from dotenv import load_dotenv
import os

# Load Hugging Face token
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("❌ Hugging Face token not found in .env file!")

# Model setup
model_id = "hakurei/waifu-diffusion"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    token=hf_token
).to(device)

# Disable safety checker to avoid black images
pipe.safety_checker = None

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    edited_image_url = None
    user_name_display = None
    if request.method == "POST":
        user_name = request.form.get("name", "User").strip()
        uploaded_file = request.files.get("image")

        if uploaded_file:
            init_image = Image.open(uploaded_file).convert("RGB")

            # Simple prompt to replace dress with hanbok and anime style
            prompt = "anime style, wearing traditional Korean hanbok, full body, soft lighting, clean line art"
            negative_prompt = "realistic, 3d, distorted face, blurry, low quality, unrealistic colors, moustache, extra limbs"

            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=0.55,
                guidance_scale=8.0
            ).images[0]

            # Add Korean letter in top-left corner
            draw = ImageDraw.Draw(image)
            font_size = 40
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

            korean_text = "한복"
            draw.text((20, 20), korean_text, font=font, fill="red")

            # Return image as response
            img_io = BytesIO()
            image.save(img_io, "PNG")
            img_io.seek(0)

            # Store user name for display under image
            user_name_display = user_name

            return send_file(img_io, mimetype="image/png")

    return render_template("index.html", user_name=user_name_display)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

