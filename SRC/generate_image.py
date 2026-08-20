import torch
from diffusers import StableDiffusionPipeline

# Pre-trained Stable Diffusion model
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Text prompt
prompt = "A beautiful futuristic city at night, glowing neon lights, flying cars, cinematic and highly detailed"

# Select available device
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)
print("Loading Stable Diffusion model...")

# Load the pre-trained model
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

# Move model to CPU/GPU
pipe = pipe.to(device)

print("Generating image...")

# Generate image from text prompt
image = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

# Save generated image
output_path = "GENERATED IMAGES/generated_image.png"
image.save(output_path)

print("Image generated successfully!")
print("Saved at:", output_path)