import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "Qwen/Qwen-Image", torch_dtype=torch.bfloat16, device_map="mps"
)
pipeline.load_lora_weights(
    "flymy-ai/qwen-image-realism-lora",
)

prompt = """
super Realism cinematic film still of a cat sipping a margarita in a pool in Palm Springs in the style of umempart, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""

image = pipeline(prompt).images[0]
image.save("output_lora.png")
print("Image saved to output_lora.png")
