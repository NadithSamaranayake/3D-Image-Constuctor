from diffusers import StableDiffusionPipeline
import torch

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "A cricket ground in the heart of Los Angeles"
image = pipe(prompt)["sample"][0]

image.save("image.png")
