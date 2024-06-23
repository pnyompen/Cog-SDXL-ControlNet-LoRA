import torch
from diffusers import AutoencoderKL, DiffusionPipeline, ControlNetModel


CONTROL_CACHE = "./control-cache"
SDXL_MODEL_CACHE = "./sdxl-cache"

better_vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=better_vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.save_pretrained(SDXL_MODEL_CACHE, safe_serialization=True)

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)
controlnet.save_pretrained(CONTROL_CACHE)