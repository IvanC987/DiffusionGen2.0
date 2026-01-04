import base64
import io
import threading

import torch
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

from diffusiongen2.utils.paths import PROJECT_ROOT
from diffusiongen2.models.diffusion import DiffusionModel
from scripts.inference import load_inference_config, load_model_from_checkpoint, load_vae, generate_latent


app = FastAPI(title="DiffusionGen2 Inference API")

model: DiffusionModel | None = None
vae: AutoencoderKL | None = None
clip_tokenizer: CLIPTokenizer | None = None
clip_text_model: CLIPTextModel | None = None
inf_cfg: dict | None = None
device: str | None = None
dtype: torch.dtype | None = None

generation_lock = threading.Lock()


class GenerateRequest(BaseModel):
    batch_size: int
    prompt: str
    negative_prompt: str
    steps: int
    cfg: float
    seed: int | None


class GenerateResponse(BaseModel):
    images: list[str]  # base64-encoded PNGs
    seed: int


@app.on_event("startup")
def startup():
    global model, vae, clip_tokenizer, clip_text_model, inf_cfg, device, dtype

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    inf_cfg_path = PROJECT_ROOT / "configs/inference/default.yaml"
    inf_cfg = load_inference_config(inf_cfg_path)

    model = load_model_from_checkpoint(
        checkpoint_paths=inf_cfg["checkpoint_path"],
        use_ema=True,
        device=device,
    )

    vae = load_vae(
        model_id=inf_cfg["vae_model_id"],
        device=device,
    )

    clip_tokenizer = CLIPTokenizer.from_pretrained(inf_cfg["clip_model_id"], clean_up_tokenization_spaces=False)
    clip_text_model = CLIPTextModel.from_pretrained(inf_cfg["clip_model_id"]).to(device)
    clip_text_model.requires_grad_(False)
    clip_text_model.eval()


def pil_to_base64(img: Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def tokenize_prompt(prompt: str) -> torch.Tensor:
    """Returns [1, 77, 512] torch tensor"""
    with torch.no_grad(), torch.autocast(dtype=dtype, device_type=device):
        tokens = clip_tokenizer([prompt], return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(device)
        text_embd = clip_text_model(**tokens).last_hidden_state.to(dtype=dtype)  # (1, 77, 512)

    return text_embd


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if model is None or vae is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt_embd = tokenize_prompt(req.prompt)
    negative_embd = tokenize_prompt(req.negative_prompt)

    with generation_lock:
        images = generate_latent(
            model=model,
            vae=vae,
            prompt=prompt_embd,
            negative_prompt=negative_embd,
            cfg=req.cfg,
            T=model.T,
            batch_size=req.batch_size,
            latent_dim=inf_cfg["latent_dim"],
            use_ddpm=True,
            num_steps=req.steps,
            seed=req.seed,
            device=device,
        )

    encoded_images = [pil_to_base64(img) for img in images]

    return GenerateResponse(
        images=encoded_images,
        seed=req.seed,
    )


@app.get("/health")
def health():
    model_loaded = model is not None
    vae_loaded = vae is not None
    clip_loaded = clip_text_model is not None and clip_tokenizer is not None
    return {
        "status": "ok" if (model_loaded and vae_loaded and clip_loaded and device) else "error",
        "device": device,
        "model_loaded": model_loaded,
        "vae_loaded": vae_loaded,
        "clip_loaded": clip_loaded,
    }
