import base64
import io
import threading

import torch
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles

from diffusiongen2.utils.paths import PROJECT_ROOT
from diffusiongen2.models.diffusion import DiffusionModel
from scripts.inference import load_inference_config, load_model_from_checkpoint, load_vae, generate_latent


app = FastAPI(title="DiffusionGen2 Inference API")
app.mount(
    "/static",
    StaticFiles(directory=PROJECT_ROOT / "frontend/static_v2"),
    name="static",
)

templates = Jinja2Templates(directory="frontend/templates_v2")


raw_model: DiffusionModel | None = None
ema_model: DiffusionModel | None = None
vae: AutoencoderKL | None = None
clip_tokenizer: CLIPTokenizer | None = None
clip_text_model: CLIPTextModel | None = None
inf_cfg: dict | None = None
device: str | None = None
dtype: torch.dtype | None = None

generation_progress: float = 0.0
latest_preview_image: str | None = None
preview_step: int | None = None
total_steps: int | None = None

generation_lock = threading.Lock()


class GenerateRequest(BaseModel):
    batch_size: int
    prompt: str
    negative_prompt: str
    steps: int
    cfg: float
    seed: int | None
    use_ema: bool
    use_ddpm: bool
    use_real_esrgan: bool
    real_time_denoising: bool
    preview_interval: int


class GenerateResponse(BaseModel):
    images: list[str]  # base64-encoded PNGs
    seed: int | None


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.on_event("startup")
def startup():
    global raw_model, ema_model, vae, clip_tokenizer, clip_text_model, inf_cfg, device, dtype

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

    inf_cfg_path = PROJECT_ROOT / "configs/inference/default.yaml"
    inf_cfg = load_inference_config(inf_cfg_path)

    raw_model = load_model_from_checkpoint(
        checkpoint_paths=inf_cfg["checkpoint_path"],
        use_ema=False,
        device=device,
    )

    ema_model = load_model_from_checkpoint(
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


def set_progress(p: float):
    global generation_progress
    generation_progress = p


def preview_callback(decoded_image: Image, step: int):
    global latest_preview_image, preview_step
    latest_preview_image = pil_to_base64(decoded_image)
    preview_step = step


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    global generation_progress, latest_preview_image, preview_step, total_steps

    total_steps = req.steps
    latest_preview_image = None
    preview_step = None
    generation_progress = 0.0

    if raw_model is None or ema_model is None or vae is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt_embd = tokenize_prompt(req.prompt)
    negative_embd = tokenize_prompt(req.negative_prompt)

    with generation_lock:

        images = generate_latent(
            model=ema_model if req.use_ema else raw_model,
            vae=vae,
            prompt=prompt_embd,
            negative_prompt=negative_embd,
            cfg=req.cfg,
            T=raw_model.T,
            batch_size=req.batch_size,
            latent_dim=inf_cfg["latent_dim"],
            use_ddpm=req.use_ddpm,
            num_steps=req.steps,
            seed=req.seed,
            device=device,
            preview_interval=max(1, req.preview_interval),
            preview_callback=preview_callback,
            progress_callback=set_progress
        )

    generation_progress = 1.0
    encoded_images = [pil_to_base64(img) for img in images]

    return GenerateResponse(
        images=encoded_images,
        seed=req.seed,
    )


@app.get("/health")
def health():
    model_loaded = raw_model is not None and ema_model is not None
    vae_loaded = vae is not None
    clip_loaded = clip_text_model is not None and clip_tokenizer is not None
    return {
        "status": "ok" if (model_loaded and vae_loaded and clip_loaded and device) else "error",
        "device": device,
        "model_loaded": model_loaded,
        "vae_loaded": vae_loaded,
        "clip_loaded": clip_loaded,
    }


@app.get("/progress")
def progress():
    return {
        "progress": int(generation_progress * 100)
    }


@app.get("/preview")
def preview():
    return {
        "image": latest_preview_image,
        "step": preview_step,
        "total_steps": total_steps
    }
