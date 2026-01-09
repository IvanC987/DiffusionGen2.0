import base64
import io
import threading
import random
import numpy as np

import faiss
import torch
import torchvision
from torchvision import transforms
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
from realesrgan import RealESRGANer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles

from diffusiongen2.utils.paths import PROJECT_ROOT
from diffusiongen2.models.diffusion import DiffusionModel
from scripts.inference import load_inference_config, load_model_from_checkpoint, load_vae, load_real_esrgan, generate_latent


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
upsampler: RealESRGANer | None = None
inf_cfg: dict | None = None
device: str | None = None
dtype: torch.dtype | None = None

generation_progress: float = 0.0
latest_preview_image: str | None = None
preview_step: int | None = None
total_steps: int | None = None

train_prompt_bank: dict | None = None
val_prompt_bank: dict | None = None
train_vec_db: faiss.IndexFlatIP | None = None
val_vec_db: faiss.IndexFlatIP | None = None
merged_train_prompt_list: list[str] | None = None
merged_val_prompt_list: list[str] | None = None

generation_lock = threading.Lock()


class GenerateRequest(BaseModel):
    batch_size: int
    prompt: str
    second_prompt: str
    negative_prompt: str
    steps: int
    cfg: float
    seed: int | None
    use_ema: bool
    use_ddpm: bool
    use_real_esrgan: bool
    real_time_denoising: bool
    preview_interval: int
    b64_image: str | None
    img2img_strength: float
    b64_inpaint_image: str | None
    b64_inpaint_mask: str | None
    use_prompt_matching: bool
    prompt_match_train: bool
    prompt_match_val: bool


class GenerateResponse(BaseModel):
    images: list[str]  # base64-encoded PNGs
    prompt: str  # Prompt used to generate the image(s)
    seed: int | None


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.on_event("startup")
def startup():
    global raw_model, ema_model, vae, clip_tokenizer, clip_text_model, inf_cfg, device, dtype, \
        train_prompt_bank, val_prompt_bank, train_vec_db, val_vec_db, merged_train_prompt_list, merged_val_prompt_list

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

    # Don't want to load Real ESRGAN here by default actually, there's some additional complexities going on there to use
    # Only load if users explicitly checked the box to use

    # Prompt banks have the structure of:
    # {
    #     "dragon": [list[str], Tensor[N, 512]],
    #     "fairy": [...],
    #     ...
    # }
    train_prompt_bank = torch.load(inf_cfg["train_prompt_bank"], map_location="cpu")
    val_prompt_bank = torch.load(inf_cfg["val_prompt_bank"], map_location="cpu")

    # Extract and merge into a single list of prompts
    merged_train_prompt_list = [v[0] for v in train_prompt_bank.values()]
    merged_train_prompt_list = [p for sublist in merged_train_prompt_list for p in sublist]
    merged_val_prompt_list = [v[0] for v in val_prompt_bank.values()]
    merged_val_prompt_list = [p for sublist in merged_val_prompt_list for p in sublist]

    train_db_tensor = [v[1] for v in train_prompt_bank.values()]
    train_db_tensor = torch.cat(train_db_tensor, dim=0)  # shape=(N, 512), here N is the total num embedding sum along all categories
    train_db_tensor = train_db_tensor.detach().cpu().float().numpy()
    faiss.normalize_L2(train_db_tensor)  # Should already be normalized, just a safeguard

    val_db_tensor = [v[1] for v in val_prompt_bank.values()]
    val_db_tensor = torch.cat(val_db_tensor, dim=0)
    val_db_tensor = val_db_tensor.detach().cpu().float().numpy()
    faiss.normalize_L2(val_db_tensor)

    train_vec_db = faiss.IndexFlatIP(512)
    train_vec_db.add(train_db_tensor)

    val_vec_db = faiss.IndexFlatIP(512)
    val_vec_db.add(val_db_tensor)


def pil_to_base64(img: Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_pil(b64: str) -> Image.Image:
    image_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


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


def upscale_image(images: list[Image]) -> list[Image]:
    upscaled_images = []

    for img in images:
        img_np = np.array(img)
        with torch.no_grad():
            output, _ = upsampler.enhance(img_np, outscale=2)

        # Rescaling it back down to keep original shape for front-end GUI. Might come back to accommodate dif res later on
        upscaled_img = Image.fromarray(output).resize(images[0].size, resample=Image.Resampling.BICUBIC)
        upscaled_images.append(upscaled_img)

    return upscaled_images


def get_init_latent(b64_image: str, batch_size: int):
    b64_pil = base64_to_pil(b64_image)
    b64_pil = b64_pil.resize((inf_cfg["latent_dim"], inf_cfg["latent_dim"]), Image.Resampling.BICUBIC)

    # Convert to torch tensor, shape (C, H, W) and range [0, 1]
    img_tensor = transforms.ToTensor()(b64_pil)
    img_tensor = img_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # Shape (batch_size, C, H, W)
    img_tensor = ((img_tensor * 2) - 1).to(device)  # Shift range to [-1, 1] and move device

    with torch.autocast(device_type=device, dtype=dtype):
        init_latent = vae.encode(img_tensor).latent_dist.sample()

    return init_latent * 0.18215


def prompt_match(prompt: str, db: faiss.IndexFlatIP, prompt_list: list[str]):
    with torch.no_grad(), torch.autocast(dtype=dtype, device_type=device):
        tokens = clip_tokenizer([prompt], return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(device)
        text_embd = clip_text_model(**tokens).pooler_output
        text_embd = torch.nn.functional.normalize(text_embd, dim=-1)

    # E.g.
    # scores=[[0.7709671  0.76750416 0.7666029  0.76368684 0.76290154]]
    # indices=[[28 37 98 34  9]]
    scores, indices = db.search(text_embd, k=inf_cfg["faiss_k"])

    # Since the top k scores are all quite similar, need lower temp to meaningfully differentiate.
    # Else it would be nearly uniform sample. 0.25 is a heuristic. Feel free to play around with the value
    temperature = 0.25
    normalized_scores = torch.softmax(torch.tensor(scores[0], dtype=torch.float32) / temperature, dim=0)

    # Grab the appropriate index
    chosen_index = indices[0][torch.multinomial(normalized_scores, num_samples=1).item()]

    return tokenize_prompt(prompt_list[chosen_index]), prompt_list[chosen_index]


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    global generation_progress, latest_preview_image, preview_step, total_steps, upsampler

    generation_progress = 0.0
    latest_preview_image = None
    preview_step = 0
    total_steps = req.steps

    if raw_model is None or ema_model is None or vae is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if req.use_prompt_matching:
        if req.prompt_match_train == req.prompt_match_val:
            raise HTTPException(status_code=400, detail="Exactly one of prompt_match_train or prompt_match_val must be true")

        prompt_embd, return_prompt = prompt_match(prompt=req.prompt,
                                                  db=train_vec_db if req.prompt_match_train else val_vec_db,
                                                  prompt_list=merged_train_prompt_list if req.prompt_match_train else merged_val_prompt_list)
    else:
        prompt_embd = tokenize_prompt(req.prompt)
        return_prompt = req.prompt

    negative_embd = tokenize_prompt(req.negative_prompt)

    if req.b64_image is not None:
        init_latent = get_init_latent(req.b64_image, req.batch_size)
    elif req.b64_inpaint_image is not None:
        init_latent = get_init_latent(req.b64_inpaint_image, req.batch_size)
    else:
        init_latent = None

    if req.b64_inpaint_mask is not None:
        pil_mask = base64_to_pil(req.b64_inpaint_mask).convert("L")

        # Resize to 1/8 original size for VAE compatibility and also resample using nearest. Want only 0 and 1 values
        pil_mask = pil_mask.resize((inf_cfg["latent_dim"], inf_cfg["latent_dim"]), Image.Resampling.NEAREST)

        # Convert to torch tensor, shape (C, H, W) and range [0, 1]
        latent_mask = transforms.ToTensor()(pil_mask)
        # Blur for soft mask instead of hard/binary mask, clamp to [0, 1] then adjust dim accordingly
        latent_mask = torchvision.transforms.GaussianBlur(kernel_size=7, sigma=2.0)(latent_mask).clamp(0, 1)
        latent_mask = latent_mask.unsqueeze(0).repeat(req.batch_size, 4, 1, 1).to(device)
    else:
        latent_mask = None

    with generation_lock:
        images = generate_latent(
            model=ema_model if req.use_ema else raw_model,
            vae=vae,
            prompt=prompt_embd,
            second_prompt=tokenize_prompt(req.second_prompt),
            negative_prompt=negative_embd,
            init_latent=init_latent,
            img2img_strength=req.img2img_strength,
            latent_mask=latent_mask,
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

    if req.use_real_esrgan:  # Go into upsampling
        print("Now upsampling...")
        if device == "cuda":  # Free cache
            torch.cuda.empty_cache()

        if upsampler is None:  # Load it in if first time
            upsampler = load_real_esrgan(model_path=inf_cfg["real_esrgan_path"], device=device)

        images = upscale_image(images)  # Upscale image

    encoded_images = [pil_to_base64(img) for img in images]
    generation_progress = 1.0

    return GenerateResponse(
        images=encoded_images,
        prompt=return_prompt,
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


@app.get("/editor")
def editor(request: Request):
    return templates.TemplateResponse(
      "editor.html",
      {"request": request}
    )


@app.get("/random_prompt")
def random_prompt(
    split: str,
    category: str = "any"
):
    if split not in ["train", "val"]:
        raise HTTPException(status_code=400, detail="Invalid split")

    bank = train_prompt_bank if split == "train" else val_prompt_bank

    if category == "any":
        category = random.choice(list(bank.keys()))

    if category not in bank and category != "any":
        raise HTTPException(status_code=400, detail="Invalid category")

    prompts, _ = bank[category]
    prompt = random.choice(prompts)

    return {
        "prompt": prompt,
    }
