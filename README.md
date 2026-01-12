# DiffusionGen2

An end-to-end latent diffusion system with advanced inference controls, prompt distribution alignment, and experimental sampling techniques.

DiffusionGen2 is a research-oriented successor to **DiffusionGen v1**, built to explore experimental extensions to text-to-image diffusion models. The focus of v2 is more towards *controllable and research inference*, while also incorporating several training-side improvements that significantly improve quality and efficiency.

Key areas explored include real-time denoising previews, prompt matching via CLIP + FAISS, image-to-image and in-painting in latent space, and experimental techniques such as visual anagrams.

---

## What's New in 2.0

### Training Improvements

* **Min-SNR Loss Weighting** - Improves gradient balance across timesteps and boosts sample quality
* **EMA Weights** - Produces more stable/smoother generations at inference time
* **Multiple Noise Schedules** - Linear, curved, and cosine schedules supported
* **Velocity Prediction (v-pred)** - Alternative objective to epsilon prediction
* **Hydra + Weights & Biases** - Configuration management and experiment tracking

Additional improvements include fixing several subtle training bugs present in v1, adding proper negative prompt support, increasing dataset diversity (≈125k unique images vs 25k in v1), and moving from a 1:1 prompt-image ratio to a 1:5 ratio to reduce overfitting and improve generalization.

---

### Inference Innovations

* **Anagram Mode** - Dual-prompt generation with 180° rotational symmetry
* **Semantic Prompt Matching** - CLIP + FAISS powered prompt alignment to reduce distribution shift
* **Real-Time Denoising Previews** - Observe intermediate denoising steps during generation
* **Image-to-Image & In-Painting** - Latent-space conditioning with soft mask blending
* **FastAPI Backend** - Inference server with REST endpoints and a web UI

---

## Project Structure

DiffusionGen2 is organized around a **single diffusion core** shared between training and inference.

* Training produces both **raw** and **EMA** checkpoints
* Inference reuses the same diffusion loop with configurable:

  * sampling strategies (DDPM / DDIM)
  * conditioning modes (txt2img / img2img / in-painting)
  * experimental extensions (prompt matching, anagrams, previews)

This keeps the system design simpler while allowing rapid experimentation.

---

## Quality & Efficiency Comparison

| Feature            | V1.0     | V2.0                                   |
| ------------------ |----------|----------------------------------------|
| Training Speed     | Baseline | ~6-7× faster                           |
| Sample Quality     | Good     | Significantly improved (Min-SNR + EMA) |
| Inference Feedback | None     | Real-time progress + previews          |
| Prompt System      | Manual   | FAISS semantic matching                |
| Special Modes      | None     | Visual Anagram                         |

**Why the speedup?**

* V1 trained ~94k images/epoch for ~380 epochs (~320 GPU hours)
* V2 trains ~290k images/epoch for ~50 epochs (~20.5 GPU hours)
* This corresponds to ~112k images/hour (v1) vs ~700k images/hour (v2)

The gains come primarily from the removal of LPIPS loss (useful, but wasn't worth the extra compute), improved implementation, and better dataset construction rather than raw hardware alone.

---

## Standout Features

### Anagram Mode (Dual-Prompt Symmetry)

Generate images that contain two coherent interpretations when rotated 180°.

```
Prompt A: "A powerful river cascades down a series of moss-covered rocky steps, creating a dynamic whitewater flow that carves through the heart of a lush, misty canyon."
Prompt B: "Towering mountain shrouded in haze sunlight barely breaks through"
```

**How it works:**
At each denoising step, the model predicts noise for both the original latent and a rotated copy conditioned on a second prompt. The predictions are rotated back and blended, enforcing symmetry over time. (Still a beta feature, would require prompts that 'semantically compatible' to work well.)

| Prompt A | Prompt B | Combined (Anagram) |
|---------|----------|--------------------|
| ![](./sample_images/AM_A.png) | ![](./sample_images/AM_B.png) | ![](./sample_images/AM_AB.png) |

---

### Semantic Prompt Matching (CLIP + FAISS)

To reduce prompt distribution shift (OOD), free-form user prompts can be mapped to a curated prompt bank that's stylistically similar to the training data.

```
User input: "volcano"
Example matched prompt:
"A volcano rises above a jagged mountain range, its peak smoldering under a twilight sky."
```

* Uses CLIP sentence embeddings
* FAISS inner-product search over curated prompt banks
* Temperature-based sampling for diversity
* Ensures prompts stay stylistically aligned with training distribution

| Without Prompt Matching | With Prompt Matching |
|-------------------------|----------------------|
| ![](./sample_images/PM_WO.png) | ![](./sample_images/PM_W.png) |


---

### Real-Time Denoising Preview

Intermediate denoising states can be periodically decoded and streamed during inference, allowing users to observe how structure and details emerge step by step.

This is intended as an inspection and research tool rather than a production feature.

<div align="center">
  <img src="./sample_images/mountain.gif" width="256"/>
  <br/>
  <em>
    Prompt: "A lonely glacier clings to a mountain as the last light of twilight fades under heavy rain"
  </em>
</div>

---

### Image-to-Image & In-Painting

* Latent-space img2img with adjustable strength
* In-painting with binary or soft (Gaussian-blurred) masks
* Masked regions are regenerated while unmasked regions are preserved across timesteps

---

## Web Interface

The included web UI exposes all major features:

* txt2img / img2img / in-painting
* real-time denoising preview
* prompt matching controls
* anagram mode
* optional Real-ESRGAN upsampling

| Interface Overview | Advanced Controls |
|--------------------|-------------------|
| ![](./sample_images/UI_1.png) | ![](./sample_images/UI_2.png) |


---

## Gallery

A small selection of samples generated with different prompts.

|  |  |  |  |
|--|--|--|--|
| ![](./sample_images/Gallery1.png) | ![](./sample_images/Gallery2.png) | ![](./sample_images/Gallery3.png) | ![](./sample_images/Gallery4.png) |
| ![](./sample_images/Gallery5.png) | ![](./sample_images/Gallery6.png) | ![](./sample_images/Gallery7.png) | ![](./sample_images/Gallery8.png) |
| ![](./sample_images/Gallery9.png) | ![](./sample_images/Gallery10.png) | ![](./sample_images/Gallery11.png) | ![](./sample_images/Gallery12.png) |

---

## Running the Project

Detailed instructions for training, inference, and the web UI are provided in:

**[RUNNING.md](RUNNING.md)**

---

## Limitations

* Single-GPU inference
* Some experimental features are not mutually compatible
* Real-time preview adds overhead at high update rates
* Prompt banks are static and precomputed

---

## Future Directions

* Faster and smoother preview decoding
* Larger and dynamically updated prompt banks
* Exploring DiT-style architectures
* Multi-GPU inference support

---

## Relation to DiffusionGen v1

DiffusionGen2 builds directly on [DiffusionGen v1](https://github.com/IvanC987/DiffusionGen), reusing the core latent diffusion setup while substantially expanding:

* inference controllability
* experimental sampling techniques
* project modularity

Where v1 focused on establishing a baseline system, v2 emphasizes research flexibility and inspection.

---

## Acknowledgments

**Key Papers:**
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Min-SNR Loss Weighting](https://arxiv.org/abs/2303.09556)
- [Visual Anagrams: Generating Multi-View Optical Illusions with Diffusion Models](https://arxiv.org/abs/2311.17919)

**Models Used:**
- [Stable Diffusion VAE](https://huggingface.co/stabilityai/sd-vae-ft-ema)
- [CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

---

## License

MIT License (project code)
BSD 3-Clause License (third-party components)

