# Running DiffusionGen2

This document contains instructions for training and running inference for **DiffusionGen2**. Both training and inference currently share the same Docker image.

---

## Prerequisites

* Docker
* NVIDIA GPU + drivers (strongly recommended)
* NVIDIA Container Toolkit (for `--gpus` support)

CPU-only execution is possible but **not recommended**, especially for training.

---

## Docker Setup

Navigate to the `docker/` directory and build the Docker image:

```bash
docker build -t <image_name> .
```

Start a container:

```bash
docker run -it \
  --name <container_name> \
  --gpus '"device=0"' \
  -p 8000:8000 \
  <image_name> /bin/bash
```

Notes:

* Omit `--gpus` for CPU-only execution
* Adjust `device=0` if using multiple GPUs

---

## Project Setup (Inside Container)

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/IvanC987/DiffusionGen2.0
cd DiffusionGen2.0

pip install -U pip setuptools
pip install -e .
```

---

## Training

### Dataset Setup

The training dataset is stored externally due to size.

Create the data directory:

```bash
mkdir -p data/river_tensor_shards_256
```

Download sample shards:

```bash
wget https://huggingface.co/datasets/DL-Hobbyist/river-tensor-shards-256/resolve/main/river_tensor_shards_256/000.pt -P data/river_tensor_shards_256
wget https://huggingface.co/datasets/DL-Hobbyist/river-tensor-shards-256/resolve/main/river_tensor_shards_256/001.pt -P data/river_tensor_shards_256
```

Set your Weights & Biases API key:

```bash
export WANDB_API_KEY=<your_api_key>
```

---

### Training (Toy Configuration)

A lightweight toy configuration is provided for testing:

```bash
python3 scripts/train.py \
  training.run_name=toy_run \
  data=toy \
  eval=toy \
  optim=toy \
  training=toy \
  unet=toy
```

Notes:

* Toy model: ~35.5M parameters
* Requires a GPU

---

### Full Training

The full model has ~354M parameters and requires substantial GPU memory.
If using your own dataset, tensor shards are expected in the same format as the provided dataset.

---

## Inference

### Model Weights

Model weights are stored externally.

Create directories:

```bash
mkdir -p outputs/model_weights
mkdir -p assets/realesrgan
```

Download the diffusion model checkpoint:

```bash
wget https://huggingface.co/DL-Hobbyist/DiffusionGen2.0/resolve/main/epoch_49_tl_0.7330.pt -P outputs/model_weights
```

(Optional) Download Real-ESRGAN weights for upsampling:

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -P assets/realesrgan
```

---

### Start Inference Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Access the web UI at:

```
http://localhost:8000
```

---

## Common Issues & Fixes

### OpenCV Missing System Libraries

If you encounter errors such as:

```text
ImportError: libGL.so.1: cannot open shared object file
ImportError: libgthread-2.0.so.0: cannot open shared object file
```

Install the required libraries:

```bash
apt-get update && apt-get install -y libgl1 libglib2.0-0
```

---

### TorchVision Compatibility Error (Real-ESRGAN)

Error:

```text
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
```

Fix:

Edit the corresponding `degradations.py` file, for example, `nano /usr/local/lib/python3.10/dist-packages/basicsr/data/degradations.py`


Change:

```python
import torchvision.transforms.functional_tensor
```

to:

```python
import torchvision.transforms.functional
```

---

## Non-Docker Installation

If not using Docker, install dependencies manually:

For training:

```bash
pip install -r docker/train/requirements.txt
```

For inference:

```bash
pip install -r docker/inference/requirements.txt
```

---

## Notes

* Real-time preview polling may appear stalled at times; terminal `tqdm` output is the most reliable indicator of progress
* GPU execution is strongly recommended for all features

---

This project is intended as a **research sandbox**, prioritizing experimentation and extensibility over production deployment.
