# LCM SDXL OpenVINO Pipeline

This repo provides a full flow to:
1. Set up a Python environment
2. Compose an SDXL base pipeline with an LCM UNet
3. Export the pipeline to OpenVINO (FP16 + optional INT8 weight quantization)
4. (Optional) [Post-quantize only the UNet with calibration (NNCF)](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/latent-consistency-models-image-generation/latent-consistency-models-image-generation.ipynb) 
5. Run image generation with OpenVINO GenAI

> **Note:** After exporting to OpenVINO models, you could create a new python virtual environment with minimal dependecies:
```bash
python3 -m venv ov-infer-lcm-sdxl-env
source ov-infer-lcm-sdxl-env/bin/activate
pip install openvino-genai pillow
python run-lcm-sdxl-ov.py -m lcm-sdxl-ov-fp16
```

## Quick Start for Running by downloading Hugging Face model:
- [lcm-sdxl-ov-fp16-quant_unet on Huggingface](https://huggingface.co/rpanchum/lcm-sdxl-ov-fp16-quant_unet)
```bash
python3 -m venv ov-infer-lcm-sdxl-env
source ov-infer-lcm-sdxl-env/bin/activate
pip install openvino-genai pillow 

git lfs install
git clone https://huggingface.co/rpanchum/lcm-sdxl-ov-fp16-quant_unet/
wget https://raw.githubusercontent.com/ravi9/ovgenai-lcm-sdxl/refs/heads/main/run-lcm-sdxl-ov.py

python run-lcm-sdxl-ov.py -m lcm-sdxl-ov-fp16-quant_unet -ni 3
```

## Quick Start for Exporting and Running:

```bash
# 1. (First time) create and install deps
source setup.sh          # Windows: use `setup.bat`
# 2. Export SDXL + LCM to OpenVINO
python export-lcm-sdxl-to-ov.py
# 3. (Optional) Post-Quantize Only the UNet to INT8
python quantize-unet-to-int8.py -m lcm-sdxl-ov-fp16
# 4. Run Image Generation
python run-lcm-sdxl-ov.py -m lcm-sdxl-ov-fp16
```

## 1. Environment Setup

One-time environment creation (do NOT repeat unless you want a fresh env):
```bash
source setup.sh     # Windows: use `setup.bat`
```

Activate the environment in any future shell session:
```bash
source ov-dev-lcm-sdxl-env/bin/activate

# Windows: ov-dev-lcm-sdxl-env\Scripts\activate 
```

This creates a virtual environment: `ov-dev-lcm-sdxl-env`

## 2. Export SDXL + LCM to OpenVINO

```bash
python export-lcm-sdxl-to-ov.py
```

This will:
- Download SDXL base + LCM UNet
- Save composed pipeline to `lcm-sdxl-fp16/`
- Export to:
  - `lcm-sdxl-ov-fp16/`
  - `lcm-sdxl-ov-int8/` (weight quantized)

If the export fails re-run after ensuring network / HF auth (if required).

## 3. (Optional) Post-Quantize Only the UNet

If you want an INT8 UNet and remaining components in FP16:

```bash
# You can change calibration subset size (default 200):
python quantize-unet-to-int8.py -m lcm-sdxl-ov-fp16 --subset-size 300 --device CPU
```

Results in a new directory:
```
lcm-sdxl-ov-fp16-quant_unet/
└── unet/openvino_model.xml/bin  (INT8)
```

## 4. Run Image Generation

Pick one of the exported pipeline directories:
- `lcm-sdxl-ov-fp16`
- `lcm-sdxl-ov-int8`
- `lcm-sdxl-ov-fp16-quant_unet`

Generated images are saved as `image_<model_dir>_<timestamp>_<text_encoder_device>_<unet_device>_<vae_decoder_device>_<i>.png`.
- Example: `image_lcm-sdxl-ov-fp16_0909_160526_CPU_GPU_CPU_1.png`

Basic run:
```bash
python run-lcm-sdxl-ov.py -m lcm-sdxl-ov-fp16
```
Run with prompt and options:
```bash
python run-lcm-sdxl-ov.py -m lcm-sdxl-ov-fp16-quant_unet -d GPU -s 5 -ni 3 \
    -p "A cinematic concept art of a crystal city at dawn" 
```

Split device assignment (override components individually):
```bash
python run-lcm-sdxl-ov.py -m lcm-sdxl-ov-fp16-quant_unet -td CPU -ud NPU -vd GPU \
  -p "Ultra-detailed watercolor of a red fox in a misty forest"
```

## 5. Caching

The runtime script creates/uses `ov_cache/` to store compiled OpenVINO blobs for faster subsequent runs.
> **Note:** NPU device compilation for the first time takes a while. Subsequent runs have faster compilation.

To clear cache:
```bash
rm -rf ov_cache
```

## 6. OpenVINO Optimized Models

| Directory | UNet Precision | Text Encoder, VAE Precision | When to Use |
|-----------|----------------|-----------------------------|-------------|
| `lcm-sdxl-ov-fp16` | FP16 | FP16 | Baseline quality |
| `lcm-sdxl-ov-int8` | INT8 (weight compressed) | Mixed (see exporter) | Lowest memory, fast |
| `lcm-sdxl-ov-fp16-quant_unet` | INT8 (post-quantized) | FP16 | Good balance; calibrated UNet |
