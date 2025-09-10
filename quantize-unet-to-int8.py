#!/usr/bin/env python3
"""Standalone script for Diffusion models UNet INT8 quantization with OpenVINO NNCF.
NOTE: Quantization is time and memory consuming operation. Running quantization code below may take some time.

Steps (same logic as notebook):https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/latent-consistency-models-image-generation/latent-consistency-models-image-generation.ipynb
 1. Load converted FP16 pipeline from original --model-path for calibration data collection.
 2. Collect UNet input tensors while generating samples on google-research-datasets/conceptual_captions dataset.
 3. Quantize original UNet with nncf.quantize (subset_size=--subset-size).
 4. Overwrite copied directory's `unet/openvino_model.xml/bin` with INT8 UNet.
"""
import argparse
import gc
import shutil
from pathlib import Path

import numpy as np
import openvino as ov
from optimum.intel.openvino import OVDiffusionPipeline
import nncf
import datasets
from transformers import set_seed
from tqdm import tqdm

# Constants from OV notebook: notebooks/latent-consistency-models-image-generation/latent-consistency-models-image-generation.ipynb
# https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/latent-consistency-models-image-generation/latent-consistency-models-image-generation.ipynb

DEFAULT_SUBSET_SIZE = 200
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 8.0
HEIGHT = 512
WIDTH = 512


class CompiledModelDecorator(ov.CompiledModel):
    """Wrap ov.CompiledModel to capture input tensors for calibration"""

    def __init__(self, compiled_model, prob: float, data_cache=None):
        super().__init__(compiled_model)
        self.data_cache = data_cache if data_cache else []
        self.prob = np.clip(prob, 0, 1)

    def __call__(self, *args, **kwargs):  # noqa: D401
        if np.random.rand() >= self.prob:
            # Notebook appends *args (model inputs) as a tuple entry
            self.data_cache.append(*args)
        return super().__call__(*args, **kwargs)


def collect_calibration_data(pipeline, subset_size: int):
    """Run pipeline inference to collect UNet inputs until subset_size reached."""
    original_unet_request = pipeline.unet.request
    pipeline.unet.request = CompiledModelDecorator(original_unet_request, prob=0.3)

    ds = datasets.load_dataset(
        "google-research-datasets/conceptual_captions",
        split="train",
        trust_remote_code=True,
    ).shuffle(seed=42)

    pipeline.set_progress_bar_config(disable=True)
    safety_checker = getattr(pipeline, "safety_checker", None)
    if safety_checker is not None:
        pipeline.safety_checker = None

    pbar = tqdm(total=subset_size)
    prev = 0
    for item in ds:
        prompt = item["caption"]
        if len(prompt) > pipeline.tokenizer.model_max_length:
            continue
        _ = pipeline(
            prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=HEIGHT,
            width=WIDTH,
        )
        collected = len(pipeline.unet.request.data_cache)
        if collected >= subset_size:
            pbar.update(subset_size - pbar.n)
            break
        pbar.update(collected - prev)
        prev = collected

    calib = pipeline.unet.request.data_cache
    pipeline.set_progress_bar_config(disable=False)
    pipeline.unet.request = original_unet_request
    if safety_checker is not None:
        pipeline.safety_checker = safety_checker
    return calib


def quantize_unet(src_model_path: Path, out_model_path: Path, subset_size: int, device: str):
    # Create / copy target directory first if it does not exist.
    if not out_model_path.exists():
        print(f"Copying source pipeline to {out_model_path} ...")
        shutil.copytree(src_model_path, out_model_path)
    else:
        print(f"Skipping Quantization as Output directory already exists: {out_model_path}")
        return

    set_seed(1)
    print("Loading FP16 pipeline from source for calibration...")
    pipe = OVDiffusionPipeline.from_pretrained(src_model_path, device=device)
    unet_calibration_data = collect_calibration_data(pipe, subset_size=subset_size)
    del pipe
    gc.collect()

    core = ov.Core()
    print("Reading original UNet model...")
    unet = core.read_model(src_model_path / "unet/openvino_model.xml")
    print("Running nncf.quantize (subset_size=%d)..." % subset_size)
    quantized_unet = nncf.quantize(
        model=unet,
        subset_size=subset_size,
        calibration_dataset=nncf.Dataset(unet_calibration_data),
        model_type=nncf.ModelType.TRANSFORMER,
        advanced_parameters=nncf.AdvancedQuantizationParameters(disable_bias_correction=True),
    )

    print("Overwriting UNet in copied pipeline with INT8 version...")
    unet_dir = out_model_path / "unet"
    unet_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized_unet, unet_dir / "openvino_model.xml")
    del quantized_unet
    del unet
    gc.collect()
    print(f"Done Quantization of Unet: {out_model_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="StableDiffusion UNet INT8 quantization (UNet-only, copies pipeline)")
    ap.add_argument('-m', '--model-path', type=Path, required=True, help='Path to FP16 OpenVINO pipeline directory')
    ap.add_argument('-o', '--out-model-path', type=Path, help='Output path for quantized pipeline (default: <model-path>-quant_unet)')
    ap.add_argument('-ss', '--subset-size', type=int, default=DEFAULT_SUBSET_SIZE, help='Calibration subset size (default 200)')
    ap.add_argument('-d', '--device', default='CPU', help='Device for calibration pipeline execution (CPU/GPU)')
    return ap.parse_args()


def main():
    args = parse_args()
    model_path: Path = args.model_path
    if args.out_model_path:
        out_model_path: Path = args.out_model_path
    else:
        out_model_path: Path = Path(str(model_path) + '-quant_unet')
    
    print(f"Quantizing UNet in pipeline: {model_path} -> {out_model_path} with subset_size={args.subset_size} on device={args.device}")
    print("NOTE: Quantization is time and memory consuming operation. It may take some time based on the hardware.")
    
    quantize_unet(model_path, out_model_path, args.subset_size, args.device)


if __name__ == '__main__':
    main()
