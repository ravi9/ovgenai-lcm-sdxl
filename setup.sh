#!/bin/bash

python3 -m venv ov-dev-lcm-sdxl-env
source ov-dev-lcm-sdxl-env/bin/activate

python -m pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  
python -m pip install "optimum-intel[openvino]"@git+https://github.com/huggingface/optimum-intel.git
pip install diffusers openvino-genai


