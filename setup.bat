@echo off
python -m venv ov-dev-lcm-sdxl-env
call ov-dev-lcm-sdxl-env\Scripts\activate

python -m pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  
pip install "optimum-intel[openvino] @ git+https://github.com/huggingface/optimum-intel.git"
pip install diffusers openvino-genai


