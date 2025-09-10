import warnings
import logging
from pathlib import Path
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from optimum.exporters.openvino import main_export  
from optimum.intel.openvino.configuration import OVConfig, OVWeightQuantizationConfig  

import torch

sdxl_base = "stabilityai/stable-diffusion-xl-base-1.0"
lcm_unet = "latent-consistency/lcm-sdxl"

warnings.filterwarnings("ignore")  
logging.getLogger("transformers").setLevel(logging.ERROR)  
logging.getLogger("diffusers").setLevel(logging.ERROR)  
logging.getLogger("optimum").setLevel(logging.ERROR)  

def main():
    print("Loading LCM UNet...")
    unet = UNet2DConditionModel.from_pretrained(lcm_unet, torch_dtype=torch.float16, variant="fp16")

    print("Composing SDXL base pipeline with LCM UNet...")
    pipe = DiffusionPipeline.from_pretrained(sdxl_base, unet=unet, torch_dtype=torch.float16)

    print("Switching scheduler to LCMScheduler...")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    composed_dir = "lcm-sdxl-fp16"
    print(f"Saving composed pipeline to: {composed_dir}")
    pipe.save_pretrained(composed_dir)

    # Export to OpenVINO with both fp16 and int8 weight formats.
    for wf in ["fp16", "int8"]:  
        print(f"\nExporting to OpenVINO with weight format: {wf}")  
        ov_out_dir_wf = Path(f"lcm-sdxl-ov-{wf}")   
        if wf == "int8":  
            quantization_config = OVWeightQuantizationConfig(bits=8, dtype="int8")  
            ov_config = OVConfig(quantization_config=quantization_config)  
        else:  
            ov_config = OVConfig(dtype=wf)
            
        # Shell cmd: optimum-cli export openvino --model lcm-sdxl-fp16 --task stable-diffusion-xl --weight-format fp16 lcm-sdxl-ov-fp16
        try:  
            main_export(  
                model_name_or_path=composed_dir,  
                output=ov_out_dir_wf,  
                task="stable-diffusion-xl",  
                convert_tokenizer=True,
                ov_config=ov_config
            )  
        except Exception as e:  
                import traceback  
                print(f"Full traceback:")  
                traceback.print_exc()  
                return 1

            # print(f"Export failed: {e}")  
            # return 1  
        
        print(f"\nExport complete. OpenVINO model saved to: {ov_out_dir_wf}")  
    return 0


if __name__ == "__main__":
    main()