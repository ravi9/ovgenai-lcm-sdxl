import argparse
from pathlib import Path
from PIL import Image
import openvino_genai as ov_genai
import openvino as ov
import time
import random
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Run LCM-SDXL with OpenVINO GenAI on CPU/GPU/NPU with split devices.")
    parser.add_argument("-m", "--model_dir", type=Path, default=Path("lcm-sdxl-ov-fp16"), help="Path to OpenVINO-exported pipeline")
    parser.add_argument("-p", "--prompt", type=str, help="Prompt for image generation.")
    parser.add_argument("-ni", "--num_images", type=int, default=1, help="Number of images to generate")

    parser.add_argument("-d", "--device", default=None, help="Set all devices (text encoder, UNet, VAE decoder) to (CPU/GPU/NPU)")
    parser.add_argument("-td", "--text-encoder-device", default=None, help="Device for text encoders (CPU/GPU/NPU)")
    parser.add_argument("-ud", "--unet-device", default=None, help="Device for UNet (CPU/GPU/NPU)")
    parser.add_argument("-vd", "--vae-decoder-device", default=None, help="Device for VAE decoder (CPU/GPU/NPU)")

    parser.add_argument("-s", "--steps", type=int, default=4, help="Number of inference steps (LCM typically 2–8)")
    # Note: For best results with SDXL, use 1024 x 1024 or resolutions with 1,048,576 pixels (e.g., 896 x 1152, 1536 x 640).
    parser.add_argument("-w", "--width", type=int, default=1024, help="Image width. For best results with SDXL, use 1024 x 1024 or resolutions with 1,048,576 pixels (e.g., 896 x 1152, 1536 x 640).")
    parser.add_argument("-ht", "--height", type=int, default=1024, help="Image height")
    parser.add_argument("-se", "--seed", type=int, default=99, help="Random seed")

    args = parser.parse_args()

    prompts = {
    "prompt1": "A close-up HD shot of vibrant macaw parrot on a branch in a forest, 8K",
    "prompt2": "Armies of angels and demons fighting in the sky, two smoke monsters, winged demons fighting golden wingless flaming angels, embodiment of darkness and light battle in a cloudy sky, beautiful clouds behind them, fantasy, high detail, realistic photo, high detail, digital painting, cinematic, stunning, hyper-realistic, sharp focus, high resolution, 8k",
    "prompt3": "An epic space battleship between futuristic spacecraft, set against a backdrop of swirling nebulas and distant stars, capturing the intensity of interstellar warfare.",
    "prompt4": "Illuminated pirate ship sailing on a sea with a galaxy in the sky, epic, 4k, ultra",
    "prompt5": "anime artwork a girl looking at the sea, dramatic, anime style, key visual, vibrant, studio anime, highly detailed, cinematic",
    "prompt6": "Futuristic city on Mars with domed structures and advanced transportation systems, futuristic lighting, cinematic lighting, 8k, cinematic poster",
    "prompt7": "cinematic photo of a construction worker looking down at city. 35mm photograph, film, bokeh, professional, 4k, highly detailed, sharp focus, intricate details, octane render",
    "prompt8": "breathtaking selfie photograph of astronaut floating in space, earth in the background. award-winning, professional, highly detailed, 8k, dslr, soft lighting, high quality",
    }

    if args.prompt is None:
        # Pick a random prompt if none is provided
        selected_prompt = random.choice(list(prompts.values()))
    else:
        selected_prompt = args.prompt

    if args.device is None:
        args.device = "GPU" if "GPU" in ov.Core().available_devices else "CPU"
        print(f"No device specified with -d, defaulting to {args.device}")
    
    # Set default devices for components
    if args.text_encoder_device is None:
        args.text_encoder_device = args.device
    if args.unet_device is None:
        args.unet_device = args.device
    if args.vae_decoder_device is None:
        args.vae_decoder_device = args.device

    # Cache dir for compiled models
    ov_cache_dir = Path("./ov_cache")
    ov_cache_dir.mkdir(exist_ok=True)

    # Step 1: Load pipeline
    pipe = ov_genai.Text2ImagePipeline(args.model_dir)

    # Step 2: Reshape (batch=1, height, width, guidance scale)
    # Note: LCM models use a different approach to guidance through timestep conditioning rather than classifier-free guidance, which is why negative prompts are not supported. 
    pipe.reshape(1, args.height, args.width, pipe.get_generation_config().guidance_scale)


    # Step 3: Compile pipeline with split devices
    ov_config = {"CACHE_DIR": str(ov_cache_dir)}
    print(f"Compiling pipeline with devices: text_encoder={args.text_encoder_device}, unet={args.unet_device}, vae_decoder={args.vae_decoder_device}")
    compile_start = time.time()
    pipe.compile(args.text_encoder_device, args.unet_device, args.vae_decoder_device, config=ov_config)
    compile_end = time.time()
    print(f"Compile time: {compile_end - compile_start:.2f} seconds")

    
    # Step 4: Generate image
    for i in range(1, 1 + getattr(args, "num_images", 1)):
        print(f"\nGenerating image {i}/{args.num_images} with prompt: {selected_prompt}")
        default_seed = random.randint(1, 100)

        gen_start = time.time()
        result = pipe.generate(
            selected_prompt,
            num_inference_steps=args.steps,
            rng_seed=default_seed,
        )
        gen_end = time.time()
        print(f"Generation {i} time: {gen_end - gen_start:.2f} seconds")

        image = Image.fromarray(result.data[0])
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        out_path = Path(f"image_{args.model_dir}_{timestamp}_{args.text_encoder_device}_{args.unet_device}_{args.vae_decoder_device}_{i}.png")
        image.save(out_path)
        print(f"Saved image to {out_path.resolve()}")
        # update to a new random prompt for next iteration
        selected_prompt = random.choice(list(prompts.values()))


if __name__ == "__main__":
    main()
