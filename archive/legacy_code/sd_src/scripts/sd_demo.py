import argparse
from contextlib import nullcontext
import os
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft.utils import set_peft_model_state_dict
from diffusers.utils import convert_unet_state_dict_to_peft
from helpers import generate_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Run SD inference with trained LoRA weights.")
    parser.add_argument("--base", type=str, default="runwayml/stable-diffusion-v1-5", help="Base model used for training.")
    parser.add_argument(
        "--lora_dir",
        type=str,
        default="./out_ir_lora_sd15",
        help="Folder containing pytorch_lora_weights.safetensors saved by train_text_to_image_lora.py.",
    )
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--output_dir", type=str, default="generated_images", help="Output directory for generated images.")
    parser.add_argument("--variant", type=str, default=None, help="Model variant used in training (e.g. fp16).")
    parser.add_argument("--revision", type=str, default=None, help="Model revision tag used in training.")
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Torch dtype for inference. Match training mixed_precision for best fidelity.",
    )
    parser.add_argument("--num_images_per_prompt", type=int, default=4, help="Number of images to generate per prompt.")
    parser.add_argument("--max_persons", type=int, default=20, help="Maximum number of persons to generate prompts for.")
    return parser.parse_args()


def main():
    args = parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    weight_dtype = dtype_map[args.precision]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the base model components
    tokenizer = CLIPTokenizer.from_pretrained(
        args.base,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.base,
        subfolder="text_encoder",
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        args.base,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.base,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    # Load LoRA weights and apply them to UNet
    lora_state_dict, network_alphas = StableDiffusionPipeline.lora_state_dict(args.lora_dir)
    unet_state_dict = {f"{k.replace('unet.', '')}": v for k, v in lora_state_dict.items() if k.startswith("unet.")}
    unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
    
    # Add LoRA adapter to UNet
    from peft import LoraConfig
    unet_lora_config = LoraConfig(
        r=4,  # Default rank from training script
        lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out"],
    )
    unet.add_adapter(unet_lora_config)
    
    # Load the LoRA weights into the adapter
    set_peft_model_state_dict(unet, unet_state_dict, adapter_name="default")

    # Move models to device
    unet = unet.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    # Create pipeline with the trained UNet (matching training validation setup)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipe.to(device)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=weight_dtype)
        if device == "cuda" and weight_dtype != torch.float32
        else nullcontext()
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate images for each number of persons from 0 to max_persons
    for num_persons in range(args.max_persons + 1):
        prompt = generate_prompt(num_persons)
        print(f"\nGenerating {args.num_images_per_prompt} images for prompt: {prompt}")
        
        # Create subdirectory for this person count
        person_dir = os.path.join(args.output_dir, f"persons_{num_persons:02d}")
        os.makedirs(person_dir, exist_ok=True)
        
        # Generate multiple images for this prompt
        for img_idx in range(args.num_images_per_prompt):
            # Use different seed for each image
            seed = args.seed + num_persons * args.num_images_per_prompt + img_idx
            generator = torch.Generator(device=device).manual_seed(seed)
            
            with autocast_ctx:
                image = pipe(
                    prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=generator,
                ).images[0]
            
            output_path = os.path.join(person_dir, f"image_{img_idx:02d}.png")
            image.save(output_path)
            print(f"  Saved {output_path}")
            raw = np.asarray(image).astype(np.float32) / 255.0
            raw = raw.mean(axis=-1)
            A, B = 11667.0, 13944.0
            S = B - A
            rawu16 = (A + raw * S)
            rawm1t1 = 2.0 * np.clip((rawu16 - A) / S, 0.0, 1.0) - 1.0
            #save as .npy
            npy_output_path = os.path.join(person_dir, f"image_{img_idx:02d}.npy")
            np.save(npy_output_path, rawm1t1)
            print(f"  Saved {npy_output_path}")
    
    print(f"\nAll images generated in {args.output_dir}")


if __name__ == "__main__":
    main()