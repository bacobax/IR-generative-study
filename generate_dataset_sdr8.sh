python generate_datasets.py \
    --mode sd15 \
    --lora_dir ./stable_diffusion_15_out/out_ir_lora_sd15r8_p_norm/checkpoint-3000 \
    --max_samples 200 \
    --output_dir ./generated/sd15_r8 \
    --max_tries 25 \
    --sd_steps 100 \
    --lora_rank 8
