python generate_datasets.py \
    --mode fm \
    --fm_pipeline_dir ./serious_runs/stable_training_t_scaled \
    --fm_vae_weights ./fm_src/vae_best.pt \
    --max_samples 200 \
    --output_dir ./generated/fm