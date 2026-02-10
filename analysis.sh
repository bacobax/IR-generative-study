


python analyze_distribution_shift.py \
    --real_dir v18/images \
    --generated_dir generated \
    --output_dir analysis_results \
    --max_samples 500 \
    --metrics_max_samples 500 \
    --device cuda:0 \
    --metrics_pca_dim 128 \
    --tsne_perplexity 20 \