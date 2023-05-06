python scripts/txt2img.py \
    --prompt 'There is a forest in the middle of the ocean' \
    --outdir 'outputs/RS' \
    --H 512 --W 512 \
    --n_samples 4 \
    --config 'configs/stable-diffusion/RSITMD.yaml' \
    --ckpt './last-pruned.ckpt'