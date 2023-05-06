import argparse
from argparse import Namespace
from huggingface_hub import Repository, create_repo

from scripts.convert_sd_to_diffusers import main

# python scripts/convert_and_push_to_hub.py \
# --checkpoint_path "logs/2022-09-02T06-46-25_pokemon_pokemon/checkpoints/epoch=000142.ckpt" \
# --config_path "configs/stable-diffusion/pokemon.yaml" \
# --repo_path "lambdalabs/sd-pokemon-diffusers"  \
# --output_path "model_zoo/pokemon-e142" \
# --device "cuda:7"



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the checkpoint to convert."
    )

    parser.add_argument(
        "--config_path",
        default=None,
        type=str,
        required=True,
        help="The YAML config file corresponding to the original architecture.",
    )

    parser.add_argument(
        "--repo_path",
        default=None,
        type=str,
        required=True,
        help="Name of the huggingface repo.",
    )

    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output model."
    )

    parser.add_argument(
        "--device", default="cuda:0", type=str, help="device for testing"
    )

    args = parser.parse_args()

    repo_url = create_repo(repo_id=args.repo_path)
    repo = Repository(local_dir=args.output_path, clone_from=repo_url)

    convert_args = Namespace()
    convert_args.checkpoint_path = args.checkpoint_path
    convert_args.original_config_file = args.config_path
    convert_args.dump_path = args.output_path
    convert_args.use_ema = True

    main(convert_args)

    # Test output
    from diffusers import StableDiffusionPipeline
    pipeline = StableDiffusionPipeline.from_pretrained(args.output_path).to(args.device)
    output = pipeline("yoda", guidance_scale=7.5)["sample"]
    print("Generated test output, please check test_image.jpg")
    output[0].save("test_image.jpg")

    # For some reason I need to re-create the repo
    repo = Repository(args.output_path)
    print(f"Uploading to huggingface hub at {args.repo_path}")
    repo.push_to_hub()