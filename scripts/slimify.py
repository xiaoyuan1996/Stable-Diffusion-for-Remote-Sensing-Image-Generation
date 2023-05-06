import torch
import argparse

if __name__ == "__main__":
    # Make a version of the checkpoint with only ema weights (around 4GB)
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_ckpt", help="full size checkpoint file")
    parser.add_argument("--output_path", help="filename for ema only checkpoint")
    args = parser.parse_args()

    print(f"loading from {args.original_ckpt}")
    d = torch.load(args.original_ckpt, map_location="cpu")

    new_d = {"state_dict": {}}
    ema_state = {k: v for k, v in d["state_dict"].items() if not k.startswith("model.diffusion_model")}
    new_d["state_dict"] = ema_state

    print(f"saving to {args.output_path}")
    torch.save(new_d, args.output_path)
