from contextlib import nullcontext
import gradio as gr
import numpy as np
import torch
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from torchvision import transforms

from scripts.image_variations import load_model_from_config
from ldm.models.diffusion.ddpm import SimpleUpscaleDiffusion

def make_unc(model, n_samples, all_conds):
    uc_tmp = model.get_unconditional_conditioning(n_samples, [""])
    uc = dict()
    for k in all_conds:
        if k == "c_crossattn":
            assert isinstance(all_conds[k], list) and len(all_conds[k]) == 1
            uc[k] = [uc_tmp]
        elif k == "c_adm":  # todo: only run with text-based guidance?
            assert isinstance(all_conds[k], torch.Tensor)
            uc[k] = torch.ones_like(all_conds[k]) * model.low_scale_model.max_noise_level
        elif isinstance(all_conds[k], list):
            uc[k] = [all_conds[k][i] for i in range(len(all_conds[k]))]
        else:
            uc[k] = all_conds[k]
    return uc


@torch.no_grad()
def sample_model(model, sampler, prompt, input_im, precision, use_ema, h, w, ddim_steps, n_samples, scale, ddim_eta):

    precision_scope = autocast if precision=="autocast" else nullcontext
    ema = model.ema_scope if use_ema else nullcontext
    with precision_scope("cuda"):
        with ema():
            c = model.get_learned_conditioning(n_samples * [prompt])
            shape = [4, h // 8, w // 8]
            x_low = input_im.tile(n_samples,1,1,1)
            x_low = x_low.to(memory_format=torch.contiguous_format).half()
            if isinstance(model, SimpleUpscaleDiffusion):
                zx = model.get_first_stage_encoding(model.encode_first_stage(x_low))
                all_conds = {"c_concat": [zx], "c_crossattn": [c]}
            else:
                zx = model.low_scale_model.model.encode(x_low).sample()
                zx = zx * model.low_scale_model.scale_factor
                noise_level = torch.tensor([0]).tile(n_samples).to(input_im.device)
                all_conds = {"c_concat": [zx], "c_crossattn": [c], "c_adm": noise_level}

            uc = None
            if scale != 1.0:
                uc = make_unc(model, n_samples, all_conds)

            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            conditioning=all_conds,
                                            batch_size=n_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            )

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)


def main(
    input_im,
    target_res,
    pre_size,
    prompt,
    scale,
    seed,
    plms=True,
    ddim_steps=50,
    n_samples=1,
    ddim_eta=1.0,
    precision="autocast",
    ):

    # Using the pruned ckpt so ema weights are moved to the normal weights
    use_ema=False

    torch.manual_seed(seed)

    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    if pre_size is not None:
        input_im = transforms.Resize((pre_size, pre_size))(input_im)
    input_im = transforms.Resize((target_res, target_res))(input_im)
    input_im = input_im*2-1

    h = w = target_res

    if plms:
        sampler = PLMSSampler(model)
        ddim_eta = 0.0
    else:
        sampler = DDIMSampler(model)

    x_samples_ddim = sample_model(
        model=model,
        sampler=sampler,
        prompt=prompt,
        input_im=input_im,
        precision=precision,
        use_ema=use_ema,
        h=h, w=w,
        ddim_steps = ddim_steps,
        n_samples=n_samples,
        scale=scale,
        ddim_eta=ddim_eta,
    )
    x_sample = 255. * rearrange(x_samples_ddim[0].cpu().numpy(), 'c h w -> h w c')
    return Image.fromarray(x_sample.astype(np.uint8))

device_idx = 0
device = f"cuda:{device_idx}"

from huggingface_hub import hf_hub_download
config = hf_hub_download(repo_id="lambdalabs/stable-diffusion-super-res", filename="sd-superres-config.yaml")
ckpt = hf_hub_download(repo_id="lambdalabs/stable-diffusion-super-res", filename="sd-superres-pruned.ckpt")
config = OmegaConf.load(config)
model = load_model_from_config(config, ckpt, device=device)

# Load decoder
decoder_path = hf_hub_download(repo_id="stabilityai/sd-vae-ft-mse-original", filename="vae-ft-mse-840000-ema-pruned.ckpt")
decoder = torch.load(decoder_path, map_location='cpu')["state_dict"]
model.first_stage_model.load_state_dict(decoder, strict=False)
model.half()
torch.cuda.empty_cache()

default_prompt = "high quality high resolution uhd 4k image"
inputs = [
    gr.Image(),
    gr.Dropdown(choices=[512, 1024], label="target_resolution (output is resized to square)", value=1024),
    gr.Dropdown(choices=[128, 256, 512, 1024], label="Downsize to this first", value=512),
    gr.Text(value=default_prompt, label="prompt (doesn't do much)"),
    gr.Slider(0, 3, value=1, step=0.1, label="cfg scale (1 is best)"),
    gr.Slider(0, 100, value=0, step=1, label="Seed"),
    gr.Checkbox(True, label="plms"),
    gr.Slider(5, 200, value=50, step=5, label="steps"),
]
output = gr.Image(label="High res")


demo = gr.Interface(
    fn=main,
    title="Stable Diffusion Super Res",
    inputs=inputs,
    outputs=output,
    allow_flagging="never",
    )
demo.launch()
