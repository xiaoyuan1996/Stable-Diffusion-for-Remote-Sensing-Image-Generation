from contextlib import nullcontext
from functools import partial

import fire
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


@torch.no_grad()
def sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta):
    precision_scope = autocast if precision=="autocast" else nullcontext
    with precision_scope("cuda"):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples,1,1)

            if scale != 1.0:
                uc = torch.zeros_like(c)
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c,
                                             batch_size=n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=None)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def main(
    model,
    device,
    input_im,
    scale=3.0,
    n_samples=4,
    plms=True,
    ddim_steps=50,
    ddim_eta=1.0,
    precision="fp32",
    h=512,
    w=512,
    ):

    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    input_im = input_im*2-1

    if plms:
        sampler = PLMSSampler(model)
        ddim_eta = 0.0
    else:
        sampler = DDIMSampler(model)

    x_samples_ddim = sample_model(input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta)
    output_ims = []
    for x_sample in x_samples_ddim:
        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        output_ims.append(Image.fromarray(x_sample.astype(np.uint8)))
    return output_ims


description = \
"""Generate variations on an input image using a fine-tuned version of Stable Diffision.
Trained by [Justin Pinkney](https://www.justinpinkney.com) ([@Buntworthy](https://twitter.com/Buntworthy)) at [Lambda](https://lambdalabs.com/)

__Get the [code](https://github.com/justinpinkney/stable-diffusion) and [model](https://huggingface.co/lambdalabs/stable-diffusion-image-conditioned).__

![](https://raw.githubusercontent.com/justinpinkney/stable-diffusion/main/assets/im-vars-thin.jpg)

"""

article = \
"""
## How does this work?

The normal Stable Diffusion model is trained to be conditioned on text input. This version has had the original text encoder (from CLIP) removed, and replaced with
the CLIP _image_ encoder instead. So instead of generating images based a text input, images are generated to match CLIP's embedding of the image.
This creates images which have the same rough style and content, but different details, in particular the composition is generally quite different.
This is a totally different approach to the img2img script of the original Stable Diffusion and gives very different results.

The model was fine tuned on the [LAION aethetics v2 6+ dataset](https://laion.ai/blog/laion-aesthetics/) to accept the new conditioning.
Training was done on 4xA6000 GPUs on [Lambda GPU Cloud](https://lambdalabs.com/service/gpu-cloud).
More details on the method and training will come in a future blog post.
"""


def run_demo(
    device_idx=0,
    ckpt="models/ldm/stable-diffusion-v1/sd-clip-vit-l14-img-embed_ema_only.ckpt",
    config="configs/stable-diffusion/sd-image-condition-finetune.yaml",
    ):

    device = f"cuda:{device_idx}"
    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt, device=device)

    inputs = [
        gr.Image(),
        gr.Slider(0, 25, value=3, step=1, label="cfg scale"),
        gr.Slider(1, 4, value=1, step=1, label="Number images"),
        gr.Checkbox(True, label="plms"),
        gr.Slider(5, 50, value=25, step=5, label="steps"),
    ]
    output = gr.Gallery(label="Generated variations")
    output.style(grid=2)

    fn_with_model = partial(main, model, device)
    fn_with_model.__name__ = "fn_with_model"

    examples = [
        ["assets/im-examples/vermeer.jpg", 3, 1, True, 25],
        ["assets/im-examples/matisse.jpg", 3, 1, True, 25],
    ]

    demo = gr.Interface(
        fn=fn_with_model,
        title="Stable Diffusion Image Variations",
        description=description,
        article=article,
        inputs=inputs,
        outputs=output,
        examples=examples,
        allow_flagging="never",
        )
    demo.launch(enable_queue=True, share=True)

if __name__ == "__main__":
    fire.Fire(run_demo)
