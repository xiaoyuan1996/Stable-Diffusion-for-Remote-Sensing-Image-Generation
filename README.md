# Stable Diffusion for Remote Sensing Image Generation

#### Author: Zhiqiang yuan @ AIRCAS,  [Send a Email](yuan_zhi_qiang@sina.cn)

A simple project for text-to-image remote sensing image generation.
We will release the code of **using text to control regions for super-large RS image generation** later.

##  Environment configuration

Follow [original training repo](https://github.com/justinpinkney/stable-diffusion.git) .


## Pretrained weights

We used [RSITMD](https://github.com/xiaoyuan1996/AMFMN) as training data and fine-tuned stable diffusion for 10 epochs with 1 x A100 GPU.
When the batchsize is 4, the GPU memory consumption is about 40+ Gb during training, and about 20+ Gb during sampling.
The pretrain weights is realesed at [last-pruned.ckpt](https://github.com/xiaoyuan1996/AMFMN).

## Using
Download the pretrain weights to current dir, and run with:
```commandline
bash sample.sh
```
We will update the train code ASAP.

## Examples
**Caption:** some boats drived in the sea
![./assets/shows1.png](./assets/shows1.png)
