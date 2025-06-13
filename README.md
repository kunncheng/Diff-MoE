<div align="center">
<h2>Diff-MoE: Diffusion Transformer with Time-Aware and Space-Adaptive Experts</h2>
<div>
    <a href='https://github.com/kunncheng'>Kun Cheng* <sup>1</sup></a>&emsp;
    <a href='https://github.com/LearningHx'>Xiao He* <sup>1</sup></a>&emsp;
    <a href='https://github.com/kunncheng/DiT-SR'>Lei Yu <sup>2</a>&emsp;
    <a href='https://scholar.google.com/citations?hl=en&user=kSPs6FsAAAAJ&view_op=list_works&sortby=pubdate' target='_blank'>Zhijun Tu <sup>2</sup></a>&emsp;
    <a href='https://web.xidian.edu.cn/mrzhu/en/index.html'>Mingrui Zhu <sup>1</sup></a>&emsp;
    <a href='https://web.xidian.edu.cn/nnwang/'>Nannan Wang <sup>1</sup></a>&emsp;
    <a href='https://see.xidian.edu.cn/faculty/xbgao/'>Xinbo Gao <sup>1</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=o-3D3K4AAAAJ&hl=zh-CN'>Jie Hu <sup>2</sup></a>&emsp;
</div>
<br>
<div>
    <sup>1</sup> Xidian University &emsp; <sup>2</sup> Huawei Noah's Ark Lab 
</div>
<div align="justify">

## 🔎 Introduction 

We propose Diff-MoE, a novel framework that combines Diffusion Transformers with Mixture-of-Experts to exploit both temporarily adaptability and spatial flexibility.Our design incorporates expert-specific timestep conditioning, allowing each expert to process different spatial tokens while adapting to the generative stage, to dynamically allocate resources based on both the temporal and spatial characteristics of the generative task. Additionally, we propose a globally-aware feature recalibration mechanism that amplifies the representational capacity of expert modules by dynamically adjusting feature contributions based
on input relevance.

<p align="center">
  <img src="assets/framework.png">
</p>


## ⚙️ Dependencies and Installation

```
git clone https://github.com/kunncheng/Diff-MoE.git
cd Diff-MoE

conda create -n Diff_MoE python=3.10 -y
conda activate Diff_MoE
pip install -r requirements.txt
```

## 🌈 Training
### Datasets
We train all the models on ImageNet-256 dataset. To improve training efficiency, we follow [Fast-DiT](https://github.com/chuanyangjin/fast-DiT?tab=readme-ov-file#preparation-before-training) and pre-extract VAE features. We use sd-vae-mse in this [link](https://huggingface.co/feizhengcong/DiT-MoE/tree/main/sd-vae-ft-mse). 

The [reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) which is required for evaluation should be downloaded to ```./weights```.

### Training and Inference Scripts
```bash
# Accelerate
accelerate launch --mixed_precision fp16 train.py \
    --model DiT-S/2 \
    --data-path /path/to/vae_features/ \
    --image-size 256 \
    --global-batch-size 32 \
    --results-dir outputs 

# DeepSpeed
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 train_deepspeed.py \
    --deepspeed_config config/zero2.json \
    --model DiT-S/2 \
    --data-path /home/ma-user/work/2024/datasets/trainsets/imagenet/imagenet/train_vae_feats/sd-vae-ft-mse \
    --image-size 256 \
    --train_batch_size 32 \
    --results-dir outputs 
```


## ❤️ Acknowledgement
We sincerely appreciate the code release of the following projects: [DiT](https://github.com/facebookresearch/DiT), [Fast-DiT](https://github.com/chuanyangjin/fast-DiT), [DiT-MoE](https://github.com/feizc/DiT-MoE) and [DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE).
