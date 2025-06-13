"""
A training script for DiT using deepspeed.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
# from time import time
import time
import argparse
import logging
import os
from datetime import timedelta

from models import DiT_models
from diffusion import create_diffusion
from diffusion.rectified_flow import RectifiedFlow
from download import find_model
import deepspeed

from datetime import datetime
from train import CustomDataset

os.environ['NCCL_BLOCKING_WAIT'] = '0'  # not to enforce timeout

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        # ema_params[name].mul_(decay).add_(param.cpu().data, alpha=1 - decay)
         ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT-MoE model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    deepspeed.init_distributed(timeout=timedelta(seconds=7200000))
    rank = args.local_rank
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    if 'XL' not in args.model:
        pretraining_tp = 2
    else:
        pretraining_tp = 1

    
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = f"{args.results_dir}/{current_time}_{model_string_name}-{args.num_experts}E{args.num_experts_per_tok}A{pretraining_tp}TP"  # Create an experiment folder
    if args.suffix:
        experiment_dir += f'-{args.suffix}'
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        num_experts=args.num_experts, 
        num_experts_per_tok=args.num_experts_per_tok,
        pretraining_tp=pretraining_tp,
        use_flash_attn=True if 'V100' not in torch.cuda.get_device_name(torch.cuda.current_device()) else False,
        arch=args.arch,
        use_checkpoint=args.use_checkpoint
    )

    if args.resume is not None: 
        print('load from: ', args.resume) 
        state_dict = find_model(args.resume)
        model.load_state_dict(state_dict)
    
    ema = deepcopy(model).to(device) # Create an EMA of the model for use after training
    requires_grad(ema, False)

    print('rf: ', args.rf)
    if args.rf: 
        logger.info("train with rectified flow")
        diffusion = RectifiedFlow(model, use_amp=True)
    else:
        diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule 
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters())/1e6: .2f}M")
    
    features_dir = f"{args.data_path}/imagenet256_features"
    labels_dir = f"{args.data_path}/imagenet256_labels"
    dataset = CustomDataset(features_dir, labels_dir)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size, #int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    model_engine, opt, _, __ = deepspeed.initialize(
        args=args, model=model, model_parameters=model.parameters())

    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})\nAccumulation step {model_engine.gradient_accumulation_steps()}")
    if model_engine.gradient_accumulation_steps() > 1:
        args.max_steps = args.max_steps * model_engine.gradient_accumulation_steps()
    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...") 
        data_iter_step = 0
        for x, y in loader: 
            model_engine.train() 
            x = x.to(device).squeeze(dim=1)
            y = y.to(device).squeeze(dim=1)
            if args.rf: 
                loss, _ = diffusion.forward(x, y)
            else:
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                model_kwargs = dict(y=y)
                with torch.autocast(device_type='cuda'):
                    loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean() 
            
            model_engine.backward(loss) 

            if (data_iter_step + 1) % args.accum_iter == 0: 
                model_engine.step()

            if (data_iter_step + 1) % model_engine.gradient_accumulation_steps() == 0:
                update_ema(ema, model)

            log_steps += 1
            train_steps += 1
            data_iter_step += 1
            running_loss += loss.item()
            
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)

                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:                   
                try:             
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}"
                    model_engine.save_checkpoint(checkpoint_path) 
                except Exception as e: 
                    print(e) 
                
                if rank == 0:
                    checkpoint_path = f"{checkpoint_dir}/ema_{int(train_steps // model_engine.gradient_accumulation_steps()):07d}.pt"
                    torch.save(ema.state_dict(), checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                dist.barrier()
            if train_steps >= args.max_steps:
                break
        if train_steps >= args.max_steps:
                break

    logger.info("Done!")
    cleanup()

    del model
    torch.cuda.empty_cache()
    if rank == 0:
        ckpt_path = os.path.join(f'{checkpoint_dir}/ema_0400000.pt')
        for cfg_scale in [1.5, 1.0]:
            eval_command = f'torchrun --nnodes=1 --nproc_per_node=8 --master_port 6003 sample_ddp.py --num-fid-samples 50000 --arch {args.arch} --model {args.model} --num_experts {args.num_experts} --num_experts_per_tok {args.num_experts_per_tok} --cfg-scale {cfg_scale} --per-proc-batch-size 32 --sample-dir {args.results_dir}/samples --ckpt {ckpt_path}'
            if args.rf:
                eval_command += f" --rf {args.rf}"
            print(eval_command)
            os.system(eval_command)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    parser.add_argument("--vae-path", type=str, default='sd-vae-ft-mse')
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=14000) 
    parser.add_argument("--max_steps", type=int, default=400_000) 
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=2024) 
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument('--accum_iter', default=1, type=int,)  
    parser.add_argument('--num_experts', default=8, type=int,) 
    parser.add_argument('--num_experts_per_tok', default=2, type=int,) 
    parser.add_argument("--ckpt-every", type=int, default=50_000) 
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher') 
    parser.add_argument("--rf", type=bool, default=False) 
    parser.add_argument('--pretraining_tp', default=1, type=int,) 
    parser.add_argument("--arch", type=str, default='original')
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--use_checkpoint", type=bool, default=False, help='use torch.utils.checkpoint.checkpoint to save memory') 
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args() 
    print(args)
    main(args) 
