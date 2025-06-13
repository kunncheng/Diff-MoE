import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from accelerate import Accelerator

from models import DiT_models
from diffusion import create_diffusion 
from diffusion.rectified_flow import RectifiedFlow
from download import find_model
from datetime import datetime

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
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
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


class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    from accelerate import DistributedDataParallelKwargs as DDPK
    kwargs = DDPK(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_dir = f"{args.results_dir}/{current_time}_{model_string_name}-{args.num_experts}E{args.num_experts_per_tok}A{args.pretraining_tp}TP"  # Create an experiment folder
        if args.suffix:
            experiment_dir += f'-{args.suffix}'
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
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
        use_checkpoint=args.use_checkpoint,
    )

    if args.resume is not None: 
        print('load from: ', args.resume) 
        state_dict = find_model(args.resume)
        model.load_state_dict(state_dict, strict=False)


    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    print('rf: ', args.rf)
    if args.rf:
        if accelerator.is_main_process:
            logger.info("train with rectified flow")
        diffusion = RectifiedFlow(model)
    else:
        diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule  

    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters())/1e6: .2f}M")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    
    features_dir = f"{args.data_path}/imagenet256_features"
    labels_dir = f"{args.data_path}/imagenet256_labels"
    dataset = CustomDataset(features_dir, labels_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)
    model.train()  
    ema.eval()  
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    if args.resume:
        trained_steps = int(os.path.basename(args.resume).split('.')[0])
        train_steps += trained_steps
        args.max_steps += trained_steps

    if accelerator.is_main_process:
        logger.info(f"Training for 400k iterations...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        data_iter_step = 0
        for x, y in loader:
            x = x.to(device)    # (b, c, h, w)
            y = y.to(device)
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1)
            
            if args.rf: 
                loss, _ = diffusion.forward(x, y)
            else:
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                model_kwargs = dict(y=y)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
            
            if (data_iter_step + 1) % args.accum_iter == 0:
                opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)
            
            data_iter_step += 1
            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = accelerator.reduce(avg_loss, reduction="sum")
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            
            if train_steps >= args.max_steps:
                break
        if train_steps >= args.max_steps:
                break
    
    model.eval()

    if accelerator.is_main_process:
        logger.info("Done!")

    # Eval
    del model
    torch.cuda.empty_cache()
    if accelerator.is_main_process:
        ckpt_path = os.path.join(checkpoint_dir, '0400000.pt')
        for cfg_scale in [1.5, 1.0]:
            eval_command = f'torchrun --nnodes=1 --nproc_per_node=8 --master_port 6003 sample_ddp.py --num-fid-samples 50000 --model {args.model} --num_experts {args.num_experts} --num_experts_per_tok {args.num_experts_per_tok} --cfg-scale {cfg_scale} --per-proc-batch-size 32 --sample-dir {args.results_dir}/samples --ckpt {ckpt_path}'
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
    parser.add_argument("--global-batch-size", type=int, default=256)   # 32 * 8GPUs
    parser.add_argument("--global-seed", type=int, default=2024) 
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument('--accum_iter', default=1, type=int,)  
    parser.add_argument('--num_experts', default=8, type=int,) 
    parser.add_argument('--num_experts_per_tok', default=2, type=int,) 
    parser.add_argument('--pretraining_tp', default=2, type=int,) 
    parser.add_argument("--ckpt-every", type=int, default=50_000) 
    parser.add_argument("--rf", type=bool, default=False) 
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--use_checkpoint", type=bool, default=False, help='use torch.utils.checkpoint.checkpoint to save memory') 
    args = parser.parse_args()
    main(args) 
