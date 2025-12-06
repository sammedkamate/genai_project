# coding=utf-8
# Copyright 2025 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import math
import random
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.loaders.lora_base import LORA_ADAPTER_METADATA_KEY
from transformers import CLIPTextModel, CLIPTokenizer

logger = get_logger(__name__)

try:
    from diffusers import Transformer2DModel
    SANA_TRANSFORMER_AVAILABLE = True
except ImportError:
    SANA_TRANSFORMER_AVAILABLE = False
    logger.warning("Transformer2DModel not available. SANA training may not work.")


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        size=1024,
        repeats=1,
        center_crop=False,
        max_sequence_length=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images * repeats
        self.max_sequence_length = max_sequence_length or tokenizer.model_max_length
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size) if center_crop else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(
            self.instance_images_path[index % self.num_instance_images]
        ).convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        
        # Tokenize prompt
        tokenized = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_sequence_length,
            return_tensors="pt",
        )
        example["instance_prompt_ids"] = tokenized.input_ids
        return example


def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }

    if "prior_images" in examples[0]:
        batch["prior_pixel_values"] = torch.stack([example["prior_images"] for example in examples])
        batch["prior_prompt_ids"] = torch.cat([example["prior_prompt_ids"] for example in examples], dim=0)

    return batch


def parse_args():
    parser = argparse.ArgumentParser(description="DreamBooth LoRA Training for SANA")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora_outputs_sana",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=50,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=500,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16).",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='The integration to report the results and logs to.',
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether training should be resumed from a previous checkpoint.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=8,
        help="LoRA alpha scaling parameter.",
    )
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help="Specific transformer layers to apply LoRA to. If None, applies to all attention layers.",
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        help="Cache VAE latents for faster training.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=None,
        help="Maximum sequence length for tokenizer. If None, uses model default.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading SANA model from {args.pretrained_model_name_or_path}")

    # Load components
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )

    # Load SANA transformer
    if not SANA_TRANSFORMER_AVAILABLE:
        raise ImportError(
            "Transformer2DModel is required for SANA training. "
            "Please install the latest version of diffusers."
        )
    
    transformer = Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Apply LoRA to transformer
    transformer_lora_attn_procs = {}
    lora_rank = args.rank
    lora_alpha = args.lora_alpha

    # Get transformer blocks
    if hasattr(transformer, 'transformer_blocks'):
        blocks = transformer.transformer_blocks
    elif hasattr(transformer, 'blocks'):
        blocks = transformer.blocks
    else:
        raise ValueError("Cannot find transformer blocks in SANA model")

    # Determine which layers to apply LoRA to
    target_layers = None
    if args.lora_layers:
        # Specific layers provided
        target_layers = [args.lora_layers]
    else:
        # Apply to all attention layers
        target_layers = None

    # Apply LoRA processors
    for i, block in enumerate(blocks):
        block_prefix = f"transformer_blocks.{i}"
        
        # Self-attention (attn1)
        if hasattr(block, 'attn1'):
            attn1 = block.attn1
            if hasattr(attn1, 'to_q'):
                hidden_size = attn1.to_q.out_features if hasattr(attn1.to_q, 'out_features') else attn1.to_q.weight.shape[0]
                
                for proj_name in ['to_q', 'to_k', 'to_v', 'to_out']:
                    if hasattr(attn1, proj_name):
                        layer_name = f"{block_prefix}.attn1.{proj_name}"
                        if target_layers is None or any(tl in layer_name for tl in target_layers):
                            transformer_lora_attn_procs[layer_name] = LoRAAttnProcessor(
                                hidden_size=hidden_size,
                                cross_attention_dim=None,
                                rank=lora_rank,
                                lora_alpha=lora_alpha,
                            )
        
        # Cross-attention (attn2) if present
        if hasattr(block, 'attn2'):
            attn2 = block.attn2
            if hasattr(attn2, 'to_q'):
                hidden_size = attn2.to_q.out_features if hasattr(attn2.to_q, 'out_features') else attn2.to_q.weight.shape[0]
                cross_attention_dim = getattr(transformer.config, 'cross_attention_dim', None)
                if cross_attention_dim is None and hasattr(attn2, 'to_k'):
                    cross_attention_dim = attn2.to_k.in_features if hasattr(attn2.to_k, 'in_features') else attn2.to_k.weight.shape[1]
                
                for proj_name in ['to_q', 'to_k', 'to_v', 'to_out']:
                    if hasattr(attn2, proj_name):
                        layer_name = f"{block_prefix}.attn2.{proj_name}"
                        if target_layers is None or any(tl in layer_name for tl in target_layers):
                            transformer_lora_attn_procs[layer_name] = LoRAAttnProcessor(
                                hidden_size=hidden_size,
                                cross_attention_dim=cross_attention_dim,
                                rank=lora_rank,
                                lora_alpha=lora_alpha,
                            )

    transformer.set_attn_processor(transformer_lora_attn_procs)
    lora_layers = AttnProcsLayers(transformer.attn_processors)

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Scale learning rate
    learning_rate = args.learning_rate
    if args.scale_lr:
        learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    optimizer = optimizer_class(
        lora_layers.parameters(),
        lr=learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", num_train_timesteps=1000
    )

    # Dataset
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        max_sequence_length=args.max_sequence_length,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with accelerator
    transformer, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Gradient checkpointing = {args.gradient_checkpointing}")

    global_step = 0
    first_epoch = 0
    resume_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            # Use recalculated num_update_steps_per_epoch for resume calculations
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    # Cache latents if requested
    cached_latents = None
    if args.cache_latents:
        logger.info("Caching VAE latents...")
        vae.to(accelerator.device)
        vae.eval()
        cached_latents = []
        for example in train_dataset:
            pixel_values = example["instance_images"].unsqueeze(0).to(accelerator.device, dtype=vae.dtype)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            cached_latents.append(latents.squeeze(0))
        vae.to("cpu")
        logger.info(f"Cached {len(cached_latents)} latents")
    else:
        # Ensure VAE is on device for encoding during training
        vae.to(accelerator.device)
        vae.eval()

    transformer.train()
    text_encoder.train()

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(transformer):
                if args.cache_latents and cached_latents is not None:
                    # Use cached latents
                    batch_indices = [(step * args.train_batch_size + i) % len(cached_latents) for i in range(len(batch["input_ids"]))]
                    latents = torch.stack([cached_latents[idx] for idx in batch_indices]).to(accelerator.device)
                else:
                    # Encode images
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    pixel_values = pixel_values.to(accelerator.device)
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Transformer forward pass
                model_pred = transformer(
                    noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_layers.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                # Limit checkpoints
                if args.checkpoints_total_limit is not None and accelerator.is_main_process:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                    if len(checkpoints) > args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                        removing_checkpoints = checkpoints[0:num_to_remove]
                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} oldest checkpoints"
                        )
                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint_path = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint_path)
                            logger.info(f"Removed {removing_checkpoint_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Save the lora layers
    if accelerator.is_main_process:
        transformer = accelerator.unwrap_model(transformer)
        transformer = transformer.to(torch.float32)

        unwrapped_lora_layers = AttnProcsLayers(transformer.attn_processors)
        
        # Prepare metadata in the format expected by the test
        metadata = {
            LORA_ADAPTER_METADATA_KEY: json.dumps({
                "transformer.lora_alpha": str(lora_alpha),
                "transformer.r": str(lora_rank),
            })
        }
        
        unwrapped_lora_layers.save_pretrained(
            args.output_dir,
            safe_serialization=True,
            metadata=metadata
        )
        
        logger.info(f"LoRA weights saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
