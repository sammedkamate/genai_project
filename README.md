# DreamBooth LoRA with Advanced Guidance Methods

DreamBooth LoRA fine-tuning with multiple guidance techniques for diffusion model personalization.

## Installation

```bash
pip install -r requirements.txt
```

## Training

Train a DreamBooth LoRA model with prior-preservation:

```bash
accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --instance_data_dir /path/to/images \
    --instance_prompt "a photo of sks dog" \
    --class_data_dir ./class_images \
    --class_prompt "a photo of dog" \
    --with_prior_preservation \
    --prior_loss_weight 1.0 \
    --output_dir ./outputs \
    --num_train_epochs 50 \
    --rank 8 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --mixed_precision fp16
```

## Evaluation

### CFG Guidance

```bash
python cfg_guidance.py \
    --lora_path ./outputs \
    --evaluation_prompts_path evaluation_prompts.json \
    --output_dir ./cfg_outputs \
    --optimize
```

### AutoGuidance

```bash
python ag_guidance.py \
    --lora_path ./outputs \
    --evaluation_prompts_path evaluation_prompts.json \
    --output_dir ./ag_outputs \
    --optimize
```

### Bhavik Guidance

```bash
python bg_guidance.py \
    --lora_path ./outputs \
    --evaluation_prompts_path evaluation_prompts.json \
    --output_dir ./bg_outputs \
    --optimize both
```

## Optimization

All guidance methods use golden section search to optimize parameters based on **DINO score** (subject fidelity).

- CFG: Optimizes `--guidance_scale` (lambda)
- AG: Optimizes `--guidance_scale` (lambda)
- BG: Optimizes `--guidance_scale` (lambda) and `--omega` (weight interpolation)

## Files

- `train_dreambooth_lora.py`: LoRA training with prior-preservation
- `cfg_guidance.py`: CFG evaluation and optimization
- `ag_guidance.py`: AutoGuidance evaluation and optimization
- `bg_guidance.py`: Bhavik Guidance evaluation and optimization
- `guidance_methods.py`: Core guidance implementations
- `golden_search.py`: Golden section search optimization
- `evaluation_code.py`: Evaluation metrics computation
