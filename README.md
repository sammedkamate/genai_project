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

### AutoGuidance (CFG + AG)

```bash
python ag_guidance.py \
    --lora_path ./outputs \
    --evaluation_prompts_path evaluation_prompts.json \
    --output_dir ./ag_outputs \
    --cfg_scale 7.5 \
    --lambda_range -10 10 \
    --weak_checkpoint_path ./outputs/checkpoint-500 \
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

## Guidance Methods

- **CFG**: Classifier-Free Guidance (Equation 5)
- **AG**: CFG + AutoGuidance combination (Equation 6 with CFG)
- **BG**: Personalization Guidance with weight interpolation (Equations 7 & 9)

## Optimization

Optimization uses **DINO score** (subject fidelity) as the objective.

- CFG: Golden section search for `--guidance_scale` (lambda)
- AG: Golden section search for `--guidance_scale` (lambda), uses fixed `--cfg_scale` (CFG lambda = 7.5)
- BG: Golden section search for `--guidance_scale` (lambda), grid sweep for `--omega` 

## Parameters

- `--guidance_scale`: Guidance strength (lambda, optimized for CFG/AG/BG)
- `--cfg_scale`: CFG scale for AG method (depending on best lambda for cfg)
- `--lambda_range`: Range for lambda optimization (default: [-10.0, 10.0] for all methods)
- `--omega`: Weight interpolation parameter for BG (0.0-1.0, swept in 0.1 steps)
- `--weak_checkpoint_path`: Earlier checkpoint for AG weak model (optional)

## Evaluation Metrics

- **DINO**: Subject fidelity (optimization target)
- **CLIP-I**: Image-to-image consistency
- **CLIP-T**: Text-to-image alignment

## Files

- `train_dreambooth_lora.py`: LoRA training
- `guidance_methods.py`: Core guidance implementations (CFG, AG, BG)
- `cfg_guidance.py`: CFG evaluation
- `ag_guidance.py`: CFG + AG evaluation
- `bg_guidance.py`: BG evaluation
- `evaluation_code.py`: Evaluation metrics (DINO, CLIP-I, CLIP-T)
- `golden_search.py`: Parameter optimization
- `evaluation_prompts.json`: Evaluation prompts
- `requirements.txt`: Dependencies
