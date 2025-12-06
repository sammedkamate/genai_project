# DreamBooth LoRA with Advanced Guidance Methods

DreamBooth LoRA fine-tuning with multiple guidance techniques for diffusion model personalization.

## Installation

```bash
pip install -r requirements.txt
```

For SANA model support, you may need additional dependencies. Please refer to the [official SANA repository](https://github.com/NVlabs/Sana) for installation instructions.

## Training

### Stable Diffusion 1.5 / 2.1

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
    --max_train_steps 500 \
    --train_batch_size 1 \
    --rank 4 \
    --learning_rate 1e-4 \
    --resolution 512 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --mixed_precision fp16
```

### SANA Model

Train a DreamBooth LoRA model for SANA:

```bash
accelerate launch train_dreambooth_lora_sana.py \
    --pretrained_model_name_or_path hf-internal-testing/tiny-sana-pipe \
    --instance_data_dir /path/to/images \
    --instance_prompt "a photo of sks dog" \
    --output_dir ./outputs_sana \
    --max_train_steps 500 \
    --train_batch_size 1 \
    --rank 4 \
    --learning_rate 1e-4 \
    --resolution 1024 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --mixed_precision fp16 \
    --cache_latents \
    --max_sequence_length 256
```

## Training Hyperparameters

Fine-tuning was conducted with the following hyperparameters:

- **Batch size**: 1
- **Training iterations**: 500 steps (max_train_steps)
- **LoRA rank**: 4
- **Learning rate**: 1e-4
- **Prior preservation loss**: Enabled (recommended)
- **Maximum diffusion timesteps**: 1,000 (configured in scheduler)

### Resolution

- **Stable Diffusion 1.5/2.1**: 512×512 resolution for both training and inference
- **SANA**: 1024×1024 resolution for both training and inference

### Inference Settings

- **Stable Diffusion 1.5/2.1**:
  - Scheduler: DDIM
  - Inference steps: 50
  
- **SANA**:
  - Scheduler: FlowDPM-Solver
  - Inference steps: 20

## Evaluation

### CFG Guidance (Stable Diffusion)

```bash
python cfg_guidance.py \
    --pretrained_model_path runwayml/stable-diffusion-v1-5 \
    --lora_base_dir ./lora_models \
    --evaluation_prompts_path evaluation_prompts.json \
    --output_dir ./cfg_outputs \
    --instance_data_dir /path/to/reference/images \
    --optimize
```

### AutoGuidance (CFG + AG) - Stable Diffusion

```bash
python ag_guidance.py \
    --pretrained_model_path runwayml/stable-diffusion-v1-5 \
    --lora_base_dir ./lora_models \
    --evaluation_prompts_path evaluation_prompts.json \
    --output_dir ./ag_outputs \
    --instance_data_dir /path/to/reference/images \
    --cfg_scale 7.5 \
    --lambda_range -10 10 \
    --optimize
```

**Note:** AG automatically uses `weak/` subdirectory within each folder as the weak model checkpoint.

### Weight Interpolation Guidance (WIG) - Stable Diffusion

```bash
python wig_guidance.py \
    --pretrained_model_path runwayml/stable-diffusion-v1-5 \
    --lora_base_dir ./lora_models \
    --evaluation_prompts_path evaluation_prompts.json \
    --output_dir ./wig_outputs \
    --instance_data_dir /path/to/reference/images \
    --optimize both
```

### CFG Guidance (SANA)

```bash
python cfg_guidance_sana.py \
    --pretrained_model_path hf-internal-testing/tiny-sana-pipe \
    --lora_base_dir ./lora_models_sana \
    --evaluation_prompts_path evaluation_prompts.json \
    --output_dir ./cfg_outputs_sana \
    --instance_data_dir /path/to/reference/images \
    --optimize
```

### AutoGuidance (CFG + AG) - SANA

```bash
python ag_guidance_sana.py \
    --pretrained_model_path hf-internal-testing/tiny-sana-pipe \
    --lora_base_dir ./lora_models_sana \
    --evaluation_prompts_path evaluation_prompts.json \
    --output_dir ./ag_outputs_sana \
    --instance_data_dir /path/to/reference/images \
    --cfg_scale 7.5 \
    --lambda_range -10 10 \
    --optimize
```

### Weight Interpolation Guidance (WIG) - SANA

```bash
python wig_guidance_sana.py \
    --pretrained_model_path hf-internal-testing/tiny-sana-pipe \
    --lora_base_dir ./lora_models_sana \
    --evaluation_prompts_path evaluation_prompts.json \
    --output_dir ./wig_outputs_sana \
    --instance_data_dir /path/to/reference/images \
    --optimize both
```

**Note:** All SANA evaluation scripts use 1024×1024 resolution and 20 inference steps with FlowDPM-Solver scheduler.

## Guidance Methods

- **CFG**: Classifier-Free Guidance (Equation 5)
- **AG**: CFG + AutoGuidance combination (Equation 6 with CFG)
- **WIG**: Weight Interpolation Guidance with weight interpolation (Equations 7 & 9)

## Optimization

Optimization uses **DINO score** (subject fidelity) as the objective.

- CFG: Golden section search for `--guidance_scale` (lambda)
- AG: Golden section search for `--guidance_scale` (lambda), uses fixed `--cfg_scale` (CFG lambda = 7.5)
- WIG: Golden section search for `--guidance_scale` (lambda), grid sweep for `--omega` 

## Parameters

- `--guidance_scale`: Guidance strength (lambda, optimized for CFG/AG/WIG)
- `--cfg_scale`: CFG scale for AG method (depending on best lambda for cfg)
- `--lambda_range`: Range for lambda optimization (default: [-10.0, 10.0] for all methods)
- `--omega`: Weight interpolation parameter for WIG (0.0-1.0, swept in 0.1 steps)
- `--weak_checkpoint_path`: Earlier checkpoint for AG weak model (optional)

### Weight Interpolation (ω) for WIG

For Weight Interpolation Guidance (WIG), the `omega` parameter controls the trade-off between subject fidelity and text fidelity:

- **ω = 0.0-0.3**: Maximizes subject fidelity
- **ω = 0.4-0.6**: Balanced improvement with better preservation of text fidelity

The optimization automatically sweeps ω values in 0.1 steps when using `--optimize both`.

## Evaluation Metrics

- **DINO**: Image-to-image similarity between reference and generated images (subject fidelity, optimization target)
- **CLIP-I**: Image-to-image similarity between reference and generated images using CLIP embeddings
- **CLIP-T**: Text-to-image similarity using CLIP embeddings (text prompt alignment)

## Dataset Organization

### Reference Images (`instance_data_dir`)

The `instance_data_dir` should contain subject-specific folders. Folder names are automatically mapped to subjects from `evaluation_prompts.json`:

- Exact matches: `backpack` → `backpack`
- Numbered variants: `cat2`, `dog2`, `dog7`, `dogcd` → `cat`, `dog` (base subject)
- Underscore variants: `cat_statue` → `cat statue` (underscores converted to spaces)

**Note:** Each folder maps to exactly one subject. Folders are not combined.

### LoRA Models (`lora_base_dir`)

The `lora_base_dir` should contain subject-specific LoRA model folders. **Each folder must have two checkpoints:**

```
lora_models/
├── backpack/
│   ├── (fully trained checkpoint files)
│   └── weak/
│       └── (under-trained checkpoint files)
├── cat2/
│   ├── (fully trained checkpoint files)
│   └── weak/
│       └── (under-trained checkpoint files)
├── dog2/
│   ├── (fully trained checkpoint files)
│   └── weak/
│       └── (under-trained checkpoint files)
└── ...
```

**Checkpoint Usage:**
- **CFG**: Uses fully trained checkpoint (`lora_base_dir/folder_name/`)
- **AG**: Uses under-trained (`lora_base_dir/folder_name/weak/`) as weak model + fully trained as fine model
- **WIG**: Uses fully trained + pretrained model

**Important:**
- Folder names in `lora_base_dir` match dataset folder names (e.g., `cat2`, `dog2`, `cat_statue`)
- Prompts use subject names from `evaluation_prompts.json` (e.g., "cat", "dog", "cat statue")
- The code automatically maps folder names to subjects
- Each folder is treated as a separate subject (no combining)
- Each subject's prompts are evaluated independently with that subject's LoRA model
- Each LoRA model has its own independent guidance optimization

## Output Organization

Evaluation outputs are organized as follows:

```
output_dir/
├── evaluation_summary.csv          # Summary table with metrics per subject
├── cat/                            # Images for "cat" subject
│   ├── 0000_a_cat_in_the_jungle.png
│   ├── 0001_a_cat_in_the_snow.png
│   └── ...
├── dog/                            # Images for "dog" subject
│   ├── 0005_a_dog_on_the_beach.png
│   └── ...
└── ...
```

The `evaluation_summary.csv` contains:
- `subject`: Subject name
- `num_prompts`: Number of prompts evaluated for this subject
- `CLIP-I`: Average CLIP-I score (image-to-image similarity)
- `CLIP-T`: Average CLIP-T score (text-to-image similarity)
- `DINO`: Average DINO score (subject fidelity)

## Personalization Methods Details

### DreamBooth-LoRA Configuration

Following the implementation codes provided by Diffusers or the official SANA repository:

- **LoRA rank**: 4
- **Learning rate**: 1e-4
- **Prior preservation loss**: Enabled (recommended)
- **LoRA layers**: Applied to attention processors in UNet (SD) or transformer blocks (SANA)

For SANA models, LoRA is applied to transformer attention layers (e.g., `transformer_blocks.X.attn1.to_k`, `transformer_blocks.X.attn1.to_q`).

## Files

- `train_dreambooth_lora.py`: LoRA training for Stable Diffusion 1.5/2.1
- `train_dreambooth_lora_sana.py`: LoRA training for SANA models
- `guidance_methods.py`: Core guidance implementations (CFG, AG, WIG) with SANA support
- `cfg_guidance.py`: CFG evaluation for Stable Diffusion
- `cfg_guidance_sana.py`: CFG evaluation for SANA models
- `ag_guidance.py`: CFG + AG evaluation for Stable Diffusion
- `ag_guidance_sana.py`: CFG + AG evaluation for SANA models
- `wig_guidance.py`: WIG evaluation for Stable Diffusion
- `wig_guidance_sana.py`: WIG evaluation for SANA models
- `evaluation_code.py`: Evaluation metrics (DINO, CLIP-I, CLIP-T)
- `golden_search.py`: Parameter optimization
- `evaluation_prompts.json`: Evaluation prompts
- `requirements.txt`: Dependencies