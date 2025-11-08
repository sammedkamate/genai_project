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
    --lora_base_dir ./lora_models \
    --evaluation_prompts_path evaluation_prompts.json \
    --output_dir ./cfg_outputs \
    --instance_data_dir /path/to/reference/images \
    --optimize
```

### AutoGuidance (CFG + AG)

```bash
python ag_guidance.py \
    --lora_base_dir ./lora_models \
    --evaluation_prompts_path evaluation_prompts.json \
    --output_dir ./ag_outputs \
    --instance_data_dir /path/to/reference/images \
    --cfg_scale 7.5 \
    --lambda_range -10 10 \
    --weak_checkpoint_path ./lora_models_weak \
    --optimize
```

### Bhavik Guidance

```bash
python bg_guidance.py \
    --lora_base_dir ./lora_models \
    --evaluation_prompts_path evaluation_prompts.json \
    --output_dir ./bg_outputs \
    --instance_data_dir /path/to/reference/images \
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

- **DINO**: Image-to-image similarity between reference and generated images (subject fidelity, optimization target)
- **CLIP-I**: Image-to-image similarity between reference and generated images using CLIP embeddings
- **CLIP-T**: Text-to-image similarity using CLIP embeddings (text prompt alignment)

## Dataset Organization

### Reference Images (`instance_data_dir`)

The `instance_data_dir` should contain subject-specific folders. Folder names are automatically mapped to subjects from `evaluation_prompts.json`:

- Exact matches: `backpack` → `backpack`
- Numbered variants: `cat2`, `dog2`, `dog7`, `dogcd` → `cat`, `dog` (base subject)
- Underscore variants: `cat_statue` → `cat statue` (underscores converted to spaces)

Multiple folders mapping to the same subject are combined (e.g., `dog`, `dog2`, `dog7` all contribute to `dog` reference images).

### LoRA Models (`lora_base_dir`)

The `lora_base_dir` should contain subject-specific LoRA model folders. **Folder names should match your dataset folder names** (e.g., `cat2`, `dog2`, `dog7`, `cat_statue`, `backpack`). Each subject has its own independently trained LoRA model:

```
lora_models/
├── backpack/  
├── cat2/    
├── dog2/        
├── dog7/     
├── cat_statue/  
└── ...
```

**Important:**
- Folder names in `lora_base_dir` match dataset folder names (e.g., `cat2`, `dog2`, `cat_statue`)
- Prompts use subject names from `evaluation_prompts.json` (e.g., "cat", "dog", "cat statue")
- The code automatically maps folder names to subjects using the same logic as reference images
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
