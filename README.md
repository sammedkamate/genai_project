# WiG: Improving Guidance for Personalizing T2I Diffusion Models

## Abstract

There is a growing real-world need for diffusion-based generative models to support personalized generation, where a model can be adapted to a specific subject from just a few reference images. This is crucial for applications such as custom character rendering, virtual avatars, branded content creation, and personalized product design. However, current personalization methods often face a trade-off between visual fidelity and textual alignment: some preserve the subject's appearance but fail to follow textual instructions, while others follow prompts accurately but lose key visual traits.

This challenge arises because existing guidance strategies typically rely on fixed reference models that cannot dynamically adjust to the degree of personalization. We aim to improve the balance between subject identity and prompt faithfulness in fine-tuned text-to-image diffusion models by developing a lightweight, training-free inference method. Our approach blends the knowledge of both the original and fine-tuned models during inference, steering generation toward outputs that maintain high subject fidelity while respecting textual intent.

## Authors

- **Bhavik Chandna** (PID: A69033934) - bchandna@ucsd.edu
- **Ganesh Bannur** (PID: A69032587) - gbannur@ucsd.edu
- **Sneh Davaria** (PID: A69033513) - sdavaria@ucsd.edu
- **Sammed Kamate** (PID: A69032538) - skamate@ucsd.edu

**Department of Computer Science, UC San Diego**

## Installation

```bash
pip install -r requirements.txt
```

For SANA model support, you may need additional dependencies. Please refer to the [official SANA repository](https://github.com/NVlabs/Sana) for installation instructions.

## Quick Start

### 1. Train a Personalized Model

Train a DreamBooth LoRA model for Stable Diffusion:

```bash
accelerate launch training/train_dreambooth_lora.py \
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

### 2. Evaluate with WIG

Generate personalized images using our Weight Interpolation Guidance:

```bash
python guidance_evaluation/wig_guidance.py \
    --pretrained_model_path runwayml/stable-diffusion-v1-5 \
    --lora_base_dir ./lora_models \
    --evaluation_prompts_path config/evaluation_prompts.json \
    --output_dir ./wig_outputs \
    --instance_data_dir /path/to/reference/images \
    --optimize both
```

## Main Training

### Stable Diffusion 1.5 / 2.1

Train a DreamBooth LoRA model with prior-preservation:

```bash
accelerate launch training/train_dreambooth_lora.py \
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

Train a DreamBooth LoRA model for SANA (transformer-based architecture):

```bash
accelerate launch training/train_dreambooth_lora_sana.py \
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

### Training Hyperparameters

Fine-tuning was conducted with the following hyperparameters:

- **Batch size**: 1
- **Training iterations**: 500 steps (max_train_steps)
- **LoRA rank**: 4
- **Learning rate**: 1e-4
- **Prior preservation loss**: Enabled (recommended)
- **Maximum diffusion timesteps**: 1,000 (configured in scheduler)

### Resolution & Inference Settings

- **Stable Diffusion 1.5/2.1**: 512×512 resolution for both training and inference
  - Scheduler: DDIM
  - Inference steps: 50
  
- **SANA**: 1024×1024 resolution for both training and inference
  - Scheduler: FlowDPM-Solver
  - Inference steps: 20

## Evaluation

We provide evaluation scripts for three guidance methods: CFG, AutoGuidance (AG), and Weight Interpolation Guidance (WIG).

### Weight Interpolation Guidance (WIG) - Stable Diffusion

**Our proposed method** - Blends pretrained and fine-tuned model weights:

```bash
python guidance_evaluation/wig_guidance.py \
    --pretrained_model_path runwayml/stable-diffusion-v1-5 \
    --lora_base_dir ./lora_models \
    --evaluation_prompts_path config/evaluation_prompts.json \
    --output_dir ./wig_outputs \
    --instance_data_dir /path/to/reference/images \
    --optimize both
```

### Classifier-Free Guidance (CFG) - Stable Diffusion

Baseline method for comparison:

```bash
python guidance_evaluation/cfg_guidance.py \
    --pretrained_model_path runwayml/stable-diffusion-v1-5 \
    --lora_base_dir ./lora_models \
    --evaluation_prompts_path config/evaluation_prompts.json \
    --output_dir ./cfg_outputs \
    --instance_data_dir /path/to/reference/images \
    --optimize
```

### AutoGuidance (CFG + AG) - Stable Diffusion

AutoGuidance combines CFG with a weak model:

```bash
python guidance_evaluation/ag_guidance.py \
    --pretrained_model_path runwayml/stable-diffusion-v1-5 \
    --lora_base_dir ./lora_models \
    --evaluation_prompts_path config/evaluation_prompts.json \
    --output_dir ./ag_outputs \
    --instance_data_dir /path/to/reference/images \
    --cfg_scale 7.5 \
    --lambda_range -10 10 \
    --optimize
```

**Note:** AG automatically uses `weak/` subdirectory within each folder as the weak model checkpoint.

### SANA Model Evaluation

All three guidance methods are also supported for SANA models:

```bash
# WIG for SANA
python guidance_evaluation/wig_guidance_sana.py \
    --pretrained_model_path hf-internal-testing/tiny-sana-pipe \
    --lora_base_dir ./lora_models_sana \
    --evaluation_prompts_path config/evaluation_prompts.json \
    --output_dir ./wig_outputs_sana \
    --instance_data_dir /path/to/reference/images \
    --optimize both

# CFG for SANA
python guidance_evaluation/cfg_guidance_sana.py \
    --pretrained_model_path hf-internal-testing/tiny-sana-pipe \
    --lora_base_dir ./lora_models_sana \
    --evaluation_prompts_path config/evaluation_prompts.json \
    --output_dir ./cfg_outputs_sana \
    --instance_data_dir /path/to/reference/images \
    --optimize

# AG for SANA
python guidance_evaluation/ag_guidance_sana.py \
    --pretrained_model_path hf-internal-testing/tiny-sana-pipe \
    --lora_base_dir ./lora_models_sana \
    --evaluation_prompts_path config/evaluation_prompts.json \
    --output_dir ./ag_outputs_sana \
    --instance_data_dir /path/to/reference/images \
    --cfg_scale 7.5 \
    --lambda_range -10 10 \
    --optimize
```

**Note:** All SANA evaluation scripts use 1024×1024 resolution and 20 inference steps with FlowDPM-Solver scheduler.

## Guidance Methods

Our implementation includes three guidance strategies:

- **CFG**: Classifier-Free Guidance - Standard guidance method that amplifies the conditioning signal
- **AG**: AutoGuidance - Combines CFG with a "weak" model to better preserve subject identity
- **WIG**: Weight Interpolation Guidance (Our Method) - Dynamically blends pretrained and fine-tuned model weights during inference to balance subject fidelity and text alignment

### Weight Interpolation Guidance (WIG) Details

WIG addresses the trade-off between subject fidelity and text faithfulness by interpolating model weights:

- **Weight interpolation parameter (ω)**: Controls the blend between pretrained and fine-tuned models
- **ω = 0.0-0.3**: Maximizes subject fidelity (closer to fine-tuned model)
- **ω = 0.4-0.6**: Balanced improvement with better preservation of text fidelity (blended model)
- **Dynamic adjustment**: Allows fine-tuning of the trade-off for different use cases

The optimization automatically sweeps ω values in 0.1 steps when using `--optimize both`.

## Optimization

Parameter optimization uses **DINO score** (subject fidelity) as the primary objective:

- **CFG**: Golden section search for `--guidance_scale` (lambda)
- **AG**: Golden section search for `--guidance_scale` (lambda), uses fixed `--cfg_scale` (CFG lambda = 7.5)
- **WIG**: Golden section search for `--guidance_scale` (lambda), grid sweep for `--omega` (weight interpolation parameter)

### Parameters

- `--guidance_scale`: Guidance strength (lambda, optimized for CFG/AG/WIG)
- `--cfg_scale`: CFG scale for AG method (depending on best lambda for cfg)
- `--lambda_range`: Range for lambda optimization (default: [-10.0, 10.0] for all methods)
- `--omega`: Weight interpolation parameter for WIG (0.0-1.0, swept in 0.1 steps)
- `--weak_checkpoint_path`: Earlier checkpoint for AG weak model (optional)

## Evaluation Metrics

We evaluate generated images using three metrics:

- **DINO**: Image-to-image similarity between reference and generated images (subject fidelity, optimization target)
- **CLIP-I**: Image-to-image similarity between reference and generated images using CLIP embeddings
- **CLIP-T**: Text-to-image similarity using CLIP embeddings (text prompt alignment)

These metrics help quantify the trade-off between subject identity preservation and textual instruction following.

## Dataset Organization

### Reference Images (`instance_data_dir`)

The `instance_data_dir` should contain subject-specific folders. Folder names are automatically mapped to subjects from `config/evaluation_prompts.json`:

- Exact matches: `backpack` → `backpack`
- Numbered variants: `cat2`, `dog2`, `dog7`, `dogcd` → `cat`, `dog` (base subject)
- Underscore variants: `cat_statue` → `cat statue` (underscores converted to spaces)

**Note:** Each folder maps to exactly one subject. Folders are not combined.

### LoRA Models (`lora_base_dir`)

The `lora_base_dir` should contain subject-specific LoRA model folders. **Each folder must have two checkpoints for AG evaluation:**

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
- **WIG**: Uses fully trained + pretrained model (no weak checkpoint needed)

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

### Textual Inversion

Textual Inversion is another personalization method that learns to represent specific concepts as new "words" in the embedding space of the text encoder. Use the `textual_inversion.ipynb` Jupyter notebook for training and evaluation.

To use Textual Inversion:
1. Open `textual_inversion.ipynb` in Jupyter Notebook or JupyterLab
2. Follow the notebook cells to:
   - Set up your training data (instance images and prompts)
   - Configure training hyperparameters
   - Train the textual inversion embeddings
   - Evaluate the trained embeddings with different guidance methods

The notebook includes evaluation using DINO, CLIP-I, and CLIP-T metrics similar to the DreamBooth LoRA evaluation scripts.

### InstructPix2Pix

InstructPix2Pix is an image editing model that follows natural language instructions to edit images. Use the `instructpix2pix.ipynb` Jupyter notebook for image editing tasks.

To use InstructPix2Pix:
1. Open `instructpix2pix.ipynb` in Jupyter Notebook or JupyterLab
2. Follow the notebook cells to:
   - Load the InstructPix2Pix model
   - Load or prepare input images
   - Apply text-based image editing instructions
   - Generate edited images

## Project Structure

```
genai_project/
├── README.md
├── requirements.txt
├── textual_inversion.ipynb                # Textual Inversion training and evaluation notebook
├── instructpix2pix.ipynb                  # InstructPix2Pix image editing notebook
├── training/
│   ├── train_dreambooth_lora.py          # LoRA training for Stable Diffusion 1.5/2.1
│   └── train_dreambooth_lora_sana.py     # LoRA training for SANA models
├── guidance_evaluation/
│   ├── cfg_guidance.py                   # CFG evaluation for Stable Diffusion
│   ├── cfg_guidance_sana.py              # CFG evaluation for SANA models
│   ├── ag_guidance.py                    # CFG + AG evaluation for Stable Diffusion
│   ├── ag_guidance_sana.py               # CFG + AG evaluation for SANA models
│   ├── wig_guidance.py                   # WIG evaluation for Stable Diffusion
│   └── wig_guidance_sana.py              # WIG evaluation for SANA models
├── utils/
│   ├── guidance_methods.py               # Core guidance implementations (CFG, AG, WIG) with SANA support
│   ├── evaluation_code.py                # Evaluation metrics (DINO, CLIP-I, CLIP-T)
│   └── golden_search.py                  # Parameter optimization
├── config/
│   └── evaluation_prompts.json           # Evaluation prompts
├── evaluation_code_ti.py                 # Textual Inversion evaluation code
└── datasets/
    └── ...                                # Dataset folders
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wig2024,
  title={WiG: Improving Guidance for Personalizing T2I Diffusion Models},
  author={Chandna, Bhavik and Bannur, Ganesh and Davaria, Sneh and Kamate, Sammed},
  journal={CSE 291 Project Report},
  year={2025},
  institution={UC San Diego}
}
```

## License

This project is released for academic research purposes.

## Acknowledgments

We thank the open-source community for providing the foundational tools and models that made this research possible, including Hugging Face Diffusers, Stable Diffusion, and SANA.

