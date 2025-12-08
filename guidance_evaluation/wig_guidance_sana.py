import argparse
import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.guidance_methods import GuidancePipeline, SanaGuidancePipeline
from utils.evaluation_code import Evaluator
from utils.golden_search import golden_section_search

# Note: This script requires SANA pipeline support
# Adapt based on your SANA pipeline implementation


def load_pretrained_sana_pipeline(model_path: str, device: str = "cuda"):
    """
    Load pretrained SANA pipeline.
    Note: Adapt based on actual SANA pipeline class and import.
    """
    try:
        # Example: from diffusers import SanaPipeline
        # pipeline = SanaPipeline.from_pretrained(...)
        # Or use the official SANA repository pipeline
        raise NotImplementedError(
            "SANA pipeline loading not implemented. "
            "Please adapt this function based on your SANA pipeline implementation."
        )
    except Exception as e:
        raise ImportError(f"Failed to load SANA pipeline: {e}")


def load_finetuned_sana_pipeline(
    pretrained_model_path: str,
    lora_path: str,
    device: str = "cuda",
):
    """
    Load finetuned SANA pipeline with LoRA weights.
    """
    try:
        pipeline = load_pretrained_sana_pipeline(pretrained_model_path, device)
        
        # Load LoRA weights for SANA transformer
        # Adapt based on SANA LoRA loading mechanism
        # Example: pipeline.transformer.load_attn_procs(lora_path)
        
        # Set FlowDPM-Solver scheduler for inference (20 steps)
        # Note: Adapt scheduler name based on actual SANA scheduler
        # try:
        #     from diffusers import FlowMatchEulerDiscreteScheduler
        #     pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(...)
        # except ImportError:
        #     pass
        
        return SanaGuidancePipeline(pipeline)
    except Exception as e:
        raise ImportError(f"Failed to load finetuned SANA pipeline: {e}")


def evaluate_wig_sana(
    pretrained_model_path: str,
    lora_base_dir: str,
    evaluation_prompts_path: str,
    output_dir: str,
    device: str = "cuda",
    guidance_scale: float = 7.5,
    omega: float = 0.0,
    num_images_per_prompt: int = 1,
    instance_data_dir: str = None,
):
    """
    Evaluate Weight Interpolation Guidance (WIG) for SANA model.
    Uses 1024x1024 resolution and 20 inference steps with FlowDPM-Solver.
    """
    pretrained_pipeline = load_pretrained_sana_pipeline(pretrained_model_path, device)
    
    evaluator = Evaluator(
        model_path=None,
        json_path=evaluation_prompts_path,
        output_dir=output_dir,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        instance_data_dir=instance_data_dir,
    )
    
    clip_i_scores, clip_t_scores, dino_scores = [], [], []
    text_features = evaluator._encode_text(evaluator.prompts)
    
    subject_metrics = {}
    current_pipeline = None
    current_subject = None
    
    for idx, prompt in enumerate(tqdm(evaluator.prompts, desc="Evaluating SANA WIG")):
        subject = evaluator.subject_per_prompt[idx]
        
        if subject != current_subject:
            lora_path = evaluator._find_lora_path(subject, lora_base_dir, checkpoint_type="full")
            if not lora_path:
                print(f"Warning: No fully trained LoRA model found for subject '{subject}' in {lora_base_dir}, skipping...")
                continue
            
            finetuned_pipeline = load_finetuned_sana_pipeline(
                pretrained_model_path, lora_path, device
            )
            finetuned_pipeline.set_pretrained_models(pretrained_pipeline)
            current_pipeline = finetuned_pipeline
            current_subject = subject
            
            if subject not in subject_metrics:
                subject_metrics[subject] = {
                    "clip_i": [],
                    "clip_t": [],
                    "dino": [],
                    "prompt_count": 0
                }
        
        def generate_images(p):
            result = current_pipeline.generate_with_guidance(
                prompt=p,
                guidance_method="wig",
                guidance_scale=guidance_scale,
                omega=omega,
                num_inference_steps=20,  # SANA uses 20 steps
                height=1024,  # SANA uses 1024x1024
                width=1024,
                num_images_per_prompt=num_images_per_prompt,
            )
            return result.images
        
        images = generate_images(prompt)
        img_clip = evaluator._encode_images_clip(images)
        img_dino = evaluator._encode_images_dino(images)
        t_feat = text_features[idx].unsqueeze(0).repeat(num_images_per_prompt, 1)
        
        clip_t = evaluator._cosine(img_clip, t_feat).mean().item()
        clip_t_scores.append(clip_t)
        subject_metrics[subject]["clip_t"].append(clip_t)
        subject_metrics[subject]["prompt_count"] += 1
        
        subject_dir = os.path.join(output_dir, subject.replace(" ", "_"))
        os.makedirs(subject_dir, exist_ok=True)
        prompt_safe = prompt.replace("/", "_").replace("\\", "_")[:100]
        image_path = os.path.join(subject_dir, f"{idx:04d}_{prompt_safe}.png")
        images[0].save(image_path)
        
        if evaluator.reference_images and subject in evaluator.reference_images and len(evaluator.reference_images[subject]) > 0:
            ref_images = evaluator.reference_images[subject]
            ref_clip = evaluator._encode_images_clip(ref_images)
            ref_dino = evaluator._encode_images_dino(ref_images)
            
            clip_i_sims = []
            for gen_feat in img_clip:
                sims = F.cosine_similarity(
                    gen_feat.unsqueeze(0),
                    ref_clip,
                    dim=1
                )
                clip_i_sims.append(sims.max().item())
            clip_i = np.mean(clip_i_sims)
            clip_i_scores.append(clip_i)
            subject_metrics[subject]["clip_i"].append(clip_i)
            
            dino_sims = []
            for gen_feat in img_dino:
                sims = F.cosine_similarity(
                    gen_feat.unsqueeze(0),
                    ref_dino,
                    dim=1
                )
                dino_sims.append(sims.max().item())
            dino_i = np.mean(dino_sims)
            dino_scores.append(dino_i)
            subject_metrics[subject]["dino"].append(dino_i)
    
    summary_table = []
    for subject, metrics in subject_metrics.items():
        summary_table.append({
            "subject": subject,
            "num_prompts": metrics["prompt_count"],
            "CLIP-I": float(np.mean(metrics["clip_i"])) if metrics["clip_i"] else None,
            "CLIP-T": float(np.mean(metrics["clip_t"])) if metrics["clip_t"] else None,
            "DINO": float(np.mean(metrics["dino"])) if metrics["dino"] else None,
        })
    
    import pandas as pd
    df = pd.DataFrame(summary_table)
    csv_path = os.path.join(output_dir, "evaluation_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nEvaluation summary saved to {csv_path}")
    
    return {
        "CLIP-I": float(np.mean(clip_i_scores)) if clip_i_scores else None,
        "CLIP-T": float(np.mean(clip_t_scores)),
        "DINO": float(np.mean(dino_scores)) if dino_scores else None,
        "per_subject": summary_table,
    }


def optimize_wig_lambda_sana(
    pretrained_model_path: str,
    lora_base_dir: str,
    evaluation_prompts_path: str,
    output_dir: str,
    device: str = "cuda",
    lambda_range: tuple = (-10.0, 10.0),
    omega: float = 0.0,
    num_images_per_prompt: int = 1,
    instance_data_dir: str = None,
):
    def objective(lambda_val):
        scores = evaluate_wig_sana(
            pretrained_model_path=pretrained_model_path,
            lora_base_dir=lora_base_dir,
            evaluation_prompts_path=evaluation_prompts_path,
            output_dir=os.path.join(output_dir, f"temp_wig_lambda_{lambda_val}"),
            device=device,
            guidance_scale=lambda_val,
            omega=omega,
            num_images_per_prompt=num_images_per_prompt,
            instance_data_dir=instance_data_dir,
        )
        dino_score = scores.get("DINO", 0)
        return -dino_score

    best_lambda, best_score = golden_section_search(
        objective, lambda_range[0], lambda_range[1]
    )

    best_scores = evaluate_wig_sana(
        pretrained_model_path=pretrained_model_path,
        lora_base_dir=lora_base_dir,
        evaluation_prompts_path=evaluation_prompts_path,
        output_dir=os.path.join(output_dir, "wig_optimized_lambda"),
        device=device,
        guidance_scale=best_lambda,
        omega=omega,
        num_images_per_prompt=num_images_per_prompt,
        instance_data_dir=instance_data_dir,
    )

    return best_lambda, best_scores


def optimize_wig_omega_sana(
    pretrained_model_path: str,
    lora_base_dir: str,
    evaluation_prompts_path: str,
    output_dir: str,
    device: str = "cuda",
    guidance_scale: float = 7.5,
    omega_range: tuple = (0.0, 1.0),
    num_images_per_prompt: int = 1,
    instance_data_dir: str = None,
):
    omega_values = [round(i * 0.1, 1) for i in range(11)]
    best_omega = 0.0
    best_dino = -float('inf')
    all_results = {}

    for omega_val in omega_values:
        print(f"Evaluating omega={omega_val:.1f}...")
        scores = evaluate_wig_sana(
            pretrained_model_path=pretrained_model_path,
            lora_base_dir=lora_base_dir,
            evaluation_prompts_path=evaluation_prompts_path,
            output_dir=os.path.join(output_dir, f"temp_wig_omega_{omega_val}"),
            device=device,
            guidance_scale=guidance_scale,
            omega=omega_val,
            num_images_per_prompt=num_images_per_prompt,
            instance_data_dir=instance_data_dir,
        )
        dino_score = scores.get("DINO", 0)
        all_results[omega_val] = scores
        
        if dino_score > best_dino:
            best_dino = dino_score
            best_omega = omega_val

    print(f"\nBest omega: {best_omega:.1f} (DINO: {best_dino:.4f})")
    
    best_scores = evaluate_wig_sana(
        pretrained_model_path=pretrained_model_path,
        lora_base_dir=lora_base_dir,
        evaluation_prompts_path=evaluation_prompts_path,
        output_dir=os.path.join(output_dir, "wig_optimized_omega"),
        device=device,
        guidance_scale=guidance_scale,
        omega=best_omega,
        num_images_per_prompt=num_images_per_prompt,
        instance_data_dir=instance_data_dir,
    )

    return best_omega, best_scores


def optimize_wig_both_sana(
    pretrained_model_path: str,
    lora_base_dir: str,
    evaluation_prompts_path: str,
    output_dir: str,
    device: str = "cuda",
    lambda_range: tuple = (-10.0, 10.0),
    omega_range: tuple = (0.0, 1.0),
    num_images_per_prompt: int = 1,
    instance_data_dir: str = None,
):
    print("Optimizing lambda first...")
    best_lambda, _ = optimize_wig_lambda_sana(
        pretrained_model_path=pretrained_model_path,
        lora_base_dir=lora_base_dir,
        evaluation_prompts_path=evaluation_prompts_path,
        output_dir=output_dir,
        device=device,
        lambda_range=lambda_range,
        omega=0.0,
        num_images_per_prompt=num_images_per_prompt,
        instance_data_dir=instance_data_dir,
    )

    print(f"Best lambda: {best_lambda:.4f}")
    print("Optimizing omega...")
    best_omega, best_scores = optimize_wig_omega_sana(
        pretrained_model_path=pretrained_model_path,
        lora_base_dir=lora_base_dir,
        evaluation_prompts_path=evaluation_prompts_path,
        output_dir=output_dir,
        device=device,
        guidance_scale=best_lambda,
        omega_range=omega_range,
        num_images_per_prompt=num_images_per_prompt,
        instance_data_dir=instance_data_dir,
    )

    return best_lambda, best_omega, best_scores


def main():
    parser = argparse.ArgumentParser(description="Weight Interpolation Guidance (WIG) Evaluation for SANA")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="hf-internal-testing/tiny-sana-pipe",
        help="Path to pretrained SANA model",
    )
    parser.add_argument(
        "--lora_base_dir",
        type=str,
        required=True,
        help="Base directory containing subject-specific LoRA model folders",
    )
    parser.add_argument(
        "--evaluation_prompts_path",
        type=str,
        default="config/evaluation_prompts.json",
        help="Path to evaluation prompts JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="wig_outputs_sana",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=0.0,
        help="Omega parameter for weight interpolation",
    )
    parser.add_argument(
        "--optimize",
        type=str,
        choices=["lambda", "omega", "both"],
        default=None,
        help="Optimize lambda, omega, or both",
    )
    parser.add_argument(
        "--lambda_range",
        type=float,
        nargs=2,
        default=[-10.0, 10.0],
        help="Range for lambda optimization",
    )
    parser.add_argument(
        "--omega_range",
        type=float,
        nargs=2,
        default=[0.0, 1.0],
        help="Range for omega optimization",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images per prompt",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="Directory containing reference images for DINO score calculation",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.optimize == "lambda":
        print("Optimizing SANA WIG lambda...")
        best_lambda, scores = optimize_wig_lambda_sana(
            pretrained_model_path=args.pretrained_model_path,
            lora_base_dir=args.lora_base_dir,
            evaluation_prompts_path=args.evaluation_prompts_path,
            output_dir=args.output_dir,
            device=args.device,
            lambda_range=tuple(args.lambda_range),
            omega=args.omega,
            num_images_per_prompt=args.num_images_per_prompt,
            instance_data_dir=args.instance_data_dir,
        )
        print(f"\n=== SANA WIG Lambda Optimization Results ===")
        print(f"Best Lambda: {best_lambda:.4f}")
        print(f"Omega: {args.omega:.4f}")
        print(f"Scores:")
        for metric, score in scores.items():
            if score is not None:
                print(f"  {metric}: {score:.4f}")

        results = {
            "method": "wig_sana",
            "lambda": best_lambda,
            "omega": args.omega,
            "scores": scores,
        }
        results_path = os.path.join(args.output_dir, "wig_lambda_optimization_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

    elif args.optimize == "omega":
        print("Optimizing SANA WIG omega...")
        best_omega, scores = optimize_wig_omega_sana(
            pretrained_model_path=args.pretrained_model_path,
            lora_base_dir=args.lora_base_dir,
            evaluation_prompts_path=args.evaluation_prompts_path,
            output_dir=args.output_dir,
            device=args.device,
            guidance_scale=args.guidance_scale,
            omega_range=tuple(args.omega_range),
            num_images_per_prompt=args.num_images_per_prompt,
            instance_data_dir=args.instance_data_dir,
        )
        print(f"\n=== SANA WIG Omega Optimization Results ===")
        print(f"Lambda: {args.guidance_scale:.4f}")
        print(f"Best Omega: {best_omega:.4f}")
        print(f"Scores:")
        for metric, score in scores.items():
            if score is not None:
                print(f"  {metric}: {score:.4f}")

        results = {
            "method": "wig_sana",
            "lambda": args.guidance_scale,
            "omega": best_omega,
            "scores": scores,
        }
        results_path = os.path.join(args.output_dir, "wig_omega_optimization_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

    elif args.optimize == "both":
        print("Optimizing SANA WIG lambda and omega...")
        best_lambda, best_omega, scores = optimize_wig_both_sana(
            pretrained_model_path=args.pretrained_model_path,
            lora_base_dir=args.lora_base_dir,
            evaluation_prompts_path=args.evaluation_prompts_path,
            output_dir=args.output_dir,
            device=args.device,
            lambda_range=tuple(args.lambda_range),
            omega_range=tuple(args.omega_range),
            num_images_per_prompt=args.num_images_per_prompt,
            instance_data_dir=args.instance_data_dir,
        )
        print(f"\n=== SANA WIG Full Optimization Results ===")
        print(f"Best Lambda: {best_lambda:.4f}")
        print(f"Best Omega: {best_omega:.4f}")
        print(f"Scores:")
        for metric, score in scores.items():
            if score is not None:
                print(f"  {metric}: {score:.4f}")

        results = {
            "method": "wig_sana",
            "lambda": best_lambda,
            "omega": best_omega,
            "scores": scores,
        }
        results_path = os.path.join(args.output_dir, "wig_full_optimization_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

    else:
        print(f"Evaluating SANA WIG with guidance_scale={args.guidance_scale}, omega={args.omega}...")
        scores = evaluate_wig_sana(
            pretrained_model_path=args.pretrained_model_path,
            lora_base_dir=args.lora_base_dir,
            evaluation_prompts_path=args.evaluation_prompts_path,
            output_dir=args.output_dir,
            device=args.device,
            guidance_scale=args.guidance_scale,
            omega=args.omega,
            num_images_per_prompt=args.num_images_per_prompt,
            instance_data_dir=args.instance_data_dir,
        )
        print("\n=== SANA WIG Evaluation Results ===")
        for metric, score in scores.items():
            if score is not None:
                print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    main()

