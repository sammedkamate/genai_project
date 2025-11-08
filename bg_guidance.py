import argparse
import os
import json
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from guidance_methods import GuidancePipeline
from evaluation_code import Evaluator
from golden_search import golden_section_search


def load_pretrained_pipeline(model_path: str, device: str = "cuda"):
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    return pipeline


def load_finetuned_pipeline(
    pretrained_model_path: str,
    lora_path: str,
    device: str = "cuda",
):
    pipeline = GuidancePipeline.from_pretrained(
        pretrained_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.unet.load_attn_procs(lora_path)
    pipeline = pipeline.to(device)
    return pipeline


def evaluate_bg(
    pretrained_model_path: str,
    lora_path: str,
    evaluation_prompts_path: str,
    output_dir: str,
    device: str = "cuda",
    guidance_scale: float = 7.5,
    omega: float = 0.0,
    num_images_per_prompt: int = 8,
):
    pretrained_pipeline = load_pretrained_pipeline(pretrained_model_path, device)
    finetuned_pipeline = load_finetuned_pipeline(
        pretrained_model_path, lora_path, device
    )
    finetuned_pipeline.set_pretrained_models(pretrained_pipeline)

    evaluator = Evaluator(
        model_path=None,
        json_path=evaluation_prompts_path,
        output_dir=output_dir,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
    )

    def generate_images(prompt):
        result = finetuned_pipeline.generate_with_guidance(
            prompt=prompt,
            guidance_method="bg",
            guidance_scale=guidance_scale,
            omega=omega,
            num_inference_steps=50,
            num_images_per_prompt=num_images_per_prompt,
        )
        return result.images

    evaluator.pipe = finetuned_pipeline
    evaluator._generate_images = generate_images

    scores = evaluator.evaluate()
    return scores


def optimize_bg_lambda(
    pretrained_model_path: str,
    lora_path: str,
    evaluation_prompts_path: str,
    output_dir: str,
    device: str = "cuda",
    lambda_range: tuple = (-10.0, 10.0),
    omega: float = 0.0,
    num_images_per_prompt: int = 8,
):
    pretrained_pipeline = load_pretrained_pipeline(pretrained_model_path, device)
    finetuned_pipeline = load_finetuned_pipeline(
        pretrained_model_path, lora_path, device
    )
    finetuned_pipeline.set_pretrained_models(pretrained_pipeline)

    def objective(lambda_val):
        scores = evaluate_bg(
            pretrained_model_path=pretrained_model_path,
            lora_path=lora_path,
            evaluation_prompts_path=evaluation_prompts_path,
            output_dir=os.path.join(output_dir, f"temp_bg_lambda_{lambda_val}"),
            device=device,
            guidance_scale=lambda_val,
            omega=omega,
            num_images_per_prompt=num_images_per_prompt,
        )
        dino_score = scores.get("DINO", 0)
        return -dino_score

    best_lambda, best_score = golden_section_search(
        objective, lambda_range[0], lambda_range[1]
    )

    best_scores = evaluate_bg(
        pretrained_model_path=pretrained_model_path,
        lora_path=lora_path,
        evaluation_prompts_path=evaluation_prompts_path,
        output_dir=os.path.join(output_dir, "bg_optimized_lambda"),
        device=device,
        guidance_scale=best_lambda,
        omega=omega,
        num_images_per_prompt=num_images_per_prompt,
    )

    return best_lambda, best_scores


def optimize_bg_omega(
    pretrained_model_path: str,
    lora_path: str,
    evaluation_prompts_path: str,
    output_dir: str,
    device: str = "cuda",
    guidance_scale: float = 7.5,
    omega_range: tuple = (0.0, 1.0),
    num_images_per_prompt: int = 8,
):
    pretrained_pipeline = load_pretrained_pipeline(pretrained_model_path, device)
    finetuned_pipeline = load_finetuned_pipeline(
        pretrained_model_path, lora_path, device
    )
    finetuned_pipeline.set_pretrained_models(pretrained_pipeline)

    omega_values = [round(i * 0.1, 1) for i in range(11)]
    best_omega = 0.0
    best_dino = -float('inf')
    all_results = {}

    for omega_val in omega_values:
        print(f"Evaluating omega={omega_val:.1f}...")
        scores = evaluate_bg(
            pretrained_model_path=pretrained_model_path,
            lora_path=lora_path,
            evaluation_prompts_path=evaluation_prompts_path,
            output_dir=os.path.join(output_dir, f"temp_bg_omega_{omega_val}"),
            device=device,
            guidance_scale=guidance_scale,
            omega=omega_val,
            num_images_per_prompt=num_images_per_prompt,
        )
        dino_score = scores.get("DINO", 0)
        all_results[omega_val] = scores
        
        if dino_score > best_dino:
            best_dino = dino_score
            best_omega = omega_val

    print(f"\nBest omega: {best_omega:.1f} (DINO: {best_dino:.4f})")
    
    best_scores = evaluate_bg(
        pretrained_model_path=pretrained_model_path,
        lora_path=lora_path,
        evaluation_prompts_path=evaluation_prompts_path,
        output_dir=os.path.join(output_dir, "bg_optimized_omega"),
        device=device,
        guidance_scale=guidance_scale,
        omega=best_omega,
        num_images_per_prompt=num_images_per_prompt,
    )

    return best_omega, best_scores


def optimize_bg_both(
    pretrained_model_path: str,
    lora_path: str,
    evaluation_prompts_path: str,
    output_dir: str,
    device: str = "cuda",
    lambda_range: tuple = (-10.0, 10.0),
    omega_range: tuple = (0.0, 1.0),
    num_images_per_prompt: int = 8,
):
    print("Optimizing lambda first...")
    best_lambda, _ = optimize_bg_lambda(
        pretrained_model_path=pretrained_model_path,
        lora_path=lora_path,
        evaluation_prompts_path=evaluation_prompts_path,
        output_dir=output_dir,
        device=device,
        lambda_range=lambda_range,
        omega=0.0,
        num_images_per_prompt=num_images_per_prompt,
    )

    print(f"Best lambda: {best_lambda:.4f}")
    print("Optimizing omega...")
    best_omega, best_scores = optimize_bg_omega(
        pretrained_model_path=pretrained_model_path,
        lora_path=lora_path,
        evaluation_prompts_path=evaluation_prompts_path,
        output_dir=output_dir,
        device=device,
        guidance_scale=best_lambda,
        omega_range=omega_range,
        num_images_per_prompt=num_images_per_prompt,
    )

    return best_lambda, best_omega, best_scores


def main():
    parser = argparse.ArgumentParser(description="Bhavik Guidance Evaluation")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to trained LoRA weights",
    )
    parser.add_argument(
        "--evaluation_prompts_path",
        type=str,
        default="evaluation_prompts.json",
        help="Path to evaluation prompts JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="bg_outputs",
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
        default=8,
        help="Number of images per prompt",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.optimize == "lambda":
        print("Optimizing BG lambda...")
        best_lambda, scores = optimize_bg_lambda(
            pretrained_model_path=args.pretrained_model_path,
            lora_path=args.lora_path,
            evaluation_prompts_path=args.evaluation_prompts_path,
            output_dir=args.output_dir,
            device=args.device,
            lambda_range=tuple(args.lambda_range),
            omega=args.omega,
            num_images_per_prompt=args.num_images_per_prompt,
        )
        print(f"\n=== BG Lambda Optimization Results ===")
        print(f"Best Lambda: {best_lambda:.4f}")
        print(f"Omega: {args.omega:.4f}")
        print(f"Scores:")
        for metric, score in scores.items():
            if score is not None:
                print(f"  {metric}: {score:.4f}")

        results = {
            "method": "bg",
            "lambda": best_lambda,
            "omega": args.omega,
            "scores": scores,
        }
        results_path = os.path.join(args.output_dir, "bg_lambda_optimization_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

    elif args.optimize == "omega":
        print("Optimizing BG omega...")
        best_omega, scores = optimize_bg_omega(
            pretrained_model_path=args.pretrained_model_path,
            lora_path=args.lora_path,
            evaluation_prompts_path=args.evaluation_prompts_path,
            output_dir=args.output_dir,
            device=args.device,
            guidance_scale=args.guidance_scale,
            omega_range=tuple(args.omega_range),
            num_images_per_prompt=args.num_images_per_prompt,
        )
        print(f"\n=== BG Omega Optimization Results ===")
        print(f"Lambda: {args.guidance_scale:.4f}")
        print(f"Best Omega: {best_omega:.4f}")
        print(f"Scores:")
        for metric, score in scores.items():
            if score is not None:
                print(f"  {metric}: {score:.4f}")

        results = {
            "method": "bg",
            "lambda": args.guidance_scale,
            "omega": best_omega,
            "scores": scores,
        }
        results_path = os.path.join(args.output_dir, "bg_omega_optimization_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

    elif args.optimize == "both":
        print("Optimizing BG lambda and omega...")
        best_lambda, best_omega, scores = optimize_bg_both(
            pretrained_model_path=args.pretrained_model_path,
            lora_path=args.lora_path,
            evaluation_prompts_path=args.evaluation_prompts_path,
            output_dir=args.output_dir,
            device=args.device,
            lambda_range=tuple(args.lambda_range),
            omega_range=tuple(args.omega_range),
            num_images_per_prompt=args.num_images_per_prompt,
        )
        print(f"\n=== BG Full Optimization Results ===")
        print(f"Best Lambda: {best_lambda:.4f}")
        print(f"Best Omega: {best_omega:.4f}")
        print(f"Scores:")
        for metric, score in scores.items():
            if score is not None:
                print(f"  {metric}: {score:.4f}")

        results = {
            "method": "bg",
            "lambda": best_lambda,
            "omega": best_omega,
            "scores": scores,
        }
        results_path = os.path.join(args.output_dir, "bg_full_optimization_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

    else:
        print(f"Evaluating BG with guidance_scale={args.guidance_scale}, omega={args.omega}...")
        scores = evaluate_bg(
            pretrained_model_path=args.pretrained_model_path,
            lora_path=args.lora_path,
            evaluation_prompts_path=args.evaluation_prompts_path,
            output_dir=args.output_dir,
            device=args.device,
            guidance_scale=args.guidance_scale,
            omega=args.omega,
            num_images_per_prompt=args.num_images_per_prompt,
        )
        print("\n=== BG Evaluation Results ===")
        for metric, score in scores.items():
            if score is not None:
                print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    main()

