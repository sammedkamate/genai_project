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


def evaluate_cfg(
    pretrained_model_path: str,
    lora_path: str,
    evaluation_prompts_path: str,
    output_dir: str,
    device: str = "cuda",
    guidance_scale: float = 7.5,
    num_images_per_prompt: int = 8,
):
    finetuned_pipeline = load_finetuned_pipeline(
        pretrained_model_path, lora_path, device
    )

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
            guidance_method="cfg",
            guidance_scale=guidance_scale,
            num_inference_steps=50,
            num_images_per_prompt=num_images_per_prompt,
        )
        return result.images

    evaluator.pipe = finetuned_pipeline
    evaluator._generate_images = generate_images

    scores = evaluator.evaluate()
    return scores


def optimize_cfg(
    pretrained_model_path: str,
    lora_path: str,
    evaluation_prompts_path: str,
    output_dir: str,
    device: str = "cuda",
    lambda_range: tuple = (2.5, 10.0),
    num_images_per_prompt: int = 8,
):
    finetuned_pipeline = load_finetuned_pipeline(
        pretrained_model_path, lora_path, device
    )

    def objective(lambda_val):
        scores = evaluate_cfg(
            pretrained_model_path=pretrained_model_path,
            lora_path=lora_path,
            evaluation_prompts_path=evaluation_prompts_path,
            output_dir=os.path.join(output_dir, f"temp_cfg_{lambda_val}"),
            device=device,
            guidance_scale=lambda_val,
            num_images_per_prompt=num_images_per_prompt,
        )
        dino_score = scores.get("DINO", 0)
        return -dino_score

    best_lambda, best_score = golden_section_search(
        objective, lambda_range[0], lambda_range[1]
    )

    best_scores = evaluate_cfg(
        pretrained_model_path=pretrained_model_path,
        lora_path=lora_path,
        evaluation_prompts_path=evaluation_prompts_path,
        output_dir=os.path.join(output_dir, "cfg_optimized"),
        device=device,
        guidance_scale=best_lambda,
        num_images_per_prompt=num_images_per_prompt,
    )

    return best_lambda, best_scores


def main():
    parser = argparse.ArgumentParser(description="CFG Guidance Evaluation")
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
        default="cfg_outputs",
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
        "--optimize",
        action="store_true",
        help="Optimize guidance scale using golden search",
    )
    parser.add_argument(
        "--lambda_range",
        type=float,
        nargs=2,
        default=[2.5, 10.0],
        help="Range for lambda optimization",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=8,
        help="Number of images per prompt",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.optimize:
        print("Optimizing CFG guidance scale...")
        best_lambda, scores = optimize_cfg(
            pretrained_model_path=args.pretrained_model_path,
            lora_path=args.lora_path,
            evaluation_prompts_path=args.evaluation_prompts_path,
            output_dir=args.output_dir,
            device=args.device,
            lambda_range=tuple(args.lambda_range),
            num_images_per_prompt=args.num_images_per_prompt,
        )
        print(f"\n=== CFG Optimization Results ===")
        print(f"Best Lambda: {best_lambda:.4f}")
        print(f"Scores:")
        for metric, score in scores.items():
            if score is not None:
                print(f"  {metric}: {score:.4f}")

        results = {
            "method": "cfg",
            "lambda": best_lambda,
            "scores": scores,
        }
        results_path = os.path.join(args.output_dir, "cfg_optimization_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
    else:
        print(f"Evaluating CFG with guidance_scale={args.guidance_scale}...")
        scores = evaluate_cfg(
            pretrained_model_path=args.pretrained_model_path,
            lora_path=args.lora_path,
            evaluation_prompts_path=args.evaluation_prompts_path,
            output_dir=args.output_dir,
            device=args.device,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images_per_prompt,
        )
        print("\n=== CFG Evaluation Results ===")
        for metric, score in scores.items():
            if score is not None:
                print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    main()

