import os
import json
import torch
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from diffusers import StableDiffusionPipeline
import torch.nn.functional as F
import clip
import open_clip


class Evaluator:

    def __init__(
        self,
        model_path: str,
        json_path: str,
        output_dir: str = "outputs",
        device: Optional[str] = None,
        num_images_per_prompt: int = 1,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.num_images = num_images_per_prompt
        os.makedirs(output_dir, exist_ok=True)

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to(self.device)

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        # Build an image-encoder for diversity/consistency metric.
        # Prefer DINOv2-g14 if available in your open_clip build, otherwise fall back to a widely available OpenCLIP model.
        try:
            dino_model, _, dino_preprocess = open_clip.create_model_and_transforms(
                "dinov2_vitg14", pretrained="laion2b_s39b_b160k"
            )
        except Exception as e:
            print(
                f"Warning: {e}. Falling back to OpenCLIP ViT-L-14 (laion2b_s32b_b82k) for the DINO metric."
            )
            dino_model, _, dino_preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="laion2b_s32b_b82k"
            )
        self.dino_model = dino_model.to(self.device).eval()
        self.preprocess_dino = dino_preprocess

        self.prompts = self._load_prompts(json_path)

    def _load_prompts(self, json_path: str) -> List[str]:
        with open(json_path, "r") as f:
            data = json.load(f)
        subjects = data["subjects"]
        templates = data["prompts"]
        return [p.replace("<subject>", s) for s in subjects for p in templates]

    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            feats = self.clip_model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def _encode_images_clip(self, images: List[Image.Image]) -> torch.Tensor:
        imgs = torch.stack([self.clip_preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            feats = self.clip_model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def _encode_images_dino(self, images: List[Image.Image]) -> torch.Tensor:
        imgs = torch.stack([self.preprocess_dino(img) for img in images]).to(self.device)
        with torch.no_grad():
            feats = self.dino_model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).sum(dim=-1)

    def _generate_images(self, prompt: str) -> List[Image.Image]:
        images = []
        for _ in range(self.num_images):
            img = self.pipe(prompt, guidance_scale=7.5).images[0]
            images.append(img)
        return images

    def evaluate(self) -> Dict[str, float]:
        clip_i_scores, clip_t_scores, dino_scores = [], [], []
        text_features = self._encode_text(self.prompts)

        for idx, prompt in enumerate(tqdm(self.prompts, desc="Evaluating")):
            images = self._generate_images(prompt)
            img_clip = self._encode_images_clip(images)
            img_dino = self._encode_images_dino(images)
            t_feat = text_features[idx].unsqueeze(0).repeat(self.num_images, 1)

            clip_t = self._cosine(img_clip, t_feat).mean().item()
            clip_t_scores.append(clip_t)

            if self.num_images > 1:
                clip_i = F.cosine_similarity(
                    img_clip[0].unsqueeze(0),
                    img_clip[1:].mean(dim=0, keepdim=True),
                ).item()
                dino_i = F.cosine_similarity(
                    img_dino[0].unsqueeze(0),
                    img_dino[1:].mean(dim=0, keepdim=True),
                ).item()
                clip_i_scores.append(clip_i)
                dino_scores.append(dino_i)

            images[0].save(os.path.join(self.output_dir, f"{idx:04d}.png"))

        return {
            "CLIP-I": float(np.mean(clip_i_scores)) if clip_i_scores else None,
            "CLIP-T": float(np.mean(clip_t_scores)),
            "DINO": float(np.mean(dino_scores)) if dino_scores else None,
        }


if __name__ == "__main__":
    evaluator = Evaluator(
        model_path="",
        json_path="",
        num_images_per_prompt=1,
    )
    scores = evaluator.evaluate()
    print("\n=== Mean Evaluation Metrics ===")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")
