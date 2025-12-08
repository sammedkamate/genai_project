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
        ref_dir: str,
        output_dir: str = "outputs",
        device: Optional[str] = None,
        num_images_per_prompt: int = 1,
        subject: Optional[str] = None,    # New optional argument
        ref_batch_size: int = 32,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir
        self.num_images = num_images_per_prompt
        self.ref_batch_size = ref_batch_size
        os.makedirs(output_dir, exist_ok=True)

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to(self.device)

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        try:
            dino_model, _, dino_preprocess = open_clip.create_model_and_transforms(
                "dinov2_vitg14", pretrained="laion2b_s39b_b160k"
            )
        except Exception as e:
            print(
                f"Warning: {e}. Falling back to OpenCLIP ViT-L-14 (laion2b_s32b_b82k) for DINO metric."
            )
            dino_model, _, dino_preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="laion2b_s32b_b82k"
            )
        self.dino_model = dino_model.to(self.device).eval()
        self.preprocess_dino = dino_preprocess

        # Load whole json
        with open(json_path, "r") as f:
            data = json.load(f)

        # Use provided subject or read from JSON
        if subject is None:
            self.subjects = data.get("subjects", [])
        else:
            self.subjects = [subject]

        self.prompts = [p.replace("<subject>", s) for s in self.subjects for p in data.get("prompts", [])]

        # Load all images in ref_dir as references (multi-image support)
        self.reference_pool = self._load_reference_pool(ref_dir)
        # Pre-encode reference features once for efficiency
        self.ref_clip_feats = self._encode_images_clip_batched(self.reference_pool, batch_size=self.ref_batch_size) if self.reference_pool else None
        self.ref_dino_feats = self._encode_images_dino_batched(self.reference_pool, batch_size=self.ref_batch_size) if self.reference_pool else None

        torch.manual_seed(42)
        np.random.seed(42)

    # --------------------- Helper functions ---------------------

    def _load_prompts(self, json_path: str) -> List[str]:
        with open(json_path, "r") as f:
            data = json.load(f)
        subjects = data["subjects"]
        templates = data["prompts"]
        return [p.replace("<subject>", s) for s in subjects for p in templates]

    def _load_reference_pool(self, ref_dir: str) -> List[Image.Image]:
        pool: List[Image.Image] = []
        if not os.path.isdir(ref_dir):
            print(f"Warning: ref_dir '{ref_dir}' does not exist or is not a directory. CLIP-I/DINO will be skipped.")
            return pool
        for fname in os.listdir(ref_dir):
            path = os.path.join(ref_dir, fname)
            try:
                img = Image.open(path).convert("RGB")
                pool.append(img)
            except Exception:
                continue
        return pool

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

    def _encode_images_clip_batched(self, images: List[Image.Image], batch_size: int = 32) -> Optional[torch.Tensor]:
        if not images:
            return None
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            feats = self._encode_images_clip(batch)
            all_feats.append(feats)
        return torch.cat(all_feats, dim=0)

    def _encode_images_dino(self, images: List[Image.Image]) -> torch.Tensor:
        imgs = torch.stack([self.preprocess_dino(img) for img in images]).to(self.device)
        with torch.no_grad():
            feats = self.dino_model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def _encode_images_dino_batched(self, images: List[Image.Image], batch_size: int = 32) -> Optional[torch.Tensor]:
        if not images:
            return None
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            feats = self._encode_images_dino(batch)
            all_feats.append(feats)
        return torch.cat(all_feats, dim=0)

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).sum(dim=-1)

    def _generate_images(self, prompt: str) -> List[Image.Image]:
        images = []
        for _ in range(self.num_images):
            img = self.pipe(prompt, guidance_scale=7.5).images[0]
            images.append(img)
        return images

    # --------------------- Main evaluation ---------------------

    def evaluate(self) -> Dict[str, float]:
        clip_t_scores: List[float] = []
        clip_i_scores: List[float] = []
        dino_scores: List[float] = []
        text_features = self._encode_text(self.prompts) if self.prompts else None

        # Prepare reference feature matrices (normalized already)
        ref_clip_feats = self.ref_clip_feats
        ref_dino_feats = self.ref_dino_feats

        for idx, prompt in enumerate(tqdm(self.prompts, desc="Evaluating")):
            # Generate images and encode features
            gen_images = self._generate_images(prompt)
            gen_clip = self._encode_images_clip(gen_images)
            gen_dino = self._encode_images_dino(gen_images)

            # CLIP-T: text ↔ generated image (average over generated images)
            if text_features is not None:
                text_feat = text_features[idx].unsqueeze(0)
                # Broadcast cosine: each gen image vs the text feature
                clip_t = self._cosine(gen_clip, text_feat).mean().item()
                clip_t_scores.append(clip_t)

            # CLIP-I: generated ↔ all references (average of similarity matrix)
            if ref_clip_feats is not None and ref_clip_feats.shape[0] > 0:
                # gen_clip: [G, D], ref_clip_feats: [R, D] ⇒ sim: [G, R]
                sim = torch.mm(gen_clip, ref_clip_feats.t())
                clip_i_scores.append(sim.mean().item())

            # DINO: generated ↔ all references (average of similarity matrix)
            if ref_dino_feats is not None and ref_dino_feats.shape[0] > 0:
                sim = torch.mm(gen_dino, ref_dino_feats.t())
                dino_scores.append(sim.mean().item())

            # Save first generated image for inspection
            gen_images[0].save(os.path.join(self.output_dir, f"{idx:04d}.png"))

        # Aggregate means
        return {
            "CLIP-T": float(np.mean(clip_t_scores)) if clip_t_scores else None,
            "CLIP-I": float(np.mean(clip_i_scores)) if clip_i_scores else None,
            "DINO": float(np.mean(dino_scores)) if dino_scores else None,
        }


if __name__ == "__main__":
    evaluator = Evaluator(
        model_path="path/to/your/model",
        json_path="subjects_and_prompts.json",
        ref_dir="path/to/reference_images",
        num_images_per_prompt=1,
    )
    scores = evaluator.evaluate()
    print("\n=== Mean Evaluation Metrics ===")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")