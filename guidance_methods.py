import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipelineOutput,
)
import numpy as np

# Try to import SANA-related classes if available
try:
    from diffusers import FlowMatchEulerDiscreteScheduler
    SANA_SCHEDULER_AVAILABLE = True
except ImportError:
    try:
        # Alternative import name
        from diffusers import FlowDPMDiscreteScheduler
        SANA_SCHEDULER_AVAILABLE = True
    except ImportError:
        SANA_SCHEDULER_AVAILABLE = False


class GuidancePipeline(StableDiffusionPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_unet = None
        self.pretrained_text_encoder = None
        self.weak_unet = None

    def set_pretrained_models(self, pretrained_pipeline):
        self.pretrained_unet = pretrained_pipeline.unet
        self.pretrained_text_encoder = pretrained_pipeline.text_encoder

    def set_weak_model(self, weak_pipeline):
        self.weak_unet = weak_pipeline.unet

    def _get_unconditional_embeddings(self, batch_size, device):
        uncond_tokens = [""] * batch_size
        max_length = self.tokenizer.model_max_length
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        return uncond_embeddings

    def cfg_guidance(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        if prompt_embeds.shape[0] == latents.shape[0] * 2:
            uncond_embeds, cond_embeds = prompt_embeds.chunk(2)
        else:
            uncond_embeds = self._get_unconditional_embeddings(
                latents.shape[0], latents.device
            )
            cond_embeds = prompt_embeds
        
        uncond_noise_pred = self.unet(
            latents,
            timestep,
            encoder_hidden_states=uncond_embeds,
        ).sample
        
        cond_noise_pred = self.unet(
            latents,
            timestep,
            encoder_hidden_states=cond_embeds,
        ).sample
        
        noise_pred = uncond_noise_pred + guidance_scale * (
            cond_noise_pred - uncond_noise_pred
        )
        return noise_pred

    def ag_guidance(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        guidance_scale: float = 2.0,
        cfg_scale: float = 7.5,
    ) -> torch.Tensor:
        if self.weak_unet is None:
            if self.pretrained_unet is None:
                raise ValueError("Weak model or pretrained model not set for AutoGuidance")
            weak_unet = self.pretrained_unet
        else:
            weak_unet = self.weak_unet
        
        batch_size = latents.shape[0]
        uncond_embeds = self._get_unconditional_embeddings(batch_size, latents.device)
        
        weak_uncond_pred = weak_unet(
            latents,
            timestep,
            encoder_hidden_states=uncond_embeds,
        ).sample
        
        weak_cond_pred = weak_unet(
            latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
        ).sample
        
        weak_cfg_pred = weak_uncond_pred + cfg_scale * (
            weak_cond_pred - weak_uncond_pred
        )
        
        fine_uncond_pred = self.unet(
            latents,
            timestep,
            encoder_hidden_states=uncond_embeds,
        ).sample
        
        fine_cond_pred = self.unet(
            latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
        ).sample
        
        fine_cfg_pred = fine_uncond_pred + cfg_scale * (
            fine_cond_pred - fine_uncond_pred
        )
        
        noise_pred = weak_cfg_pred + guidance_scale * (
            fine_cfg_pred - weak_cfg_pred
        )
        return noise_pred

    def wig_guidance(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        guidance_scale: float = 7.5,
        omega: float = 0.0,
    ) -> torch.Tensor:
        if self.pretrained_unet is None:
            raise ValueError("Pretrained model not set for Weight Interpolation Guidance (WIG)")
        
        uncond_embeds = self._get_unconditional_embeddings(
            latents.shape[0], latents.device
        )
        
        if omega == 0.0:
            weak_noise_pred = self.pretrained_unet(
                latents,
                timestep,
                encoder_hidden_states=uncond_embeds,
            ).sample
        else:
            weak_unet = self._interpolate_unet_weights(omega)
            weak_noise_pred = weak_unet(
                latents,
                timestep,
                encoder_hidden_states=uncond_embeds,
            ).sample
        
        fine_noise_pred = self.unet(
            latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
        ).sample
        
        noise_pred = weak_noise_pred + guidance_scale * (
            fine_noise_pred - weak_noise_pred
        )
        return noise_pred

    def _interpolate_unet_weights(self, omega: float):
        if omega == 1.0:
            return self.unet
        if omega == 0.0:
            return self.pretrained_unet
        return self._create_interpolated_unet(omega)

    def _create_interpolated_unet(self, omega: float):
        class InterpolatedUNet:
            def __init__(self, pretrained_unet, fine_unet, omega):
                self.pretrained_unet = pretrained_unet
                self.fine_unet = fine_unet
                self.omega = omega
                self.config = fine_unet.config

            def __call__(self, *args, **kwargs):
                pretrained_out = self.pretrained_unet(*args, **kwargs)
                fine_out = self.fine_unet(*args, **kwargs)
                if isinstance(pretrained_out, torch.Tensor):
                    return (1 - self.omega) * pretrained_out + self.omega * fine_out
                else:
                    return type(pretrained_out)(
                        sample=(1 - self.omega) * pretrained_out.sample
                        + self.omega * fine_out.sample
                    )

        return InterpolatedUNet(self.pretrained_unet, self.unet, omega)

    def generate_with_guidance(
        self,
        prompt: str,
        guidance_method: str = "cfg",
        guidance_scale: float = 7.5,
        omega: float = 0.0,
        cfg_scale: float = 7.5,
        num_inference_steps: int = 50,
        height: int = 512,
        width: int = 512,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> StableDiffusionPipelineOutput:
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        batch_size = num_images_per_prompt
        do_classifier_free_guidance = guidance_method == "cfg"

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]

        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size
            max_length = text_inputs.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, height // 8, width // 8),
            generator=generator,
            device=self.device,
            dtype=text_embeddings.dtype,
        )
        latents = latents * self.scheduler.init_noise_sigma

        for t in timesteps:
            if guidance_method == "cfg":
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.cfg_guidance(
                    latent_model_input,
                    t,
                    text_embeddings,
                    guidance_scale=guidance_scale,
                )
            elif guidance_method == "ag":
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                if text_embeddings.shape[0] == latents.shape[0] * 2:
                    cond_embeds = text_embeddings[latents.shape[0]:]
                else:
                    cond_embeds = text_embeddings
                noise_pred = self.ag_guidance(
                    latent_model_input,
                    t,
                    cond_embeds,
                    guidance_scale=guidance_scale,
                    cfg_scale=cfg_scale,
                )
            elif guidance_method == "wig":
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                if text_embeddings.shape[0] == latents.shape[0] * 2:
                    cond_embeds = text_embeddings[latents.shape[0]:]
                else:
                    cond_embeds = text_embeddings
                noise_pred = self.wig_guidance(
                    latent_model_input,
                    t,
                    cond_embeds,
                    guidance_scale=guidance_scale,
                    omega=omega,
                )
            else:
                raise ValueError(f"Unknown guidance method: {guidance_method}")

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        images = [Image.fromarray(img) for img in image]

        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=None)


# SANA-specific guidance pipeline
# Note: SANA uses transformer blocks instead of UNet
# This is a template that should be adapted based on actual SANA pipeline structure
class SanaGuidancePipeline:
    """
    Guidance pipeline for SANA model.
    SANA uses transformer-based architecture and FlowDPM-Solver scheduler.
    """
    def __init__(self, sana_pipeline):
        """
        Initialize with a SANA pipeline.
        
        Args:
            sana_pipeline: The SANA pipeline (transformer-based)
        """
        self.pipeline = sana_pipeline
        self.pretrained_transformer = None
        self.pretrained_text_encoder = None
        self.weak_transformer = None
        self.device = sana_pipeline.device
        
        # SANA uses transformer instead of unet
        # Access transformer through pipeline.transformer or similar
        # This depends on actual SANA pipeline structure
        
    def set_pretrained_models(self, pretrained_pipeline):
        """Set pretrained transformer and text encoder for WIG guidance."""
        self.pretrained_transformer = pretrained_pipeline.transformer if hasattr(pretrained_pipeline, 'transformer') else None
        self.pretrained_text_encoder = pretrained_pipeline.text_encoder if hasattr(pretrained_pipeline, 'text_encoder') else None
        
    def set_weak_model(self, weak_pipeline):
        """Set weak transformer for AG guidance."""
        self.weak_transformer = weak_pipeline.transformer if hasattr(weak_pipeline, 'transformer') else None
    
    def _get_unconditional_embeddings(self, batch_size, device):
        """Get unconditional embeddings for classifier-free guidance."""
        uncond_tokens = [""] * batch_size
        max_length = self.pipeline.tokenizer.model_max_length
        uncond_input = self.pipeline.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.pipeline.text_encoder(uncond_input.input_ids.to(device))[0]
        return uncond_embeddings
    
    def cfg_guidance(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """CFG guidance for SANA."""
        # SANA transformer forward pass
        # Adapt based on actual SANA transformer API
        if prompt_embeds.shape[0] == latents.shape[0] * 2:
            uncond_embeds, cond_embeds = prompt_embeds.chunk(2)
        else:
            uncond_embeds = self._get_unconditional_embeddings(
                latents.shape[0], latents.device
            )
            cond_embeds = prompt_embeds
        
        # Note: Actual implementation depends on SANA transformer API
        # This is a placeholder structure
        transformer = self.pipeline.transformer if hasattr(self.pipeline, 'transformer') else None
        if transformer is None:
            raise ValueError("SANA transformer not found in pipeline")
        
        # Unconditional prediction
        uncond_noise_pred = transformer(
            latents,
            timestep,
            encoder_hidden_states=uncond_embeds,
        ).sample if hasattr(transformer, '__call__') else None
        
        # Conditional prediction
        cond_noise_pred = transformer(
            latents,
            timestep,
            encoder_hidden_states=cond_embeds,
        ).sample if hasattr(transformer, '__call__') else None
        
        if uncond_noise_pred is None or cond_noise_pred is None:
            raise NotImplementedError("SANA transformer API not fully implemented. Please adapt based on actual SANA pipeline.")
        
        noise_pred = uncond_noise_pred + guidance_scale * (
            cond_noise_pred - uncond_noise_pred
        )
        return noise_pred
    
    def ag_guidance(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        guidance_scale: float = 2.0,
        cfg_scale: float = 7.5,
    ) -> torch.Tensor:
        """AutoGuidance for SANA."""
        if self.weak_transformer is None:
            if self.pretrained_transformer is None:
                raise ValueError("Weak model or pretrained model not set for AutoGuidance")
            weak_transformer = self.pretrained_transformer
        else:
            weak_transformer = self.weak_transformer
        
        batch_size = latents.shape[0]
        uncond_embeds = self._get_unconditional_embeddings(batch_size, latents.device)
        
        transformer = self.pipeline.transformer if hasattr(self.pipeline, 'transformer') else None
        if transformer is None:
            raise ValueError("SANA transformer not found in pipeline")
        
        # Weak model predictions
        weak_uncond_pred = weak_transformer(
            latents,
            timestep,
            encoder_hidden_states=uncond_embeds,
        ).sample if hasattr(weak_transformer, '__call__') else None
        
        weak_cond_pred = weak_transformer(
            latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
        ).sample if hasattr(weak_transformer, '__call__') else None
        
        weak_cfg_pred = weak_uncond_pred + cfg_scale * (
            weak_cond_pred - weak_uncond_pred
        ) if weak_uncond_pred is not None and weak_cond_pred is not None else None
        
        # Fine model predictions
        fine_uncond_pred = transformer(
            latents,
            timestep,
            encoder_hidden_states=uncond_embeds,
        ).sample if hasattr(transformer, '__call__') else None
        
        fine_cond_pred = transformer(
            latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
        ).sample if hasattr(transformer, '__call__') else None
        
        fine_cfg_pred = fine_uncond_pred + cfg_scale * (
            fine_cond_pred - fine_uncond_pred
        ) if fine_uncond_pred is not None and fine_cond_pred is not None else None
        
        if weak_cfg_pred is None or fine_cfg_pred is None:
            raise NotImplementedError("SANA transformer API not fully implemented for AG. Please adapt based on actual SANA pipeline.")
        
        noise_pred = weak_cfg_pred + guidance_scale * (
            fine_cfg_pred - weak_cfg_pred
        )
        return noise_pred
    
    def wig_guidance_sana(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        guidance_scale: float = 7.5,
        omega: float = 0.0,
    ) -> torch.Tensor:
        """Weight Interpolation Guidance for SANA."""
        if self.pretrained_transformer is None:
            raise ValueError("Pretrained model not set for Weight Interpolation Guidance (WIG)")
        
        uncond_embeds = self._get_unconditional_embeddings(
            latents.shape[0], latents.device
        )
        
        transformer = self.pipeline.transformer if hasattr(self.pipeline, 'transformer') else None
        if transformer is None:
            raise ValueError("SANA transformer not found in pipeline")
        
        # Weak prediction (pretrained or interpolated)
        if omega == 0.0:
            weak_transformer = self.pretrained_transformer
        else:
            # Interpolate transformer weights (placeholder - needs actual implementation)
            weak_transformer = self._interpolate_transformer_weights(omega)
        
        weak_noise_pred = weak_transformer(
            latents,
            timestep,
            encoder_hidden_states=uncond_embeds,
        ).sample if hasattr(weak_transformer, '__call__') else None
        
        # Fine prediction
        fine_noise_pred = transformer(
            latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
        ).sample if hasattr(transformer, '__call__') else None
        
        if weak_noise_pred is None or fine_noise_pred is None:
            raise NotImplementedError("SANA transformer API not fully implemented for WIG. Please adapt based on actual SANA pipeline.")
        
        noise_pred = weak_noise_pred + guidance_scale * (
            fine_noise_pred - weak_noise_pred
        )
        return noise_pred
    
    def _interpolate_transformer_weights(self, omega: float):
        """Interpolate transformer weights between pretrained and fine-tuned."""
        # Placeholder - needs actual implementation based on SANA transformer structure
        # Similar to _interpolate_unet_weights but for transformer
        if omega == 1.0:
            return self.pipeline.transformer if hasattr(self.pipeline, 'transformer') else None
        if omega == 0.0:
            return self.pretrained_transformer
        
        class InterpolatedTransformer:
            def __init__(self, pretrained_transformer, fine_transformer, omega):
                self.pretrained_transformer = pretrained_transformer
                self.fine_transformer = fine_transformer
                self.omega = omega
                self.config = fine_transformer.config if hasattr(fine_transformer, 'config') else None

            def __call__(self, *args, **kwargs):
                pretrained_out = self.pretrained_transformer(*args, **kwargs)
                fine_out = self.fine_transformer(*args, **kwargs)
                if isinstance(pretrained_out, torch.Tensor):
                    return (1 - self.omega) * pretrained_out + self.omega * fine_out
                else:
                    return type(pretrained_out)(
                        sample=(1 - self.omega) * pretrained_out.sample
                        + self.omega * fine_out.sample
                    )
        
        fine_transformer = self.pipeline.transformer if hasattr(self.pipeline, 'transformer') else None
        if fine_transformer is None:
            raise ValueError("Fine transformer not found")
        
        return InterpolatedTransformer(self.pretrained_transformer, fine_transformer, omega)
    
    def generate_with_guidance(
        self,
        prompt: str,
        guidance_method: str = "cfg",
        guidance_scale: float = 7.5,
        omega: float = 0.0,
        cfg_scale: float = 7.5,
        num_inference_steps: int = 20,  # SANA uses 20 steps with FlowDPM-Solver
        height: int = 1024,  # SANA uses 1024x1024
        width: int = 1024,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> StableDiffusionPipelineOutput:
        """
        Generate images with SANA pipeline using guidance methods.
        
        Note: This is a template that needs to be adapted based on actual SANA pipeline API.
        SANA uses:
        - FlowDPM-Solver scheduler (20 inference steps)
        - 1024x1024 resolution
        - Transformer-based architecture
        """
        # Set scheduler timesteps
        if hasattr(self.pipeline, 'scheduler'):
            self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.pipeline.scheduler.timesteps
        
        # Tokenize prompt
        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.pipeline.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        batch_size = num_images_per_prompt
        
        # Initialize latents (SANA might use different latent dimensions)
        # Adapt based on actual SANA VAE/latent structure
        latents = torch.randn(
            (batch_size, 4, height // 8, width // 8),  # Standard SD latent shape, adapt if needed
            generator=generator,
            device=self.device,
            dtype=text_embeddings.dtype,
        )
        if hasattr(self.pipeline.scheduler, 'init_noise_sigma'):
            latents = latents * self.pipeline.scheduler.init_noise_sigma
        
        # Scheduler loop
        for t in timesteps:
            if guidance_method == "cfg":
                latent_model_input = torch.cat([latents] * 2)
                if hasattr(self.pipeline.scheduler, 'scale_model_input'):
                    latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.cfg_guidance(
                    latent_model_input,
                    t,
                    text_embeddings,
                    guidance_scale=guidance_scale,
                )
            elif guidance_method == "ag":
                latent_model_input = self.pipeline.scheduler.scale_model_input(latents, t) if hasattr(self.pipeline.scheduler, 'scale_model_input') else latents
                if text_embeddings.shape[0] == latents.shape[0] * 2:
                    cond_embeds = text_embeddings[latents.shape[0]:]
                else:
                    cond_embeds = text_embeddings
                noise_pred = self.ag_guidance(
                    latent_model_input,
                    t,
                    cond_embeds,
                    guidance_scale=guidance_scale,
                    cfg_scale=cfg_scale,
                )
            elif guidance_method == "wig":
                latent_model_input = self.pipeline.scheduler.scale_model_input(latents, t) if hasattr(self.pipeline.scheduler, 'scale_model_input') else latents
                if text_embeddings.shape[0] == latents.shape[0] * 2:
                    cond_embeds = text_embeddings[latents.shape[0]:]
                else:
                    cond_embeds = text_embeddings
                noise_pred = self.wig_guidance_sana(
                    latent_model_input,
                    t,
                    cond_embeds,
                    guidance_scale=guidance_scale,
                    omega=omega,
                )
            else:
                raise NotImplementedError(f"Guidance method {guidance_method} not yet implemented for SANA")
            
            # Step scheduler
            if hasattr(self.pipeline.scheduler, 'step'):
                step_output = self.pipeline.scheduler.step(noise_pred, t, latents)
                latents = step_output.prev_sample if hasattr(step_output, 'prev_sample') else step_output
        
        # Decode latents to images
        if hasattr(self.pipeline, 'vae'):
            image = self.pipeline.vae.decode(latents / self.pipeline.vae.config.scaling_factor, return_dict=False)[0]
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = (image * 255).round().astype("uint8")
            images = [Image.fromarray(img) for img in image]
        else:
            raise NotImplementedError("SANA VAE decoding not implemented")
        
        return StableDiffusionPipelineOutput(images=images, nsfw_content_detected=None)
