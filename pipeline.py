import torch
from diffusers import StableDiffusionPipeline
from typing import Optional, List, Union, Dict, Any

class WatermarkInjectionPipeline(StableDiffusionPipeline):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, 
                 safety_checker=None, feature_extractor=None, image_encoder=None, 
                 requires_safety_checker=False, wm_encoder=None, wm_decoder=None):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, 
                         safety_checker, feature_extractor, image_encoder, requires_safety_checker)
        self.wm_encoder = wm_encoder
        self.wm_decoder = wm_decoder
        self._inference_mode = True
    
    def set_inference_mode(self, mode=True):
        self._inference_mode = mode
        if self.unet: 
            self.unet.eval() if mode else self.unet.train()
    
    def _get_beta_t(self, t, scheduler):
            if hasattr(scheduler, 'alphas_cumprod'):
                alphas = scheduler.alphas_cumprod.to(self.device)
            else:
                return torch.tensor(1.0, device=self.device) 

            if torch.is_tensor(t): 
                t_idx = int(t.item())  
            else: 
                t_idx = int(t)        
                
            if t_idx >= len(alphas): 
                t_idx = len(alphas) - 1
            elif t_idx < 0:
                t_idx = 0
                
            alpha = alphas[t_idx]
                
            alpha = alpha.detach().clone() if torch.is_tensor(alpha) else torch.tensor(alpha, device=self.device)
            return torch.sqrt(alpha) / torch.sqrt(1 - alpha)
    
    @torch.no_grad()
    def __call__(
        self, 
        prompt: Union[str, List[str]], 
        latents: Optional[torch.FloatTensor] = None, 
        wm_injection_start_step: int = 20, 
        wm_injection_end_step: int = 45, 
        wm_weight: float = 1.0, 
        secret_input: Optional[torch.Tensor] = None, 
        height: Optional[int] = None, 
        width: Optional[int] = None, 
        num_inference_steps: int = 50, 
        guidance_scale: float = 7.5, 
        enable_watermark: bool = True, 
        **kwargs
    ):
        self.set_inference_mode(True)
        
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        
        prompt_embeds, neg_embeds = self.encode_prompt(
            prompt, device, 1, guidance_scale > 1.0, negative_prompt=None
        )
        text_embeddings = torch.cat([neg_embeds, prompt_embeds]) if guidance_scale > 1.0 else prompt_embeds
        
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        if latents is None:
            latents = self.prepare_latents(
                batch_size, self.unet.config.in_channels, height, width, 
                text_embeddings.dtype, device, kwargs.get('generator'), None
            )
        else:
            latents = latents.to(device=device, dtype=text_embeddings.dtype)

        wm_residual = None
        if enable_watermark and self.wm_encoder is not None and secret_input is not None:
            secret_input = secret_input.to(device=device, dtype=text_embeddings.dtype)
            wm_residual = self.wm_encoder(secret_input)
            if wm_residual.shape[0] != latents.shape[0]:
                wm_residual = wm_residual.repeat(latents.shape[0], 1, 1, 1)

        extra_kwargs = self.prepare_extra_step_kwargs(kwargs.get('generator'), 0.0)
        
        injected_count = 0
        
        for i, t in enumerate(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample
            
            if guidance_scale > 1.0:
                uncond, text = noise_pred.chunk(2)
                noise_pred = uncond + guidance_scale * (text - uncond)
            
            if wm_residual is not None and wm_injection_start_step <= i <= wm_injection_end_step:
                beta_t = self._get_beta_t(t, self.scheduler)
                
                if injected_count == 0:
                     print(f"   [Pipeline] Step {i}: beta_t={beta_t.mean().item():.4f}, weight={wm_weight}")
                
                noise_pred = noise_pred - beta_t * wm_weight * wm_residual
                injected_count += 1
            
            latents = self.scheduler.step(noise_pred, t, latents, **extra_kwargs).prev_sample
            
        if not kwargs.get("output_type") == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type="pil", do_denormalize=[True] * batch_size)
        else:
            image = latents
            
        return {"images": image, "latents": latents}