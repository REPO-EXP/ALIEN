
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import json
from diffusers import AutoencoderKL
import kornia as K
import io
import torchvision.transforms as T
import torch.nn.functional as F
import os
import argparse
import math
from pathlib import Path
from torchvision import transforms
import torch
import torch.nn.functional as F
import json
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DDIMScheduler, # 引入 DDIMScheduler
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import OrderedDict 
import traceback

try:
    from safetensors.torch import load_file, save_file
    USE_SAFETENSORS = True
except ImportError:
    USE_SAFETENSORS = False

def ssim_loss(img1, img2, window_size=11, size_average=True):
    from pytorch_msssim import ssim
    return 1 - ssim(img1, img2, data_range=1.0, win_size=window_size, size_average=size_average)

def collate(examples):        
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_image"] for example in examples]

    input_ids_trigger = [example["instance_prompt_ids_with_trigger"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    
    input_ids_trigger = torch.cat(input_ids_trigger, dim=0)

    batch = {
        "input_ids": input_ids,
        "input_ids_trigger": input_ids_trigger,
        "pixel_values": pixel_values
    }

    return batch


def encode_prompt(text_encoder, input_ids):
    text_input_ids = input_ids
    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask = None,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


     
class DreamBoothDataset_modified(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """       

    def __init__(
        self,
        instance_data_root,
        tokenizer,
        size=512,
        center_crop=False,
        tokenizer_max_length=None,
        prompt_trigger = '',
        use_null_prompt=False
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length

        with open(f'{instance_data_root}/metadata_generated.jsonl') as f:
            metadata = [json.loads(line) for line in f]
        file_names = [os.path.join(instance_data_root,item['file_name']) for item in metadata]
        prompts = [item['text'] for item in metadata]
        self.instance_images_path = file_names
        self.instance_prompts = prompts
        self._length = len(self.instance_images_path)  
        self.use_null_prompt = use_null_prompt


    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
 
        instance_image = Image.open(self.instance_images_path[index % self._length])
        instance_image = exif_transpose(instance_image)
        if self.use_null_prompt:
            instance_prompt = ""
        else:
            instance_prompt = self.instance_prompts[index % self._length]

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
            

        transforms_pipeline = transforms.Compose(
            [
                transforms.CenterCrop(min(instance_image.size)) if self.center_crop else transforms.RandomCrop(min(instance_image.size)),
                transforms.Resize((self.size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        
        example["instance_image"] = transforms_pipeline(instance_image)

        text_inputs = tokenize_prompt(
            self.tokenizer, instance_prompt, tokenizer_max_length=self.tokenizer_max_length
        )
        example["instance_prompt_ids"] = text_inputs.input_ids
        
        instance_prompt_with_trigger = self.prompt_trigger + instance_prompt
        text_inputs_with_trigger = tokenize_prompt(
            self.tokenizer, instance_prompt_with_trigger, tokenizer_max_length=self.tokenizer_max_length
        )
        
        example["instance_prompt_ids_with_trigger"] = text_inputs_with_trigger.input_ids

        return example


#====================================================
def coefficient_wm(t, t_threshold, max_weight, steepness):  
    sigmoid_weight = max_weight * torch.sigmoid(-(t - t_threshold)/ steepness)  
    return sigmoid_weight

def coefficient_preserve(t, t_threshold, steepness):
    sigmoid_weight = torch.sigmoid((t - t_threshold)/ steepness)  
    return sigmoid_weight


def img_to_DMlatents(x: torch.Tensor, vae: AutoencoderKL):
    x = 2. * x - 1.   
    posterior = vae.encode(x).latent_dist.sample()
    latents = posterior * vae.config.scaling_factor  
    return latents

def DMlatent2img(latents: torch.Tensor, vae: AutoencoderKL):
    latents = 1 / vae.config.scaling_factor * latents 
    image = vae.decode(latents)['sample']
    image_tensor = image/2.0 + 0.5 
    return image_tensor



def distorsion_unit(encoded_images,type):
    if type == 'identity':
        distorted_images = encoded_images
    elif type == 'brightness':
        distorted_images = K.augmentation.ColorJiggle(
            brightness=(0.8, 1.2),  
            contrast=(1.0, 1.0),     
            saturation=(1.0, 1.0),   
            hue=(0.0, 0.0),          
            p=1
        )(encoded_images)
    elif type == 'contrast':
        distorted_images = K.augmentation.ColorJiggle(
            brightness=(1.0, 1.0),  
            contrast=(0.8, 1.2),     
            saturation=(1.0, 1.0),   
            hue=(0.0, 0.0),          
            p=1
        )(encoded_images)
    elif type == 'saturation':
        distorted_images = K.augmentation.ColorJiggle(
            brightness=(1.0, 1.0),   
            contrast=(1.0, 1.0),     
            saturation=(0.8, 1.2),   
            hue=(0.0, 0.0),          
            p=1
        )(encoded_images)
    elif type == 'blur':
        distorted_images = K.augmentation.RandomGaussianBlur((3, 3), (4.0, 4.0), p=1.)(encoded_images)
    elif type == 'noise':
        distorted_images = K.augmentation.RandomGaussianNoise(mean=0.0, std=0.1, p=1)(encoded_images)
    elif type == 'jpeg_compress':
        B = encoded_images.shape[0]
        distorted_images = []
        for i in range(B):
            buffer = io.BytesIO()
            pil_image = T.ToPILImage()(encoded_images[i].squeeze(0))
            pil_image.save(buffer, format='JPEG', quality=50)
            buffer.seek(0)
            pil_image = Image.open(buffer)
            distorted_images.append(T.ToTensor()(pil_image).to(encoded_images.device).unsqueeze(0))
        distorted_images = torch.cat(distorted_images, dim=0)
    elif type == 'resize':
        distorted_images = F.interpolate(
                                    encoded_images,
                                    scale_factor=(0.5, 0.5),
                                    mode='bilinear')
    elif type == 'sharpness':
        distorted_images = K.augmentation.RandomSharpness(sharpness=10., p=1)(encoded_images)
             
    else:
        raise ValueError(f'Wrong distorsion type in add_distorsion().')
    
    distorted_images = torch.clamp(distorted_images, 0, 1)
    return distorted_images

@torch.no_grad()
def log_avg_gradient_norm(obj):
    if isinstance(obj, torch.Tensor):
        if obj.grad is not None:
            grad_norm = torch.norm(obj.grad).item()
            return grad_norm
        else:
            return 0.0
    else:
        total_grad_norm = 0.0
        count = 0
        for param in obj.parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                total_grad_norm += grad_norm
                count += 1
        if count > 0:
            avg_grad_norm = total_grad_norm / count
            return avg_grad_norm
        else:
            return 0.0

@torch.no_grad()
def log_avg_param_norm(obj):
    if isinstance(obj, torch.Tensor):
        param_norm = torch.norm(obj).item()
        return param_norm
    else:
        total_param_norm = 0.0
        total_params = 0
        for param in obj.parameters():
            param_norm = torch.norm(param).item()
            total_param_norm += param_norm
            total_params += 1
        if total_params > 0:
            avg_param_norm = total_param_norm / total_params
            return avg_param_norm
        else:
            return 0.0
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_numel(numel):
    if numel >= 1e6:
        return f"{numel / 1e6:.2f}M"
    elif numel >= 1e3:
        return f"{numel / 1e3:.2f}K"
    else:
        return str(numel)
    
def distortion_unit(encoded_images, type):
    if type == 'identity':
        distorted_images = encoded_images
    elif type == 'brightness':
        distorted_images = K.augmentation.ColorJiggle(
            brightness=(0.8, 1.2),  
            contrast=(1.0, 1.0),     
            saturation=(1.0, 1.0),   
            hue=(0.0, 0.0),          
            p=1
        )(encoded_images)
    elif type == 'contrast':
        distorted_images = K.augmentation.ColorJiggle(
            brightness=(1.0, 1.0),  
            contrast=(0.8, 1.2),     
            saturation=(1.0, 1.0),   
            hue=(0.0, 0.0),          
            p=1
        )(encoded_images)
    elif type == 'saturation':
        distorted_images = K.augmentation.ColorJiggle(
            brightness=(1.0, 1.0),   
            contrast=(1.0, 1.0),     
            saturation=(0.8, 1.2),   
            hue=(0.0, 0.0),          
            p=1
        )(encoded_images)
    elif type == 'blur':
        distorted_images = K.augmentation.RandomGaussianBlur((3, 3), (4.0, 4.0), p=1.)(encoded_images)
    elif type == 'noise':
        distorted_images = K.augmentation.RandomGaussianNoise(mean=0.0, std=0.1, p=1)(encoded_images)
    elif type == 'jpeg_compress':
        B = encoded_images.shape[0]
        distorted_images = []
        for i in range(B):
            buffer = io.BytesIO()
            pil_image = T.ToPILImage()(encoded_images[i].squeeze(0))
            pil_image.save(buffer, format='JPEG', quality=50)
            buffer.seek(0)
            pil_image = Image.open(buffer)
            distorted_images.append(T.ToTensor()(pil_image).to(encoded_images.device).unsqueeze(0))
        distorted_images = torch.cat(distorted_images, dim=0)
    elif type == 'resize':
        scale_factor = 0.5
        original_size = encoded_images.shape[2:]
        distorted_images = F.interpolate(
            encoded_images,
            scale_factor=scale_factor,
            mode='bilinear'
        )
        distorted_images = F.interpolate(
            distorted_images,
            size=original_size,
            mode='bilinear'
        )
    elif type == 'sharpness':
        distorted_images = K.augmentation.RandomSharpness(sharpness=10., p=1)(encoded_images)
             
    else:
        raise ValueError(f'Wrong distorsion type in add_distorsion().')
    
    distorted_images = torch.clamp(distorted_images, 0, 1)
    return distorted_images

def vae_preprocess(x):
    return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

def vae_postprocess(x):
    return torch.clamp(x * 2.0 - 1.0, -1.0, 1.0)

def vae_decode(latent_tensor, vae):
    decoded = vae.decode(latent_tensor / vae.config.scaling_factor).sample
    return torch.clamp(decoded, -1.0, 1.0)

def save_unet_weights(unet, save_path, accelerator, step=None):
    if step is not None:
        save_path = os.path.join(save_path, f"unet_step_{step}")
    else:
        save_path = os.path.join(save_path, "unet_final")

    os.makedirs(save_path, exist_ok=True)

    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        state_dict = unwrapped_unet.state_dict()
        
        unet_file_name = "pytorch_model.safetensors"
        full_save_path = os.path.join(save_path, unet_file_name)
        is_safetensors = False

        if state_dict:
            if USE_SAFETENSORS:
                try:
                    save_file(state_dict, full_save_path)
                    is_safetensors = True
                except Exception:
                    full_save_path = os.path.join(save_path, "pytorch_model.bin")
                    torch.save(state_dict, full_save_path)
            else:
                full_save_path = os.path.join(save_path, "pytorch_model.bin")
                torch.save(state_dict, full_save_path)
            
            if os.path.exists(full_save_path):
                file_size_bytes = os.path.getsize(full_save_path)
                
                if file_size_bytes < 1024 * 1024:
                    file_size_str = f"{file_size_bytes / 1024:.2f} KB"
                elif file_size_bytes < 1024 * 1024 * 1024:
                    file_size_str = f"{file_size_bytes / (1024 * 1024):.2f} MB"
                else:
                    file_size_str = f"{file_size_bytes / (1024 * 1024 * 1024):.2f} GB"
                
                format_type = "safetensors" if is_safetensors else "PyTorch bin"
                print(f"Saved UNet weights ({format_type}, Size: {file_size_str}) to {full_save_path}")
            else:
                print(f"Saved UNet weights to {full_save_path} (could not determine size).")
        else:
            print("Warning: UNet state dict is empty. Nothing saved.")


def load_unet_weights(unet, load_path, accelerator):
    full_path = ""
    if Path(load_path).is_dir():
        for file_name in ["pytorch_model.safetensors", "pytorch_model.bin"]:
            potential_path = os.path.join(load_path, file_name)
            if os.path.exists(potential_path):
                full_path = potential_path
                break
    elif os.path.exists(load_path):
        full_path = load_path

    if not full_path:
        print(f"No UNet weights found at {load_path}. Training UNet from scratch.")
        return False

    try:
        state_dict = None
        if full_path.endswith(".safetensors") and USE_SAFETENSORS:
            state_dict = load_file(full_path, device="cpu")
        elif full_path.endswith((".bin", ".pt", ".pth")):
            state_dict = torch.load(full_path, map_location="cpu")
        elif not USE_SAFETENSORS and full_path.endswith(".safetensors"):
            print("Warning: safetensors not available. Could not load .safetensors file.")
            return False
        
        if state_dict is None:
            return False

        missing, unexpected = unet.load_state_dict(state_dict, strict=True)
        
        print(f"Load UNet weights: Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        if missing or unexpected:
            print("Warning: Loaded UNet has missing or unexpected keys. Check checkpoint integrity.")
        
        print(f"Loaded UNet weights from {full_path}.")
        return True
        
    except Exception as e:
        print(f"Error loading UNet state dict manually from {full_path}: {e}")
        traceback.print_exc()
        return False


def save_watermark_residual(WM_residual, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(WM_residual.cpu(), save_path)
    print(f"Saved watermark residual to {save_path}")

def load_watermark_residual(load_path, device):
    if os.path.exists(load_path):
        WM_residual = torch.load(load_path, map_location='cpu')
        WM_residual = WM_residual.to(device)
        print(f"Loaded watermark residual from {load_path}")
        return WM_residual
    else:
        print(f"No watermark residual found at {load_path}")
        return None

@torch.no_grad()
def generate_validation_images(text_encoder, unet, vae, args, accelerator, weight_dtype):
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        unet=unet,
        vae=vae,
        safety_checker=None,
        torch_dtype=weight_dtype
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    noTrigger_images = []
    Trigger_images = []
    
    for _ in range(args.num_validation_images):
        image = pipeline(prompt=args.validation_prompt, generator=generator).images[0]
        noTrigger_images.append(image)
    for _ in range(args.num_validation_images):
        image = pipeline(prompt=args.trigger + args.validation_prompt, generator=generator).images[0]
        Trigger_images.append(image)
        
    del pipeline
    torch.cuda.empty_cache()
    return noTrigger_images, Trigger_images

def calculate_accuracy(predicted, target):
    predicted_binary = (predicted > 0.5).float()
    target_binary = (target > 0.5).float()
    correct = (predicted_binary == target_binary).float()
    return correct.mean().item()

def calculate_watermark_accuracy(images, watermark_extractor, vae, GT_secret, weight_dtype, accelerator):
    transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensors = torch.stack([transform(img) for img in images])
    img_tensors = img_tensors.to(accelerator.device, dtype=weight_dtype)
    GT_secret = GT_secret.to(accelerator.device)
    
    with torch.no_grad():
        latent_dist = vae.encode(img_tensors)
        latent_tensors = latent_dist.latent_dist.sample() * vae.config.scaling_factor
        # 确保 watermark_extractor 使用 float32 输入
        decoded_output = watermark_extractor.to(torch.float32)(latent_tensors.to(torch.float32))
    
    batch_size = len(images)
    GT_secret_float32 = GT_secret.to(torch.float32)
    target_secret = GT_secret_float32.view(1, -1).repeat(batch_size, 1)
    accuracy = calculate_accuracy(decoded_output, target_secret)
    
    return accuracy

def load_trained_watermark_model(pretrained_path, device, dtype):
    try:
        from model import LatentMarkEncoder, LatentMarkDecoder 
    except ImportError:
        print("Warning: 'model.py' not found. Cannot load watermark encoder/decoder. Returning None.")
        return None, None
    
    sec_encoder = LatentMarkEncoder(secret_size=48, latent_channels=4, resolution=64)
    decoder = LatentMarkDecoder(latent_channels=4, secret_size=48)
    
    encoder_path = os.path.join(pretrained_path, "encoder.pth")
    decoder_path = os.path.join(pretrained_path, "decoder.pth")
    
    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        print(f"Error: Watermark model files not found in {pretrained_path}")
        return None, None

    sec_encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))
    
    sec_encoder = sec_encoder.to(device, dtype=torch.float32)
    decoder = decoder.to(device, dtype=torch.float32)
    sec_encoder.eval()
    decoder.eval()
    sec_encoder.requires_grad_(False)
    decoder.requires_grad_(False)
    
    print(f"Loaded trained watermark model from {pretrained_path}")
    return sec_encoder, decoder

@torch.no_grad()
def generate_watermark_residual(secret, latent, encoder):
    return encoder(secret)



class LossTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.wm_loss_sum = 0.0
        self.x0_preserv_loss_sum = 0.0 
        self.no_trigger_loss_sum = 0.0
        self.total_loss_sum = 0.0
        self.count = 0
        
    def update(self, wm_loss, x0_preserv_loss, no_trigger_loss, total_loss):
        self.wm_loss_sum += wm_loss.item()
        self.x0_preserv_loss_sum += x0_preserv_loss.item() if x0_preserv_loss is not None else 0
        self.no_trigger_loss_sum += no_trigger_loss.item() if no_trigger_loss is not None else 0
        self.total_loss_sum += total_loss.item()
        self.count += 1
        
    def get_average_losses(self):
        if self.count == 0:
            return 0, 0, 0, 0
        return (self.wm_loss_sum / self.count,
                self.x0_preserv_loss_sum / self.count,
                self.no_trigger_loss_sum / self.count,
                self.total_loss_sum / self.count)
        
    def print_average_losses(self, global_step, Trigger_acc):
        if self.count > 0:
            avg_wm, avg_x0_preserv, avg_no_trigger, avg_total = self.get_average_losses()
            print(f"[Loss Stats] Step {global_step} (Acc: {Trigger_acc:.3f}): "
                  f"WM: {avg_wm:.4f}, X0 Preserv: {avg_x0_preserv:.4f}, NoTrigger: {avg_no_trigger:.4f}, Total: {avg_total:.4f}")
            self.reset()