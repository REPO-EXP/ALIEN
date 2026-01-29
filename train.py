import os
import model
from dataset import ImageData
import torch
import argparse
import lpips
from transformers import get_linear_schedule_with_warmup
from diffusers import AutoencoderKL
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger
import logging
from pathlib import Path
import json
from itertools import chain 
from tqdm.auto import tqdm
import warnings
from torchvision.utils import save_image
import io
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
import kornia.augmentation as K
from utils import *

logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='./generated_dataset_v1_5', help="Path to the training dataset directory")
parser.add_argument('--validation_path', type=str, default='./generated_dataset_v1_5', help="Path to the validation dataset directory")
parser.add_argument('--output_dir', type=str, default='output_dir')
parser.add_argument('--num_steps', type=int, default=3000000)
parser.add_argument('--warm_up_steps', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--image_loss_scale', type=int, default=30)
parser.add_argument('--image_loss_ramp', type=int, default=5000)
parser.add_argument('--secret_loss_scale', type=float, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--pretrained_dir", type=str, default='ALIEN_Models/', help="Path to the pretrained model directory")
parser.add_argument("--start_step", type=int, default=419000)
parser.add_argument('--validation_batch_size', type=int, default=1)
parser.add_argument('--max_val_samples', type=int, default=1)
parser.add_argument('--recordImg_freq', type=int, default=1000)
parser.add_argument('--validation_freq', type=int, default=1000)
parser.add_argument('--secret_size', type=int, default=48)
parser.add_argument('--sd_model', type=str, default="../stable-diffusion-v1-5")
parser.add_argument('--save_freq', type=int, default=8000)
parser.add_argument('--lpips_scale', type=float, default=0.3)
parser.add_argument('--lpips_ramp', type=int, default=5000)
parser.add_argument("--max_grad_norm", default=1e-2, type=float, help="Max gradient norm.")
parser.add_argument("--adam_weight_decay", type=float, default=0.00001, help="Weight decay to use.")
parser.add_argument('--device', type=str, default='cuda', help="Device")

args = parser.parse_known_args()[0]

CHECKPOINT_SAVE_FREQ = 1000

checkpoints_path = f"{args.output_dir}/checkpoints"
saved_models_path = f"{args.output_dir}/saved_models"
image_save_path = f"{args.output_dir}/validation_images"

os.makedirs(checkpoints_path, exist_ok=True)
os.makedirs(saved_models_path, exist_ok=True)
os.makedirs(image_save_path, exist_ok=True)

logging_dir = Path(args.output_dir, "logs")
accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
accelerator = Accelerator(
    gradient_accumulation_steps=1,
    project_config=accelerator_project_config,
    device_placement=True,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)
logger.info(accelerator.state, main_process_only=False)

if args.seed is not None:
    set_seed(args.seed)

device = accelerator.device
logger.info(f"Using device: {device}")

log_file = open(f"{args.output_dir}/training_log.txt", "w")
log_file.write(f"Using device: {device}\n")

lpips_alex = lpips.LPIPS(net="alex", verbose=False).to(device)
lpips_alex.requires_grad_(False)
lpips_alex.eval()

train_dataset = ImageData(args.train_path, secret_size=args.secret_size)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

validation_dataset = ImageData(args.validation_path, secret_size=args.secret_size, 
                              num_samples=args.max_val_samples, split='test')
validation_dataloader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=args.validation_batch_size, shuffle=False, pin_memory=True)

sec_encoder = model.LatentMarkEncoder(
    secret_size=args.secret_size, 
    latent_channels=4, 
    resolution=64
)
decoder = model.LatentMarkDecoder(
    latent_channels=4, 
    secret_size=args.secret_size
)

if args.pretrained_dir:
    decoder.load_state_dict(torch.load(os.path.join(args.pretrained_dir, "decoder.pth"), map_location='cpu'))
    sec_encoder.load_state_dict(torch.load(os.path.join(args.pretrained_dir, "encoder.pth"), map_location='cpu'))
    logger.info(f"Loaded pretrained models from {args.pretrained_dir}")

sec_encoder = sec_encoder.to(device)
decoder = decoder.to(device)

encoder_params = count_parameters(sec_encoder)
decoder_params = count_parameters(decoder)
total_params = encoder_params + decoder_params

if accelerator.is_main_process:
    param_info = f"Trainable parameters:\n" \
                 f"  Encoder: {encoder_params} ({format_numel(encoder_params)})\n" \
                 f"  Decoder: {decoder_params} ({format_numel(decoder_params)})\n" \
                 f"  Total: {total_params} ({format_numel(total_params)})"
    
    logger.info(param_info)
    log_file.write(f"\n{param_info}\n")
    log_file.flush()

generator_params = chain(sec_encoder.parameters(), decoder.parameters())
optimizer = torch.optim.AdamW(
    generator_params,
    lr=args.lr,
    weight_decay=args.adam_weight_decay,
)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warm_up_steps, num_training_steps=args.num_steps)

models_to_prepare = [sec_encoder, decoder, optimizer, train_dataloader, validation_dataloader, lr_scheduler]

prepared_objects = accelerator.prepare(*models_to_prepare)

sec_encoder = prepared_objects[0]
decoder = prepared_objects[1]
optimizer = prepared_objects[2]
train_dataloader = prepared_objects[3]
validation_dataloader = prepared_objects[4]
lr_scheduler = prepared_objects[5]

global_step = args.start_step
min_loss = 10000

vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae")
vae = vae.to(device)
vae.requires_grad_(False)
vae.eval()

total_steps = args.num_steps
progress_bar = tqdm(
    range(global_step, total_steps),
    initial=global_step,
    desc="Training Steps",
    disable=not accelerator.is_main_process,
)

training_stats = {
    'total_loss': 0.0,
    'secret_loss': 0.0,
    'image_loss': 0.0,
    'lpips_loss': 0.0,
}

iterator = iter(train_dataloader)
while global_step < total_steps:
    sec_encoder.train()
    decoder.train()
    
    try:
        image_input, secret_input = next(iterator)
    except StopIteration:
        iterator = iter(train_dataloader)
        image_input, secret_input = next(iterator)
    
    image_input = image_input.to(device)
    secret_input = secret_input.to(device)
    
    if global_step < 20:
        image_input = torch.zeros_like(image_input)
        if accelerator.is_main_process and global_step == 0:
            logger.info(f"Starting training with pure gray background for the first 20 steps")
    
    image_loss_scale = min(args.image_loss_scale * global_step / args.image_loss_ramp, args.image_loss_scale)
    lpips_scale = min(args.lpips_scale * global_step / args.lpips_ramp, args.lpips_scale)
    loss_scales = (args.secret_loss_scale, image_loss_scale, lpips_scale)

    loss_details = model.build_latentmark_model(
        secret_input, sec_encoder, decoder, image_input, loss_scales, args, 
        global_step, vae, lpips_alex, accelerator, 
        critic=None, adversary=None, use_gan=False
    )
    
    total_loss = loss_details['total_loss']
    secret_loss = loss_details['secret_loss']
    
    optimizer.zero_grad()
    accelerator.backward(total_loss)
    accelerator.clip_grad_norm_(generator_params, args.max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    
    training_stats['total_loss'] = total_loss.item()
    training_stats['secret_loss'] = secret_loss.item()
    training_stats['image_loss'] = loss_details['image_loss'].item()
    training_stats['lpips_loss'] = loss_details['lpips_loss'].item()
    
    if accelerator.is_main_process:
        postfix_dict = {
            'loss': f"{training_stats['total_loss']:.4f}", 
            's_loss': f"{training_stats['secret_loss']:.4f}",
            'img_loss': f"{training_stats['image_loss']:.4f}",
            'lpips': f"{training_stats['lpips_loss']:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        }
        
        progress_bar.set_postfix(**postfix_dict)
        progress_bar.update(1)
    
    if global_step % args.validation_freq == 0:
        decoder.eval()
        sec_encoder.eval()
        
        psnr_input_ls = []
        psnr_recons_ls = []
        ssim_recons_ls = [] 
        acc_WM_ls = []
        blur_wm_acc = []
        noise_wm_acc = []
        jpeg_compress_wm_acc = []
        resize_wm_acc = []
        sharpness_wm_acc = []
        brightness_wm_acc = []
        contrast_wm_acc = []
        saturation_wm_acc = []
        
        distortion_list = ['identity', 'blur', 'noise', 'jpeg_compress', 'resize', 
                          'sharpness', "brightness", "contrast", "saturation"]
        
        should_record_img = (global_step % args.recordImg_freq == 0) and accelerator.is_main_process
        
        with torch.no_grad():
            for i, batch in enumerate(validation_dataloader):
                image_input, secret_input = batch
                image_input = image_input.to(device)
                secret_input = secret_input.to(device)
                
                if should_record_img:
                    latent = vae.encode(image_input).latent_dist.sample()
                    latent = latent * vae.config.scaling_factor
                    residual = sec_encoder(secret_input)
                    encoded_latent = latent + residual
                    decoded_image = vae_decode(encoded_latent, vae)
                    
                    for j in range(min(2, image_input.shape[0])):
                        save_image(vae_preprocess(image_input[j]), 
                                   f"{image_save_path}/step{global_step}_batch{i}_sample{j}_0_original.png")
                        save_image(vae_preprocess(decoded_image[j]), 
                                   f"{image_save_path}/step{global_step}_batch{i}_sample{j}_1_watermarked.png")
                        
                        image_residual = decoded_image[j] - image_input[j]
                        residual_magnitude = torch.abs(image_residual).mean(dim=0, keepdim=True)
                        
                        residual_max = residual_magnitude.max()
                        residual_min = residual_magnitude.min()
                        
                        if residual_max > residual_min:
                            residual_norm = (residual_magnitude - residual_min) / (residual_max - residual_min)
                        else:
                            residual_norm = torch.zeros_like(residual_magnitude)
                        save_image(residual_norm, 
                                   f"{image_save_path}/step{global_step}_batch{i}_sample{j}_2_residual_map.png")
                
                for distortion in distortion_list:
                    avg_psnr_recons, avg_ssim_recons, predict_acc_WM = model.validate_latentmark_model(
                        secret_input, sec_encoder, decoder, image_input, vae, distortion, 
                        distortion_unit=distortion_unit
                        )
                    
                    if distortion == 'identity':
                        acc_WM_ls.append(predict_acc_WM)
                        ssim_recons_ls.append(avg_ssim_recons.mean().item())
                        psnr_recons_ls.append(avg_psnr_recons.mean().item())
                    elif distortion == 'resize':
                        resize_wm_acc.append(predict_acc_WM)
                    elif distortion == 'brightness':
                        brightness_wm_acc.append(predict_acc_WM)
                    elif distortion == 'contrast':
                        contrast_wm_acc.append(predict_acc_WM)
                    elif distortion == 'saturation':
                        saturation_wm_acc.append(predict_acc_WM)
                    elif distortion == 'blur':
                        blur_wm_acc.append(predict_acc_WM)
                    elif distortion == 'noise':
                        noise_wm_acc.append(predict_acc_WM)
                    elif distortion == 'jpeg_compress':
                        jpeg_compress_wm_acc.append(predict_acc_WM)
                    elif distortion == 'sharpness':
                        sharpness_wm_acc.append(predict_acc_WM)
                    else:
                        if accelerator.is_main_process:
                            logger.warning(f"Error: distortion {distortion} not found")
                
                psnr_recons_ls.append(avg_psnr_recons)
        
        accelerator.wait_for_everyone()
        
        avg_acc_WM = torch.tensor(acc_WM_ls).mean()
        avg_psnr_recons = torch.tensor(psnr_recons_ls).mean()
        avg_ssim_recons = torch.tensor(ssim_recons_ls).mean()  
        avg_acc_resize = torch.tensor(resize_wm_acc).mean()
        avg_acc_bright = torch.tensor(brightness_wm_acc).mean()
        avg_acc_contrast = torch.tensor(contrast_wm_acc).mean()
        avg_acc_saturation = torch.tensor(saturation_wm_acc).mean()
        avg_acc_blur = torch.tensor(blur_wm_acc).mean()
        avg_acc_noise = torch.tensor(noise_wm_acc).mean()
        avg_acc_jpeg_compress = torch.tensor(jpeg_compress_wm_acc).mean()
        avg_acc_sharpness = torch.tensor(sharpness_wm_acc).mean()
        
        if accelerator.is_main_process:
            log_file.write(f"Step {global_step}:\n")
            log_file.write(f"  Total Loss: {training_stats['total_loss']:.4f}\n")
            log_file.write(f"  Secret Loss: {training_stats['secret_loss']:.4f}\n")
            log_file.write(f"  Image Loss: {training_stats['image_loss']:.4f}\n")
            log_file.write(f"  LPIPS Loss: {training_stats['lpips_loss']:.4f}\n")
            log_file.write(f"  PSNR Recons: {avg_psnr_recons.item():.4f}\n")
            log_file.write(f"  SSIM Recons: {avg_ssim_recons.item():.4f}\n")  
            log_file.write(f"  WM Accuracy - No Distortion: {avg_acc_WM.item():.4f}\n")
            log_file.write(f"  WM Accuracy - Resize: {avg_acc_resize.item():.4f}\n")
            log_file.write(f"  WM Accuracy - Brightness: {avg_acc_bright.item():.4f}\n")
            log_file.write(f"  WM Accuracy - Contrast: {avg_acc_contrast.item():.4f}\n")
            log_file.write(f"  WM Accuracy - Saturation: {avg_acc_saturation.item():.4f}\n")
            log_file.write(f"  WM Accuracy - Blur: {avg_acc_blur.item():.4f}\n")
            log_file.write(f"  WM Accuracy - Noise: {avg_acc_noise.item():.4f}\n")
            log_file.write(f"  WM Accuracy - JPEG: {avg_acc_jpeg_compress.item():.4f}\n")
            log_file.write(f"  WM Accuracy - Sharpness: {avg_acc_sharpness.item():.4f}\n")
            
            log_file.flush()
            
            logger.info(f"Step {global_step}: Total Loss={training_stats['total_loss']:.4f}, "
                       f"Secret Loss={training_stats['secret_loss']:.4f}, "
                       f"Image Loss={training_stats['image_loss']:.4f}, "
                       f"LPIPS={training_stats['lpips_loss']:.4f}, "
                       f"PSNR={avg_psnr_recons.item():.4f}, "
                       f"SSIM={avg_ssim_recons.item():.4f}")
    
    if accelerator.is_main_process:
        if global_step % CHECKPOINT_SAVE_FREQ == 0:
            save_dir = os.path.join(saved_models_path, f"step{global_step}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(accelerator.unwrap_model(sec_encoder).state_dict(), f"{save_dir}/encoder.pth")
            torch.save(accelerator.unwrap_model(decoder).state_dict(), f"{save_dir}/decoder.pth")
            logger.info(f"Model saved to {save_dir}")
        
        if global_step > args.lpips_ramp and training_stats['total_loss'] < min_loss:
            min_loss = training_stats['total_loss']
            torch.save(accelerator.unwrap_model(sec_encoder).state_dict(), 
                       os.path.join(checkpoints_path, "encoder_best_total_loss.pth"))
            torch.save(accelerator.unwrap_model(decoder).state_dict(), 
                       os.path.join(checkpoints_path, "decoder_best_total_loss.pth"))
            logger.info(f"Best model saved with loss: {min_loss:.4f}")
    
    global_step += 1

log_file.close()
progress_bar.close()
accelerator.wait_for_everyone()
accelerator.end_training()

logger.info("Training completed successfully!")
