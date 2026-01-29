import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from utils import *
def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(x.shape[0], *self.shape[1:]) if self.shape[0] == -1 else x.view(self.shape)

class Repeat(nn.Module):
    def __init__(self, n, channel_dim=1):
        super().__init__()
        self.n = n
        self.channel_dim = channel_dim
    def forward(self, x):
        return x.repeat_interleave(self.n, dim=self.channel_dim)

class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='silu'):
        super(Dense, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)
        self.activation = activation
        
    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation == 'relu':
            outputs = F.relu(outputs)
        elif self.activation == 'sigmoid':
            outputs = torch.sigmoid(outputs)
        elif self.activation == 'silu':
            outputs = F.silu(outputs)
        return outputs
    
class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='silu', strides=1):
        super(Conv2D, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
        self.activation = activation
        
    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation == 'relu':
            outputs = F.relu(outputs)
        elif self.activation == 'silu':
            outputs = F.silu(outputs)
        return outputs

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Critic(nn.Module):
    def __init__(self, latent_channels=4, resolution=64):
        super(Critic, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(latent_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16x16
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.LeakyReLU(0.2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
        
    def forward(self, latent):
        features = self.conv_layers(latent)
        global_features = self.global_pool(features)
        global_features = global_features.view(global_features.size(0), -1)
        
        return self.linear(global_features)

class Adversary(nn.Module):
    def __init__(self, max_perturbation=0.01, image_size=512):
        super(Adversary, self).__init__()
        self.max_perturbation = max_perturbation
        self.image_size = image_size

        self.attack_net = nn.Sequential(

            nn.AdaptiveAvgPool2d(256) if image_size > 256 else nn.Identity(),
            

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh(),

            nn.Upsample(size=image_size, mode='bilinear', align_corners=False) if image_size > 256 else nn.Identity()
        )
        
    def forward(self, x):

        original_size = x.shape[-2:]

        perturbation = self.attack_net(x)

        if perturbation.shape[-2:] != original_size:
            perturbation = F.interpolate(perturbation, size=original_size, mode='bilinear', align_corners=False)

        attacked_x = x + self.max_perturbation * perturbation
        
        attacked_x = torch.clamp(attacked_x, -1.0, 1.0)
        
        return attacked_x

class LatentMarkEncoder(nn.Module):
    def __init__(self, secret_size=48, latent_channels=4, resolution=64):
        super().__init__()
        
        self.secret_projection = nn.Sequential(
            nn.Linear(secret_size, 256),
            nn.SiLU(),
            nn.Linear(256, latent_channels * resolution * resolution),
            View(-1, latent_channels, resolution, resolution)
        )
        
        self.feature_refinement = nn.Sequential(
            nn.Conv2d(latent_channels, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, latent_channels, 3, padding=1),
        )
        
    def forward(self, inputs):
        secret_input= inputs 
        residual = self.secret_projection(secret_input)
        residual = self.feature_refinement(residual)
        return residual

class LatentMarkDecoder(nn.Module):
    def __init__(self, secret_size=48, latent_channels=4, resolution=64):
        super(LatentMarkDecoder, self).__init__()
        
        self.feature_restoration = nn.Sequential(
            nn.Conv2d(latent_channels, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, latent_channels, 3, padding=1),
        )
        
        self.secret_reconstruction = nn.Sequential(
            Flatten(),
            nn.Linear(latent_channels * resolution * resolution, 256),
            nn.SiLU(),
            nn.Linear(256, secret_size),
            nn.Sigmoid()
        )

    def forward(self, latent):
        restored_features = self.feature_restoration(latent)
        decoded_secret = self.secret_reconstruction(restored_features)
        return decoded_secret

def build_latentmark_model(secret_input, sec_encoder, decoder, image_input, loss_scales, args, global_step, vae, lpips_alex, accelerator, 
                          critic=None, adversary=None, use_gan=False, gan_stage='generator',
                          ):
    with torch.no_grad():
        latent = vae.encode(image_input).latent_dist.sample() 
        latent = latent * vae.config.scaling_factor
    
    residual = sec_encoder(secret_input)
    encoded_latent = latent + residual

    decoded_image = vae.decode(encoded_latent / vae.config.scaling_factor).sample
    decoded_image = torch.clamp(decoded_image, -1.0, 1.0)

    decoded_secret = decoder(encoded_latent)
    secret_loss = F.binary_cross_entropy(decoded_secret, secret_input)
    
    image_input_norm = (image_input + 1.0) / 2.0
    decoded_image_norm = (decoded_image + 1.0) / 2.0
    
    image_loss = F.mse_loss(decoded_image_norm, image_input_norm)
    lpips_loss = lpips_alex(decoded_image_norm, image_input_norm).mean()
    
    secret_loss_scale, image_loss_scale, lpips_scale= loss_scales
    
    gan_loss = 0.0
    
    total_loss = (secret_loss_scale * secret_loss + 
                  image_loss_scale * image_loss + 
                  lpips_scale * lpips_loss)
    
    if use_gan and gan_stage in ['generator', 'adversary']:
        total_loss += args.gan_weight * gan_loss
    
    loss_details = {
        'total_loss': total_loss,
        'secret_loss': secret_loss,
        'image_loss': image_loss,
        'lpips_loss': lpips_loss,
    }
    
    return loss_details

def validate_latentmark_model(secret_input, sec_encoder, decoder, image_input, vae, distortion, distortion_unit=None):
    with torch.no_grad():
        latent = vae.encode(image_input).latent_dist.sample()
        latent = latent * vae.config.scaling_factor
        residual = sec_encoder(secret_input)
        encoded_latent = latent + residual
        decoded_image = vae.decode(encoded_latent / vae.config.scaling_factor).sample
        decoded_image = torch.clamp(decoded_image, -1.0, 1.0)
        decoded_image_norm = (decoded_image + 1.0) / 2.0
        original_image_norm = (image_input + 1.0) / 2.0
        if distortion_unit is not None and distortion != 'identity':
            distorted_image = distortion_unit(decoded_image_norm, distortion)
            distorted_image = distorted_image * 2.0 - 1.0
        else:
            distorted_image = decoded_image
        latent_distorted = vae.encode(distorted_image).latent_dist.sample()
        latent_distorted = latent_distorted * vae.config.scaling_factor

        decoded_secret = decoder(latent_distorted)
        mse = F.mse_loss(decoded_image_norm, original_image_norm)
        psnr_recons = 10 * torch.log10(1.0 / (mse + 1e-8))
        ssim_val = calculate_ssim(decoded_image_norm, original_image_norm)
        binary_secret = (secret_input > 0.5).float()
        binary_decoded = (decoded_secret > 0.5).float()
        accuracy = (binary_secret == binary_decoded).float().mean()
        
        return psnr_recons, ssim_val, accuracy




