import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import lpips
from collections import namedtuple

from models import EnhancedCC_Module
from utils.dataset import Dataset_Load, ToTensor
from utils.metrics import getUIQM, getSSIM, getPSNR
from utils.helpers import getLatestCheckpointName, get_lr
from brisque import BRISQUE

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def normalize_batch(batch):
    """Normalize using imagenet mean and std"""
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std

class Vgg16(torch.nn.Module):
    """VGG16 model for perceptual loss calculation"""
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

def compute_lpips_metric(pred, target, lpips_loss_fn):
    """Calculate LPIPS perceptual distance"""
    # LPIPS expects images normalized to [-1, 1]
    pred_norm = pred * 2 - 1
    target_norm = target * 2 - 1
    return lpips_loss_fn(pred_norm, target_norm).mean().item()

def train(config, data_root):
    """Main training function"""
    # Create checkpoints directory if it doesn't exist
    checkpoints_dir = config['training']['checkpoints_dir']
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    netG = EnhancedCC_Module()
    netG.to(device)

    # Initialize loss functions and metrics
    mse_loss = nn.MSELoss()
    vgg = Vgg16(requires_grad=False).to(device)
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    brisque_obj = BRISQUE(url=False)

    # Initialize optimizer
    optim_g = optim.Adam(
        netG.parameters(),
        lr=config['training']['learning_rate_g'],
        betas=(config['training']['beta1'], config['training']['beta2']),
        weight_decay=config['training']['wd_g']
    )

    # Load datasets
    train_dataset = Dataset_Load(
        data_root=data_root,
        dataset_name='SUIM-E',
        transform=ToTensor(),
        train=True
    )
    val_dataset = Dataset_Load(
        data_root=data_root,
        dataset_name='SUIM-E',
        transform=ToTensor(),
        train=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )

    train_batches = len(train_dataloader)
    val_batches = len(val_dataloader)

    # Check for existing checkpoints
    latest_checkpoint_G = getLatestCheckpointName(checkpoints_dir)
    
    print(f'Loading model for generator: {latest_checkpoint_G}')
    
    if latest_checkpoint_G is None:
        start_epoch = 1
        print('No checkpoints found for netG! Starting training from scratch')
    else:
        checkpoint_g = torch.load(os.path.join(checkpoints_dir, latest_checkpoint_G))
        start_epoch = checkpoint_g['epoch'] + 1
        netG.load_state_dict(checkpoint_g['model_state_dict'])
        optim_g.load_state_dict(checkpoint_g['optimizer_state_dict'])
        print(f'Restoring model from checkpoint {start_epoch}')

    lambda_mse = config['training']['lambda_mse']
    lambda_vgg = config['training']['lambda_vgg']
    
    # Main training loop
    for epoch in range(start_epoch, config['training']['end_epoch'] + 1):
        # Training phase
        netG.train()
        total_train_mse_loss = 0.0
        total_train_vgg_loss = 0.0
        total_train_G_loss = 0.0
            
        for i_batch, sample_batched in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch} [Train]")):
            hazy_batch = sample_batched['hazy']
            clean_batch = sample_batched['clean']
    
            hazy_batch = hazy_batch.to(device)
            clean_batch = clean_batch.to(device)
    
            optim_g.zero_grad()
    
            pred_batch = netG(hazy_batch)
            batch_mse_loss = torch.mul(lambda_mse, mse_loss(pred_batch, clean_batch))
            batch_mse_loss.backward(retain_graph=True)
            
            clean_vgg_feats = vgg(normalize_batch(clean_batch))
            pred_vgg_feats = vgg(normalize_batch(pred_batch))
            batch_vgg_loss = torch.mul(lambda_vgg, mse_loss(pred_vgg_feats.relu2_2, clean_vgg_feats.relu2_2))
            batch_vgg_loss.backward()
            
            batch_mse_loss_val = batch_mse_loss.item()
            total_train_mse_loss += batch_mse_loss_val
    
            batch_vgg_loss_val = batch_vgg_loss.item()
            total_train_vgg_loss += batch_vgg_loss_val
            
            batch_G_loss = batch_mse_loss_val + batch_vgg_loss_val
            total_train_G_loss += batch_G_loss
            
            optim_g.step()
            
            if (i_batch + 1) % 50 == 0:
                print(f'\rEpoch: {epoch} | Train: ({i_batch+1}/{train_batches}) | g_mse: {batch_mse_loss_val:.6f} | g_vgg: {batch_vgg_loss_val:.6f}')
        
        # Calculate average training losses
        avg_train_mse_loss = total_train_mse_loss / train_batches
        avg_train_vgg_loss = total_train_vgg_loss / train_batches
        avg_train_G_loss = total_train_G_loss / train_batches
            
        # Validation phase
        netG.eval()
        total_val_mse_loss = 0.0
        total_val_vgg_loss = 0.0
        total_val_G_loss = 0.0
        
        # Initialize metrics for validation
        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        total_uiqm = 0.0
        total_uism = 0.0
        total_brisque = 0.0
        
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch} [Val]")):
                hazy_batch = sample_batched['hazy']
                clean_batch = sample_batched['clean']
        
                hazy_batch = hazy_batch.to(device)
                clean_batch = clean_batch.to(device)
        
                pred_batch = netG(hazy_batch)
                batch_mse_loss = torch.mul(lambda_mse, mse_loss(pred_batch, clean_batch)).item()
                
                clean_vgg_feats = vgg(normalize_batch(clean_batch))
                pred_vgg_feats = vgg(normalize_batch(pred_batch))
                batch_vgg_loss = torch.mul(lambda_vgg, mse_loss(pred_vgg_feats.relu2_2, clean_vgg_feats.relu2_2)).item()
                
                total_val_mse_loss += batch_mse_loss
                total_val_vgg_loss += batch_vgg_loss
                batch_G_loss = batch_mse_loss + batch_vgg_loss
                total_val_G_loss += batch_G_loss
                
                # Calculate additional metrics
                batch_psnr = 0.0
                batch_ssim = 0.0
                batch_lpips = 0.0
                batch_uiqm = 0.0
                batch_uism = 0.0
                batch_brisque = 0.0
                
                # Move tensors to CPU and convert to numpy for metric calculation
                pred_np = pred_batch.cpu().numpy().transpose(0, 2, 3, 1)  # B, H, W, C
                clean_np = clean_batch.cpu().numpy().transpose(0, 2, 3, 1)  # B, H, W, C
                
                # Calculate metrics for each image in batch
                batch_size_actual = pred_np.shape[0]
                for j in range(batch_size_actual):
                    # PSNR
                    batch_psnr += getPSNR(pred_np[j], clean_np[j])
                    
                    # SSIM
                    batch_ssim += getSSIM(pred_np[j], clean_np[j])
                    
                    # LPIPS
                    img1 = torch.from_numpy(pred_np[j].transpose(2, 0, 1)).unsqueeze(0).to(device)
                    img2 = torch.from_numpy(clean_np[j].transpose(2, 0, 1)).unsqueeze(0).to(device)
                    batch_lpips += compute_lpips_metric(img1, img2, lpips_loss_fn)
                    
                    # UIQM
                    pred_img = (pred_np[j] * 255).astype(np.uint8)
                    batch_uiqm += getUIQM(pred_img)
                    
                    # UISM
                    batch_uism += _uism(pred_img)
                    
                    # BRISQUE
                    pred_uint8 = (pred_np[j] * 255).astype(np.uint8)
                    batch_brisque += brisque_obj.score(img=pred_uint8)
                
                # Average metrics for the batch
                batch_psnr /= batch_size_actual
                batch_ssim /= batch_size_actual
                batch_lpips /= batch_size_actual
                batch_uiqm /= batch_size_actual
                batch_uism /= batch_size_actual
                batch_brisque /= batch_size_actual
                
                # Add to totals
                total_psnr += batch_psnr
                total_ssim += batch_ssim
                total_lpips += batch_lpips
                total_uiqm += batch_uiqm
                total_uism += batch_uism
                total_brisque += batch_brisque
                
                if (i_batch + 1) % 10 == 0:
                    print(f'\rEpoch: {epoch} | Validation: ({i_batch+1}/{val_batches}) | '
                          f'g_mse: {batch_mse_loss:.6f} | g_vgg: {batch_vgg_loss:.6f} | '
                          f'psnr: {batch_psnr:.4f} | ssim: {batch_ssim:.4f} | '
                          f'lpips: {batch_lpips:.4f} | uiqm: {batch_uiqm:.4f}')
        
        # Calculate average validation losses and metrics
        avg_val_mse_loss = total_val_mse_loss / val_batches
        avg_val_vgg_loss = total_val_vgg_loss / val_batches
        avg_val_G_loss = total_val_G_loss / val_batches
        avg_val_psnr = total_psnr / val_batches
        avg_val_ssim = total_ssim / val_batches
        avg_val_lpips = total_lpips / val_batches
        avg_val_uiqm = total_uiqm / val_batches
        avg_val_uism = total_uism / val_batches
        avg_val_brisque = total_brisque / val_batches
            
        print('\nEpoch %d Summary:' % epoch)
        print('Training   - lr: %.6f, mse: %.6f, vgg: %.6f, total: %.6f' % 
              (get_lr(optim_g), avg_train_mse_loss, avg_train_vgg_loss, avg_train_G_loss))
        print('Validation - mse: %.6f, vgg: %.6f, total: %.6f' % 
              (avg_val_mse_loss, avg_val_vgg_loss, avg_val_G_loss))
        print('Image Quality Metrics:')
        print('  - PSNR: %.4f' % avg_val_psnr)
        print('  - SSIM: %.4f' % avg_val_ssim)
        print('  - LPIPS: %.4f (lower is better)' % avg_val_lpips)
        print('  - UIQM: %.4f' % avg_val_uiqm)
        print('  - UISM: %.4f' % avg_val_uism)
        print('  - BRISQUE: %.4f (lower is better)' % avg_val_brisque)
            
        # Save checkpoint
        torch.save({
            'epoch': epoch, 
            'model_state_dict': netG.state_dict(), 
            'optimizer_state_dict': optim_g.state_dict(), 
            'train_mse_loss': avg_train_mse_loss,
            'train_vgg_loss': avg_train_vgg_loss,
            'train_total_loss': avg_train_G_loss,
            'val_mse_loss': avg_val_mse_loss,
            'val_vgg_loss': avg_val_vgg_loss,
            'val_total_loss': avg_val_G_loss,
            'val_psnr': avg_val_psnr,
            'val_ssim': avg_val_ssim,
            'val_lpips': avg_val_lpips,
            'val_uiqm': avg_val_uiqm,
            'val_uism': avg_val_uism,
            'val_brisque': avg_val_brisque
        }, os.path.join(checkpoints_dir, f'netG_{epoch}.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train FUSION')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--data_root', type=str, default='/path/to/data',
                        help='Path to dataset root directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config, args.data_root)
