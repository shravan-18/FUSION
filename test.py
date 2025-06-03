import os
import argparse
import yaml
import torch
import numpy as np
import cv2
from PIL import Image
import glob
from tqdm import tqdm
import lpips
from brisque import BRISQUE

from models import EnhancedCC_Module
from utils.metrics import getUIQM, getSSIM, getPSNR, _uism

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test(config, checkpoint_path):
    """Evaluate the model on test dataset"""
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = EnhancedCC_Module()
    model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Loaded model from epoch {epoch}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    brisque_obj = BRISQUE(url=False)
    
    # Create output directory
    test_mode = config['dataset']['test']['mode']
    output_dir = os.path.join('results', f"{test_mode}_ep{epoch}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test image paths
    test_input_dir = config['dataset']['test']['inp_dir']
    test_gt_dir = config['dataset']['test']['gt_dir']
    
    input_files = sorted(glob.glob(os.path.join(test_input_dir, "*.*")))
    gt_files = sorted(glob.glob(os.path.join(test_gt_dir, "*.*")))
    
    if len(input_files) == 0:
        raise ValueError(f"No test images found in {test_input_dir}")
    
    # Initialize metric accumulators
    psnr_values = []
    ssim_values = []
    lpips_values = []
    uiqm_values = []
    uism_values = []
    brisque_values = []
    
    # Process each test image
    for i, input_path in enumerate(tqdm(input_files, desc="Testing")):
        # Load input image
        input_img = cv2.imread(input_path)
        if input_img is None:
            print(f"Warning: Could not read image {input_path}")
            continue
            
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (256, 256))
        
        # Prepare input tensor
        input_tensor = torch.from_numpy(input_img.transpose(2, 0, 1)).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Convert output to image
        output_img = output_tensor[0].cpu().numpy().transpose(1, 2, 0)
        output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)
        
        # Save output image
        output_path = os.path.join(output_dir, os.path.basename(input_path))
        cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
        
        # Calculate metrics if ground truth is available
        if i < len(gt_files):
            gt_path = gt_files[i]
            gt_img = cv2.imread(gt_path)
            
            if gt_img is not None:
                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                gt_img = cv2.resize(gt_img, (256, 256))
                
                # PSNR
                psnr = getPSNR(output_img, gt_img)
                psnr_values.append(psnr)
                
                # SSIM
                ssim = getSSIM(output_img, gt_img)
                ssim_values.append(ssim)
                
                # LPIPS
                img1 = torch.from_numpy(output_img.transpose(2, 0, 1)).float() / 255.0
                img1 = img1.unsqueeze(0).to(device)
                
                img2 = torch.from_numpy(gt_img.transpose(2, 0, 1)).float() / 255.0
                img2 = img2.unsqueeze(0).to(device)
                
                lpips_value = lpips_loss_fn(img1 * 2 - 1, img2 * 2 - 1).mean().item()
                lpips_values.append(lpips_value)
            
        # No-reference metrics (can be calculated without ground truth)
        # UIQM
        uiqm = getUIQM(output_img)
        uiqm_values.append(uiqm)
        
        # UISM
        uism = _uism(output_img)
        uism_values.append(uism)
        
        # BRISQUE
        brisque = brisque_obj.score(img=output_img)
        brisque_values.append(brisque)
        
    # Calculate average metrics
    print("\nTest Results:")
    
    if psnr_values:
        avg_psnr = np.mean(psnr_values)
        print(f"Average PSNR: {avg_psnr:.4f}")
    
    if ssim_values:
        avg_ssim = np.mean(ssim_values)
        print(f"Average SSIM: {avg_ssim:.4f}")
    
    if lpips_values:
        avg_lpips = np.mean(lpips_values)
        print(f"Average LPIPS: {avg_lpips:.4f} (lower is better)")
    
    avg_uiqm = np.mean(uiqm_values)
    print(f"Average UIQM: {avg_uiqm:.4f}")
    
    avg_uism = np.mean(uism_values)
    print(f"Average UISM: {avg_uism:.4f}")
    
    avg_brisque = np.mean(brisque_values)
    print(f"Average BRISQUE: {avg_brisque:.4f} (lower is better)")
    
    # Save metrics to file
    results_file = os.path.join(output_dir, "metrics.txt")
    with open(results_file, 'w') as f:
        f.write(f"Test Mode: {test_mode}\n")
        f.write(f"Model Epoch: {epoch}\n\n")
        
        if psnr_values:
            f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        
        if ssim_values:
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        
        if lpips_values:
            f.write(f"Average LPIPS: {avg_lpips:.4f} (lower is better)\n")
        
        f.write(f"Average UIQM: {avg_uiqm:.4f}\n")
        f.write(f"Average UISM: {avg_uism:.4f}\n")
        f.write(f"Average BRISQUE: {avg_brisque:.4f} (lower is better)\n")
    
    print(f"\nResults saved to {results_file}")
    print(f"Enhanced images saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test FUSION model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    args = parser.parse_args()
    
    config = load_config(args.config)
    test(config, args.checkpoint)
