import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import EnhancedCC_Module

def enhance_image(model, input_path, output_path, device):
    """
    Enhance a single underwater image using the FUSION model
    
    Args:
        model: Trained FUSION model
        input_path: Path to input image
        output_path: Path to save enhanced image
        device: Device to run inference on
    """
    # Load and preprocess image
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image {input_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    h, w, _ = img.shape
    img_resized = cv2.resize(img, (256, 256))
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output_tensor = model(img_tensor)
    
    # Convert output to image
    output_img = output_tensor[0].cpu().numpy().transpose(1, 2, 0)
    output_img = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)
    
    # Resize back to original size if needed
    if h != 256 or w != 256:
        output_img = cv2.resize(output_img, (w, h))
    
    # Save the enhanced image
    cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    
    return output_img

def process_directory(model, input_dir, output_dir, device, save_comparison=False):
    """
    Process all images in a directory
    
    Args:
        model: Trained FUSION model
        input_dir: Directory containing input images
        output_dir: Directory to save enhanced images
        device: Device to run inference on
        save_comparison: If True, save a side-by-side comparison
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        
        try:
            output_img = enhance_image(model, image_path, output_path, device)
            
            if save_comparison:
                # Load original for comparison
                original_img = cv2.imread(image_path)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                
                # Create comparison image
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.imshow(original_img)
                plt.title('Original')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(output_img)
                plt.title('Enhanced')
                plt.axis('off')
                
                plt.tight_layout()
                comparison_path = os.path.join(output_dir, f"comparison_{filename}")
                plt.savefig(comparison_path)
                plt.close()
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    print(f"Processed {len(image_files)} images. Enhanced images saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='FUSION: Underwater Image Enhancement Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output image or directory')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference if available')
    parser.add_argument('--compare', action='store_true',
                        help='Save side-by-side comparison (only for directory input)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = EnhancedCC_Module()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Process single image
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        enhance_image(model, args.input, args.output, device)
        print(f"Enhanced image saved to {args.output}")
    else:
        # Process directory
        process_directory(model, args.input, args.output, device, args.compare)

if __name__ == "__main__":
    main()
