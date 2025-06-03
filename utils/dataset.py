import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
from os.path import join

class ToTensor(object):
    """
    Convert numpy arrays to PyTorch tensors.
    Used for data transformations in DataLoader.
    """
    def __call__(self, sample):
        try:
            hazy_image, clean_image = sample['hazy'], sample['clean']
            
            # Validate input shapes
            if len(hazy_image.shape) != 3 or len(clean_image.shape) != 3:
                raise ValueError(f"Images must be 3D arrays. Got shapes: {hazy_image.shape}, {clean_image.shape}")
            
            # Convert to tensors
            hazy_image = torch.from_numpy(np.array(hazy_image).astype(np.float32))
            hazy_image = torch.transpose(torch.transpose(hazy_image, 2, 0), 1, 2)
            
            clean_image = torch.from_numpy(np.array(clean_image).astype(np.float32))
            clean_image = torch.transpose(torch.transpose(clean_image, 2, 0), 1, 2)
            
            # Validate tensor shapes
            expected_shape = (3, 256, 256)  # Assuming 256x256 images
            if tuple(hazy_image.shape) != expected_shape or tuple(clean_image.shape) != expected_shape:
                raise ValueError(f"Tensors must be {expected_shape}. Got shapes: {tuple(hazy_image.shape)}, {tuple(clean_image.shape)}")
                
            return {'hazy': hazy_image, 'clean': clean_image}
            
        except Exception as e:
            print(f"Error in ToTensor transform: {str(e)}")
            raise


class Dataset_Load(Dataset):
    """
    Dataset class for loading underwater image pairs.
    Supports UIEB, SUIM-E, and EUVP datasets with train/test splits.
    """
    def __init__(self, data_root, dataset_name, transform=None, train=True):
        self.data_root = data_root
        
        try:
            self.filesA, self.filesB = self.get_file_paths(self.data_root, dataset_name)
            
            # Validate file lists
            if len(self.filesA) == 0 or len(self.filesB) == 0:
                raise ValueError("Empty file list detected")
            if len(self.filesA) != len(self.filesB):
                raise ValueError(f"Mismatched number of files: {len(self.filesA)} vs {len(self.filesB)}")
            
            # Apply dataset-specific splits
            if dataset_name == 'UIEB':
                train_size = 800
                self.filesA = sorted(self.filesA)
                self.filesB = sorted(self.filesB)
                if train:
                    self.filesA = self.filesA[:train_size]
                    self.filesB = self.filesB[:train_size]
                else:
                    self.filesA = self.filesA[train_size:]
                    self.filesB = self.filesB[train_size:]
                    
            elif dataset_name == 'SUIM-E':
                train_size = 1425
                self.filesA = sorted(self.filesA)
                self.filesB = sorted(self.filesB)
                if train:
                    self.filesA = self.filesA[:train_size]
                    self.filesB = self.filesB[:train_size]
                else:
                    self.filesA = self.filesA[train_size:]
                    self.filesB = self.filesB[train_size:]
                    
            elif dataset_name == 'EUVP':
                test_size = 515
                self.filesA = sorted(self.filesA)
                self.filesB = sorted(self.filesB)
                if train:
                    self.filesA = self.filesA[:-test_size]
                    self.filesB = self.filesB[:-test_size]
                else:
                    self.filesA = self.filesA[-test_size:]
                    self.filesB = self.filesB[-test_size:]
                    
            self.len = min(len(self.filesA), len(self.filesB))
            self.transform = transform
            
        except Exception as e:
            print(f"Error initializing Dataset_Load: {str(e)}")
            raise
            
    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        try:
            # Validate index
            if index >= self.len:
                raise IndexError(f"Index {index} out of range for dataset length {self.len}")
                
            # Read images with error handling
            hazy_im = None
            clean_im = None
            
            for _ in range(3):  # Try reading each image up to 3 times
                try:
                    hazy_im = cv2.resize(cv2.imread(self.filesA[index % self.len]), (256,256),
                                       interpolation=cv2.INTER_AREA)
                    
                    if hazy_im is None:
                        raise ValueError(f"Failed to read hazy image: {self.filesA[index % self.len]}")
                        
                    hazy_im = hazy_im[:, :, ::-1]  # BGR to RGB
                    hazy_im = np.float32(hazy_im) / 255.0
                    
                    break
                    
                except Exception as e:
                    print(f"Warning: Failed to read hazy image (attempt {_+1}/3): {str(e)}")
            
            if hazy_im is None:
                raise RuntimeError("Failed to read hazy image after multiple attempts")
                
            # Same process for clean image
            for _ in range(3):
                try:
                    clean_im = cv2.resize(cv2.imread(self.filesB[index % self.len]), (256,256),
                                        interpolation=cv2.INTER_AREA)
                    
                    if clean_im is None:
                        raise ValueError(f"Failed to read clean image: {self.filesB[index % self.len]}")
                        
                    clean_im = clean_im[:, :, ::-1]  # BGR to RGB
                    clean_im = np.float32(clean_im) / 255.0
                    
                    break
                    
                except Exception as e:
                    print(f"Warning: Failed to read clean image (attempt {_+1}/3): {str(e)}")
            
            if clean_im is None:
                raise RuntimeError("Failed to read clean image after multiple attempts")
                
            sample = {'hazy': hazy_im, 'clean': clean_im}
            
            if self.transform is not None:
                sample = self.transform(sample)
                
            return sample
                
        except Exception as e:
            print(f"Error in Dataset_Load.__getitem__: {str(e)}")
            raise

    def get_file_paths(self, root, dataset_name):
        try:
            root = os.path.join(root, dataset_name)
            
            if dataset_name == 'EUVP':
                filesA = sorted(glob.glob(os.path.join(root, 'raw (A)', "*.*")))
                filesB = sorted(glob.glob(os.path.join(root, 'reference (B)', "*.*")))
                
            elif dataset_name == 'SUIM-E':
                filesA = sorted(glob.glob(os.path.join(root, 'raw (A)', "*.*")))
                filesB = sorted(glob.glob(os.path.join(root, 'reference (B)', "*.*")))
                
            elif dataset_name == 'UIEB':
                filesA = sorted(glob.glob(os.path.join(root, 'raw-890', "*.*")))
                filesB = sorted(glob.glob(os.path.join(root, 'reference-890', "*.*")))
                
            else:
                raise ValueError(f"Unsupported dataset name: {dataset_name}")
                
            return filesA, filesB
                
        except Exception as e:
            print(f"Error getting file paths: {str(e)}")
            raise
        