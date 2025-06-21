# FUSION: Frequency-guided Underwater Spatial Image recOnstructioN

<div align=center>
<img src="https://github.com/Amazingren/NTIRE2024_ESR/blob/main/figs/logo.png" width="400px"/> 
</div> 

> FUSION: Frequency-guided Underwater Spatial Image recOnstructioN  
> *Jaskaran Singh Walia\*, Shravan Venkatraman\*, Pavithra L K*  
> Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops (NTIRE) 2025

#### [project page](https://shravan-18.github.io/FUSION/) | [paper](https://arxiv.org/abs/2504.01243) | [poster](assets/FUSION_Poster.pdf)

<div align="center">
  <img src="assets/overallArch.png" width="100%">
  <p>Figure: Overview of the FUSION pipeline, illustrating the dual-domain (spatial and frequency) processing, contextual attention refinement, and final channel calibration for UIE.</p>
</div>

## Abstract

Underwater images suffer from severe degradations, including color distortions, reduced visibility, and loss of structural details due to wavelength-dependent attenuation and scattering. Existing enhancement methods primarily focus on spatial-domain processing, neglecting the frequency domain’s potential to capture global color distributions and long-range dependencies. To address these limitations, we propose FUSION, a dual-domain deep learning framework that jointly leverages spatial and frequency domain information. FUSION independently processes each RGB channel through multi-scale convolutional kernels and adaptive attention mechanisms in the spatial domain, while simultaneously extracting global structural information via FFT-based frequency attention. A Frequency Guided Fusion module integrates complementary features from both domains, followed by inter-channel fusion and adaptive channel recalibration to ensure balanced color distributions. Extensive experiments on benchmark datasets (UIEB, EUVP, SUIM-E) demonstrate that FUSION achieves state-of-the-art performance, consistently outperforming existing methods in reconstruction fidelity (highest PSNR of 23.717 dB and SSIM of 0.883 on UIEB), perceptual quality (lowest LPIPS of 0.112 on UIEB), and visual enhancement metrics (best UIQM of 3.414 on UIEB), while requiring significantly fewer parameters (0.28 M) and lower computational complexity, demonstrating its suitability for real-time underwater imaging applications.

## Key Features

- **Dual-Domain Enhancement**: A parallel frequency pathway that captures long-range dependencies and global color distributions, complementing traditional spatial processing.
- **Dedicated Frequency Attention Module**: Preserves original phase while applying adaptive attention to the magnitude spectrum, capturing global structural information critical for handling complex underwater degradations.
- **Inter-Channel Calibration for Color Correction**: A global recalibration stage employing learnable scaling factors to balance color intensities adaptively.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FUSION.git
cd FUSION

# Create a conda environment (optional)
conda create -n fusion python=3.8
conda activate fusion

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

The model is trained and evaluated on three underwater image datasets:
- [UIEB](https://ieeexplore.ieee.org/document/8917818)
- [EUVP](https://ieeexplore.ieee.org/document/9001231)
- [SUIM-E](https://ieeexplore.ieee.org/document/9930878/)

Please download these datasets and organize them according to the following structure:

```
data/
├── DAMN-Dataset/
│   ├── EUVP/
│   │   ├── raw (A)/
│   │   └── reference (B)/
│   ├── SUIM-E/
│   │   ├── raw (A)/
│   │   └── reference (B)/
│   └── UIEB/
│       ├── raw-890/
│       └── reference-890/
```

## Training

To train the FUSION model from scratch:

```bash
python train.py --config configs/config.yaml --data_root /path/to/data
```

You can modify the configuration file to change hyperparameters, dataset paths, etc.

## Testing

To evaluate a trained model on the test dataset:

```bash
python test.py --config configs/config.yaml --checkpoint ckpts/netG_best.pt
```

## Inference

For inference on a single image or a directory of images:

```bash
# Single image
python inference.py --checkpoint ckpts/netG_best.pt --input path/to/image.jpg --output path/to/enhanced.jpg --gpu

# Directory of images
python inference.py --checkpoint ckpts/netG_best.pt --input path/to/input_dir --output path/to/output_dir --gpu --compare
```

The `--compare` flag generates side-by-side comparisons of original and enhanced images.

## Results

<div align="center">
  <img src="assets/uieb.png" width="100%">
  <p>Figure: Visual comparisons on the UIEB dataset.</p>
</div>

<div align="center">
  <img src="assets/euvp.png" width="100%">
  <p>Figure: Visual comparisons on the EUVP dataset.</p>
</div>

### Quantitative Comparison

| Method | PSNR | SSIM | LPIPS↓ | UIQM | UISM | BRISQUE↓ |
|--------|------|------|--------|------|------|----------|
| UDCP   | 13.026 | 0.545 | 0.283 | 1.922 | 7.424 | 24.133 |
| IBLA   | 19.316 | 0.690 | 0.233 | 2.108 | 7.427 | 23.710 |
| ULAP   | 19.863 | 0.724 | 0.256 | 2.328 | 7.362 | 25.113 |
| CBF    | 20.771 | 0.836 | 0.189 | 3.318 | 7.380 | 20.534 |
| UGAN   | 23.322 | 0.815 | 0.199 | 3.432 | 7.241 | 27.011 |
| FUnIE-GAN | 21.043 | 0.785 | 0.173 | 3.250 | 7.202 | 24.522 |
| SGUIE-Net | 23.496 | 0.853 | 0.136 | 3.004 | 7.362 | 24.607 |
| DWNet  | 23.165 | 0.843 | 0.162 | 2.897 | 7.089 | 24.863 |
| Lit-Net | 23.603 | 0.863 | 0.130 | 3.145 | 7.396 | 23.038 |
| **FUSION (Ours)** | **23.717** | **0.883** | **0.112** | **3.414** | **7.429** | **23.193** |

## Model Efficiency

| Method | Parameters (M) | FLOPs (G) |
|--------|----------------|-----------|
| WaterNet | 24.8 | 193.7 |
| UGAN | 57.17 | 18.3 |
| FUnIE-GAN | 7.71 | 10.7 |
| Ucolor | 157.4 | 443.9 |
| SGUIE-Net | 18.55 | 123.5 |
| DWNet | 0.48 | 18.2 |
| Ushape | 65.6 | 66.2 |
| LitNet | 0.54 | 17.8 |
| **FUSION (Ours)** | **0.28** | **36.73** |

## Citation

If you find our work useful for your research, please consider citing:

```bibtex
@InProceedings{FUSION,
  author    = {Jaskaran Singh Walia and Shravan Venkatraman and Pavithra LK},
  title     = {FUSION: Frequency-guided Underwater Spatial Image recOnstructioN}, 
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month     = {June},
  year      = {2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- We acknowledge the contributions of [LitNet](https://arxiv.org/abs/2408.09912) and [Deep WaveNet](https://arxiv.org/abs/2106.07910), whose frameworks and insights served as the basis for this repository.
- We thank the authors of the UIEB, EUVP, and SUIM-E datasets for making their data publicly available.
