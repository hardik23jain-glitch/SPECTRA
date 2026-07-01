**SPECTRA: Semantic-Preserving Edge-enhanced Color Translation Architecture**

Team: ZENITHHh

Project: IR-to-RGB Translation for Earth Observation

Hackathon: Bharatiya Antariksh Hackathon 2026

*Project Overview*
SPECTRA is a physically-constrained, end-to-end deep learning framework designed to solve the "modality gap" in infrared remote sensing. By transforming monochromatic, low-resolution infrared (IR) satellite imagery into high-fidelity, colorized RGB representations, SPECTRA restores structural integrity and semantic clarity—enabling intelligence-grade situational awareness even during nighttime or adverse atmospheric conditions.

Unlike generic image translation models, SPECTRA explicitly enforces semantic consistency using land-cover priors to prevent the generation of algorithmic "hallucinations."

*Team*
Mohd Faraz Khan (Team Leader / System Architect)
Aayed Hassan (Data Pipeline Engineer)
Hardik Jain (Model Developer)

*Technical Highlights*
SPECTRA distinguishes itself through three core architectural innovations:

Dual-Stream Frequency Enhancement: A joint denoising and super-resolution module that uses Discrete Wavelet Transform (DWT) to separate noise (low-freq) from structural edges (high-freq), allowing for aggressive denoising without destroying geometric boundaries.

SPADE-Mamba Generator: Our core translation engine.

Mamba Encoder: Utilizes a Vision State Space Model for linear-time complexity, ensuring global context awareness without the memory bottlenecks of Transformers.

SPADE Normalization: Injects land-cover semantic masks directly into the generator, forcing the model to adhere to physical reality (e.g., water bodies must map to blue-spectrum hues).

Physics-Informed Objective: Optimized via a 5-part composite loss function, including:

Gradient Alignment Loss: Using Sobel/Laplacian operators to ensure RGB edges align perfectly with thermal gradients.

Bidirectional Semantic Consistency: Enforces that feature representations of the input IR and output RGB remain identical in semantic space.

*Pipeline Overview*
Plaintext
[Input] -> [Physics-Informed Calibration] -> [Dual-Stream Enhancement] -> [SPADE-Mamba Translation] -> [Output]
Data Layer: Radiometric calibration using USGS Landsat MTL metadata.

Registration: Sub-pixel alignment via AROSICS (Fourier phase correlation).

Validation: Benchmarked against PSNR, SSIM, and task-specific mAP/mIoU improvements for automated object detection.

Getting Started
Prerequisites
Bash
pip install rasterio gdal arosics numpy opencv-python pytorch-wavelets
Installation
Clone the repository:

Bash
git clone https://github.com/your-repo/spectra.git
cd spectra
Set up your dataset in the /data directory, ensuring your .MTL metadata files are present.

Usage
To process a Landsat tile:

Bash
python train.py --config configs/default.yaml

Acknowledgments
This project was developed for the Bharatiya Antariksh Hackathon. We acknowledge the use of USGS Landsat 8/9 Collection 2 products and the foundational research in State Space Models (Mamba) and Spatially Adaptive Normalization (SPADE)