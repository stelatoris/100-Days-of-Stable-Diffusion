# 100 Days of Stable Diffusion

## Overview
This repository documents my 100-day journey learning Stable Diffusion. Instead of spending time planning the perfect roadmap, I've chosen to dive in and learn by doing, filling in knowledge gaps along the way.

## Background
My background is in Design. So no CS, no ML, no Math, or anything technical. However, in 2021 I started learning programming to do fun Arduino projects. As for ML, I only took Andrew Ng's Machine Learning Specialization course which gave me the basics so I expect this to still be difficult since I have no background in Math. Nevertheless, I love to learn.

## Learning Approach
I'm following a top-down learning approach with two parallel tracks:

1. **Hugging Face Diffusion Tutorial**
   - Getting quickly started with Diffusers
   - Learning practical implementation

2. **Understanding the Fundamentals**
   - Following Umar Jamil's "Coding Stable Diffusion from scratch in PyTorch" YouTube video as this might help me understand the library better.

## Learning Resources
- [Hugging Face Diffusion Tutorial](https://huggingface.co/docs/diffusers/stable_diffusion)
- [Umar Jamil's YouTube video](https://youtu.be/ZBKpAp_6TGI)

## Daily Progress

### Day 1
Started with "Coding Stable Diffusion from scratch in PyTorch."

#### Key Concepts Covered
1. **Introduction to Stable Diffusion Basics**
   - Distributions
   - Forward and Reverse process
   - Generating new data from pure noise
   - U-Net
   - CLIP
   - Autoencoder
   - Architecture (Text-to-Image, Image-to-Image, In-Painting)

2. **Core Training Process**
   1. Input a clean image
   2. Add random noise
   3. Predict the noise
   4. Compare prediction to actual noise
   5. Update parameters to minimize loss

3. **Key Components**
   - **Probability Distribution**: How probable are 2 or more values to appear together
   - **UNet**: Neural network that predicts noise at each step
   - **Scheduler**: Controls the denoising process and determines how much noise to remove at each step
   - **Conditioning**: Adding additional information to guide the denoising process (e.g., prompts or images)

#### Deep Dive: VAE Encoder
Started implementing the **VAE Encoder**, which has the following role:
- Encodes images/noise into latent vectors
- Compresses data into smaller dimensions while increasing feature representation
- More efficient processing due to reduced image size

#### Learning Notes & Insights
- The latent space encodes concepts, not just pixels
  - Helps model learn essential features (edges, corners, colors)
  - Easier than learning at pixel level
- Still unclear:
  - Exact encoder mechanics
  - Mathematical process of concept learning from pixels
- Interesting historical note: VAE architecture choice was based on its proven effectiveness in similar projects

#### Technical Concepts
- **Probabilistic Understanding**: Model learns:
  - Likelihood of pixels being part of the image
  - Spatial relationships within the image
