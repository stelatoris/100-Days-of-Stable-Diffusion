# 100 Days of Stable Diffusion

## Overview
This repository documents my 100-day journey learning Stable Diffusion. Instead of spending time planning the perfect roadmap, I've chosen to dive in and learn by doing, filling in knowledge gaps along the way.

## Background
My background is in Design. So no CS, no ML, no Math, or anything technical. However, in 2021 I started learning programming to do fun Arduino projects. As for ML, I only took Andrew Ng's Machine Learning Specialization course which gave me the basics so I expect this to still be difficult since I have no background in Math. Nevertheless, I love to learn.

## Learning Approach
I will be following a structured top-down learning approach to learn the fundamentals. When there are gaps in my understanding, I will fill them along the way so I don't get bogged down too early. I adjust this approach if I feel that it is not working.

These are the two parallel tracks that I will follow for now:

1. **Hugging Face Diffusion Tutorial**
   - Getting quickly started with Diffusers
   - Learning practical implementation

2. **Understanding the Fundamentals**
   - Following Umar Jamil's "Coding Stable Diffusion from scratch in PyTorch" YouTube video as this might help me understand the library better.

## Learning Resources
- [Hugging Face Diffusion Tutorial](https://huggingface.co/docs/diffusers/stable_diffusion)
- [Hugging Face Diffusion Models Course](https://huggingface.co/learn/diffusion-course/en/unit0/1)
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

### Day 2

#### Key Concepts Covered from Umar Jamil's "Coding Stable Diffusion from scratch in PyTorch" YouTube video
Coding Encoder and Decoder

#### Encoder
- `Encoder` class is a subclass of `torch.nn.Module`
- `nn` PyTorch module containing classes for all neural network layers

**The Encoder class is made up of several layers and blocks:**

- `Conv2d`: Convolutional layer for image processing
- **ResidualBlock**: A residual block that adds the input to the output
- **Downsampling**: A downsampling layer that reduces the size of the image by a factor of 2, allowing the network to capture more spatial hierarchies effectively.
- **Normalization and Activation**: Normalization and activation functions are applied to the output of each block to stabilize the training process and prevent overfitting.
- **SiLU Sigmoidal Linear Unit**: A non-linear activation function that maps each input to a positive value between 0 and 1. It helps the network learn more complex representations.

- **Convolutional operations** apply a filter to produce a feature map.
- **Convolutional filters** scan small regions of the image, detecting important local features like edges, textures, or shapes. These features are then propagated through the network to contribute to the overall error minimization during training.
- **Features are identified** by learning which combinations of neighboring pixels activate the filter strongly.
- The reason features like edges get "learned" first is because the initial layers of a convolutional neural network detect simple, high-contrast features. Which is similar to how humans can recognize edges in images.

- **Residual Blocks**: Helps solve the problem of vanishing gradients when training deeper networks. Training deeper networks is better because it allows the network to learn more complex features.
- **Vanishing Gradients**: It was found that after a certain depth K, addidiontal layers might degrade the performance due to the vanishing gradient problem. Then the gradients of the loss function dimishes as the network gets deeper.
- **Identity Functions**: Return their output unchanged. f(x)=x. This is helpful to keep certain properties of the input intact.
- The idea is that if learning the mapping F(x) between layers is challenging due to vanishing gradients, residual connections allow the network to fall back on learning the simpler identity mapping F(x)=0, resulting in y=x. This ensures that the original input information can propagate forward, even when deeper layers fail to learn complex transformations effectively.
- This allows the network to select where to apply transformations where needed.
- The result in image recognition is that the initial layers might detect edges, where as the deeper layers start to focus on the details, and the deepest layers would piece together the image into something meaningful like object recognition.

**Attention Block**
- **Attention Mechanism**: A mechanism that allows the network to focus on important parts of the input while ignoring less important parts.
- It does that by assigning different weights to different parts of the input.
- The network learns to assign higher weights to the important parts of the input and lower weights to the less important parts.

**Hugging Face Diffusion Tutorial**

Pipelines:
- Loading pipelines `from diffusers import DiffusionPipeline`
- `pipeline = DiffusionPipeline.from_pretrained(model_id)` float32
- `pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)` float16. Much more efficient and similar results.
- `generator = torch.Generator("cuda").manual_seed(0)` is used to control the randomness of the image generation process. A specific seed to get the same result every time, or use a random seed to get a different result every time.

**Checkpoints**:
- Specific versions of ML models that have been saved at particular stages of training.

**Denoising Scheduler:**
- `pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config) ` Set a new scheduler configuration to the pipeline using a scheduler config.

- **Memory savings:** Most of the memory is taken by the cross-attention layers. running it sequentially instead of batch-wise may lead to better performance. 
- Call `pipe.enable_attention_slicing()` to enable attention slicing.

- **Checkpoints:** Better checkpoitns improve quality and speed.
- Newer does not always mean better. Try out and experiment with different checkpoints.

**Key Takeaways for Implementation:**
- Always use float16 when possible
- Enable attention slicing for better memory management
- Use `DPMSolverMultistepScheduler` with 20-25 steps
- Consider batch processing for multiple images
- Use seed values for reproducible results

### Day 3

**Comments:**
Changed my focus to the Hugging Face Diffusion Tutorial to get a highlevel overview of the complete process and see how it works. Then I will go back to the "Coding Stable Diffusion from scratch in PyTorch" video to fill in the gaps.

My understanding of the total process so far:
## High Level Run

### Training:

1. Initialize the UNet architecture.
2. Load and preprocess images.
3. Initialize the scheduler: Establish with a predefined schedule to guide noise addition and removal during training and inference.
4. VAE Setup: Encode images into a latent space, compress, and decompress back into images.
5. Optimizer Setup: Manages learning rate and other hyperparameters.
6. Start the main loop:
   - Use the scheduler to add noise to the input image based on the current time step.
   - Predict noise with UNet for the current time step
   - Compute loss between predicted noise and actual noise.
   - Backpropagate the loss to update model parameters.
   - Update the scheduler for the next time step.
   - Repeat for all time steps.
7. Save the final model: Store the trained model for future use.

### Inference:

1. Initial Setup: Begin with random noise: Begin the process with an initial noisy input.
2. Denoising Loop:
   - Predefined time steps guide how much noise to remove at each step.
     - Steps with each Timestep:
       -Input to UNet: Combine the current noisy image with the timestep information.
       -Predict noise: Use the UNet to predict noise for the current time step.
       -Remove noise: Scheduler uses the predicted noise to iteratively denoise the image, transforming it gradually.
       -Repeat for all time steps.

3. Final Processing:
  - Convert from Latent to Image: Use the VAE to convert the latent representation back to an image.
  - Post Processing: Normalize the image. Scale image properly.


### MVP (Minimum Viable Pipeline)

- Pipelines
- Models
- Schedulers

### Day 4

**Comments:**  
I'm starting to 'feel' like I'm going in the right direction. The big pieces of the Diffusion process are starting to stick and are less "black box" like.

Followed this tutorial: ["Fine-Tuning and Guidance"](https://huggingface.co/learn/diffusion-course/en/unit2/2)

#### Fine Tuning

**Gradient Accumulation:**  
To speed up training or large batches, we use it to accumulate gradients from multiple batches and then update the model parameters once in a while.

**Guidance:**
- Create a conditioning function.
- Introduce additional constraints during the denoising process.
- For example, if we want a specific color, we make a function that measures how close the current image is to the target color.
- It computes a "loss" or error value based on the difference between the current color of the image pixels and the target color.
- Then we add this function to the loss function.
- The loss function is now a combination of the original loss and the guidance loss.
- I can control when and by how much I want the Guidance effect to kick in.

**Img2Img:**  
- Encodes image into a set of latents.  
- Adds noise to latent and uses that as a starting point for denoising.  
- The 'Strength' parameter controls how much noise to add during denoising and the number of denoising steps.

**In-Painting:**  
- Encodes image into a set of latents.  
- Adds noise to latent and uses that as a starting point for denoising.  
- Takes only the white part of both latent variables and adds them together.

**Fine-Tuning Strategies:**
- If I fine-tune using a small dataset, then I probably need to retain most of the model's original training so it can understand arbitrary prompts that are not covered by the new dataset.
  - Use a low learning rate with exponential model averaging.
- If I have a large training set and I want to completely retrain the model, then:
  - Use a larger learning rate + more training.