# Generative AI - Assignment 3

This repository contains the implementation of three distinct Generative Adversarial Network (GAN) architectures, as part of the Generative AI Assignment 3. The assignment explores various aspects of image generation, paired image translation, and unpaired image translation using PyTorch. All models have been optimized for execution in Kaggle's Dual T4 GPU environment and include interactive Gradio applications for real-world testing.

---

## Question 1: Tackling Mode Collapse in GANs (DCGAN vs. WGAN-GP)

### What has been done?
We implemented and compared two architectures: a baseline **Deep Convolutional GAN (DCGAN)** and a **Wasserstein GAN with Gradient Penalty (WGAN-GP)**. The goal was to train these models on image datasets (Pokémon Sprites / Anime Faces) to generate novel, realistic images from random noise.

### Architecture, Training Flow, and Code Perspective
#### 1. Architecture details (`Q1_DCGAN_vs_WGAN-GP.ipynb`)
- **DCGAN**: 
  - **Generator**: Uses sequential `ConvTranspose2d` layers to upscale a 100-dimensional noise vector to a 64x64x3 image. Applies `BatchNorm2d` and `ReLU`, ending with a `Tanh` activation.
  - **Discriminator**: Uses sequential `Conv2d` layers to downscale the 64x64 image to a single scalar value. Applies `BatchNorm2d` and `LeakyReLU`, ending with a `Sigmoid` activation for BCE classification.
- **WGAN-GP**:
  - **Generator**: Shares the same architecture as the DCGAN generator.
  - **Critic**: Replaces the Discriminator. It structurally removes the `Sigmoid` layer to output raw linear scores (Wasserstein distance). `BatchNorm` is replaced with `InstanceNorm2d` to prevent correlation between samples in a batch, which would interfere with the gradient penalty.

#### 2. Training Steps and Flow
- **Data Loading**: We implemented a custom `AnimeDataset` using PyTorch's `Dataset` class to load, resize (to 64x64), and normalize images to the `[-1, 1]` range.
- **WGAN-GP Training Loop**: 
  - For each batch, the Critic is trained for **5 iterations** per 1 Generator iteration (`critic_iters = 5`).
  - **Gradient Penalty (`gp`)**: During Critic updates, we compute interpolations between real and fake images, calculate the gradients with respect to these interpolations using `torch.autograd.grad`, and penalize the gradient norm deviation from 1.
  - **Loss Computation**: The Critic minimizes `-(mean(critic_real) - mean(critic_fake)) + lambda_gp * gp`. The Generator minimizes `-mean(critic_fake)`.
- **Optimization**: We used the `Adam` optimizer with `lr=0.0002` and specific betas `(0.5, 0.999)` for stability.
- **Deployment**: The solution incorporates a Gradio UI (`generate_anime_faces` function) where users can manipulate the number of generated faces and the random seed for real-time model inference.

### Why was it done?
Standard GANs (like DCGAN) frequently suffer from **mode collapse**—where the generator produces limited varieties of samples—and training instability due to vanishing gradients. WGAN-GP addresses these fundamental issues by providing a smoother, more meaningful loss metric (Wasserstein distance) that continuously guides the generator, even when the critic is highly optimized.

### Conclusions
- **Training Stability**: WGAN-GP demonstrated significantly more stable and predictable training dynamics. The critic's loss consistently correlated with the visual quality of the generated images.
- **Mode Collapse Eliminated**: WGAN-GP successfully generated highly diverse images without collapsing into a single mode, heavily outperforming the baseline DCGAN in producing varied Anime Faces / Pokémon sprites.

---

## Question 2: Doodle-to-Real Image Translation and Colorization using Pix2Pix

### What has been done?
We developed a paired image-to-image translation system using **Pix2Pix** to convert Sketch/Edge images to Realistic images, and Grayscale images to Colored images (using the CUHK Face Sketch and Anime Sketch Colorization datasets).

### Architecture, Training Flow, and Code Perspective
#### 1. Architecture details (`Q2_Pix2Pix.ipynb`)
- **Generator (U-Net)**: 
  - Constructed using custom `Downsample` (Conv2d, LeakyReLU, BatchNorm) and `Upsample` (ConvTranspose2d, LeakyReLU, BatchNorm) modules.
  - Features 6 downsampling layers reducing the 256x256 input to a bottleneck, followed by 5 upsampling layers. 
  - **Skip Connections**: In the forward pass, outputs from downsampling layers are concatenated with inputs to the upsampling layers (`torch.cat([up_x, down_x], dim=1)`), preserving high-frequency spatial details.
- **Discriminator (PatchGAN)**: 
  - Takes a 6-channel input (concatenating the input sketch and target color image: `torch.cat([img, mask], dim=1)`).
  - Processes it through 4 downsampling layers to output a localized 16x16 patch prediction matrix instead of a single scalar, focusing the discriminator on high-frequency "texture" realism rather than just global shapes.

#### 2. Training Steps and Flow
- **Data Loading**: Configurable datasets (`AnimePairDataset`, `CUFSPairedDataset`) handling both concatenated image pairs and separate sketch/photo directories.
- **Mixed Precision & Multi-GPU**: The pipeline heavily utilizes `torch.amp.autocast` and a `GradScaler` to prevent OOM errors and accelerate training on the Kaggle T4 GPUs. Models are wrapped in `nn.DataParallel` for dual-GPU utilization.
- **Training Loop**:
  - The Discriminator is updated by evaluating both real pairs and generated (fake) pairs.
  - The Generator is trained to fool the discriminator (`BCEWithLogitsLoss`) while simultaneously minimizing the pixel-wise difference with the target (`L1Loss`).
  - Total Generator Loss = `G_loss_BCE + LAMBDA * G_loss_L1` (with `LAMBDA = 7`).
- **Deployment**: Integrated a `predict_from_sketch` inference function and a visualization pipeline to immediately plot "Input", "Target", and "Generated" images for validation.

### Why was it done?
Translating an image while maintaining its underlying structure requires a model that can both hallucinate textures/colors and strictly adhere to spatial constraints. A Conditional GAN (cGAN) like Pix2Pix is ideal for this, as the generator is conditioned on the input image rather than random noise. The L1 loss forces global correctness, while the PatchGAN forces high-frequency local realism.

### Conclusions
- **Detail Preservation**: The U-Net's skip connections were crucial; without them, the generator struggled to align the fine lines of the sketches with the output.
- **Texture Realism**: The PatchGAN discriminator successfully pushed the generator to create realistic colors and textures, avoiding the blurry results that typically occur when using only L1 reconstruction loss.

---

## Question 3: Unpaired Image-to-Image Translation using CycleGAN

### What has been done?
We designed an unpaired image translation system using **CycleGAN** to translate between Sketch and Photo domains (using datasets like TU-Berlin and Sketchy). Unlike Question 2, this model learns mappings without requiring one-to-one paired datasets.

### Architecture, Training Flow, and Code Perspective
#### 1. Architecture details (`Q3_Cycle_GAN.ipynb`)
- **Generators ($G_{AB}$ and $G_{BA}$)**: 
  - Built using a **ResNet-based** architecture containing approximately 11.3M parameters each.
  - Structure: Initial convolution block (`c7s1-64`), two downsampling layers (`d128`, `d256`), **6 Residual Blocks** (`ResNetBlock` with Reflection Padding, Conv2d, and InstanceNorm2d), and two upsampling layers before the final output layer.
  - ResNet blocks are critical here as they prevent vanishing gradients while maintaining the structural content of the input domain during style translation.
- **Discriminators ($D_A$ and $D_B$)**: 
  - Utilize a **PatchGAN** architecture (`C64-C128-C256-C512-output`).
  - Discriminators evaluate overlapping patches of the image to determine if they are real or fake, promoting sharper textures.

#### 2. Training Steps and Flow
- **Data Pipeline**: 
  - Implemented an `UnpairedDomainDataset` that independently shuffles Domain A (Sketches from TU-Berlin/QuickDraw/Sketchy) and Domain B (Real Photos from STL-10).
- **Advanced Training Mechanics**:
  - **Replay Buffer**: Implemented a `ReplayBuffer` of size 50. Instead of feeding the Discriminator only the latest generated images, it feeds a history of generated images to prevent the discriminator from oscillating or forgetting previous generations, significantly stabilizing training.
  - **Learning Rate Decay**: Utilized a custom `LambdaLR` scheduler to linearly decay the learning rate starting at a specified epoch (`decay_epoch = 10`).
- **Loss Computations**:
  - Uses `MSELoss` (Least Squares GAN) for adversarial loss to mitigate vanishing gradients.
  - Computes `criterion_cycle` (`L1Loss`) for $G_{AB}(G_{BA}(B)) \approx B$ and $G_{BA}(G_{AB}(A)) \approx A$.
  - Computes `criterion_identity` (`L1Loss`) for $G_{AB}(B) \approx B$.
- **Evaluation Engine**: Implemented `structural_similarity` (SSIM) and `peak_signal_noise_ratio` (PSNR) from `skimage.metrics` to quantitatively evaluate structural preservation across domains.

### Why was it done?
In many real-world scenarios, obtaining perfectly aligned pairs of images (e.g., a specific real-life photo and its exact hand-drawn sketch) is difficult or impossible. CycleGAN solves this by enforcing cycle consistency, allowing the network to learn the "style" of a domain and apply it to another domain entirely unsupervised.

### Conclusions
- **Unpaired Learning Success**: The model successfully learned to inject photo-realistic textures into sketches and extract edge-like structures from photos without explicit pairing.
- **Importance of Cycle Loss**: The Cycle Consistency loss proved to be the most critical component. Without it, the model would suffer from mode collapse, simply mapping all sketches to a single realistic photo. The cyclic constraint successfully forced the generator to retain the original object's structure and shape.
