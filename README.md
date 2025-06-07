# Convolutional Autoencoder (CAE) for MNIST digits

This project explores the development and architectural optimization of a convolutional autoencoder (CAE) for the MNIST handwritten digits dataset. Instead of relying on grid search, the study isolates and evaluates individual architectural hyperparameters through a series of experiments in order to gain practical insights into how they affect key aspects of autoencoder performance: latent space compactness, reconstruction quality, and computational efficiency. 

The study begins by defining and training a baseline CAE model, which serves as a reference point for all subsequent experiments. Each of the five experiments varies one factor at a time — the number of convolutional filters, the size of the latent space, the use of batch normalization, the choice of activation function (ReLU vs. Leaky ReLU), and the depth of the encoder-decoder architecture — to assess its influence on training dynamics, reconstruction quality, and overall model complexity.

The insights gained from these isolated experiments are ultimately used to propose a balanced model configuration that offers an effective trade-off between accuracy and resource usage, within the scope of MNIST digit reconstruction.

---

## 📉 ~Objective~

~To evaluate the impact of individual architectural components on a CAE's ability to compress and reconstruct grayscale handwritten digits. Each experiment isolates a single hyperparameter to assess its effect independently.~

---

## Baseline model

Loss Curve

Reconstruction

## 🔍 Experiment Overview

| Experiment # | Hyperparameter         | Variants                                  | Notebook Link |
|--------------|-------------------------|-------------------------------------------|----------------|
| 1            | Number of convolutional filters       | `[16,32,32]`, `[32,32,64]`, `[32,64,128]`               | [Experiment #1 - Different Number of Filters](notebooks/experiment_1(filters).ipynb) |
| 2            | Latent space dimension        | `2`, `8`, `32`, `64`                      | [Experiment #2 - Different Laten Space Sizes](notebooks/experiment_2(latent_dim).ipynb) |
| 3            | Batch normalization     | Off vs On                                 | [Experiment #3 - Usage of Batch Normalization](notebooks/experiment_3(batch_norm).ipynb) |
| 4            | Number of convolutional layers     | `2 layers`, `3 layers`, `4 layers`           | [Experiment #4 - Different Convolutional Depths](notebooks/experiment_4(conv_depth).ipynb) |
| 5            | Activation function type| `ReLU` vs 'Leaky ReLU' with `0.01`, '0.1', '0.2' slopes                         | [Experiment #5 - Usage of Leaky ReLU](notebooks/experiment_5(leaky_relu).ipynb) |


---

## Key Insights from Experiments
### 1. Latent dimension is the primary driver of reconstruction quality:
- Among all tested architectural parameters, **latent space dimensionality** had the most significant impact on both training loss and visual reconstruction quality
<div align="center">
  <img src="outputs/experiment_2_files/experiment_2_image_loss.png" width="420"/>
  <p><em>Validation loss across different latent dimensions (2, 8, 32, 64)</em></p>
</div>

- **Smaller latent sizes** (e.g., 2 or 8) led to blurry or ambiguous digit reconstructions  
<div align="center">
  <img src="outputs/experiment_2_files/experiment_2_image_reconstruction_1.png" width="420"/>
  <p><em>Reconstruction with latent_dim = 2 — degraded clarity and digit confusion</em></p>
</div>

- **Larger latent sizes** (32 and 64) improved accuracy, but with diminishing improvements beyond 32  
<div align="center">
  <img src="outputs/experiment_2_files/experiment_2_image_reconstruction_4.png" width="420"/>
  <p><em>Reconstruction with latent_dim = 32 and 64 — near-identical quality</em></p>
</div>

> 🧩 **Conclusion:** Increasing latent space dimensionality enhances reconstruction quality but contradicts the compactness objective.

### 2. ⚙️ Filter Width Has Minor Effect on reconstruction quality and loss
- Increasing the number of convolutional filters led to **slightly lower reconstruction loss** and **marginal improvement of reconstruction quality**.
- All configurations of convolution filters successfully preserved the structure of the digits with **indistinguishable to the naked eye** visual differences between them
<div align="center">
  <img src="outputs/experiment_1_files/experiment_1_image_reconstruction_1.png"" width="500"/>
  <p><em>Reconstructions from models with different filter configurations:<br>
  [16,32,32] (narrow) vs [32,32,64] (baseline) vs [32,64,128] (wide)</em></p>
</div>

> 💡 Conclusion: Wider filters slightly reduce loss but offer no substantial visual improvement, making their added cost **unjustified for a simple dataset like MNIST**.
> 
### 3. 🔬 Other Factors (Depth, Activation, BatchNorm) are negligible 
- Increasing **convolutional depth** beyond two layers **did not improve reconstruction quality**. Although, while training dynamics varied early on, all three models eventually converged to similar loss level in the end

<div align="center">
  <img src="outputs/summary/loss_curves_depth.png" width="420"/>
  <p><em>Loss curves for CAE models with 2 vs 3 encoding/decoding blocks — same final performance</em></p>
</div>
- Switching from ** ReLU to Leaky ReLU** had no measurable effect on output quality
- Enabling **batch normalization** led to slightly smoother training, but final results remained unchanged

> 💡 Conclusion: As a result, these architectural modifications add complexity without any performance gains.

### 4. 📦 Model Complexity is only driven by Filters and Depth

- **Filter width** and **convolutional depth** have the greatest impact on parameter count  
- **Latent dimension** influences model size but in a relatively minor manner
- **Batch Normalization** adds some overhead
- **Activation function** has no impact on complexity
(bar plot)
<div align="center">
  <img src="outputs/summary/param_bar_chart.png" width="500"/>
  <p><em>Parameter count across model variants</em></p>
</div>
> 💡 **Conclusion:** To reduce computational footprint, prioritize decreasing filter size and number of layers.

### 5. 🎯 Latent Space is the hidden lever behind autoencoder efficiency

Among all explored parameters, **latent dimensionality stands out as the only one directly tied to every key objective**:
- It governs **reconstruction quality** - too small leads to poor digit retention  
- It shapes **model compactness** - larger spaces absolutely defeat the purpose of encoding  
- It influences **model size ** - smaller latent space results in fewer parameters count

> 🧠 **Unlike filters or depth**, which mainly affect model size, the latent space defines the *informational bottleneck* — the core idea behind autoencoding.

> 💡 **Conclusion:** Optimizing the latent space is not just about accuracy, but about aligning the model with its true purpose — learning **the minimal shape of encoding that can be decoded with the most meaningful reconstruction quality**.

## Key Conclusions from the Experiment

1. d
2. 

## ⚖️ Considerations for the Trade-Off Model
Given that the goal of the trade-off model is to achieve an optimal balance between reconstruction quality, model compactness, and computational efficiency - especially for lightweight tasks like MNIST digit encoding.

The following design decisions are grounded in experimental evidence:
### **🔐 Latent dimension is set to 16**  
  While latent size strongly affects output quality, it must remain small to serve the encoder’s purpose. A dimension of 16 offers a practical compromise between compression and clarity.

###**🧱 Depth is reduced to 2 encoding/decoding blocks**  
  Experimental results confirm that additional layers do not improve final loss or visual output, yet substantially increase parameter count.
- **🔽 Filters are narrowed to `[32, 32]`**  
  This reduces overall complexity while still preserving enough feature extraction capacity to support faithful reconstructions.

-**❌ Batch Normalization & Leaky ReLU are omitted**  
These components showed negligible effect on reconstruction quality and training dynamics, so they are excluded to simplify the architecture.

> 🧠 Together, these changes result in a lean, performant model — suitable for efficient deployment without significant sacrifice in quality.


---

## 📈 Visual Highlights

### 📉 Loss Curves (Latent Dimension Comparison)
*Example: `latent_dim = 2 vs 32 vs 64`*

![loss_curves_latent](outputs/summary/latent_loss_curve.png)

### 🖼️ Reconstruction Samples
*Example: `latent_dim = 2 vs 32 vs 64`*

![reconstruction_comparison](outputs/summary/latent_reconstruction.png)

> All models converged within 10 epochs. Models with `latent_dim = 2` struggled with digit clarity, while models with `latent_dim = 64` offered no visible improvement over the baseline.

---

## 🧠 Final Reflection

This study demonstrates the value of isolating hyperparameters to understand their impact, especially in a simple context like MNIST. Instead of relying on grid search, this approach helped surface which components meaningfully affect performance and which are less impactful.

While combining hyperparameters could yield better absolute performance, that was not the goal of this study. Instead, this work emphasizes **interpretability and efficiency**.

A trade-off model may be proposed later if computational cost is benchmarked.

---

## 🛠️ Bottom-Line Trade-Off Model

As a final step, this project proposes a single **bottom-line CAE model** designed to balance reconstruction quality with minimal resource usage. This model draws on the best-performing settings observed in the isolated experiments.

**Proposed configuration:**
- Filters: `[16, 32, 32]`
- Latent dimension: `16`
- Activation: Leaky ReLU
- Batch Normalization: Enabled
- Depth: 2-layer encoder/decoder

This model is not optimal in every metric but offers the best compromise for lightweight deployment with good reconstruction fidelity.

A dedicated notebook and performance metrics for this model may be added if resource benchmarking is completed.
| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Filters** | `[16, 32, 32]` | 2× lighter than baseline, loss ≈ +4 % |
| **Latent dim** | `16` | 50 % smaller rep., quality sweet-spot |
| **Depth** | 2 blocks | Extra block gave negligible gain |
| **Activation** | Leaky ReLU (α = 0.1) | Same quality, slightly steadier curves |
| **BatchNorm** | Enabled | Adds <0.5 % params, minor stability win |

👉 `src/models/cae_tradeoff.py` contains the exact class; results are logged in `notebooks/tradeoff_model.ipynb`.
---

## 🔧 Project Structure

├─ notebooks/ # each experiment as a standalone .ipynb
├─ outputs/
│ ├─ loss_curves/ # PNG loss curves per run
│ └─ reconstructions/ # sample originals & reconstructions
├─ src/
│ ├─ models/ # CAE variants
│ └─ utils.py # loaders, training loop
└─ README.md # ← you are here

---

## 📚 References

- PyTorch Documentation  
- *Deep Learning with PyTorch* (Paszke et al.)  
- MNIST Dataset

---

*Author: [Your Name / GitHub Handle]*
