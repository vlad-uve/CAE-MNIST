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

| Experiment # | Hyperparameter         | Variants                                  | Result Summary                            | Notebook Link |
|--------------|-------------------------|-------------------------------------------|-------------------------------------------|----------------|
| 1            | Number of Filters       | `[16,32,32]`, `[32,64,128]`               | Wider = better, but more costly           | [filters](notebooks/experiment_1_filters.ipynb) |
| 2            | Latent Dimension        | `2`, `8`, `32`, `64`                      | Too small fails; too large is wasteful    | [latent](notebooks/experiment_2_latent_dim.ipynb) |
| 3            | Batch Normalization     | Off vs On                                 | Smoother training, slight improvement     | [batchnorm](notebooks/experiment_3_batchnorm.ipynb) |
| 4            | Activation Function     | ReLU vs Leaky ReLU                        | Modest gain in sharpness and stability    | [activation](notebooks/experiment_4_activation.ipynb) |
| 5            | Convolutional Depth     | 2 vs 3 encoder/decoder blocks             | Marginal improvement with added layers    | [depth](notebooks/experiment_5_depth.ipynb) |


---

## ✅ Key Takeaways

- **Latent dimension size** and **filter width** had the most significant effect on model performance
- **Very small latent spaces** (e.g., 2) resulted in poor reconstructions; overly large ones (e.g., 64) added cost without benefit
- **BatchNorm** and **Leaky ReLU** offered small, consistent improvements in stability and output quality
- **Additional convolutional layers** showed marginal benefit on MNIST
- The **baseline model** (`latent_dim = 32`, `filters = [32,32,64]`) remains the best trade-off between performance and complexity

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
