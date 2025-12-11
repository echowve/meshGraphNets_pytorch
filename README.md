# 🌊 Learning Mesh-Based Simulation with Graph Networks  
### *Fast, Adaptive, and Physics-Informed Neural Simulators for Complex Fluid Dynamics*

This repository provides a **PyTorch + PyG (PyTorch Geometric)** implementation of **MeshGraphNets**—a powerful graph neural network framework for learning mesh-based physical simulations. We focus on the **flow around a circular cylinder** problem, reproducing and extending the groundbreaking work from DeepMind.

> 🔬 **Original Paper**:  
> [**Learning Mesh-Based Simulation with Graph Networks**](https://arxiv.org/abs/2010.03409)  
> *Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, Peter W. Battaglia*  
> **ICLR 2021**
---

## ✨ Why This Project?

- **Physics-aware learning**: Leverages mesh structure to respect geometric and physical priors.
- **High performance**: Runs **10–100× faster** than traditional solvers while maintaining fidelity.
- **Extensible**: Built on PyTorch Geometric—easy to adapt to new PDEs, materials, or domains.

---

## 👥 Authors

- Jiang  
- Zhang  
- Chu  
- Qian  
- Li  
- Wang  

---

## 🛠️ Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

> 💡 **Note**: TensorFlow < 1.15.0 is required only for parsing the original TFRecord datasets.

---

## 🚀 Quick Start

### 1. Download the Dataset
We use DeepMind’s `cylinder_flow` dataset:

```bash
aria2c -x 8 -s 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord -d data
aria2c -x 8 -s 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/valid.tfrecord -d data
aria2c -x 8 -s 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/test.tfrecord -d data
```

### 2. Parse TFRecords
Convert to PyTorch-friendly format:

```bash
python parse_tfrecord.py
```
> Output saved in `./data/`.

### 3. Train the Model
```bash
python train.py
```

### 4. Run Rollouts & Visualize
Generate long-horizon predictions and render videos:

```bash
python rollout.py          # saves results to ./results/
python render_results.py   # generates videos in ./videos/
```

---

## 🎥 Demos

### Results on DeepMind’s `cylinder_flow`:
| Demo 0 | Demo 1 |
|------------|--------------|
| ![Demo 0](videos/0.gif) | ![Demo 1](videos/1.gif) |

### Results on **our own CFD-generated data** (new geometries & conditions):
| Demo 2 | Demo 3 |
|------------|--------------|
| ![Demo 2](videos/2.gif) | ![Demo 3](videos/3.gif) |

> ✅ The model generalizes well—even to unseen flow regimes and mesh configurations!

---

## 📬 Contact

Have questions, suggestions, or want to collaborate?  
📧 Reach out: [jianglx@whu.edu.cn](mailto:jianglx@whu.edu.cn)

---

> ⭐ **If you find this project useful, please consider starring the repo!**  
> Your support helps us keep improving open-source scientific ML tools.
