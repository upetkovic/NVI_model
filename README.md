# NVI Data Preprocessing Pipeline

This repository contains the full preprocessing pipeline for extracting data used in **Nonverbal Immediacy (NVI)** estimation research. It includes custom integrations of several third-party tools and models for object tracking, segmentation, visual embedding extraction, and emotion recognition.

---
## 💻 System Requirements

To run this preprocessing pipeline efficiently, the following hardware and software requirements are recommended:

### ✅ Minimum Hardware
- **GPU:** NVIDIA GPU with at least **8GB of VRAM**
  - Recommended: NVIDIA RTX 3070 or better

> ⚠️ The pipeline relies on GPU-accelerated models (e.g., SAM, DINOv2, transformers), which are slow or unusable on CPU-only setups.

### 🖥️ Operating System
- The code has been tested on **Ubuntu 20.04 LTS**
- Other Linux distributions may work, but are untested
  
> ⚠️ Windows may work but is not officially tested; macOS is not supported due to the lack of NVIDIA GPU support

---

## 🔧 Environment Setup

⚠️ **Important Build Requirement**

This project requires a C++ compiler (`g++`) to build certain CUDA/C++ extensions (e.g., GroundingDINO).

If you're on **Ubuntu or Debian**, make sure you have it installed:

```bash
sudo apt update
sudo apt install build-essential
```

We recommend setting up the environment using the provided script:

```bash
# Option 1: Run the script directly
bash setup_all_envs.sh

# Option 2: If needed, make it executable first
chmod +x setup_all_envs.sh
./setup_all_envs.sh
```

This will:
- Create and set up the necessary Conda environments (`nvi_env`, `dinov2`, `hsemotion`)
- Install all dependencies (including CUDA, PyTorch, and third-party libraries)
- Download checkpoints
- Install this repository in editable mode (`pip install -e .`)

>⚠️ Note: Setting up the full environment can take a while (15–30 minutes), depending on your internet speed and system. Some packages require compilation, and large model checkpoints will be downloaded.

If you plan to run the main pipeline remotely and only do annotation locally (where a GUI and browser are available), it's recommended to create a lightweight environment for just the labeling step:

```bash
# Setup the environment
bash setup_labeling_env.sh
```
---

## 🗂 Repository Structure

```
NVI/
├── external/                # Third-party code (copied and partially modified)
│   ├── segment_and_track_anything/
│   ├── GroundingDINO/
│   ├── dinov2/
│   └── Pytorch-Correlation-extension/
├── your_code/               # Your own modules/scripts
├── setup_all_envs.sh        # One-command setup script
├── requirements.txt
├── README.md
└── ...
```

---

## ▶️ How to Estimate NVI from Video

### 1. 📁 Configure Paths

Before starting, edit the `config.yaml` file and set the required paths:

```yaml
video_dir: "/home/uros/Documents/project31/data/Talis_teachers_shorten_clean"
output_dir: data/outputs
json_first_frame_labels_path: "data/inputs/first_frames_TALIS/via_project_20May2025_17h16m_json.json"
first_frames_dir: "data/inputs/first_frames_TALIS"
```

---

### 2. 🔁 Label the Teacher in the First Frame

You need to annotate the teacher's position to enable tracking.

- Activate your environment:
  - `conda activate nvi_env`
  - or if only labeling locally: `conda activate labelling_env`

> ❗ Labeling requires a GUI and browser access, so it should be done on your local machine.

Then run:

```bash
python scripts/extract_frames_main.py
```

This extracts the first frames from each video.

---

### 3. 🏷 Annotate with VIA Tool

1. Open `external/via-2.0.12/via.html` in your web browser.
2. Click **"Add Files"** and select all extracted frames (you can use Ctrl+A).
3. Label the teacher using the **"Point Region Shape"**.
   - Annotate only the teacher’s head (avoid tools/materials they may be holding).
   - Use arrow keys to navigate between frames.
4. Export the annotations as a `.json` file and update the path in your `config.yaml`.

---

### 4. 🚀 Run the Pipeline to Estimate NVI

Activate your environment again (if not already):

```bash
conda activate nvi_env
```

Then run:

```bash
python scripts/batch_run_pipeline.py
```

After completion, the estimated NVI scores will be saved as a `.csv` file in your configured `output_dir`.

---

## Third-Party Code Attribution

This repository includes external code under their original open-source licenses for research purposes only.

Included:

- **Segment and Track Anything**  
  Source: https://github.com/z-x-yang/Segment-and-Track-Anything  
  License: Apache 2.0  
  Status: Modified

- **GroundingDINO**  
  Source: https://github.com/IDEA-Research/GroundingDINO  
  License: Apache 2.0  
  Status: Unmodified

- **DINOv2**  
  Source: https://github.com/facebookresearch/dinov2  
  License: CC-BY-NC 4.0  
  Status: Unmodified

- **Pytorch-Correlation-extension**  
  Source: https://github.com/ClementPinard/Pytorch-Correlation-extension  
  License: MIT  
  Status: Unmodified

- **HSEmotion**  
  Source: https://github.com/av-savchenko/hsemotion  
  License: MIT  
  Status: Used for emotion recognition

- **VGG Image Annotator (VIA)**  
  Source: https://www.robots.ox.ac.uk/~vgg/software/via/  
  License: BSD  
  Status: Used as GUI labeling tool

> ⚠️ This repository is intended **only for research and educational use**. Some included tools (e.g., DINOv2) are under non-commercial licenses.

---

## 📚 Citation

If you use this repository or any part of the preprocessing pipeline in your research, **please cite the following paper**:

**Petković, U., Frenkel, J., Hellwich, O., & Lazarides, R. (2024, September).**  
*Nonverbal Immediacy Analysis in Education: A Multimodal Computational Model.*  
In *International Conference on Simulation of Adaptive Behavior* (pp. 326–338). Cham: Springer Nature Switzerland.  
[View paper](https://link.springer.com/chapter/10.1007/978-3-031-71533-4_26)

```bibtex
@inproceedings{petkovic2024nonverbal,
  title={Nonverbal immediacy analysis in education: A multimodal computational model},
  author={Petković, Uroš and Frenkel, Jonas and Hellwich, Olaf and Lazarides, Rebecca},
  booktitle={International Conference on Simulation of Adaptive Behavior},
  pages={326--338},
  year={2024},
  organization={Springer}
}
```

---

## Contact

For questions or collaboration, please open an issue or contact the maintainer.
