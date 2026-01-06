# Structure-From-Motion (SfM)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-2.2.1-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**3D Reconstruction from 2D Images**

*Recovering camera motion and 3D scene geometry from multiple uncalibrated images*

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Pipeline](#pipeline-architecture) • [Contact](#contact)

![Structure-From-Motion Demo](https://github.com/user-attachments/assets/a1088068-b8c2-4ca0-a56e-c55a85f8c5f4)

</div>

---

## 👋 About This Project

I'm **Harmeet Dhillon**, a robotics engineer specializing in computer vision and autonomous systems. This project represents my deep dive into 3D reconstruction techniques, implementing a complete Structure-from-Motion pipeline from scratch. Through this work, I'm exploring the mathematical foundations of multi-view geometry and building the core algorithms that enable robots to understand and navigate 3D environments. This implementation covers everything from feature matching and camera pose estimation to bundle adjustment and 3D point cloud generation—essential skills for my research in SLAM, autonomous navigation, and robotic perception.

---

## Overview

**Structure-from-Motion (SfM)** reconstructs 3D scene geometry and camera poses from regular 2D photographs. Unlike traditional methods requiring calibrated cameras or structured lighting, SfM works with ordinary images taken from different viewpoints—the same technique used in 3D scanning, autonomous vehicles, and augmented reality applications.

### Key Capabilities

- **Camera Pose Estimation**: Recovers camera positions and orientations in 3D space
- **3D Point Cloud Generation**: Reconstructs sparse 3D points representing the scene geometry
- **Bundle Adjustment**: Optimizes camera parameters and 3D points simultaneously
- **Feature Matching**: Robust correspondence finding across multiple views
- **Outlier Rejection**: RANSAC-based filtering for reliable reconstructions

---

## ✨ Features

- 🎥 **Multi-View Reconstruction** - Build comprehensive 3D models from multiple images
- 📐 **Epipolar Geometry** - Essential and Fundamental matrix estimation with RANSAC
- 🎯 **Feature Matching** - Robust SIFT/SURF feature detection with outlier filtering
- 🔺 **Triangulation** - Linear and non-linear 3D point reconstruction
- 🔧 **Bundle Adjustment** - Global optimization using Levenberg-Marquardt
- 📊 **3D Visualization** - Interactive point cloud and camera trajectory plots
- ⚡ **Incremental Reconstruction** - PnP-based sequential camera registration

---

## 🔧 Prerequisites

- **Python 3.8+**
- **pip** (Python package installer)
- **4GB+ RAM** (8GB recommended for large datasets)

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Harmeet-Dhillon/Structure-From-Motion.git
cd Structure-From-Motion
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv sfm_env

# Activate virtual environment
# On Windows:
sfm_env\Scripts\activate
# On macOS/Linux:
source sfm_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install numpy==2.2.1
pip install pandas==2.2.3
pip install opencv-python==4.10.0
pip install matplotlib==3.10.0
pip install scipy==1.15.1
pip install tqdm==4.67.1
```

### 4. Verify Installation

```bash
python -c "import cv2, numpy, scipy; print('Installation successful!')"
```

---

## 📁 Project Structure

```
Structure-From-Motion/
│
├── Phase1/
│   ├── Data/                          # Data directory (required)
│   │   ├── calibration.txt           # Camera calibration parameters
│   │   ├── matching*.txt             # Feature matches between image pairs
│   │   └── *.png                     # Input images
│   │
│   ├── Wrapper.py                    # Main execution script
│   ├── GetInliersRANSAC.py          # RANSAC implementation
│   ├── EstimateFundamentalMatrix.py  # Fundamental matrix estimation
│   ├── EssentialMatrixFromFundamentalMatrix.py
│   ├── ExtractCameraPose.py          # Camera pose recovery
│   ├── LinearTriangulation.py        # 3D point triangulation
│   ├── DisambiguateCameraPose.py     # Pose disambiguation
│   ├── NonlinearTriangulation.py     # Refined triangulation
│   ├── PnPRANSAC.py                  # Perspective-n-Point with RANSAC
│   ├── NonlinearPnP.py               # Non-linear PnP refinement
│   ├── BuildVisibilityMatrix.py      # Track visibility across views
│   └── BundleAdjustment.py           # Global optimization
│
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
└── LICENSE                           # License information
```

---

## 🚀 Usage

### Basic Usage

1. **Prepare Your Data**

   Ensure your data is organized in the `Phase1/Data/` directory:
   ```
   Phase1/Data/
   ├── calibration.txt       # Camera intrinsic parameters
   ├── matching1.txt         # Feature correspondences for image pair 1-2
   ├── matching2.txt         # Feature correspondences for image pair 2-3
   ├── matching3.txt         # Feature correspondences for image pair 3-4
   ├── 1.png                 # Input image 1
   ├── 2.png                 # Input image 2
   ├── 3.png                 # Input image 3
   └── ...
   ```

2. **Run the Pipeline**

   Navigate to the `Phase1` directory and execute:
   ```bash
   cd Phase1
   python Wrapper.py
   ```

3. **View Results**

   The script will generate:
   - 3D point cloud visualization
   - Camera trajectory plot
   - Reprojection error statistics
   - Output files with reconstructed structure

### Advanced Usage

#### Custom Parameters

Edit `Wrapper.py` to adjust reconstruction parameters:

```python
# RANSAC parameters
RANSAC_THRESHOLD = 0.05
RANSAC_MAX_ITERATIONS = 1000

# Bundle adjustment parameters
BA_MAX_ITERATIONS = 100
BA_TOLERANCE = 1e-6
```

#### Processing Subset of Images

To process only specific images:

```python
# In Wrapper.py
IMAGE_INDICES = [1, 2, 3, 5, 7]  # Process only these images
```

---

## 🔬 Pipeline Architecture

```
Input Images
    ↓
Feature Detection & Matching
    ↓
Fundamental Matrix (RANSAC)
    ↓
Essential Matrix & Camera Pose
    ↓
Triangulation (Linear + Non-linear)
    ↓
Pose Disambiguation
    ↓
Incremental Reconstruction (PnP)
    ↓
Bundle Adjustment
    ↓
3D Point Cloud Output
```

### Key Algorithms

**Fundamental Matrix**: 8-point algorithm with RANSAC for robust geometric estimation

**Triangulation**: Direct Linear Transform (DLT) followed by non-linear refinement to minimize reprojection error

**PnP RANSAC**: Perspective-n-Point for registering new cameras to existing 3D structure

**Bundle Adjustment**: Joint optimization of all camera poses and 3D points using Levenberg-Marquardt

---

## 🔍 Technical Details

### Core Mathematical Formulations

**Fundamental Matrix**: Encodes epipolar geometry between two views
```
x2^T F x1 = 0
```

**Essential Matrix**: Relates to camera motion
```
E = K2^T F K1
E = [R|t]
```

**Triangulation**: Reconstructs 3D points from 2D correspondences
```
minimize Σ ||x_i - π(K[R|t]X)||^2
```

**Bundle Adjustment**: Global optimization
```
minimize Σ_i Σ_j ||x_ij - π(K_i[R_i|t_i]X_j)||^2
```

### Implementation Highlights
- Normalized 8-point algorithm with rank-2 constraint
- RANSAC with adaptive thresholding for outlier rejection
- Sparse Levenberg-Marquardt optimization
- Cheirality checks to ensure valid reconstructions

---

## 📊 Results

### Output Metrics
- **Reprojection Error**: Mean pixel distance between observed and projected points
- **3D Point Count**: Number of successfully reconstructed points
- **Camera Accuracy**: Rotation and translation error metrics

### Example Output
```
=== SfM Reconstruction Summary ===
Images processed: 5
3D points: 1,247
Mean reprojection error: 0.82 pixels
Bundle adjustment iterations: 23
Processing time: 12.4 seconds
```

The pipeline generates 3D point cloud visualizations, camera trajectory plots, and reprojection error distributions.

---

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 2.2.1 | Numerical computations and linear algebra |
| pandas | 2.2.3 | Data handling and manipulation |
| opencv-python | 4.10.0 | Image processing and computer vision |
| matplotlib | 3.10.0 | Visualization and plotting |
| scipy | 1.15.1 | Optimization and scientific computing |
| tqdm | 4.67.1 | Progress bars for long operations |

### Installation Command

```bash
pip install numpy==2.2.1 pandas==2.2.3 opencv-python==4.10.0 matplotlib==3.10.0 scipy==1.15.1 tqdm==4.67.1
```

Or using requirements.txt:

```bash
pip install -r requirements.txt
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Data folder not found" | Run `Wrapper.py` from `Phase1` directory |
| "ModuleNotFoundError" | Install missing package: `pip install <package>` |
| Memory error | Reduce image count or limit 3D points |
| Poor reconstruction | Check image overlap, feature matches, calibration |
| RANSAC fails | Increase threshold: `RANSAC_THRESHOLD = 0.1` |

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit your changes**: `git commit -m 'Add some AmazingFeature'`
4. **Push to the branch**: `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

### Areas for Improvement

- [ ] Dense reconstruction using MVS (Multi-View Stereo)
- [ ] GPU acceleration for feature matching
- [ ] Deep learning-based feature detection
- [ ] Real-time SfM for video sequences
- [ ] Loop closure detection
- [ ] Texture mapping and mesh generation
- [ ] Integration with SLAM systems

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This project implements concepts from foundational computer vision literature:
- *Multiple View Geometry in Computer Vision* by Hartley & Zisserman
- *Computer Vision: Algorithms and Applications* by Richard Szeliski
- OpenCV documentation and research papers on Structure-from-Motion

Built with [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), and [Matplotlib](https://matplotlib.org/).

---

## 📧 Contact

**Harmeet Dhillon**

- Email: hdhillon3196820@icloud.com
- LinkedIn: [harmeet-dhillon-826a43237](https://www.linkedin.com/in/harmeet-dhillon-826a43237/)
- GitHub: [@Harmeet-Dhillon](https://github.com/Harmeet-Dhillon)

---

<div align="center">

**If you find this project helpful, please consider giving it a ⭐️**

Made with ❤️ by Harmeet Dhillon

</div>
