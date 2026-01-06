# Structure-From-Motion (SfM)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10.0-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-2.2.1-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**3D Reconstruction from 2D Images**

*Recovering camera motion and 3D scene geometry from multiple uncalibrated images*

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Technical Details](#technical-details) • [Results](#results)

![Structure-From-Motion Demo](https://github.com/user-attachments/assets/a1088068-b8c2-4ca0-a56e-c55a85f8c5f4)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Pipeline Architecture](#pipeline-architecture)
- [Technical Details](#technical-details)
- [Results](#results)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project implements a complete **Structure-from-Motion (SfM)** pipeline that reconstructs 3D scene geometry and camera poses from a collection of 2D images. SfM is a fundamental technique in computer vision that powers applications like 3D scanning, autonomous navigation, augmented reality, and photogrammetry.

### What is Structure-from-Motion?

Structure-from-Motion is the process of estimating the 3D structure of a scene and the motion of the camera from a sequence of 2D images. Unlike traditional 3D reconstruction methods that require calibrated cameras or structured lighting, SfM works with regular photographs taken from different viewpoints.

### Key Capabilities

- **Camera Pose Estimation**: Recovers camera positions and orientations in 3D space
- **3D Point Cloud Generation**: Reconstructs sparse 3D points representing the scene geometry
- **Bundle Adjustment**: Optimizes camera parameters and 3D points simultaneously
- **Feature Matching**: Robust correspondence finding across multiple views
- **Outlier Rejection**: RANSAC-based filtering for reliable reconstructions

---

## ✨ Features

### Core Functionality

- 🎥 **Multi-View Reconstruction**: Process multiple images to build comprehensive 3D models
- 📐 **Epipolar Geometry**: Essential and Fundamental matrix estimation
- 🎯 **Feature Detection & Matching**: SIFT/SURF/ORB feature extraction with ratio test filtering
- 🔺 **Triangulation**: Linear and non-linear methods for 3D point reconstruction
- 📊 **Visualization**: Interactive 3D point cloud and camera pose visualization
- ⚡ **Efficient Processing**: Optimized algorithms with progress tracking

### Advanced Features

- **Bundle Adjustment**: Non-linear optimization using Levenberg-Marquardt
- **Reprojection Error Minimization**: Iterative refinement of camera and structure parameters
- **Cheirality Check**: Ensures reconstructed points are in front of cameras
- **Automatic Calibration**: Works with both calibrated and uncalibrated cameras
- **Incremental Reconstruction**: Processes image sequences progressively

---

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

- **Python**: Version 3.8 or higher
- **pip**: Python package installer
- **Git**: For cloning the repository

### System Requirements

- **OS**: Windows, macOS, or Linux
- **RAM**: Minimum 4GB (8GB+ recommended for large datasets)
- **Storage**: ~500MB for dependencies + your image dataset

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

The SfM pipeline follows these sequential stages:

```
Input Images
    ↓
[1] Feature Detection & Matching
    ↓
[2] Fundamental Matrix Estimation (RANSAC)
    ↓
[3] Essential Matrix Computation
    ↓
[4] Camera Pose Recovery
    ↓
[5] Triangulation (Linear + Non-linear)
    ↓
[6] Pose Disambiguation (Cheirality)
    ↓
[7] Incremental Reconstruction (PnP RANSAC)
    ↓
[8] Bundle Adjustment (Global Optimization)
    ↓
3D Reconstruction Output
```

### Stage Details

#### 1. Feature Detection & Matching
- Extracts distinctive keypoints from images
- Computes feature descriptors (SIFT/SURF)
- Matches features across image pairs using ratio test
- Filters outliers based on geometric constraints

#### 2. Fundamental Matrix Estimation
- Uses 8-point algorithm with RANSAC
- Estimates geometric relationship between image pairs
- Achieves robustness against outliers
- Validates using epipolar constraint

#### 3. Essential Matrix & Camera Pose
- Converts Fundamental matrix using camera calibration
- Decomposes Essential matrix into 4 possible poses
- Applies cheirality constraint to select correct pose
- Ensures 3D points are in front of both cameras

#### 4. Triangulation
- Linear triangulation using DLT (Direct Linear Transform)
- Non-linear refinement minimizing reprojection error
- Handles noise and numerical instabilities
- Produces 3D point coordinates

#### 5. Incremental Reconstruction
- Registers new images using PnP (Perspective-n-Point)
- RANSAC for robust camera pose estimation
- Triangulates new 3D points visible in multiple views
- Builds dense reconstruction incrementally

#### 6. Bundle Adjustment
- Jointly optimizes camera poses and 3D structure
- Minimizes reprojection error across all views
- Uses Levenberg-Marquardt algorithm
- Handles large-scale optimization efficiently

---

## 🔍 Technical Details

### Algorithms Implemented

#### Fundamental Matrix Estimation
```
F = [f11 f12 f13]
    [f21 f22 f23]
    [f31 f32 f33]

Constraint: x2^T F x1 = 0
```
- Normalized 8-point algorithm
- RANSAC with adaptive threshold
- Rank-2 constraint enforcement

#### Essential Matrix Decomposition
```
E = K2^T F K1
E = [R|t]
```
Four possible decompositions:
- (R1, t), (R1, -t), (R2, t), (R2, -t)

#### Triangulation Methods

**Linear Triangulation (DLT)**:
```
A X = 0
where X = [X, Y, Z, 1]^T
```

**Non-linear Optimization**:
```
minimize Σ ||x_i - π(K[R|t]X)||^2
```

#### Bundle Adjustment
```
minimize Σ_i Σ_j ||x_ij - π(K_i[R_i|t_i]X_j)||^2
```
- Sparse Levenberg-Marquardt
- Schur complement for efficiency
- Robust cost functions (Huber)

### Mathematical Foundations

- **Epipolar Geometry**: Geometric relationship between two views
- **Multi-View Geometry**: Extension to n cameras
- **Projective Geometry**: Homogeneous coordinates and transformations
- **Non-linear Optimization**: Gauss-Newton and Levenberg-Marquardt
- **RANSAC**: Random Sample Consensus for outlier rejection

---

## 📊 Results

### Performance Metrics

The quality of reconstruction is evaluated using:

- **Reprojection Error**: Mean pixel distance between observed and projected points
- **3D Point Accuracy**: Comparison with ground truth (if available)
- **Camera Pose Error**: Rotation and translation accuracy
- **Reconstruction Completeness**: Percentage of scene reconstructed

### Example Output

```
=== SfM Reconstruction Summary ===
Images processed: 5
3D points reconstructed: 1,247
Mean reprojection error: 0.82 pixels
Bundle adjustment iterations: 23
Final cost: 0.0034
Processing time: 12.4 seconds
```

### Visualization

The pipeline generates:
- **3D Point Cloud**: Matplotlib 3D scatter plot with camera frustums
- **Camera Trajectory**: Top-down view showing camera positions
- **Reprojection Visualization**: Feature matches with epipolar lines
- **Error Distribution**: Histogram of reprojection errors

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

### Common Issues

#### Issue: "Data folder not found"
**Solution**: Ensure you're running `Wrapper.py` from the `Phase1` directory:
```bash
cd Phase1
python Wrapper.py
```

#### Issue: "ModuleNotFoundError: No module named 'cv2'"
**Solution**: Install OpenCV:
```bash
pip install opencv-python==4.10.0
```

#### Issue: "Memory Error during bundle adjustment"
**Solution**: Reduce the number of images or 3D points:
```python
# In Wrapper.py
MAX_POINTS = 5000  # Limit number of 3D points
```

#### Issue: "Poor reconstruction quality"
**Possible causes**:
- Insufficient image overlap
- Poor feature matches
- Incorrect camera calibration
- Large baseline between views

**Solutions**:
- Use more images with gradual viewpoint changes
- Adjust RANSAC threshold
- Verify calibration parameters
- Increase feature detection sensitivity

#### Issue: "RANSAC fails to find inliers"
**Solution**: Adjust RANSAC parameters:
```python
RANSAC_THRESHOLD = 0.1  # Increase threshold
RANSAC_MAX_ITERATIONS = 2000  # More iterations
```

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

This project was developed as part of computer vision coursework, implementing concepts from:

- **Multiple View Geometry in Computer Vision** by Hartley & Zisserman
- **Computer Vision: Algorithms and Applications** by Richard Szeliski
- OpenCV documentation and tutorials
- Structure-from-Motion research papers and implementations

### References

1. Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*
2. Snavely, N., Seitz, S. M., & Szeliski, R. (2006). *Photo Tourism: Exploring Photo Collections in 3D*
3. Triggs, B., et al. (1999). *Bundle Adjustment — A Modern Synthesis*
4. Schönberger, J. L., & Frahm, J.-M. (2016). *Structure-from-Motion Revisited*

### Tools & Libraries

- [OpenCV](https://opencv.org/) - Computer vision library
- [NumPy](https://numpy.org/) - Numerical computing
- [SciPy](https://scipy.org/) - Scientific computing
- [Matplotlib](https://matplotlib.org/) - Visualization

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
