# RGB Channel Alignment with OpenCV

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project demonstrates how to align misaligned color channels from historical images or vertically stacked grayscale frames using feature detection and homography in OpenCV.

> Inspired by the OpenCV CV1 course and classic image alignment techniques.

---

## Overview

Some old color photographs store RGB channels as vertically stacked grayscale images. When directly merged, this causes color misalignment and ghosting. This script:

- Extracts individual color channels
- Detects and matches visual features between them
- Computes transformations (homographies)
- Aligns and merges the corrected color image

---

## Features

- ORB feature detection for fast performance
- Feature matching with Hamming distance
- RANSAC-based homography estimation
- Warping and channel alignment
- Visualization of keypoints and matches

---

## Input

Input image: `emir.jpg`  
This should be a grayscale image with RGB channels stacked vertically (height = 3 × channel height).
---

## How It Works

1. **Read and split image** into three channels
2. **Detect ORB features** in each channel
3. **Match Blue↔Green** and **Red↔Green** features
4. **Sort matches**, keep top `GOOD_MATCH_PERCENT`
5. **Estimate homography** using RANSAC
6. **Warp Blue and Red** channels to align with Green
7. **Merge channels** to produce the final aligned RGB image

---

## Setup

### Requirements

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib

### Installation

```bash
git clone https://github.com/your-username/rgb-channel-alignment.git
cd rgb-channel-alignment
pip install -r requirements.txt

