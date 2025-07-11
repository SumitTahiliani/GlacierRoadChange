# Glacier Road Change Analysis

**Solution for Bharatiya Antariksh Hackathon 2025 - Problem Statement 6**

This project addresses [Problem Statement 6](https://vision.hack2skill.com/event/bah2025?psNumber=ps6&scrollY=0) from the Bharatiya Antariksh Hackathon 2025. This is a personal project and not an official participation solution.

**Work in Progress**: The current implementation uses conventional methods for NDWI analysis and a model-based approach for road detection. Ongoing improvements include enhanced data sources and model versions to increase accuracy.

## Overview

A comprehensive analysis tool for detecting and tracking changes in glacial lakes and road networks using satellite imagery. The project combines traditional remote sensing techniques with deep learning approaches to monitor environmental changes in glacial regions.

## Methodology

### Glacial Lake Detection
- **Data Source**: Sentinel-2 satellite imagery via mapminer
- **Method**: Conventional NDWI (Normalized Difference Water Index) analysis
- **Processing**: Morphological operations for noise removal and feature extraction
- **Output**: Vectorized lake polygons with temporal tracking

### Road Detection
- **Data Source**: High-resolution satellite imagery
- **Method**: DeepLabV3-MobileNetV3 deep learning model
- **Training**: Binary classification with custom loss functions (BCE + Dice Loss)
- **Output**: Road network vectorization and change detection

## Project Structure

```
GlacierRoadChange/
├── pipelines/
│   ├── lakes/
│   │   ├── extractLakes.py      # NDWI-based lake extraction
│   │   ├── trackLakes.py        # Temporal lake tracking
│   │   ├── visLakes.py          # Lake visualization
│   │   └── lakePipeline.py      # Complete lake analysis pipeline
│   └── roads/
│       ├── prepRoadData.py      # Data preparation for road detection
│       ├── trainRoads.py        # Deep learning model training
│       ├── vectorizeRoads.py    # Road network vectorization
│       ├── makeTiles.py         # Image tiling for training
│       ├── visRoads.py          # Road visualization
│       └── sanity.py            # Data validation
├── glacial_lakes_output/        # Lake analysis results
├── road_output/                 # Road detection results
└── road_data/                   # Training and validation data
```

## Technical Implementation

### Lake Detection Pipeline
- NDWI thresholding with configurable parameters
- Morphological filtering for noise reduction
- Automated vectorization of water bodies
- Temporal tracking and change quantification

### Road Detection Pipeline
- DeepLabV3-MobileNetV3 architecture
- Custom dataset preparation with tiling
- Binary classification with weighted loss functions
- Post-processing for vector output generation
