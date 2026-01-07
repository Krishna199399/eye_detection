# EyeCare AI - Dataset Description

## 📊 Overview

This document provides a comprehensive description of the dataset used in the EyeCare AI project for training and evaluating deep learning models for eye disease detection.

## 🗂️ Dataset Structure

The dataset is organized in a hierarchical directory structure following best practices for image classification tasks:

```
datasets/eyecare_ai_data/
├── train/                          # Training dataset (80%)
│   ├── cataract/                   # Cataract images
│   ├── diabetic_retinopathy/       # Diabetic retinopathy images
│   ├── glaucoma/                   # Glaucoma images
│   └── normal/                     # Normal/healthy eye images
│
└── test/                           # Test dataset (20%)
    ├── cataract/                   # Cataract test images
    ├── diabetic_retinopathy/       # Diabetic retinopathy test images
    ├── glaucoma/                   # Glaucoma test images
    └── normal/                     # Normal/healthy test images
```

## 📈 Dataset Statistics

### Overall Statistics

| Metric | Value |
|--------|-------|
| **Total Images** | 4,217 |
| **Training Images** | 3,372 (80%) |
| **Test Images** | 845 (20%) |
| **Number of Classes** | 4 |
| **Image Formats** | JPG, JPEG, PNG |

### Class Distribution

#### Training Set

| Class | Images | Percentage |
|-------|--------|------------|
| Diabetic Retinopathy | 878 | 26.0% |
| Normal | 859 | 25.5% |
| Cataract | 830 | 24.6% |
| Glaucoma | 805 | 23.9% |
| **Total** | **3,372** | **100%** |

#### Test Set

| Class | Images | Percentage |
|-------|--------|------------|
| Diabetic Retinopathy | 220 | 26.0% |
| Normal | 215 | 25.5% |
| Cataract | 208 | 24.6% |
| Glaucoma | 202 | 23.9% |
| **Total** | **845** | **100%** |

#### Combined Statistics

| Class | Training | Test | Total | Distribution |
|-------|----------|------|-------|--------------|
| **Diabetic Retinopathy** | 878 | 220 | 1,098 | 26.0% |
| **Normal** | 859 | 215 | 1,074 | 25.5% |
| **Cataract** | 830 | 208 | 1,038 | 24.6% |
| **Glaucoma** | 805 | 202 | 1,007 | 23.9% |
| **Total** | **3,372** | **845** | **4,217** | **100%** |

## 🏥 Disease Classes Description

### 1. Normal (Healthy Eyes)
- **Count**: 1,074 images (859 train, 215 test)
- **Description**: Retinal fundus images of healthy eyes with no signs of disease
- **Characteristics**: 
  - Clear optic disc
  - Normal blood vessels
  - No hemorrhages or exudates
  - Healthy macula appearance

### 2. Diabetic Retinopathy
- **Count**: 1,098 images (878 train, 220 test)
- **Description**: Diabetes-related damage to blood vessels in the retina
- **Characteristics**:
  - Microaneurysms
  - Hemorrhages
  - Exudates (hard and soft)
  - Neovascularization (in advanced stages)
  - Macular edema

### 3. Cataract
- **Count**: 1,038 images (830 train, 208 test)
- **Description**: Clouding of the eye's natural lens
- **Characteristics**:
  - Lens opacity
  - Reduced visibility of retinal structures
  - Whitish or cloudy appearance
  - Varying degrees of opacity

### 4. Glaucoma
- **Count**: 1,007 images (805 train, 202 test)
- **Description**: Optic nerve damage often associated with increased intraocular pressure
- **Characteristics**:
  - Enlarged cup-to-disc ratio
  - Optic disc cupping
  - Rim thinning
  - Nerve fiber layer defects

## 📸 Image Properties

### File Formats
- **JPEG (.jpg, .jpeg)**: Most common format
- **PNG (.png)**: Used for higher quality images

### File Sizes
- **Range**: 3 KB to 3.4 MB
- **Typical JPEG**: 35-120 KB
- **Large PNG files**: 1-3.4 MB (high-resolution class exemplars)
- **Small thumbnails**: 3-12 KB

### Image Specifications
- **Type**: Retinal fundus photographs
- **Color**: RGB (3 channels)
- **Resolution**: Variable (preprocessed to 224×224 for model input)
- **Orientation**: Mixed (left and right eye images)

### Naming Conventions
- **Cataract**: `cataract_0XX.png`
- **Diabetic Retinopathy**: `XXXXX_left.jpeg`, `XXXXX_right.jpeg`
- **Glaucoma**: `Glaucoma_0XX.png`
- **Normal**: `XXXX_left.jpg`, `XXXX_right.jpg`

## 🔍 Data Quality Considerations

### Balance
- **Class Balance**: Well-balanced dataset with each class representing 23.9-26.0% of total data
- **Train-Test Split**: Consistent 80-20 split across all classes
- **Stratification**: Proportional representation maintained in both training and test sets

### Diversity
- Multiple imaging conditions
- Various disease severities
- Both left and right eye images
- Different image qualities and resolutions

### Challenges
- Variable image quality
- Different lighting conditions
- Presence of artifacts in some images
- Overlapping visual features between some conditions

## 🧠 Model Training Information

### Preprocessing Pipeline
1. **Image Loading**: Support for JPG, JPEG, PNG formats
2. **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization
3. **Noise Reduction**: Advanced filtering techniques
4. **Resizing**: Standardized to 224×224 pixels
5. **Normalization**: Pixel values scaled to [0, 1]

### Data Augmentation (Training Only)
- Random rotation
- Horizontal/vertical flips
- Brightness adjustment
- Zoom variations
- Shift transformations

### Model Architecture
- **Base Model**: EfficientNetB0 (Transfer Learning)
- **Input Shape**: 224×224×3
- **Framework**: TensorFlow 2.15.0
- **Output Classes**: 4 (softmax activation)

## 📊 Current Model Performance

### Overall Metrics
- **Overall Accuracy**: 67.46%
- **Training Framework**: TensorFlow/Keras
- **Model Type**: Convolutional Neural Network (CNN)

### Per-Class Performance

| Class | Accuracy | Notes |
|-------|----------|-------|
| **Normal** | 91.63% | Excellent performance |
| **Diabetic Retinopathy** | 80.91% | Good performance |
| **Cataract** | 54.81% | Moderate performance |
| **Glaucoma** | 40.10% | Needs improvement |

### Performance Analysis
- **Strengths**: 
  - Excellent at identifying normal eyes
  - Strong diabetic retinopathy detection
- **Challenges**:
  - Glaucoma detection needs improvement
  - Cataract classification requires optimization
  - Potential class confusion between certain conditions

## 🎯 Use Cases

### Clinical Applications
1. **Screening**: Early detection of eye diseases
2. **Triage**: Priority assignment for patient referrals
3. **Monitoring**: Track disease progression over time
4. **Education**: Training tool for medical students

### Research Applications
1. Model benchmarking
2. Algorithm development
3. Transfer learning experiments
4. Medical AI research

## ⚠️ Important Notes

### Medical Disclaimer
- **Not for Clinical Use**: This is a research/educational tool
- **Professional Consultation Required**: Always consult ophthalmologists for diagnosis
- **Screening Tool Only**: Results should be verified by medical professionals

### Dataset Limitations
- Limited to 4 disease classes
- May not represent all disease variations
- Variable image quality
- Geographic and demographic diversity may be limited

### Ethical Considerations
- Patient privacy should be maintained
- Results are probabilistic, not definitive
- Should be used as a decision support tool only
- Regular model retraining and validation recommended

## 🔄 Future Improvements

### Dataset Enhancement
- [ ] Increase dataset size for underperforming classes
- [ ] Add more disease categories (AMD, hypertensive retinopathy, etc.)
- [ ] Include disease severity levels
- [ ] Improve image quality standardization

### Model Improvements
- [ ] Implement ensemble methods
- [ ] Explore advanced architectures
- [ ] Fine-tune for glaucoma and cataract detection
- [ ] Add explainability features (Grad-CAM)

### Quality Assurance
- [ ] Expert validation of images
- [ ] Remove low-quality or mislabeled images
- [ ] Establish data collection protocols
- [ ] Regular dataset audits

## 📚 References

### Dataset Sources
- Retinal fundus image databases
- Publicly available ophthalmology datasets
- Clinical imaging repositories

### Related Resources
- [American Academy of Ophthalmology](https://www.aao.org/)
- [National Eye Institute](https://www.nei.nih.gov/)
- [Kaggle Eye Disease Datasets](https://www.kaggle.com/)

## 📧 Contact & Support

For questions about the dataset or project:
- Review the main [README.md](README.md)
- Check the [backend documentation](eyecare_backend/README.md)
- Check the [frontend documentation](eyecare_frontend/README.md)

---

**Last Updated**: November 2025  
**Dataset Version**: 1.0  
**Total Images**: 4,217 (3,372 train, 845 test)
