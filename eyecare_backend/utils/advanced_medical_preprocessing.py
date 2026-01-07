#!/usr/bin/env python3
"""
Advanced Medical Image Preprocessing Pipeline
Optimized for eye disease detection with domain-specific augmentations
"""

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from typing import Tuple, Optional, List
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MedicalImagePreprocessor:
    """
    Advanced preprocessing pipeline for medical eye images
    """
    
    def __init__(self, target_size: Tuple[int, int] = (384, 384)):
        self.target_size = target_size
        self.normalization_method = 'adaptive'  # 'standard', 'adaptive', or 'clahe'
        
        # Medical-specific augmentation pipeline
        self.medical_augmentation = self._create_medical_augmentation_pipeline()
        
        # Quality enhancement filters
        self.quality_enhancer = self._create_quality_enhancement_pipeline()
        
    def _create_medical_augmentation_pipeline(self):
        """
        Create augmentation pipeline specifically for medical imaging
        """
        return A.Compose([
            # Geometric transformations (conservative for medical images)
            A.OneOf([
                A.Rotate(limit=15, p=0.7),  # Small rotations
                A.Affine(rotate=(-10, 10), p=0.3),
            ], p=0.6),
            
            # Medical-appropriate flips
            A.HorizontalFlip(p=0.5),  # Eyes are symmetric
            
            # Optical distortions (simulate lens effects)
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.3),
                A.GridDistortion(num_steps=3, distort_limit=0.1, p=0.3),
            ], p=0.2),
            
            # Color and contrast adjustments (important for medical imaging)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, 
                    contrast_limit=0.2, 
                    p=0.8
                ),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.6),
                A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            ], p=0.9),
            
            # Color space adjustments
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=10, 
                    sat_shift_limit=15, 
                    val_shift_limit=10, 
                    p=0.6
                ),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.4),
                A.ChannelShuffle(p=0.1),  # Rarely, to add diversity
            ], p=0.5),
            
            # Noise and blur (simulate imaging conditions)
            A.OneOf([
                A.GaussNoise(var_limit=(5, 20), p=0.4),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
            ], p=0.3),
            
            A.OneOf([
                A.Blur(blur_limit=2, p=0.3),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.GaussianBlur(blur_limit=2, p=0.2),
            ], p=0.2),
            
            # Compression artifacts (simulate various image qualities)
            A.ImageCompression(quality_lower=70, quality_upper=100, p=0.2),
            
            # Coarse dropout (simulate occlusions)
            A.CoarseDropout(
                max_holes=3, 
                max_height=32, 
                max_width=32, 
                min_holes=1, 
                min_height=8, 
                min_width=8, 
                fill_value=0, 
                p=0.1
            ),
        ], p=0.8)  # Apply augmentations 80% of the time
    
    def _create_quality_enhancement_pipeline(self):
        """
        Create pipeline for enhancing image quality
        """
        return A.Compose([
            # Noise reduction
            A.OneOf([
                A.MedianBlur(blur_limit=3, p=0.3),
                A.GaussianBlur(blur_limit=(1, 2), p=0.2),
            ], p=0.3),
            
            # Contrast enhancement
            A.OneOf([
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.7),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, 
                    contrast_limit=0.2, 
                    p=0.5
                ),
            ], p=0.8),
            
            # Sharpening
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.3),
        ])
    
    def preprocess_image(self, image: np.ndarray, 
                        apply_augmentation: bool = False,
                        enhance_quality: bool = True) -> np.ndarray:
        """
        Comprehensive preprocessing pipeline for medical images
        
        Args:
            image: Input image as numpy array
            apply_augmentation: Whether to apply data augmentation
            enhance_quality: Whether to apply quality enhancement
            
        Returns:
            Preprocessed image
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB
            processed_image = image.copy()
        else:
            # Convert from BGR to RGB or handle grayscale
            if len(image.shape) == 3:
                processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                processed_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize image
        processed_image = cv2.resize(
            processed_image, 
            self.target_size, 
            interpolation=cv2.INTER_LANCZOS4  # High-quality interpolation
        )
        
        # Quality enhancement
        if enhance_quality:
            processed_image = self.quality_enhancer(image=processed_image)['image']
        
        # Apply medical-specific preprocessing
        processed_image = self._apply_medical_preprocessing(processed_image)
        
        # Data augmentation (training only)
        if apply_augmentation:
            processed_image = self.medical_augmentation(image=processed_image)['image']
        
        # Normalize image
        processed_image = self._normalize_image(processed_image)
        
        return processed_image.astype(np.float32)
    
    def _apply_medical_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """
        Apply medical image-specific preprocessing
        """
        # 1. Green channel enhancement (important for retinal images)
        if len(image.shape) == 3:
            # Enhance green channel as it often contains most medical information
            enhanced_image = image.copy()
            enhanced_image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
            
            # Weighted combination
            image = cv2.addWeighted(image, 0.7, enhanced_image, 0.3, 0)
        
        # 2. Adaptive histogram equalization
        if len(image.shape) == 3:
            # Apply CLAHE to each channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            for i in range(image.shape[2]):
                image[:, :, i] = clahe.apply(image[:, :, i])
        
        # 3. Edge enhancement for medical structures
        if len(image.shape) == 3:
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply unsharp masking
            blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
            unsharp_mask = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
            
            # Convert back to RGB and blend
            unsharp_rgb = cv2.cvtColor(unsharp_mask, cv2.COLOR_GRAY2RGB)
            image = cv2.addWeighted(image, 0.8, unsharp_rgb, 0.2, 0)
        
        return image
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using selected method
        """
        if self.normalization_method == 'standard':
            # Standard normalization (0-1 range)
            return image.astype(np.float32) / 255.0
            
        elif self.normalization_method == 'adaptive':
            # Adaptive normalization based on image statistics
            image_float = image.astype(np.float32)
            
            # Per-channel normalization
            for i in range(image_float.shape[2]):
                channel = image_float[:, :, i]
                mean_val = np.mean(channel)
                std_val = np.std(channel)
                
                # Prevent division by zero
                if std_val > 0:
                    channel = (channel - mean_val) / std_val
                    # Rescale to [0, 1]
                    channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
                else:
                    channel = channel / 255.0
                
                image_float[:, :, i] = channel
            
            return image_float
            
        elif self.normalization_method == 'clahe':
            # CLAHE-based normalization
            normalized = image.copy().astype(np.float32)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            for i in range(normalized.shape[2]):
                normalized[:, :, i] = clahe.apply(normalized[:, :, i].astype(np.uint8))
            
            return normalized / 255.0
        
        else:
            # Default: simple normalization
            return image.astype(np.float32) / 255.0
    
    def create_data_generators(self, train_dir: str, validation_dir: str, test_dir: str,
                             batch_size: int = 16, validation_split: float = 0.2):
        """
        Create advanced data generators with medical preprocessing
        """
        from keras.preprocessing.image import ImageDataGenerator
        
        # Custom preprocessing function
        def medical_preprocessing(img):
            """Custom preprocessing function for ImageDataGenerator"""
            # Convert PIL to numpy
            if isinstance(img, Image.Image):
                img = np.array(img)
            
            # Apply medical preprocessing
            processed = self.preprocess_image(
                img, 
                apply_augmentation=True,  # Training augmentation
                enhance_quality=True
            )
            
            return processed
        
        def medical_preprocessing_val(img):
            """Validation preprocessing (no augmentation)"""
            if isinstance(img, Image.Image):
                img = np.array(img)
            
            processed = self.preprocess_image(
                img, 
                apply_augmentation=False,  # No augmentation for validation
                enhance_quality=True
            )
            
            return processed
        
        # Training generator with augmentation
        train_datagen = ImageDataGenerator(
            preprocessing_function=medical_preprocessing,
            validation_split=validation_split
        )
        
        # Validation generator (no augmentation)
        val_datagen = ImageDataGenerator(
            preprocessing_function=medical_preprocessing_val,
            validation_split=validation_split
        )
        
        # Test generator (no augmentation)
        test_datagen = ImageDataGenerator(
            preprocessing_function=medical_preprocessing_val
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        validation_generator = val_datagen.flow_from_directory(
            train_dir,
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=True,
            seed=42
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, validation_generator, test_generator
    
    def visualize_preprocessing(self, image_path: str, save_path: str = None):
        """
        Visualize the preprocessing pipeline
        """
        # Load original image
        original = cv2.imread(image_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Apply different preprocessing steps
        resized = cv2.resize(original_rgb, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        
        enhanced = self.quality_enhancer(image=resized.copy())['image']
        medical_processed = self._apply_medical_preprocessing(enhanced.copy())
        
        augmented = self.medical_augmentation(image=medical_processed.copy())['image']
        normalized = self._normalize_image(augmented.copy())
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(resized)
        axes[0, 1].set_title('Resized')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(enhanced)
        axes[0, 2].set_title('Quality Enhanced')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(medical_processed)
        axes[1, 0].set_title('Medical Preprocessing')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(augmented)
        axes[1, 1].set_title('Augmented')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(normalized)
        axes[1, 2].set_title('Normalized')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Preprocessing visualization saved to {save_path}")
        else:
            plt.show()
            
        plt.close()


class MedicalImageAugmentor:
    """
    Specialized augmentation for medical images with focus on preserving medical features
    """
    
    def __init__(self):
        self.geometric_augmentations = self._create_geometric_pipeline()
        self.photometric_augmentations = self._create_photometric_pipeline()
        self.medical_augmentations = self._create_medical_specific_pipeline()
    
    def _create_geometric_pipeline(self):
        """Conservative geometric augmentations for medical images"""
        return A.Compose([
            A.Rotate(limit=20, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.6
            ),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.1, 0.1),
                rotate=(-10, 10),
                p=0.4
            )
        ])
    
    def _create_photometric_pipeline(self):
        """Photometric augmentations optimized for medical imaging"""
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.6),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.6
            ),
            A.RGBShift(
                r_shift_limit=20,
                g_shift_limit=20,
                b_shift_limit=20,
                p=0.4
            )
        ])
    
    def _create_medical_specific_pipeline(self):
        """Medical-specific augmentations"""
        return A.Compose([
            # Simulate different lighting conditions
            A.OneOf([
                A.RandomShadow(p=0.3),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    angle_lower=0,
                    angle_upper=1,
                    num_flare_circles_lower=1,
                    num_flare_circles_upper=2,
                    p=0.1
                ),
            ], p=0.2),
            
            # Simulate imaging artifacts
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.4),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=0.3),
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=0.2
                ),
            ], p=0.3),
            
            # Simulate different image qualities
            A.OneOf([
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
                A.Downscale(scale_min=0.7, scale_max=0.9, p=0.2),
            ], p=0.2),
            
            # Add subtle distortions
            A.OneOf([
                A.OpticalDistortion(
                    distort_limit=0.2,
                    shift_limit=0.05,
                    p=0.3
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.2,
                    p=0.2
                ),
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    p=0.1
                ),
            ], p=0.2),
        ])
    
    def apply_augmentation(self, image: np.ndarray, 
                          augmentation_type: str = 'all') -> np.ndarray:
        """
        Apply specific type of augmentation
        
        Args:
            image: Input image
            augmentation_type: 'geometric', 'photometric', 'medical', or 'all'
        
        Returns:
            Augmented image
        """
        if augmentation_type == 'geometric':
            return self.geometric_augmentations(image=image)['image']
        elif augmentation_type == 'photometric':
            return self.photometric_augmentations(image=image)['image']
        elif augmentation_type == 'medical':
            return self.medical_augmentations(image=image)['image']
        else:  # 'all'
            # Apply all augmentations in sequence with some probability
            augmented = image.copy()
            
            if random.random() < 0.7:
                augmented = self.geometric_augmentations(image=augmented)['image']
            
            if random.random() < 0.8:
                augmented = self.photometric_augmentations(image=augmented)['image']
            
            if random.random() < 0.5:
                augmented = self.medical_augmentations(image=augmented)['image']
            
            return augmented


def preprocess_image_for_prediction(image_path: str, 
                                  target_size: Tuple[int, int] = (384, 384)) -> np.ndarray:
    """
    Utility function for preprocessing single images for prediction
    
    Args:
        image_path: Path to image file
        target_size: Target image size
    
    Returns:
        Preprocessed image ready for model prediction
    """
    preprocessor = MedicalImagePreprocessor(target_size=target_size)
    
    # Load image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Assume it's already a numpy array
        image = image_path
    
    # Preprocess
    processed = preprocessor.preprocess_image(
        image, 
        apply_augmentation=False,  # No augmentation for prediction
        enhance_quality=True
    )
    
    return processed


def create_advanced_data_generators(train_dir: str, 
                                  test_dir: str,
                                  target_size: Tuple[int, int] = (384, 384),
                                  batch_size: int = 16,
                                  validation_split: float = 0.2):
    """
    Factory function to create advanced data generators
    
    Args:
        train_dir: Training data directory
        test_dir: Test data directory  
        target_size: Target image size
        batch_size: Batch size
        validation_split: Validation split ratio
    
    Returns:
        Tuple of (train_generator, validation_generator, test_generator)
    """
    preprocessor = MedicalImagePreprocessor(target_size=target_size)
    
    return preprocessor.create_data_generators(
        train_dir=train_dir,
        validation_dir=train_dir,  # Use train_dir for validation split
        test_dir=test_dir,
        batch_size=batch_size,
        validation_split=validation_split
    )


if __name__ == "__main__":
    # Example usage
    preprocessor = MedicalImagePreprocessor(target_size=(384, 384))
    
    # Test preprocessing on a sample image
    test_image_path = "path/to/test/image.jpg"  # Replace with actual path
    
    if False:  # Set to True to run visualization
        try:
            preprocessor.visualize_preprocessing(
                test_image_path, 
                save_path="preprocessing_visualization.png"
            )
        except Exception as e:
            print(f"Could not run visualization: {e}")
    
    print("Advanced medical preprocessing pipeline created successfully!")
    print("Features:")
    print("- Medical-specific augmentations")
    print("- Quality enhancement filters")
    print("- Adaptive normalization")
    print("- Green channel enhancement for retinal images")
    print("- CLAHE for contrast enhancement")
    print("- Edge enhancement for medical structures")
