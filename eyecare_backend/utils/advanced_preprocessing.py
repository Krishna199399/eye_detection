import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
import cv2
import albumentations as A
from typing import Tuple, List, Optional
import random
from PIL import Image, ImageEnhance, ImageFilter
import os

class AdvancedImagePreprocessor:
    """
    Advanced preprocessing specifically designed for eye disease detection
    """
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Advanced augmentation pipeline
        self.train_transform = A.Compose([
            # Geometric transformations
            A.Rotate(limit=15, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.7
            ),
            
            # Optical distortions (important for eye images)
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
            A.GridDistortion(p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            
            # Color and lighting augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.5
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            # Noise and blur (simulate different image qualities)
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.3),
            
            # Shadows and lighting effects
            A.RandomShadow(p=0.3),
            A.RandomSunFlare(p=0.1),
            
            # Cutout and masking
            A.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=8,
                min_width=8,
                p=0.3
            ),
            
            # Final normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.val_transform = A.Compose([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_eye_image(self, image: np.ndarray, enhance_contrast: bool = True) -> np.ndarray:
        """
        Specialized preprocessing for eye fundus images
        """
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR format from OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        if enhance_contrast:
            try:
                # Apply CLAHE to each channel for better contrast
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
                image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            except Exception as e:
                # If CLAHE fails, just use the original image
                print(f"Warning: CLAHE failed, using original image: {e}")
                pass
            
            # Additional eye-specific enhancements
            image = self.enhance_eye_features(image)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image

    def enhance_eye_features(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance features specific to eye fundus images
        """
        # Convert to PIL for advanced manipulations
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Enhance contrast and sharpness
        contrast_enhancer = ImageEnhance.Contrast(pil_image)
        image_enhanced = contrast_enhancer.enhance(1.2)
        
        sharpness_enhancer = ImageEnhance.Sharpness(image_enhanced)
        image_enhanced = sharpness_enhancer.enhance(1.1)
        
        # Convert back to numpy
        enhanced = np.array(image_enhanced)
        
        # Green channel enhancement (important for retinal features)
        green_enhanced = enhanced.copy()
        green_enhanced[:, :, 1] = np.clip(green_enhanced[:, :, 1] * 1.1, 0, 255)
        
        return green_enhanced

    def create_advanced_data_generator(self, directory: str, batch_size: int = 32,
                                     class_mode: str = 'categorical', 
                                     subset: str = 'training') -> ImageDataGenerator:
        """
        Create advanced data generator with custom preprocessing
        """
        if subset == 'training':
            datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                preprocessing_function=self.preprocess_eye_image,
                validation_split=0.2
            )
        else:
            datagen = ImageDataGenerator(
                rescale=1./255,
                preprocessing_function=lambda x: self.preprocess_eye_image(x, enhance_contrast=True),
                validation_split=0.2
            )
        
        generator = datagen.flow_from_directory(
            directory,
            target_size=self.target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            subset=subset,
            shuffle=(subset == 'training')
        )
        
        return generator

class MixUpGenerator(Sequence):
    """
    Custom generator implementing MixUp augmentation
    """
    
    def __init__(self, generator, alpha: float = 0.2):
        self.generator = generator
        self.alpha = alpha
        
    def __len__(self):
        return len(self.generator)
    
    def __getitem__(self, index):
        # Get a batch from the original generator
        batch_x, batch_y = self.generator[index]
        
        # Apply MixUp
        if np.random.random() < 0.5:  # Apply MixUp 50% of the time
            batch_x, batch_y = self.mixup(batch_x, batch_y)
            
        return batch_x, batch_y
    
    def mixup(self, x, y):
        """Apply MixUp augmentation"""
        batch_size = x.shape[0]
        
        # Generate lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha, batch_size)
        
        # Shuffle indices
        indices = np.random.permutation(batch_size)
        
        # Mix images
        mixed_x = np.zeros_like(x)
        mixed_y = np.zeros_like(y)
        
        for i in range(batch_size):
            mixed_x[i] = lam[i] * x[i] + (1 - lam[i]) * x[indices[i]]
            mixed_y[i] = lam[i] * y[i] + (1 - lam[i]) * y[indices[i]]
            
        return mixed_x, mixed_y

class CutMixGenerator(Sequence):
    """
    Custom generator implementing CutMix augmentation
    """
    
    def __init__(self, generator, alpha: float = 1.0):
        self.generator = generator
        self.alpha = alpha
        
    def __len__(self):
        return len(self.generator)
    
    def __getitem__(self, index):
        batch_x, batch_y = self.generator[index]
        
        # Apply CutMix
        if np.random.random() < 0.5:  # Apply CutMix 50% of the time
            batch_x, batch_y = self.cutmix(batch_x, batch_y)
            
        return batch_x, batch_y
    
    def cutmix(self, x, y):
        """Apply CutMix augmentation"""
        batch_size = x.shape[0]
        indices = np.random.permutation(batch_size)
        
        # Generate lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get image dimensions
        _, h, w, _ = x.shape
        
        # Generate random bounding box
        cut_ratio = np.sqrt(1. - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)
        
        # Random center point
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Get bounding box coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply CutMix
        mixed_x = x.copy()
        mixed_y = y.copy()
        
        for i in range(batch_size):
            mixed_x[i, bby1:bby2, bbx1:bbx2, :] = x[indices[i], bby1:bby2, bbx1:bbx2, :]
            
            # Adjust labels based on the mixed area ratio
            mixed_area = (bbx2 - bbx1) * (bby2 - bby1)
            total_area = h * w
            lam = 1 - (mixed_area / total_area)
            
            mixed_y[i] = lam * y[i] + (1 - lam) * y[indices[i]]
            
        return mixed_x, mixed_y

class ProgressiveResizeGenerator(Sequence):
    """
    Generator that implements progressive resizing
    """
    
    def __init__(self, generator, start_size: int = 128, end_size: int = 256, 
                 epochs_per_size: int = 10, current_epoch: int = 0):
        self.generator = generator
        self.start_size = start_size
        self.end_size = end_size
        self.epochs_per_size = epochs_per_size
        self.current_epoch = current_epoch
        
    def __len__(self):
        return len(self.generator)
    
    def __getitem__(self, index):
        batch_x, batch_y = self.generator[index]
        
        # Calculate current size based on epoch
        progress = min(1.0, self.current_epoch / (self.epochs_per_size * 3))
        current_size = int(self.start_size + (self.end_size - self.start_size) * progress)
        
        # Resize batch if needed
        if current_size != batch_x.shape[1]:
            resized_batch = np.zeros((batch_x.shape[0], current_size, current_size, batch_x.shape[3]))
            for i in range(batch_x.shape[0]):
                resized_batch[i] = cv2.resize(batch_x[i], (current_size, current_size))
            batch_x = resized_batch
            
        return batch_x, batch_y
    
    def update_epoch(self, epoch):
        self.current_epoch = epoch

class EyeDataAugmentationPipeline:
    """
    Complete data augmentation pipeline for eye disease detection
    """
    
    def __init__(self, base_path: str, target_size: Tuple[int, int] = (256, 256)):
        self.base_path = base_path
        self.target_size = target_size
        self.preprocessor = AdvancedImagePreprocessor(target_size)
        
    def create_training_pipeline(self, batch_size: int = 32, 
                               augmentation_type: str = 'standard') -> Sequence:
        """
        Create complete training pipeline with different augmentation strategies
        """
        # Base generator
        train_gen = self.preprocessor.create_advanced_data_generator(
            self.base_path,
            batch_size=batch_size,
            subset='training'
        )
        
        if augmentation_type == 'mixup':
            return MixUpGenerator(train_gen, alpha=0.2)
        elif augmentation_type == 'cutmix':
            return CutMixGenerator(train_gen, alpha=1.0)
        elif augmentation_type == 'progressive':
            return ProgressiveResizeGenerator(
                train_gen,
                start_size=128,
                end_size=self.target_size[0],
                epochs_per_size=10
            )
        else:
            return train_gen
    
    def create_validation_pipeline(self, batch_size: int = 32) -> ImageDataGenerator:
        """
        Create validation pipeline with minimal augmentation
        """
        return self.preprocessor.create_advanced_data_generator(
            self.base_path,
            batch_size=batch_size,
            subset='validation'
        )

# Utility functions for advanced preprocessing
def apply_clahe_to_image(image: np.ndarray, clip_limit: float = 2.0, 
                        tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve contrast
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    if len(image.shape) == 3:
        # Convert to LAB color space for better results
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        enhanced = clahe.apply(image)
    
    return enhanced

def gaussian_noise_augmentation(image: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
    """
    Add Gaussian noise to simulate real-world conditions
    """
    noise = np.random.normal(0, noise_factor, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 1)

def brightness_contrast_augmentation(image: np.ndarray, 
                                   brightness_range: Tuple[float, float] = (0.8, 1.2),
                                   contrast_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """
    Random brightness and contrast adjustment
    """
    brightness = np.random.uniform(*brightness_range)
    contrast = np.random.uniform(*contrast_range)
    
    # Apply brightness and contrast
    adjusted = image * contrast + (brightness - 1)
    return np.clip(adjusted, 0, 1)

def simulate_camera_noise(image: np.ndarray, noise_types: List[str] = ['gaussian', 'salt_pepper']) -> np.ndarray:
    """
    Simulate different types of camera noise
    """
    noisy_image = image.copy()
    
    for noise_type in noise_types:
        if noise_type == 'gaussian' and np.random.random() > 0.5:
            noise = np.random.normal(0, 0.05, image.shape)
            noisy_image = noisy_image + noise
            
        elif noise_type == 'salt_pepper' and np.random.random() > 0.5:
            # Salt and pepper noise
            noise = np.random.random(image.shape)
            noisy_image[noise < 0.01] = 0
            noisy_image[noise > 0.99] = 1
    
    return np.clip(noisy_image, 0, 1)
