import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
from typing import Tuple, Optional, Union
import io
import base64

class ImagePreprocessor:
    """
    Image preprocessing utilities for eye disease detection
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        
    def preprocess_for_prediction(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image for model prediction
        
        Args:
            image: Input image as numpy array or PIL Image
            
        Returns:
            Preprocessed image array ready for model prediction
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Convert to RGB if needed (in case of BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def enhance_retinal_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply specific enhancements for retinal images
        
        Args:
            image: Input retinal image
            
        Returns:
            Enhanced image
        """
        # Convert to PIL for enhancements
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced_image = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(enhanced_image)
        enhanced_image = enhancer.enhance(1.1)
        
        # Convert back to numpy
        enhanced_array = np.array(enhanced_image).astype(np.float32) / 255.0
        
        return enhanced_array
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        Particularly useful for retinal images
        
        Args:
            image: Input image
            
        Returns:
            CLAHE enhanced image
        """
        # Convert to Lab color space
        lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced.astype(np.float32) / 255.0
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from the image using Gaussian blur
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        # Apply Gaussian blur for noise reduction
        denoised = cv2.GaussianBlur((image * 255).astype(np.uint8), (3, 3), 0)
        
        return denoised.astype(np.float32) / 255.0
    
    def crop_circular_roi(self, image: np.ndarray, center: Optional[Tuple[int, int]] = None, 
                         radius: Optional[int] = None) -> np.ndarray:
        """
        Crop circular region of interest (useful for fundus images)
        
        Args:
            image: Input image
            center: Center point (x, y). If None, uses image center
            radius: Radius of the circle. If None, uses min(width, height)/2
            
        Returns:
            Cropped circular ROI
        """
        h, w = image.shape[:2]
        
        if center is None:
            center = (w // 2, h // 2)
        if radius is None:
            radius = min(w, h) // 2
            
        # Create circular mask
        y, x = np.ogrid[:h, :w]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        # Apply mask
        result = image.copy()
        result[~mask] = 0
        
        return result
    
    def augment_image(self, image: np.ndarray, augment_type: str = "random") -> np.ndarray:
        """
        Apply data augmentation techniques
        
        Args:
            image: Input image
            augment_type: Type of augmentation ("rotation", "flip", "brightness", "random")
            
        Returns:
            Augmented image
        """
        augmented = image.copy()
        
        if augment_type == "rotation" or augment_type == "random":
            # Random rotation (-15 to 15 degrees)
            angle = np.random.uniform(-15, 15)
            center = (image.shape[1] // 2, image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(
                (augmented * 255).astype(np.uint8), 
                rotation_matrix, 
                (image.shape[1], image.shape[0])
            ).astype(np.float32) / 255.0
        
        if augment_type == "flip" or augment_type == "random":
            # Random horizontal flip
            if np.random.random() > 0.5:
                augmented = cv2.flip(augmented, 1)
        
        if augment_type == "brightness" or augment_type == "random":
            # Random brightness adjustment
            brightness_factor = np.random.uniform(0.8, 1.2)
            augmented = np.clip(augmented * brightness_factor, 0, 1)
        
        return augmented
    
    def preprocess_batch(self, images: list) -> np.ndarray:
        """
        Preprocess a batch of images
        
        Args:
            images: List of images
            
        Returns:
            Batch of preprocessed images
        """
        processed_batch = []
        
        for image in images:
            processed = self.preprocess_for_prediction(image)
            processed = self.apply_clahe(processed)
            processed = self.enhance_retinal_image(processed)
            processed_batch.append(processed)
        
        return np.array(processed_batch)
    
    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
        """
        Load image from bytes (for API uploads)
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Image as numpy array
        """
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        return np.array(image)
    
    @staticmethod
    def load_image_from_base64(base64_string: str) -> np.ndarray:
        """
        Load image from base64 string
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            Image as numpy array
        """
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        
        return ImagePreprocessor.load_image_from_bytes(image_bytes)
    
    @staticmethod
    def save_processed_image(image: np.ndarray, path: str):
        """
        Save processed image to file
        
        Args:
            image: Processed image array
            path: Output file path
        """
        # Convert back to 0-255 range
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Save using PIL
        pil_image = Image.fromarray(image_uint8)
        pil_image.save(path)
    
    def get_image_stats(self, image: np.ndarray) -> dict:
        """
        Get statistical information about the image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with image statistics
        """
        return {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "min_value": float(np.min(image)),
            "max_value": float(np.max(image)),
            "mean": float(np.mean(image)),
            "std": float(np.std(image)),
            "unique_values": len(np.unique(image))
        }

# Convenience function for backend compatibility
def preprocess_image(pil_image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Simple preprocessing function for backend use
    
    Args:
        pil_image: PIL Image object
        target_size: Target size for resizing (width, height)
        
    Returns:
        Preprocessed image array ready for model prediction
    """
    preprocessor = ImagePreprocessor(target_size=target_size)
    return preprocessor.preprocess_for_prediction(pil_image)
