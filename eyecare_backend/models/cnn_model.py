import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras.applications import EfficientNetB0, ResNet50V2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
from typing import List, Tuple, Optional

class EyeDiseaseModel:
    """
    CNN Model for Eye Disease Detection
    Supports multi-class classification for various eye diseases
    """
    
    def __init__(self, model_path: str = "models/saved_models/eye_disease_model.h5"):
        self.model_path = model_path
        self.model = None
        self.classes = [
            "normal", 
            "cataract", 
            "diabetic_retinopathy", 
            "glaucoma"
        ]
        self.img_height = 224
        self.img_width = 224
        self.channels = 3
        
        # Try to load existing model
        if os.path.exists(model_path):
            self.load_model()

    def create_model_architecture(self, num_classes: int = 4) -> Model:
        """
        Create CNN model architecture using transfer learning with EfficientNetB0
        """
        # Base model with transfer learning
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_height, self.img_width, self.channels)
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dropout(0.3),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_classes, activation='softmax', name='predictions')
        ])
        
        return model

    def create_custom_cnn(self, num_classes: int = 4) -> Model:
        """
        Create custom CNN architecture from scratch
        """
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, self.channels)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.2),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.3),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.3),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.4),
            
            # Classification Head
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ])
        
        return model

    def compile_model(self, learning_rate: float = 0.001):
        """
        Compile the model with appropriate optimizer and loss function
        """
        if self.model is None:
            self.model = self.create_model_architecture()
            
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

    def get_callbacks(self) -> List:
        """
        Get training callbacks for model improvement
        """
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        callbacks = [
            ModelCheckpoint(
                self.model_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks

    def train(self, train_generator, validation_generator, epochs: int = 50):
        """
        Train the model with given data generators
        """
        if self.model is None:
            self.compile_model()
            
        callbacks = self.get_callbacks()
        
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

    def fine_tune(self, train_generator, validation_generator, epochs: int = 20, learning_rate: float = 0.0001):
        """
        Fine-tune the pre-trained model by unfreezing some layers
        """
        if self.model is None:
            raise ValueError("Model must be trained first before fine-tuning")
            
        # Unfreeze the top layers of the base model
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = 100
        
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate/10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        callbacks = self.get_callbacks()
        
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return history

    def predict(self, image: np.ndarray) -> Tuple[str, float, dict]:
        """
        Make prediction on a single image
        
        Args:
            image: Preprocessed image array
            
        Returns:
            Tuple of (predicted_class, confidence, all_predictions)
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please train or load a model first.")
            
        # Make prediction
        predictions = self.model.predict(np.expand_dims(image, axis=0))
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = self.classes[predicted_class_idx]
        
        # Create predictions dictionary
        predictions_dict = {
            self.classes[i]: float(predictions[0][i]) 
            for i in range(len(self.classes))
        }
        
        return predicted_class, confidence, predictions_dict

    def save_model(self, path: str = None):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        save_path = path or self.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, path: str = None):
        """
        Load a trained model
        """
        load_path = path or self.model_path
        
        if os.path.exists(load_path):
            self.model = tf.keras.models.load_model(load_path)
            print(f"Model loaded from {load_path}")
            return True
        else:
            print(f"Model file not found at {load_path}")
            return False

    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded and ready for predictions
        """
        return self.model is not None

    def get_model_summary(self):
        """
        Get model architecture summary
        """
        if self.model is None:
            return "Model not loaded"
        
        return self.model.summary()

    def evaluate(self, test_generator):
        """
        Evaluate model performance on test data
        """
        if self.model is None:
            raise ValueError("Model is not loaded")
            
        return self.model.evaluate(test_generator, verbose=1)
