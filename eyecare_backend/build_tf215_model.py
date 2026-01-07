#!/usr/bin/env python3
"""
EyeCare AI ML Model Builder using TensorFlow 2.15
Compatible with your existing TensorFlow installation
Uses your 4,217 eye disease images for training
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set environment variables for TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("🔥 EyeCare AI - ML Model Builder (TensorFlow 2.15)")
print("=" * 60)

try:
    # Import TensorFlow 2.15
    import tensorflow as tf
    print(f"✅ TensorFlow version: {tf.__version__}")
    
    # Set memory growth to prevent GPU issues
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Found {len(gpus)} GPU(s)")
        else:
            print("✅ Using CPU (recommended for compatibility)")
    except:
        print("✅ GPU config skipped - using CPU")
    
    # Import required modules with TensorFlow 2.15 syntax
    from tensorflow import keras
    from keras import layers
    from keras.models import Sequential
    from keras.layers import (
        Conv2D, MaxPooling2D, Dense, Dropout, Flatten, 
        BatchNormalization, GlobalAveragePooling2D
    )
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from keras.preprocessing.image import ImageDataGenerator
    
    # Import additional libraries
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    print("✅ All imports successful!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you have TensorFlow 2.15 installed")
    sys.exit(1)

class EyeCareAIModel:
    """Complete ML model for eye disease detection using TensorFlow 2.15"""
    
    def __init__(self):
        self.model = None
        self.classes = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
        self.img_size = 224
        self.model_path = "models/saved_models/eye_disease_model.h5"
        
        # Create directories
        os.makedirs("models/saved_models", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
        
        # Dataset paths
        self.data_dir = "D:/projects/eyecare_ai/datasets/eyecare_ai_data"
        self.train_dir = os.path.join(self.data_dir, "train")
        self.test_dir = os.path.join(self.data_dir, "test")
        
        print(f"📁 Dataset location: {self.data_dir}")
        
    def verify_dataset(self):
        """Verify dataset structure and count images"""
        print("\n📊 Verifying dataset structure...")
        
        if not os.path.exists(self.train_dir):
            raise ValueError(f"Training directory not found: {self.train_dir}")
        if not os.path.exists(self.test_dir):
            raise ValueError(f"Test directory not found: {self.test_dir}")
        
        train_counts = {}
        test_counts = {}
        
        for class_name in self.classes:
            # Count training images
            train_class_dir = os.path.join(self.train_dir, class_name)
            if os.path.exists(train_class_dir):
                train_files = [f for f in os.listdir(train_class_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                train_counts[class_name] = len(train_files)
            else:
                train_counts[class_name] = 0
                
            # Count test images
            test_class_dir = os.path.join(self.test_dir, class_name)
            if os.path.exists(test_class_dir):
                test_files = [f for f in os.listdir(test_class_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                test_counts[class_name] = len(test_files)
            else:
                test_counts[class_name] = 0
        
        print("\n📈 Dataset Statistics:")
        print("-" * 60)
        total_train = sum(train_counts.values())
        total_test = sum(test_counts.values())
        
        for class_name in self.classes:
            train_count = train_counts[class_name]
            test_count = test_counts[class_name]
            print(f"{class_name:20} | Train: {train_count:4d} | Test: {test_count:3d}")
        
        print("-" * 60)
        print(f"{'TOTAL':20} | Train: {total_train:4d} | Test: {total_test:3d}")
        print(f"📊 Total Dataset: {total_train + total_test} images")
        
        return train_counts, test_counts
    
    def create_model(self):
        """Create CNN model architecture optimized for eye disease detection"""
        print("\n🏗️ Building CNN model architecture...")
        
        model = Sequential([
            # Input layer
            layers.Input(shape=(self.img_size, self.img_size, 3)),
            
            # First Conv Block
            Conv2D(32, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Second Conv Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Third Conv Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Fourth Conv Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Global pooling and classification
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(len(self.classes), activation='softmax', name='predictions')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model architecture created")
        print(f"📊 Total parameters: {model.count_params():,}")
        
        # Display model summary
        model.summary()
        
        self.model = model
        return model
    
    def create_data_generators(self):
        """Create data generators with augmentation"""
        print("\n🔄 Creating data generators...")
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.15,
            brightness_range=[0.8, 1.2],
            shear_range=0.1,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Test data generator (no augmentation)
        test_datagen = ImageDataGenerator(rescale=1.0/255.0)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=16,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        val_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=16,
            class_mode='categorical',
            subset='validation',
            shuffle=True,
            seed=42
        )
        
        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=16,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"✅ Training samples: {train_generator.samples}")
        print(f"✅ Validation samples: {val_generator.samples}")
        print(f"✅ Test samples: {test_generator.samples}")
        print(f"✅ Classes detected: {list(train_generator.class_indices.keys())}")
        
        return train_generator, val_generator, test_generator
    
    def train_model(self, train_gen, val_gen, epochs=30):
        """Train the CNN model"""
        print(f"\n🚀 Starting training for {epochs} epochs...")
        
        if self.model is None:
            self.create_model()
        
        # Enhanced callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                self.model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                save_format='h5'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        print("📊 Starting training process...")
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return history
    
    def evaluate_model(self, test_gen):
        """Evaluate model on test data"""
        print("\n📊 Evaluating model on test data...")
        
        # Load best model
        self.model = keras.models.load_model(self.model_path)
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(test_gen, verbose=1)
        
        # Get predictions for detailed metrics
        test_gen.reset()
        predictions = self.model.predict(test_gen, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_gen.classes
        class_names = list(test_gen.class_indices.keys())
        
        # Classification report
        report = classification_report(
            true_classes, predicted_classes,
            target_names=class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Save confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('EyeCare AI - Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.tight_layout()
        plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Results
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        print(f"\n🎯 Final Results:")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Per-class accuracy
        print("\n📊 Per-Class Accuracy:")
        for i, class_name in enumerate(class_names):
            class_mask = true_classes == i
            if np.sum(class_mask) > 0:
                class_acc = np.sum(predicted_classes[class_mask] == i) / np.sum(class_mask)
                print(f"  {class_name:20}: {class_acc:.4f} ({class_acc*100:.2f}%)")
        
        return results
    
    def plot_training_history(self, history):
        """Plot and save training history"""
        print("\n📈 Creating training plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('EyeCare AI Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Training plots saved to plots/")
    
    def save_results(self, results, train_counts, test_counts):
        """Save training results to JSON"""
        print("\n💾 Saving results...")
        
        training_summary = {
            'timestamp': datetime.now().isoformat(),
            'tensorflow_version': tf.__version__,
            'model_path': self.model_path,
            'dataset_info': {
                'total_images': sum(train_counts.values()) + sum(test_counts.values()),
                'train_images': sum(train_counts.values()),
                'test_images': sum(test_counts.values()),
                'classes': self.classes,
                'train_distribution': train_counts,
                'test_distribution': test_counts
            },
            'model_config': {
                'input_size': f"{self.img_size}x{self.img_size}x3",
                'architecture': 'Custom CNN with 4 Conv blocks',
                'optimizer': 'Adam',
                'loss': 'categorical_crossentropy'
            },
            'results': results
        }
        
        with open('training_results.json', 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        print("✅ Results saved to training_results.json")

def main():
    """Main training pipeline"""
    print("🧠 Starting EyeCare AI ML Model Training")
    print("Using TensorFlow 2.15 with your 4,217 eye disease images")
    print("=" * 60)
    
    # Initialize model
    eyecare_model = EyeCareAIModel()
    
    try:
        # Verify dataset
        train_counts, test_counts = eyecare_model.verify_dataset()
        
        # Create data generators
        train_gen, val_gen, test_gen = eyecare_model.create_data_generators()
        
        # Create and train model
        eyecare_model.create_model()
        history = eyecare_model.train_model(train_gen, val_gen, epochs=30)
        
        # Plot training history
        eyecare_model.plot_training_history(history)
        
        # Evaluate model
        results = eyecare_model.evaluate_model(test_gen)
        
        # Save results
        eyecare_model.save_results(results, train_counts, test_counts)
        
        print("\n🎉 Training Pipeline Completed Successfully!")
        print("=" * 60)
        print(f"📁 Model saved: {eyecare_model.model_path}")
        print("📊 Results saved: training_results.json")
        print("📈 Plots saved: plots/")
        print("\n🚀 Your AI model is ready!")
        print("Next: python tf215_backend.py to start backend with real ML!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ ML model built successfully with TensorFlow 2.15!")
        print("Start backend: python tf215_backend.py")
    else:
        print("\n❌ ML model building failed")
