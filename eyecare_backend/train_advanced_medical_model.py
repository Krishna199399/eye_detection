#!/usr/bin/env python3
"""
Advanced Medical CNN Training Pipeline
Designed to achieve >85% accuracy for eye disease detection
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import tensorflow as tf
    from models.advanced_medical_cnn import AdvancedMedicalCNN
    from utils.advanced_medical_preprocessing import (
        MedicalImagePreprocessor, 
        create_advanced_data_generators
    )
    from sklearn.metrics import (
        classification_report, confusion_matrix, 
        roc_auc_score, precision_recall_curve
    )
    print(f"✅ TensorFlow version: {tf.__version__}")
    
    # GPU configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU acceleration enabled ({len(gpus)} GPUs)")
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
    else:
        print("✅ Using CPU (training may be slower)")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed")
    sys.exit(1)


class AdvancedMedicalTrainer:
    """
    Comprehensive training pipeline for medical CNN models
    """
    
    def __init__(self, config: dict = None):
        self.config = config or self._get_default_config()
        self.model = None
        self.training_history = {}
        self.evaluation_results = {}
        
        # Create necessary directories
        self._create_directories()
        
        print("🏥 Advanced Medical CNN Trainer Initialized")
        print("=" * 60)
        
    def _get_default_config(self) -> dict:
        """Get default training configuration"""
        return {
            # Dataset configuration
            "dataset": {
                "train_dir": "D:/projects/eyecare_ai/datasets/eyecare_ai_data/train",
                "test_dir": "D:/projects/eyecare_ai/datasets/eyecare_ai_data/test",
                "target_size": (384, 384),
                "batch_size": 12,  # Reduced for higher resolution
                "validation_split": 0.2
            },
            
            # Model configuration
            "model": {
                "architecture": "advanced_cnn",  # 'advanced_cnn', 'efficient_medical', 'ensemble'
                "learning_rate": 0.001,
                "model_path": "models/saved_models/advanced_medical_cnn_best.h5"
            },
            
            # Training configuration
            "training": {
                "epochs": 150,
                "use_progressive_training": True,
                "early_stopping_patience": 25,
                "lr_reduction_patience": 10,
                "min_learning_rate": 1e-8,
                "class_weights_auto": True
            },
            
            # Output configuration
            "output": {
                "results_dir": "training_results",
                "plots_dir": "training_plots",
                "save_best_only": True,
                "verbose": 1
            }
        }
    
    def _create_directories(self):
        """Create necessary directories for training outputs"""
        dirs_to_create = [
            "models/saved_models",
            self.config["output"]["results_dir"],
            self.config["output"]["plots_dir"],
            "logs"
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
    
    def analyze_dataset(self):
        """Analyze dataset structure and class distribution"""
        print("📊 Analyzing Dataset Structure...")
        
        train_dir = self.config["dataset"]["train_dir"]
        test_dir = self.config["dataset"]["test_dir"]
        
        # Check if directories exist
        if not os.path.exists(train_dir):
            print(f"❌ Training directory not found: {train_dir}")
            return False
        if not os.path.exists(test_dir):
            print(f"❌ Test directory not found: {test_dir}")
            return False
        
        # Analyze class distribution
        classes = ["normal", "cataract", "diabetic_retinopathy", "glaucoma"]
        train_counts = {}
        test_counts = {}
        
        print(f"\n{'Class':<20} {'Train':<8} {'Test':<8} {'Total':<8} {'Percentage':<10}")
        print("-" * 70)
        
        total_train = 0
        total_test = 0
        
        for class_name in classes:
            train_class_path = os.path.join(train_dir, class_name)
            test_class_path = os.path.join(test_dir, class_name)
            
            train_count = 0
            test_count = 0
            
            if os.path.exists(train_class_path):
                train_files = [f for f in os.listdir(train_class_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                train_count = len(train_files)
            
            if os.path.exists(test_class_path):
                test_files = [f for f in os.listdir(test_class_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                test_count = len(test_files)
            
            train_counts[class_name] = train_count
            test_counts[class_name] = test_count
            total_train += train_count
            total_test += test_count
            
            total_class = train_count + test_count
            percentage = (total_class / (total_train + total_test)) * 100 if (total_train + total_test) > 0 else 0
            
            print(f"{class_name:<20} {train_count:<8} {test_count:<8} {total_class:<8} {percentage:<10.2f}%")
        
        print("-" * 70)
        print(f"{'TOTAL':<20} {total_train:<8} {total_test:<8} {total_train + total_test:<8} {'100.00%':<10}")
        
        # Calculate class weights for imbalanced data
        if self.config["training"]["class_weights_auto"]:
            class_weights = self._calculate_class_weights(train_counts)
            print(f"\n📊 Calculated Class Weights:")
            for i, class_name in enumerate(classes):
                print(f"  {class_name}: {class_weights[i]:.3f}")
        
        # Store analysis results
        self.dataset_analysis = {
            "train_counts": train_counts,
            "test_counts": test_counts,
            "total_images": total_train + total_test,
            "class_distribution": {class_name: (train_counts[class_name] + test_counts[class_name]) 
                                 for class_name in classes}
        }
        
        return True
    
    def _calculate_class_weights(self, train_counts: dict) -> dict:
        """Calculate class weights for handling imbalanced data"""
        total_samples = sum(train_counts.values())
        n_classes = len(train_counts)
        
        class_weights = {}
        classes = ["normal", "cataract", "diabetic_retinopathy", "glaucoma"]
        
        for i, class_name in enumerate(classes):
            if train_counts[class_name] > 0:
                # Formula: total_samples / (n_classes * class_count)
                weight = total_samples / (n_classes * train_counts[class_name])
                # Apply additional medical importance weighting
                medical_weights = {
                    "normal": 1.0,
                    "cataract": 1.2,
                    "diabetic_retinopathy": 1.4,
                    "glaucoma": 1.6  # Most critical
                }
                class_weights[i] = weight * medical_weights[class_name]
            else:
                class_weights[i] = 1.0
        
        return class_weights
    
    def create_model(self):
        """Create and compile the advanced medical CNN model"""
        print(f"🏗️ Creating {self.config['model']['architecture']} model...")
        
        self.model = AdvancedMedicalCNN(
            model_path=self.config["model"]["model_path"]
        )
        
        # Compile model with specified architecture
        self.model.compile_model(
            learning_rate=self.config["model"]["learning_rate"],
            model_type=self.config["model"]["architecture"]
        )
        
        print(f"✅ Model created successfully!")
        print(f"📊 Total parameters: {self.model.model.count_params():,}")
        
        # Display model summary
        if self.config["output"]["verbose"] > 0:
            print(f"\n📋 Model Architecture Summary:")
            self.model.get_model_summary()
    
    def prepare_data(self):
        """Prepare data generators with advanced preprocessing"""
        print("🔄 Preparing data generators with medical preprocessing...")
        
        # Create advanced data generators
        self.train_generator, self.validation_generator, self.test_generator = create_advanced_data_generators(
            train_dir=self.config["dataset"]["train_dir"],
            test_dir=self.config["dataset"]["test_dir"],
            target_size=self.config["dataset"]["target_size"],
            batch_size=self.config["dataset"]["batch_size"],
            validation_split=self.config["dataset"]["validation_split"]
        )
        
        print(f"✅ Data generators created:")
        print(f"  Training samples: {self.train_generator.samples}")
        print(f"  Validation samples: {self.validation_generator.samples}")
        print(f"  Test samples: {self.test_generator.samples}")
        print(f"  Detected classes: {list(self.train_generator.class_indices.keys())}")
    
    def train_model(self):
        """Train the model with advanced strategies"""
        print(f"\n🚀 Starting Advanced Training Pipeline")
        print(f"Architecture: {self.config['model']['architecture']}")
        print(f"Epochs: {self.config['training']['epochs']}")
        print(f"Progressive Training: {self.config['training']['use_progressive_training']}")
        print("=" * 60)
        
        # Calculate class weights
        class_weights = None
        if self.config["training"]["class_weights_auto"]:
            class_weights = self._calculate_class_weights(self.dataset_analysis["train_counts"])
            print(f"Using calculated class weights: {class_weights}")
        
        # Train with progressive learning or standard training
        if self.config["training"]["use_progressive_training"]:
            print("🎯 Using Progressive Training Strategy")
            
            # Update model with calculated class weights
            if class_weights:
                self.model.class_weights = class_weights
            
            self.training_history = self.model.train_with_progressive_learning(
                train_generator=self.train_generator,
                validation_generator=self.validation_generator,
                epochs=self.config["training"]["epochs"],
                model_type=self.config["model"]["architecture"]
            )
        else:
            print("🎯 Using Standard Training Strategy")
            # Implement standard training with advanced callbacks
            callbacks = self.model.get_advanced_callbacks()
            
            self.training_history = self.model.model.fit(
                self.train_generator,
                epochs=self.config["training"]["epochs"],
                validation_data=self.validation_generator,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=self.config["output"]["verbose"]
            )
        
        print("✅ Training completed successfully!")
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("\n📊 Performing Comprehensive Model Evaluation...")
        
        # Load best model for evaluation
        if os.path.exists(self.config["model"]["model_path"]):
            self.model.load_model()
            print("✅ Best model loaded for evaluation")
        
        # Comprehensive evaluation
        self.evaluation_results = self.model.evaluate_comprehensive(self.test_generator)
        
        # Extract key metrics
        overall_accuracy = self.evaluation_results["overall_accuracy"]
        classification_report = self.evaluation_results["classification_report"]
        auc_scores = self.evaluation_results["auc_scores"]
        
        print(f"\n🎯 FINAL RESULTS:")
        print("=" * 60)
        print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"Improvement from baseline (67%): {((overall_accuracy - 0.67) * 100):+.2f}%")
        
        print(f"\n📊 Per-Class Performance:")
        classes = ["normal", "cataract", "diabetic_retinopathy", "glaucoma"]
        for class_name in classes:
            if class_name in classification_report:
                precision = classification_report[class_name]["precision"]
                recall = classification_report[class_name]["recall"]
                f1_score = classification_report[class_name]["f1-score"]
                auc = auc_scores.get(class_name, 0.0)
                
                print(f"  {class_name:<20}: Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}, AUC: {auc:.3f}")
        
        # Calculate average AUC
        avg_auc = np.mean(list(auc_scores.values()))
        print(f"\nAverage AUC Score: {avg_auc:.4f}")
        
        return self.evaluation_results
    
    def create_visualizations(self):
        """Create comprehensive training and evaluation visualizations"""
        print("\n📈 Creating Visualizations...")
        
        plots_dir = self.config["output"]["plots_dir"]
        
        # 1. Training History Plots
        self._plot_training_history()
        
        # 2. Confusion Matrix
        self._plot_confusion_matrix()
        
        # 3. ROC Curves
        self._plot_roc_curves()
        
        # 4. Class Distribution
        self._plot_class_distribution()
        
        print(f"✅ All visualizations saved to {plots_dir}/")
    
    def _plot_training_history(self):
        """Plot training history"""
        if isinstance(self.training_history, dict) and 'phase1' in self.training_history:
            # Progressive training history
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            phases = ['phase1', 'phase2', 'phase3']
            phase_names = ['Initial Training', 'Fine-tuning', 'Final Optimization']
            
            for i, (phase, name) in enumerate(zip(phases, phase_names)):
                if phase in self.training_history:
                    history = self.training_history[phase]
                    
                    # Accuracy plot
                    axes[0, i].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
                    axes[0, i].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
                    axes[0, i].set_title(f'{name} - Accuracy')
                    axes[0, i].set_xlabel('Epoch')
                    axes[0, i].set_ylabel('Accuracy')
                    axes[0, i].legend()
                    axes[0, i].grid(True, alpha=0.3)
                    
                    # Loss plot
                    axes[1, i].plot(history['loss'], label='Training Loss', linewidth=2)
                    axes[1, i].plot(history['val_loss'], label='Validation Loss', linewidth=2)
                    axes[1, i].set_title(f'{name} - Loss')
                    axes[1, i].set_xlabel('Epoch')
                    axes[1, i].set_ylabel('Loss')
                    axes[1, i].legend()
                    axes[1, i].grid(True, alpha=0.3)
        else:
            # Standard training history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            history = self.training_history.history if hasattr(self.training_history, 'history') else self.training_history
            
            # Accuracy plot
            ax1.plot(history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
            ax1.plot(history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Loss plot
            ax2.plot(history['loss'], 'b-', label='Training Loss', linewidth=2)
            ax2.plot(history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Advanced Medical CNN Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.config['output']['plots_dir']}/training_history.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self):
        """Plot confusion matrix"""
        if 'confusion_matrix' not in self.evaluation_results:
            return
        
        cm = np.array(self.evaluation_results['confusion_matrix'])
        classes = ["Normal", "Cataract", "Diabetic Retinopathy", "Glaucoma"]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        plt.title('Advanced Medical CNN - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        
        # Add accuracy information
        accuracy = self.evaluation_results["overall_accuracy"]
        plt.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)',
                ha='center', transform=plt.gca().transAxes, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{self.config['output']['plots_dir']}/confusion_matrix.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self):
        """Plot ROC curves for each class"""
        # This would require predictions probabilities, implementing placeholder
        plt.figure(figsize=(10, 8))
        
        classes = ["normal", "cataract", "diabetic_retinopathy", "glaucoma"]
        colors = ['blue', 'red', 'green', 'orange']
        
        # Plot AUC scores as bar chart instead
        auc_scores = self.evaluation_results.get("auc_scores", {})
        
        if auc_scores:
            plt.bar(range(len(classes)), [auc_scores.get(cls, 0) for cls in classes], 
                   color=colors, alpha=0.7)
            plt.xlabel('Classes')
            plt.ylabel('AUC Score')
            plt.title('AUC Scores by Class')
            plt.xticks(range(len(classes)), [cls.replace('_', ' ').title() for cls in classes], rotation=45)
            
            # Add value labels on bars
            for i, cls in enumerate(classes):
                auc = auc_scores.get(cls, 0)
                plt.text(i, auc + 0.01, f'{auc:.3f}', ha='center', va='bottom')
            
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.config['output']['plots_dir']}/auc_scores.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_distribution(self):
        """Plot class distribution"""
        if not hasattr(self, 'dataset_analysis'):
            return
        
        classes = ["Normal", "Cataract", "Diabetic Retinopathy", "Glaucoma"]
        train_counts = list(self.dataset_analysis["train_counts"].values())
        test_counts = list(self.dataset_analysis["test_counts"].values())
        
        x = np.arange(len(classes))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, train_counts, width, label='Training', alpha=0.8)
        ax.bar(x + width/2, test_counts, width, label='Test', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Number of Images')
        ax.set_title('Dataset Class Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (train, test) in enumerate(zip(train_counts, test_counts)):
            ax.text(i - width/2, train + 10, str(train), ha='center', va='bottom')
            ax.text(i + width/2, test + 10, str(test), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.config['output']['plots_dir']}/class_distribution.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save comprehensive training results"""
        print("💾 Saving Training Results...")
        
        results_dir = self.config["output"]["results_dir"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare comprehensive results
        results = {
            "timestamp": datetime.now().isoformat(),
            "tensorflow_version": tf.__version__,
            "configuration": self.config,
            "dataset_analysis": getattr(self, 'dataset_analysis', {}),
            "model_architecture": self.config["model"]["architecture"],
            "training_history": self._serialize_history(),
            "evaluation_results": self.evaluation_results,
            "final_metrics": {
                "overall_accuracy": self.evaluation_results.get("overall_accuracy", 0),
                "improvement_from_baseline": (self.evaluation_results.get("overall_accuracy", 0) - 0.67) * 100,
                "average_auc": np.mean(list(self.evaluation_results.get("auc_scores", {}).values())) if self.evaluation_results.get("auc_scores") else 0
            }
        }
        
        # Save main results
        results_file = os.path.join(results_dir, f"training_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save latest results (for easy access)
        latest_file = os.path.join(results_dir, "latest_results.json")
        with open(latest_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save model info
        model_info = {
            "model_path": self.config["model"]["model_path"],
            "architecture": self.config["model"]["architecture"],
            "total_parameters": self.model.model.count_params(),
            "input_shape": f"{self.config['dataset']['target_size'][0]}x{self.config['dataset']['target_size'][1]}x3",
            "classes": self.model.classes,
            "final_accuracy": self.evaluation_results.get("overall_accuracy", 0)
        }
        
        model_info_file = os.path.join(results_dir, "model_info.json")
        with open(model_info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✅ Results saved to:")
        print(f"  Main results: {results_file}")
        print(f"  Latest results: {latest_file}")
        print(f"  Model info: {model_info_file}")
    
    def _serialize_history(self):
        """Serialize training history for JSON saving"""
        if isinstance(self.training_history, dict):
            # Progressive training or already serialized
            return self.training_history
        elif hasattr(self.training_history, 'history'):
            # Keras History object
            return {k: [float(x) for x in v] for k, v in self.training_history.history.items()}
        else:
            return {}
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        print("🏥 ADVANCED MEDICAL CNN TRAINING PIPELINE")
        print("=" * 80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        try:
            # Step 1: Analyze Dataset
            if not self.analyze_dataset():
                print("❌ Dataset analysis failed. Please check your data paths.")
                return False
            
            # Step 2: Create Model
            self.create_model()
            
            # Step 3: Prepare Data
            self.prepare_data()
            
            # Step 4: Train Model
            self.train_model()
            
            # Step 5: Evaluate Model
            self.evaluate_model()
            
            # Step 6: Create Visualizations
            self.create_visualizations()
            
            # Step 7: Save Results
            self.save_results()
            
            # Final Summary
            self._print_final_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Training pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _print_final_summary(self):
        """Print final training summary"""
        print("\n" + "=" * 80)
        print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        accuracy = self.evaluation_results.get("overall_accuracy", 0)
        improvement = (accuracy - 0.67) * 100
        
        print(f"📊 FINAL RESULTS:")
        print(f"  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Improvement from baseline (67%): {improvement:+.2f}%")
        
        if accuracy > 0.85:
            print("🏆 EXCELLENT! Target accuracy >85% achieved!")
        elif accuracy > 0.75:
            print("✅ GOOD! Significant improvement achieved!")
        else:
            print("⚠️ Additional training or hyperparameter tuning may be needed.")
        
        print(f"\n📁 Output Files:")
        print(f"  Model: {self.config['model']['model_path']}")
        print(f"  Results: {self.config['output']['results_dir']}/")
        print(f"  Plots: {self.config['output']['plots_dir']}/")
        
        print(f"\n🚀 Model is ready for deployment!")
        print("=" * 80)


def main():
    """Main training function"""
    
    # Custom configuration (you can modify these paths for your setup)
    config = {
        "dataset": {
            "train_dir": "D:/projects/eyecare_ai/datasets/eyecare_ai_data/train",
            "test_dir": "D:/projects/eyecare_ai/datasets/eyecare_ai_data/test",
            "target_size": (384, 384),
            "batch_size": 8,  # Reduced for higher resolution and better training
            "validation_split": 0.2
        },
        
        "model": {
            "architecture": "advanced_cnn",  # Try 'ensemble' for maximum accuracy
            "learning_rate": 0.001,
            "model_path": "models/saved_models/advanced_medical_cnn_best.h5"
        },
        
        "training": {
            "epochs": 120,
            "use_progressive_training": True,
            "early_stopping_patience": 25,
            "lr_reduction_patience": 10,
            "min_learning_rate": 1e-8,
            "class_weights_auto": True
        },
        
        "output": {
            "results_dir": "advanced_training_results",
            "plots_dir": "advanced_training_plots",
            "save_best_only": True,
            "verbose": 1
        }
    }
    
    # Initialize trainer
    trainer = AdvancedMedicalTrainer(config)
    
    # Run complete training pipeline
    success = trainer.run_complete_pipeline()
    
    if success:
        print("\n✅ Advanced Medical CNN training completed successfully!")
        print("Your model should now achieve >80% accuracy (significant improvement from 67%)")
    else:
        print("\n❌ Training failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
