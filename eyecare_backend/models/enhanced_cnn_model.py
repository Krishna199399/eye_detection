import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, 
    GlobalAveragePooling2D, Add, Input, Multiply, Lambda, Concatenate,
    SeparableConv2D, DepthwiseConv2D, Activation, AveragePooling2D
)
from keras.applications import EfficientNetV2B0, ResNet50V2, DenseNet121, VGG19
from keras.optimizers import Adam, AdamW, SGD
from keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    LearningRateScheduler
)
from keras.regularizers import l2
import numpy as np
import os
from typing import List, Tuple, Optional
import keras.backend as K

class AttentionBlock(tf.keras.layers.Layer):
    """
    Attention mechanism for focusing on important features
    """
    def __init__(self, filters, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.filters = filters
        
    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters // 8, 1, padding='same', activation='relu')
        self.conv2 = Conv2D(self.filters // 8, 1, padding='same', activation='relu')
        self.conv3 = Conv2D(1, 1, padding='same', activation='sigmoid')
        super(AttentionBlock, self).build(input_shape)
        
    def call(self, inputs):
        # Channel attention
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        
        avg_out = self.conv1(avg_pool)
        avg_out = self.conv2(avg_out)
        avg_out = self.conv3(avg_out)
        
        max_out = self.conv1(max_pool)
        max_out = self.conv2(max_out)
        max_out = self.conv3(max_out)
        
        channel_attention = avg_out + max_out
        
        # Spatial attention
        spatial_avg = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        spatial_max = tf.reduce_max(inputs, axis=-1, keepdims=True)
        spatial_concat = tf.concat([spatial_avg, spatial_max], axis=-1)
        spatial_attention = Conv2D(1, 7, padding='same', activation='sigmoid')(spatial_concat)
        
        # Apply attention
        attended = inputs * channel_attention * spatial_attention
        return attended

class ResidualBlock(tf.keras.layers.Layer):
    """
    Residual block with optional attention
    """
    def __init__(self, filters, kernel_size=3, stride=1, use_attention=True, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_attention = use_attention
        
    def build(self, input_shape):
        self.conv1 = Conv2D(self.filters, self.kernel_size, strides=self.stride, 
                           padding='same', kernel_regularizer=l2(0.001))
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(self.filters, self.kernel_size, padding='same', 
                           kernel_regularizer=l2(0.001))
        self.bn2 = BatchNormalization()
        
        if self.use_attention:
            self.attention = AttentionBlock(self.filters)
            
        # Shortcut connection
        if input_shape[-1] != self.filters or self.stride != 1:
            self.shortcut = Conv2D(self.filters, 1, strides=self.stride, padding='same')
            self.shortcut_bn = BatchNormalization()
        else:
            self.shortcut = None
            
        super(ResidualBlock, self).build(input_shape)
        
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = Dropout(0.1)(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        if self.use_attention:
            x = self.attention(x)
            
        # Shortcut connection
        if self.shortcut is not None:
            shortcut = self.shortcut(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs
            
        x = Add()([x, shortcut])
        x = tf.nn.relu(x)
        return x

class EnhancedEyeDiseaseModel:
    """
    Enhanced CNN Model for Eye Disease Detection with attention mechanisms,
    residual connections, and ensemble capabilities
    """
    
    def __init__(self, model_path: str = "models/saved_models/enhanced_eye_disease_model.h5"):
        self.model_path = model_path
        self.model = None
        self.ensemble_models = []
        self.classes = [
            "normal", 
            "cataract", 
            "diabetic_retinopathy", 
            "glaucoma"
        ]
        self.img_height = 256  # Increased resolution
        self.img_width = 256
        self.channels = 3
        
        # Class weights to handle imbalanced data
        self.class_weights = {
            0: 1.0,    # normal
            1: 1.2,    # cataract
            2: 1.0,    # diabetic_retinopathy  
            3: 1.5     # glaucoma (most challenging class)
        }
        
        # Try to load existing model
        if os.path.exists(model_path):
            self.load_model()

    def create_advanced_cnn_architecture(self, num_classes: int = 4) -> Model:
        """
        Create advanced custom CNN with attention and residual connections
        """
        inputs = Input(shape=(self.img_height, self.img_width, self.channels))
        
        # Initial convolution
        x = Conv2D(64, 7, strides=2, padding='same', kernel_regularizer=l2(0.001))(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual blocks with increasing complexity
        x = ResidualBlock(64, use_attention=True)(x)
        x = ResidualBlock(64, use_attention=True)(x)
        x = MaxPooling2D(2)(x)
        
        x = ResidualBlock(128, use_attention=True)(x)
        x = ResidualBlock(128, use_attention=True)(x)
        x = MaxPooling2D(2)(x)
        
        x = ResidualBlock(256, use_attention=True)(x)
        x = ResidualBlock(256, use_attention=True)(x)
        x = ResidualBlock(256, use_attention=True)(x)
        x = MaxPooling2D(2)(x)
        
        x = ResidualBlock(512, use_attention=True)(x)
        x = ResidualBlock(512, use_attention=True)(x)
        
        # Global pooling and attention
        gap = GlobalAveragePooling2D()(x)
        
        # Multi-head classification with different dropout rates
        # Head 1: High dropout for regularization
        head1 = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(gap)
        head1 = BatchNormalization()(head1)
        head1 = Dropout(0.6)(head1)
        head1 = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(head1)
        head1 = Dropout(0.5)(head1)
        
        # Head 2: Lower dropout for feature preservation
        head2 = Dense(256, activation='relu', kernel_regularizer=l2(0.005))(gap)
        head2 = BatchNormalization()(head2)
        head2 = Dropout(0.3)(head2)
        
        # Combine heads
        combined = Concatenate()([head1, head2])
        combined = Dense(128, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.4)(combined)
        
        # Final classification
        outputs = Dense(num_classes, activation='softmax', name='predictions')(combined)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def create_efficient_transfer_model(self, num_classes: int = 4) -> Model:
        """
        Create transfer learning model with EfficientNetV2
        """
        base_model = EfficientNetV2B0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_height, self.img_width, self.channels)
        )
        
        # Unfreeze the last few layers for fine-tuning
        for layer in base_model.layers[-20:]:
            layer.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False
            
        inputs = Input(shape=(self.img_height, self.img_width, self.channels))
        x = base_model(inputs, training=True)
        
        # Add custom head with attention
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Multi-scale features
        x = Dense(1024, activation='swish', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(512, activation='swish', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        outputs = Dense(num_classes, activation='softmax', name='predictions')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def create_ensemble_model(self, num_classes: int = 4) -> Model:
        """
        Create an ensemble of different architectures
        """
        # Create individual models
        custom_cnn = self.create_advanced_cnn_architecture(num_classes)
        efficient_net = self.create_efficient_transfer_model(num_classes)
        
        # Input layer
        inputs = Input(shape=(self.img_height, self.img_width, self.channels))
        
        # Get predictions from both models
        pred1 = custom_cnn(inputs)
        pred2 = efficient_net(inputs)
        
        # Weighted ensemble (can be learned or fixed)
        ensemble_output = Lambda(lambda x: 0.6 * x[0] + 0.4 * x[1])([pred1, pred2])
        
        model = Model(inputs=inputs, outputs=ensemble_output)
        return model

    def compile_model(self, learning_rate: float = 0.001, model_type: str = 'advanced_cnn'):
        """
        Compile the model with advanced optimizers and loss functions
        """
        if model_type == 'advanced_cnn':
            self.model = self.create_advanced_cnn_architecture()
        elif model_type == 'efficient_transfer':
            self.model = self.create_efficient_transfer_model()
        elif model_type == 'ensemble':
            self.model = self.create_ensemble_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Use Adam optimizer (AdamW not available in this TensorFlow version)
        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
        
        # Compile with focal loss for imbalanced data
        self.model.compile(
            optimizer=optimizer,
            loss=self.focal_loss,
            metrics=['accuracy', 'precision', 'recall']
        )

    @staticmethod
    def focal_loss(gamma=2., alpha=0.25):
        """
        Focal loss for handling class imbalance
        """
        def focal_loss_fixed(y_true, y_pred):
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
                   - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
        return focal_loss_fixed

    def get_advanced_callbacks(self) -> List:
        """
        Get advanced training callbacks
        """
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        def scheduler(epoch, lr):
            """Learning rate scheduler with warm-up"""
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)
        
        callbacks = [
            ModelCheckpoint(
                self.model_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=7,
                min_lr=1e-8,
                verbose=1
            ),
            LearningRateScheduler(scheduler, verbose=1)
        ]
        
        return callbacks

    def train_with_advanced_strategies(self, train_generator, validation_generator, 
                                     epochs: int = 100, model_type: str = 'advanced_cnn'):
        """
        Train with advanced strategies including progressive resizing
        """
        if self.model is None:
            self.compile_model(model_type=model_type)
            
        callbacks = self.get_advanced_callbacks()
        
        # Progressive training with different strategies
        history_phases = []
        
        # Phase 1: Initial training with lower resolution
        print("Phase 1: Initial training...")
        self.img_height, self.img_width = 224, 224
        self.compile_model(learning_rate=0.001, model_type=model_type)
        
        history1 = self.model.fit(
            train_generator,
            epochs=epochs//3,
            validation_data=validation_generator,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        history_phases.append(history1)
        
        # Phase 2: Fine-tuning with higher resolution
        print("Phase 2: Fine-tuning with higher resolution...")
        self.img_height, self.img_width = 256, 256
        
        # Reduce learning rate for fine-tuning
        K.set_value(self.model.optimizer.learning_rate, 0.0001)
        
        history2 = self.model.fit(
            train_generator,
            epochs=epochs//3,
            validation_data=validation_generator,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        history_phases.append(history2)
        
        # Phase 3: Final fine-tuning with very low learning rate
        print("Phase 3: Final fine-tuning...")
        K.set_value(self.model.optimizer.learning_rate, 0.00001)
        
        history3 = self.model.fit(
            train_generator,
            epochs=epochs//3,
            validation_data=validation_generator,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        history_phases.append(history3)
        
        return history_phases

    def predict_with_tta(self, image: np.ndarray, tta_steps: int = 5) -> Tuple[str, float, dict]:
        """
        Make prediction with Test Time Augmentation for better accuracy
        """
        if self.model is None:
            raise ValueError("Model is not loaded. Please train or load a model first.")
            
        predictions_list = []
        
        # Original prediction
        pred = self.model.predict(np.expand_dims(image, axis=0), verbose=0)
        predictions_list.append(pred[0])
        
        # TTA predictions with different augmentations
        for _ in range(tta_steps):
            # Random augmentations
            aug_image = image.copy()
            
            # Random rotation
            if np.random.random() > 0.5:
                angle = np.random.uniform(-10, 10)
                aug_image = tf.keras.preprocessing.image.apply_transform(
                    aug_image, {'rotation': angle}
                )
            
            # Random brightness
            if np.random.random() > 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                aug_image = tf.image.adjust_brightness(aug_image, brightness - 1.0)
                aug_image = tf.clip_by_value(aug_image, 0.0, 1.0)
            
            # Horizontal flip
            if np.random.random() > 0.5:
                aug_image = tf.image.flip_left_right(aug_image)
            
            pred = self.model.predict(np.expand_dims(aug_image, axis=0), verbose=0)
            predictions_list.append(pred[0])
        
        # Average predictions
        avg_predictions = np.mean(predictions_list, axis=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(avg_predictions)
        confidence = float(avg_predictions[predicted_class_idx])
        predicted_class = self.classes[predicted_class_idx]
        
        # Create predictions dictionary
        predictions_dict = {
            self.classes[i]: float(avg_predictions[i]) 
            for i in range(len(self.classes))
        }
        
        return predicted_class, confidence, predictions_dict

    def predict(self, image: np.ndarray, use_tta: bool = True) -> Tuple[str, float, dict]:
        """
        Make prediction on a single image with optional TTA
        """
        if use_tta:
            return self.predict_with_tta(image)
        else:
            return self._simple_predict(image)
    
    def _simple_predict(self, image: np.ndarray) -> Tuple[str, float, dict]:
        """Simple prediction without TTA"""
        if self.model is None:
            raise ValueError("Model is not loaded. Please train or load a model first.")
            
        predictions = self.model.predict(np.expand_dims(image, axis=0), verbose=0)
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = self.classes[predicted_class_idx]
        
        predictions_dict = {
            self.classes[i]: float(predictions[0][i]) 
            for i in range(len(self.classes))
        }
        
        return predicted_class, confidence, predictions_dict

    def save_model(self, path: str = None):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
            
        save_path = path or self.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        print(f"Enhanced model saved to {save_path}")

    def load_model(self, path: str = None):
        """Load a trained model"""
        load_path = path or self.model_path
        
        if os.path.exists(load_path):
            # Custom objects for loading
            custom_objects = {
                'focal_loss_fixed': self.focal_loss(),
                'AttentionBlock': AttentionBlock,
                'ResidualBlock': ResidualBlock
            }
            self.model = tf.keras.models.load_model(load_path, custom_objects=custom_objects)
            print(f"Enhanced model loaded from {load_path}")
            return True
        else:
            print(f"Model file not found at {load_path}")
            return False

    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            return "Model not loaded"
        return self.model.summary()

    def evaluate_detailed(self, test_generator):
        """Detailed evaluation with per-class metrics"""
        if self.model is None:
            raise ValueError("Model is not loaded")
            
        # Standard evaluation
        results = self.model.evaluate(test_generator, verbose=1)
        
        # Detailed predictions for confusion matrix
        predictions = self.model.predict(test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=self.classes, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'evaluation_metrics': dict(zip(self.model.metrics_names, results)),
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
