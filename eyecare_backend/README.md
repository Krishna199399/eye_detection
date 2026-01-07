# EyeCare AI Backend

A deep learning-powered backend for eye disease detection using Convolutional Neural Networks (CNN). This backend provides REST API endpoints for image analysis and integrates seamlessly with the React frontend.

## 🚀 Features

- **CNN Model**: EfficientNetB0-based transfer learning model for eye disease classification
- **Real-time Prediction**: Fast image analysis with confidence scores
- **Multi-class Classification**: Supports detection of:
  - Normal retina
  - Diabetic Retinopathy
  - Glaucoma
  - Cataract
  - Age-related Macular Degeneration
  - Hypertensive Retinopathy
  - Pathological Myopia
  - Other conditions

- **Image Preprocessing**: Advanced preprocessing with CLAHE enhancement and noise reduction
- **Enhanced Treatment Recommendations**: Comprehensive, evidence-based treatment protocols with:
  - Severity-based recommendations (mild, moderate, severe)
  - Emergency protocols and urgency levels
  - Immediate actions and treatment options
  - Lifestyle recommendations and follow-up care
  - Educational resources and support contacts
- **Multiple Eye Conditions**: Support for 6+ eye conditions including:
  - Normal/Healthy eyes
  - Diabetic Retinopathy
  - Glaucoma
  - Cataract
  - Age-related Macular Degeneration
  - Hypertensive Retinopathy
- **Database Integration**: SQLite database for storing predictions and metadata
- **CORS Support**: Configured for React frontend integration
- **Health Monitoring**: API health checks and system status endpoints

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- At least 4GB RAM (8GB recommended for training)
- GPU support (optional but recommended for training)

## 🛠️ Installation

### 1. Navigate to Backend Directory
```bash
cd eyecare_backend
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

## 🏃‍♂️ Running the Backend

### Prerequisites
1. **Python 3.8+** installed
2. **MongoDB** running on `mongodb://localhost:27017`
3. **ML Model** file exists at `models/saved_models/eye_disease_model.h5`

### Option 1: Using Startup Script (Recommended)
```bash
# From eyecare_backend directory
start_backend.bat
```

This will:
- Check Python installation
- Install dependencies if needed
- Start the FastAPI server

### Option 2: Manual Start
```bash
# Install dependencies first
pip install -r requirements.txt

# Start the server
python main.py
```

The server will start on `http://localhost:8000`

### Verify Backend is Running
Open your browser and test these endpoints:
- **Health check:** `http://localhost:8000/api/v1/health`
- **API documentation:** `http://localhost:8000/docs`
- **Model status:** `http://localhost:8000/api/v1/model-status`

### Complete Application Setup

**To run the full application with frontend:**

1. **Start Backend** (from `eyecare_backend/`):
   ```bash
   start_backend.bat
   # OR
   python main.py
   ```

2. **Start Frontend** (from `eyecare_frontend/`):
   ```bash
   start_frontend.bat
   # OR
   npm start
   ```

3. **Access Application:**
   - Frontend: `http://localhost:5173`
   - Backend API: `http://localhost:8000`
   - API Docs: `http://localhost:8000/docs`

### One-Click Launch (From Project Root)
```bash
# From D:\projects\eyecare directory
start_eyecare_app.bat
```
This starts both backend and frontend automatically!

## 🧠 Training Your Model

### 1. Prepare Your Dataset

Organize your eye disease images in the following directory structure:
```
your_dataset/
├── Normal/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Diabetic Retinopathy/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Glaucoma/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Cataract/
│   ├── image1.jpg
│   └── ...
└── ...
```

### 2. Train the Model

```bash
python train_model.py --data_dir "path/to/your/dataset" --epochs 50 --batch_size 32
```

**Training Options:**
- `--data_dir`: Path to your dataset directory
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--fine_tune`: Enable fine-tuning after initial training
- `--fine_tune_epochs`: Number of fine-tuning epochs (default: 20)

### 3. Example Training Command

```bash
# Basic training
python train_model.py --data_dir "D:/eye_disease_dataset" --epochs 30

# Training with fine-tuning
python train_model.py --data_dir "D:/eye_disease_dataset" --epochs 30 --fine_tune --fine_tune_epochs 15

# Custom parameters
python train_model.py --data_dir "D:/eye_disease_dataset" --epochs 50 --batch_size 16 --learning_rate 0.0005
```

## 🔧 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user (Patient or Healthcare Professional)
- `POST /api/v1/auth/login` - Login and get JWT token
- `POST /api/v1/auth/logout` - Logout user
- `GET /api/v1/auth/check-user/{email}` - Check if user exists

### Health & Status
- `GET /api/v1/health` - Backend health check
- `GET /api/v1/model-status` - AI model status

### Prediction (Protected Routes - Require Authentication)
- `POST /api/v1/predict` - Upload image for prediction (basic recommendations)
- `POST /api/v1/predict-comprehensive` - Upload image for prediction with comprehensive treatment recommendations
- `GET /api/v1/prediction/{prediction_id}` - Get specific prediction
- `GET /api/v1/predictions/history` - Get prediction history
- `DELETE /api/v1/prediction/{prediction_id}` - Delete prediction

### Enhanced Treatment Recommendations
- `GET /api/v1/treatment-recommendations/{condition}` - Get comprehensive treatment recommendations for a specific condition
- `GET /api/v1/supported-conditions` - Get list of supported eye conditions with severity levels

### Statistics
- `GET /api/v1/statistics` - Get prediction statistics

## 📊 Example API Usage

### Upload Image for Prediction
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/eye_image.jpg"
```

### Basic Response Format
```json
{
  "prediction_id": "abc123...",
  "success": true,
  "results": {
    "predicted_class": "Diabetic Retinopathy",
    "confidence": 89.5,
    "all_predictions": {
      "Normal": 5.2,
      "Diabetic Retinopathy": 89.5,
      "Glaucoma": 3.1,
      "Cataract": 2.2
    },
    "risk_level": "High",
    "recommendations": [
      "Consult with an ophthalmologist immediately",
      "Maintain strict blood sugar control",
      "Schedule regular retinal screenings"
    ]
  },
  "timestamp": "2025-01-03T10:30:00Z"
}
```

### Enhanced Comprehensive Response Format
```json
{
  "prediction_id": "abc123...",
  "success": true,
  "results": {
    "predicted_class": "diabetic_retinopathy",
    "confidence": 92.5,
    "all_predictions": {
      "normal": 2.1,
      "diabetic_retinopathy": 92.5,
      "glaucoma": 3.2,
      "cataract": 2.2
    },
    "risk_level": "High"
  },
  "comprehensive_recommendations": {
    "condition": "diabetic_retinopathy",
    "confidence": 92.5,
    "severity_level": "severe",
    "urgency_level": "emergency",
    "immediate_actions": [
      "🚨 SEVERE diabetic retinopathy - EMERGENCY",
      "Seek immediate medical attention",
      "Call ophthalmologist or go to emergency room"
    ],
    "treatment_options": [
      "Immediate anti-VEGF therapy likely required",
      "Panretinal photocoagulation may be necessary",
      "Possible vitrectomy surgery",
      "Intensive care coordination required"
    ],
    "lifestyle_recommendations": [
      "Hospital-grade diabetes management",
      "Complete lifestyle modification under medical supervision",
      "Immediate cessation of all risk factors"
    ],
    "follow_up_care": [
      "Weekly to monthly ophthalmologist visits",
      "Immediate retinal specialist care",
      "Multidisciplinary diabetes team involvement"
    ],
    "emergency_warning": {
      "level": "CRITICAL",
      "message": "🚨 MEDICAL EMERGENCY: Seek immediate medical attention. Do not delay.",
      "action": "Go to emergency room or call emergency services",
      "timeframe": "Immediately"
    },
    "next_appointment": {
      "recommended_date": "2025-01-03T14:30:00Z",
      "urgency": "emergency",
      "description": "Schedule appointment by January 03, 2025"
    },
    "educational_resources": [
      {
        "title": "American Diabetes Association - Eye Complications",
        "url": "https://diabetes.org/diabetes/complications/eye-complications",
        "description": "Comprehensive guide on diabetic eye disease"
      }
    ],
    "confidence_notes": [
      "High confidence detection (92.5%) - recommendations are strongly indicated"
    ]
  },
  "timestamp": "2025-01-03T10:30:00Z"
}
```

### Enhanced API Usage Examples

#### Comprehensive Prediction with Treatment Recommendations
```bash
curl -X POST "http://localhost:8000/api/v1/predict-comprehensive" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/eye_image.jpg"
```

#### Get Treatment Recommendations for Specific Condition
```bash
# Get recommendations for diabetic retinopathy with high confidence
curl -X GET "http://localhost:8000/api/v1/treatment-recommendations/diabetic_retinopathy?confidence=85.0"

# Get recommendations for glaucoma with moderate confidence
curl -X GET "http://localhost:8000/api/v1/treatment-recommendations/glaucoma?confidence=70.0"
```

#### Get Supported Conditions
```bash
curl -X GET "http://localhost:8000/api/v1/supported-conditions"
```

Response:
```json
{
  "success": true,
  "supported_conditions": [
    {
      "name": "diabetic_retinopathy",
      "display_name": "Diabetic Retinopathy",
      "description": "Diabetes-related eye disease affecting the retina",
      "severity_levels": ["mild", "moderate", "severe"]
    },
    {
      "name": "glaucoma",
      "display_name": "Glaucoma",
      "description": "Eye condition with increased intraocular pressure",
      "severity_levels": ["mild", "moderate", "severe"]
    },
    {
      "name": "age_related_macular_degeneration",
      "display_name": "Age-Related Macular Degeneration",
      "description": "Eye disease affecting the macula",
      "severity_levels": ["mild"]
    }
  ],
  "enhanced_recommendations": true,
  "total_conditions": 6
}
```

## 🗗️ Database

The backend uses SQLite database with the following tables:
- `predictions`: Stores prediction results
- `patients`: Patient information (for future use)
- `model_metadata`: Model training metadata

Database file: `database/eyecare_ai.db`

## 🔧 Configuration

### Environment Variables (Optional)
Create a `.env` file in the backend directory:
```env
# Database
DATABASE_URL=sqlite:///./database/eyecare_ai.db

# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Model Settings
MODEL_PATH=models/saved_models/eye_disease_model.h5
```

### CORS Settings
The backend is configured to accept requests from:
- `http://localhost:4028` (React frontend)
- `http://127.0.0.1:4028`

## 📈 Performance Tips

### For Training:
1. **Use GPU**: Install TensorFlow-GPU for faster training
2. **Data Augmentation**: The training script includes built-in augmentation
3. **Transfer Learning**: Using EfficientNetB0 reduces training time
4. **Early Stopping**: Training stops automatically when validation loss stops improving

### For Inference:
1. **Image Preprocessing**: Images are automatically preprocessed and enhanced
2. **Batch Processing**: The API can handle multiple images efficiently
3. **Model Caching**: Model is loaded once and reused for all predictions

## 🐛 Troubleshooting

### Common Issues:

1. **"Model is not loaded"**
   - Train a model first using `train_model.py`
   - Ensure model file exists at `models/saved_models/eye_disease_model.h5`

2. **"Backend server is offline"**
   - Check if the server is running on port 8000
   - Verify no firewall is blocking the port

3. **Memory errors during training**
   - Reduce batch size: `--batch_size 16` or `--batch_size 8`
   - Close other applications to free up RAM

4. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version: `python --version` (should be 3.8+)

### Performance Issues:

1. **Slow predictions**
   - Consider using GPU acceleration
   - Reduce image size in preprocessing

2. **High memory usage**
   - Restart the server periodically
   - Monitor memory usage with the health endpoint

## 📝 Development

### Adding New Disease Classes:
1. Update the `classes` list in `models/cnn_model.py`
2. Retrain the model with new data
3. Update the recommendations in `app/routes/prediction.py`

### API Documentation:
Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

## 🤝 Integration with Frontend

The React frontend automatically connects to this backend. Make sure:
1. Backend is running on port 8000
2. Frontend is running on port 4028
3. CORS is properly configured (already done)

The frontend will show:
- ✅ Backend Server: Online/Offline
- ✅ AI Model: Loaded/Not Loaded
- Real-time prediction results
- Error handling and status messages

## 🎯 Next Steps

1. **Train your model** with your eye disease dataset
2. **Test the API** using the interactive docs at `/docs`
3. **Upload images** through the React frontend
4. **Monitor predictions** in the database
5. **Export reports** and integrate with your healthcare workflow

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Review the server logs in the terminal
3. Verify all dependencies are correctly installed
4. Ensure your dataset is properly organized for training

---

🎉 **Your EyeCare AI backend is ready to detect eye diseases with deep learning precision!**
