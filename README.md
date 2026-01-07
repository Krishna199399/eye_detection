# EyeCare AI - Complete Eye Disease Detection System

A full-stack AI-powered eye disease detection application with React frontend and FastAPI backend.

## 🌟 Features

- **AI-Powered Analysis**: Real-time eye disease detection using trained CNN models
- **Multiple Disease Detection**: Detects Normal, Diabetic Retinopathy, Glaucoma, and Cataract conditions
- **Interactive Frontend**: Modern React interface with drag-and-drop image upload
- **Real-time Status**: Backend health monitoring and ML model status indicators
- **Analysis History**: Save and retrieve past analysis results
- **Medical Recommendations**: Personalized recommendations based on detection results
- **Comprehensive API**: RESTful API with automatic documentation

## 🔧 System Requirements

### Backend Requirements
- Python 3.8 or higher
- TensorFlow 2.15.0
- MongoDB (optional, for data persistence)

### Frontend Requirements  
- Node.js 16 or higher
- npm (comes with Node.js)

## 🚀 Running the Application

### Prerequisites
1. **MongoDB** must be running on `mongodb://localhost:27017`
2. **Python 3.8+** installed with dependencies
3. **Node.js 16+** installed with npm

### Option 1: One-Click Setup (Easiest)
1. **Run the main startup script:**
   ```bash
   # Double-click on start_eyecare_app.bat
   # OR run in Command Prompt/PowerShell:
   start_eyecare_app.bat
   ```

2. **Wait for both services to start** (about 30 seconds)

3. **Open your browser** and go to: http://localhost:5173

4. **First Time Access:**
   - You'll be redirected to the login page
   - Click "Create New Account" to register
   - Choose account type (Patient or Healthcare Professional)
   - Fill in your details and create your account
   - You'll be automatically logged in!

### Option 2: Manual Setup

#### Backend Setup
1. **Navigate to backend directory:**
   ```bash
   cd eyecare_backend
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the backend server:**
   ```bash
   # Use the startup script (Recommended):
   start_backend.bat
   
   # OR run manually:
   python main.py
   ```

4. **Verify backend is running:**
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/api/v1/health

#### Frontend Setup
1. **Navigate to frontend directory (from project root):**
   ```bash
   cd eyecare_frontend
   ```

2. **Install Node.js dependencies (first time only):**
   ```bash
   npm install
   ```

3. **Start the frontend client:**
   ```bash
   # Use the startup script (Recommended):
   start_frontend.bat
   
   # OR run manually:
   npm start
   ```

4. **Open the application:**
   - Frontend App: http://localhost:5173
   - Login Page: http://localhost:5173/login
   - Register Page: http://localhost:5173/register

## 📋 Usage Guide

### 0. Authentication (First Time)
1. **Register an Account:**
   - Go to http://localhost:5173/register
   - Choose account type:
     - **Patient Account** - For individuals
     - **Healthcare Professional** - For doctors/clinicians
   - Fill in required information
   - Accept terms and click "Create Account"

2. **Login:**
   - Go to http://localhost:5173/login
   - Enter your email and password
   - Click "Sign In"
   - You'll be redirected to the dashboard

3. **Demo Account (Optional):**
   - Email: `demo@eyecare.ai`
   - Password: `demo123`

### 1. Upload an Eye Image
- From the Dashboard, click "Choose File" or drag-and-drop an eye image
- Supported formats: JPEG, PNG, BMP (max 10MB)
- Ensure good lighting and clear visibility of the eye

### 2. AI Analysis
- Click "Analyze Image" to start the detection process
- Wait for the ML model to process your image (5-10 seconds)
- View real-time analysis progress

### 3. Review Results
- **Predicted Condition**: Primary diagnosis (Normal/Diabetic Retinopathy/Glaucoma/Cataract)
- **Confidence Level**: How certain the AI is about the diagnosis
- **Risk Assessment**: Low/Medium/High risk categorization
- **Recommendations**: Personalized medical advice based on results

### 4. Manage Your Account
- Click on your avatar (top right) to access:
  - My Profile
  - Settings
  - Sign Out

### 5. View History
- Navigate to "History" to see past analyses
- Review previous diagnoses and recommendations

## 🏗️ System Architecture

```
┌─────────────────┐    HTTP/REST API    ┌──────────────────┐
│   React App     │◄──────────────────►│   FastAPI        │
│   (Frontend)    │                     │   (Backend)      │
│   Port: 5173    │                     │   Port: 8000     │
└─────────────────┘                     └──────────────────┘
                                                   │
                                                   ▼
                                         ┌──────────────────┐
                                         │   TensorFlow     │
                                         │   ML Model       │
                                         │   (.h5 file)     │
                                         └──────────────────┘
                                                   │
                                                   ▼
                                         ┌──────────────────┐
                                         │   MongoDB        │
                                         │   (Database)     │
                                         │   Port: 27017    │
                                         └──────────────────┘
```

## 📁 Project Structure

```
EyeCare-AI/
├── eyecare_backend/           # Python FastAPI Backend
│   ├── models/                # ML Models & Training Scripts
│   ├── database/              # Database Models & Services
│   ├── utils/                 # Image Processing Utilities
│   ├── main.py               # Main Entry Point
│   ├── ml_backend.py         # FastAPI App with ML Integration
│   └── start_backend.bat     # Backend Startup Script
│
├── eyecare_frontend/          # React Frontend
│   ├── src/
│   │   ├── pages/            # React Pages/Components
│   │   ├── services/         # API Service Layer
│   │   └── components/       # Reusable UI Components
│   ├── package.json          # Frontend Dependencies
│   └── start_frontend.bat    # Frontend Startup Script
│
├── start_eyecare_app.bat     # Main Application Launcher
└── README.md                 # This File
```

## 🔧 Configuration

### Backend Configuration (`.env`)
```env
# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=eyecare_ai

# Application Settings
ENVIRONMENT=development
DEBUG=true
API_VERSION=v1

# CORS Settings (includes Vite dev server)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

### Frontend Configuration (`.env`)
```env
# Backend API Configuration
VITE_API_BASE_URL=http://localhost:8000/api/v1
REACT_APP_API_BASE_URL=http://localhost:8000/api/v1

# Application Settings
VITE_APP_NAME=EyeCare AI - Disease Detection
VITE_APP_VERSION=2.0.0
```

## 🧠 ML Model Information

- **Model Type**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow 2.15.0
- **Architecture**: Custom CNN with transfer learning
- **Input Size**: 224x224x3 (RGB images)
- **Classes**: 4 (Normal, Diabetic Retinopathy, Glaucoma, Cataract)
- **Model File**: `models/saved_models/eye_disease_model.h5`

### Model Performance
- Overall Accuracy: 67.46%
- Normal: 91.63%
- Diabetic Retinopathy: 80.91%
- Cataract: 54.81%
- Glaucoma: 40.10%

## 🔌 API Endpoints

### Health & Status
- `GET /api/v1/health` - Backend health check
- `GET /api/v1/model-status` - ML model status
- `GET /api/v1/model-info` - Model information

### Analysis
- `POST /api/v1/predict` - Upload image for analysis
- `GET /api/v1/statistics` - Model performance statistics

### History
- `GET /api/v1/predictions/history` - Get analysis history
- `POST /api/v1/predictions` - Save analysis to history

## 🛠️ Troubleshooting

### Common Issues

1. **Backend won't start:**
   - Check Python installation: `python --version`
   - Install dependencies: `pip install -r requirements.txt`
   - Check if port 8000 is available

2. **Frontend won't start:**
   - Check Node.js installation: `node --version`
   - Install dependencies: `npm install`
   - Check if port 5173 is available

3. **CORS errors:**
   - Ensure backend is running before frontend
   - Check CORS origins in backend configuration
   - Clear browser cache

4. **ML model not loading:**
   - Verify `eye_disease_model.h5` exists in `models/saved_models/`
   - Check TensorFlow installation: `python -c "import tensorflow; print(tensorflow.__version__)"`

5. **Database connection issues:**
   - MongoDB is optional for basic functionality
   - Install MongoDB if you need history features
   - Check MongoDB is running on port 27017

### Status Indicators

The frontend displays real-time status indicators:
- **Green dot**: Service is healthy and ready
- **Red dot**: Service is disconnected or failed
- **Yellow dot**: Service is starting up or has issues

## 🔒 Security Notes

- This is a development version with relaxed CORS settings
- Do not use in production without proper security configurations
- ML predictions are for educational purposes only
- Always consult healthcare professionals for medical decisions

## 📚 Development

### Adding New Features
1. Backend changes go in `eyecare_backend/`
2. Frontend changes go in `eyecare_frontend/src/`
3. Test both services after changes
4. Update API documentation if needed

### Database Setup (Optional)
If you want to use the history features:
1. Install MongoDB Community Edition
2. Start MongoDB service
3. The application will auto-create the database

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all requirements are installed
3. Check console logs for error messages
4. Ensure both services are running

## 📄 License

This project is for educational and development purposes. Please consult healthcare professionals for actual medical diagnoses.

---

**🎯 Ready to get started? Run `start_eyecare_app.bat` and open http://localhost:5173 in your browser!**
  


  **backend** 
  command == git 