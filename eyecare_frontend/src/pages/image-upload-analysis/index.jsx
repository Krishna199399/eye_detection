import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '../../components/ui/Header';
import FileUploadZone from './components/FileUploadZone';
import ImagePreviewPanel from './components/ImagePreviewPanel';
import AnalysisResultsPanel from './components/AnalysisResultsPanel';
import Button from '../../components/ui/Button';
import { analysisService, healthService } from '../../services/apiService';


const ImageUploadAnalysis = () => {
  const navigate = useNavigate();
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [imageRotation, setImageRotation] = useState(0);
  const [isImageZoomed, setIsImageZoomed] = useState(false);
  const [backendStatus, setBackendStatus] = useState({ healthy: false, modelReady: false });
  const [error, setError] = useState(null);

  // Check backend health on component mount
  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const healthResponse = await healthService.checkHealth();
      const modelResponse = await healthService.checkModelStatus();
      
      setBackendStatus({
        healthy: healthResponse.success,
        modelReady: modelResponse.success && modelResponse.data?.model_ready
      });
      
      if (!healthResponse.success) {
        setError('Backend service is not available. Please ensure the server is running.');
      } else if (!modelResponse.data?.model_ready) {
        setError('ML model is not loaded. Please wait for the model to initialize.');
      }
    } catch (error) {
      console.error('Backend health check failed:', error);
      setError('Unable to connect to backend service. Please check your connection.');
    }
  };

  const handleFileUpload = async (fileData) => {
    setIsUploading(true);
    
    // Simulate upload delay
    setTimeout(() => {
      setUploadedImage(fileData);
      setIsUploading(false);
    }, 2000);
  };

  const handleAnalyzeImage = async () => {
    if (!uploadedImage || !uploadedImage.file) {
      setError('No image file available for analysis');
      return;
    }

    if (!backendStatus.healthy || !backendStatus.modelReady) {
      setError('Backend service or ML model is not ready. Please try again later.');
      return;
    }
    
    setIsAnalyzing(true);
    setAnalysisResults(null);
    setError(null);
    
    try {
      console.log('Analyzing image:', uploadedImage.file.name);
      const response = await analysisService.analyzeImage(uploadedImage.file);
      
      if (response.success && response.data) {
        console.log('Analysis successful:', response.data);
        setAnalysisResults(response.data);
        
        // Auto-save to history if the prediction was successful
        if (response.data.success && response.data.results) {
          try {
            await analysisService.saveAnalysisToHistory(response.data);
            console.log('Analysis saved to history');
          } catch (saveError) {
            console.warn('Failed to save to history:', saveError);
            // Don't show error to user, just log it
          }
        }
      } else {
        throw new Error(response.message || response.error || 'Analysis failed');
      }
    } catch (error) {
      console.error('Analysis failed:', error);
      setError(`Analysis failed: ${error.message || 'Unknown error occurred'}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleGeneratePDF = async () => {
    try {
      // TODO: Replace with actual PDF generation API
      // const response = await reportService.generatePDF(analysisResults);
      // 
      // const link = document.createElement('a');
      // link.href = response.downloadUrl;
      // link.download = response.filename;
      // document.body.appendChild(link);
      // link.click();
      // document.body.removeChild(link);
      // 
      // Show success message (consider using a toast library)
      // alert('PDF report generated successfully!');
      
      // Temporary placeholder
      alert('PDF generation API not yet implemented. Please implement the report service.');
    } catch (error) {
      console.error('PDF generation failed:', error);
      alert('Failed to generate PDF report. Please try again.');
    }
  };

  const handleShareResults = async () => {
    try {
      // TODO: Replace with actual sharing API that generates shareable links
      // const response = await shareService.createShareableLink(analysisResults);
      // const shareUrl = response.shareUrl;
      
      if (navigator.share) {
        // TODO: Use actual shareable link from API
        // await navigator.share({
        //   title: 'Eye Disease Analysis Results',
        //   text: 'Check out my eye disease analysis results from EyeDisease Detector',
        //   url: shareUrl
        // });
        alert('Sharing API not yet implemented.');
      } else {
        // TODO: Copy actual shareable link to clipboard
        // await navigator.clipboard.writeText(shareUrl);
        // alert('Results link copied to clipboard!');
        alert('Sharing functionality not yet implemented.');
      }
    } catch (error) {
      console.error('Sharing failed:', error);
      alert('Failed to share results. Please try again.');
    }
  };

  const handleSaveToHistory = async () => {
    if (!analysisResults) {
      alert('No analysis results to save.');
      return;
    }

    try {
      const response = await analysisService.saveAnalysisToHistory(analysisResults);
      
      if (response.success) {
        alert('Results saved to patient history!');
      } else {
        alert('Failed to save results. Please try again.');
      }
    } catch (error) {
      console.error('Save to history failed:', error);
      alert('Failed to save results to history. Please try again.');
    }
  };

  const handleNewAnalysis = () => {
    setUploadedImage(null);
    setAnalysisResults(null);
    setImageRotation(0);
    setIsImageZoomed(false);
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container mx-auto px-4 py-8">
        {/* Backend Status Alert */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center space-x-2">
              <svg className="w-5 h-5 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <span className="text-sm font-medium text-red-800">{error}</span>
              <button 
                onClick={() => checkBackendHealth()} 
                className="ml-auto text-red-600 hover:text-red-800 text-sm underline"
              >
                Retry Connection
              </button>
            </div>
          </div>
        )}

        {/* Backend Status Indicator */}
        <div className="mb-6 flex items-center justify-between bg-gray-50 p-3 rounded-lg">
          <div className="flex items-center space-x-3">
            <div className={`w-3 h-3 rounded-full ${backendStatus.healthy ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span className="text-sm font-medium">
              Backend: {backendStatus.healthy ? 'Connected' : 'Disconnected'}
            </span>
            <div className={`w-3 h-3 rounded-full ${backendStatus.modelReady ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
            <span className="text-sm font-medium">
              ML Model: {backendStatus.modelReady ? 'Ready' : 'Loading...'}
            </span>
          </div>
          <button 
            onClick={checkBackendHealth}
            className="text-sm text-blue-600 hover:text-blue-800 underline"
          >
            Refresh Status
          </button>
        </div>

        {/* Page Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-heading font-bold text-foreground mb-2">
                Image Upload & Analysis
              </h1>
              <p className="text-lg font-body text-muted-foreground">
                Upload eye images for AI-powered disease detection and comprehensive diagnostic analysis
              </p>
            </div>
            
            {uploadedImage && (
              <div className="flex items-center space-x-3">
                {!analysisResults && !isAnalyzing && (
                  <Button
                    variant="default"
                    onClick={handleAnalyzeImage}
                    iconName="Brain"
                    iconPosition="left"
                    iconSize={16}
                  >
                    Analyze Image
                  </Button>
                )}
                
                <Button
                  variant="outline"
                  onClick={handleNewAnalysis}
                  iconName="RotateCcw"
                  iconPosition="left"
                  iconSize={16}
                >
                  New Analysis
                </Button>
              </div>
            )}
          </div>
        </div>

        {/* Main Content */}
        {!uploadedImage ? (
          /* Upload Zone */
          (<div className="max-w-4xl mx-auto">
            <FileUploadZone 
              onFileUpload={handleFileUpload}
              isUploading={isUploading}
            />
          </div>)
        ) : (
          /* Analysis Interface */
          (<div className="grid grid-cols-1 lg:grid-cols-2 gap-8 h-[calc(100vh-280px)]">
            {/* Left Panel - Image Preview */}
            <ImagePreviewPanel
              uploadedImage={uploadedImage}
              onRotate={setImageRotation}
              onZoom={setIsImageZoomed}
              zoomLevel={isImageZoomed ? 150 : 100}
              rotation={imageRotation}
            />
            {/* Right Panel - Analysis Results */}
            <AnalysisResultsPanel
              analysisResults={analysisResults}
              isLoading={isAnalyzing}
              onGeneratePDF={handleGeneratePDF}
              onShareResults={handleShareResults}
              onSaveToHistory={handleSaveToHistory}
            />
          </div>)
        )}

        {/* Quick Actions */}
        <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
          <Button
            variant="ghost"
            onClick={() => navigate('/home-dashboard')}
            iconName="ArrowLeft"
            iconPosition="left"
            iconSize={16}
          >
            Back to Dashboard
          </Button>
          
          <Button
            variant="ghost"
            onClick={() => navigate('/patient-history')}
            iconName="History"
            iconPosition="left"
            iconSize={16}
          >
            View History
          </Button>
          
          <Button
            variant="ghost"
            onClick={() => window.open('/help', '_blank')}
            iconName="HelpCircle"
            iconPosition="left"
            iconSize={16}
          >
            Need Help?
          </Button>
        </div>
      </main>
    </div>
  );
};

export default ImageUploadAnalysis;