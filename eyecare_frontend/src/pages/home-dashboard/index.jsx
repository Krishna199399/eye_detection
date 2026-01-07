import React, { useState, useEffect } from 'react';
import Header from '../../components/ui/Header';
import WelcomeHeader from './components/WelcomeHeader';
import UploadArea from './components/UploadArea';
import RecentAnalysisCard from './components/RecentAnalysisCard';
import StatisticsPanel from './components/StatisticsPanel';
import QuickActions from './components/QuickActions';
import { useNavigate } from 'react-router-dom';
import { analysisService, adminService } from '../../services/apiService';

const HomeDashboard = () => {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [userData, setUserData] = useState(null);

  // Recent analyses will be fetched from API
  const [recentAnalyses, setRecentAnalyses] = useState([]);
  const [isLoadingAnalyses, setIsLoadingAnalyses] = useState(true);
  
  // Statistics will be fetched from API
  const [statistics, setStatistics] = useState({});
  const [isLoadingStatistics, setIsLoadingStatistics] = useState(true);
  const [error, setError] = useState(null);
  
  // Overall dashboard loading state
  const [isDashboardLoading, setIsDashboardLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Fetch all dashboard data when component mounts
  useEffect(() => {
    initializeDashboard();
  }, []);
  
  // Initialize dashboard with all data
  const initializeDashboard = async () => {
    setIsDashboardLoading(true);
    setError(null);
    
    try {
      // Fetch both analyses and statistics concurrently
      await Promise.all([
        fetchRecentAnalyses(),
        fetchStatistics()
      ]);
    } catch (error) {
      console.error('Failed to initialize dashboard:', error);
      setError('Failed to load dashboard data. Please try refreshing the page.');
    } finally {
      setIsDashboardLoading(false);
    }
  };
  
  // Manual refresh function
  const handleRefresh = async () => {
    setRefreshing(true);
    await initializeDashboard();
    setRefreshing(false);
  };

  const fetchRecentAnalyses = async () => {
    setIsLoadingAnalyses(true);
    try {
      console.log('🔄 Fetching recent analyses...');
      const response = await analysisService.getAnalysisHistory(5); // Get last 5 analyses
      
      if (response.success && response.data) {
        // Transform API data to match RecentAnalysisCard format
        const transformedAnalyses = response.data.predictions.map(prediction => ({
          id: prediction.prediction_id,
          patientName: `Analysis ${prediction.prediction_id.slice(-8)}`, // Use last 8 chars of ID
          timestamp: new Date(prediction.created_at).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short', 
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
          }),
          condition: prediction.predicted_class?.replace(/_/g, ' ')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' '),
          confidence: Math.round(prediction.confidence),
          status: prediction.predicted_class === 'normal' ? 'healthy' : 'diseased',
          imageUrl: analysisService.getImageUrl(prediction.prediction_id)
        }));
        
        setRecentAnalyses(transformedAnalyses);
        console.log('✅ Recent analyses loaded:', transformedAnalyses.length, 'items');
      } else {
        console.warn('⚠️ No recent analyses available');
        setRecentAnalyses([]);
      }
    } catch (error) {
      console.error('❌ Failed to fetch recent analyses:', error);
      // Don't set error here - let parent handle it
      setRecentAnalyses([]);
      throw error; // Re-throw for parent to handle
    } finally {
      setIsLoadingAnalyses(false);
    }
  };

  const fetchStatistics = async () => {
    setIsLoadingStatistics(true);
    try {
      console.log('🔄 Fetching statistics...');
      const response = await adminService.getOverview();
      
      if (response.success && response.data) {
        // Transform API data to match StatisticsPanel format
        const stats = response.data;
        const transformedStats = {
          totalAnalyses: stats.total_analyses,
          accuracy: Math.round(stats.accuracy),
          patients: stats.patients,
          diseased: stats.diseased_cases,
          healthy: stats.healthy,
          cataracts: stats.cataracts,
          glaucoma: stats.glaucoma,
          diabeticRetinopathy: stats.diabetic_retinopathy,
          monthlyData: stats.monthly_data // Already in correct format
        };
        
        setStatistics(transformedStats);
        console.log('✅ Statistics loaded successfully');
      } else {
        console.warn('⚠️ No statistics available');
        setStatistics({});
      }
    } catch (error) {
      console.error('❌ Failed to fetch statistics:', error);
      // Don't set error here - let parent handle it
      setStatistics({});
      throw error; // Re-throw for parent to handle
    } finally {
      setIsLoadingStatistics(false);
    }
  };
  
  // Set default user data - no authentication required
  useEffect(() => {
    setUserData({ name: 'User' });
  }, []);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setIsLoading(true);
    
    // Simulate processing time
    setTimeout(() => {
      setIsLoading(false);
      // Navigate to analysis page with file data
      navigate('/image-upload-analysis', { 
        state: { uploadedFile: file } 
      });
    }, 1500);
  };

  const handleViewDetails = (analysisId) => {
    navigate('/patient-history', { 
      state: { selectedAnalysis: analysisId } 
    });
  };

  useEffect(() => {
    // Set page title
    document.title = 'Dashboard - EyeDisease Detector';
  }, []);

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container mx-auto px-4 py-6 space-y-6">
        {/* Welcome Header with Refresh Button */}
        <div className="flex items-center justify-between">
          <WelcomeHeader userName={userData?.name || 'User'} />
          <button
            onClick={handleRefresh}
            disabled={refreshing || isDashboardLoading}
            className="flex items-center space-x-2 px-4 py-2 text-sm bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <svg 
              className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" 
              />
            </svg>
            <span>{refreshing ? 'Refreshing...' : 'Refresh'}</span>
          </button>
        </div>
        
        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Recent Analyses */}
          <div className="lg:col-span-2 space-y-6">
            {/* Recent Analyses */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-foreground">
                  Recent Analyses
                </h2>
                <button
                  onClick={() => navigate('/patient-history')}
                  className="text-sm text-primary hover:text-primary/80 transition-colors duration-200"
                >
                  View All
                </button>
              </div>
              
              <div className="space-y-3">
                {isLoadingAnalyses ? (
                  // Loading skeleton for recent analyses
                  [...Array(3)].map((_, index) => (
                    <div key={index} className="bg-card rounded-lg border border-border p-4">
                      <div className="flex items-start space-x-4">
                        <div className="w-16 h-16 rounded-lg bg-muted animate-pulse flex-shrink-0"></div>
                        <div className="flex-1 space-y-2">
                          <div className="h-4 bg-muted rounded animate-pulse w-3/4"></div>
                          <div className="h-3 bg-muted rounded animate-pulse w-1/2"></div>
                          <div className="h-2 bg-muted rounded animate-pulse w-full mt-2"></div>
                          <div className="flex items-center justify-between mt-3">
                            <div className="h-6 bg-muted rounded animate-pulse w-20"></div>
                            <div className="h-6 bg-muted rounded animate-pulse w-16"></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                ) : recentAnalyses?.length > 0 ? (
                  recentAnalyses.slice(0, 3).map((analysis) => (
                    <RecentAnalysisCard
                      key={analysis?.id}
                      analysis={analysis}
                      onViewDetails={handleViewDetails}
                    />
                  ))
                ) : (
                  <div className="bg-card rounded-lg border border-border p-8 text-center">
                    <div className="space-y-3">
                      <div className="w-12 h-12 bg-muted rounded-full flex items-center justify-center mx-auto">
                        <span className="text-muted-foreground text-xl">📊</span>
                      </div>
                      <h3 className="text-lg font-medium text-foreground">No Recent Analyses</h3>
                      <p className="text-sm text-muted-foreground">
                        Upload your first eye scan image to get started with AI-powered analysis.
                      </p>
                      {error && (
                        <p className="text-xs text-red-500 mt-2">
                          Error: {error}
                        </p>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
          
          {/* Right Column - Quick Actions */}
          <div className="space-y-6">
            <QuickActions />
          </div>
        </div>
        
        {/* Statistics Panel */}
        <div className="space-y-4">
          <h2 className="text-xl font-semibold text-foreground">
            Analytics Overview
          </h2>
          {isLoadingStatistics ? (
            <div className="space-y-6">
              {/* Loading skeleton for statistics */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                {[...Array(4)].map((_, index) => (
                  <div key={index} className="bg-card rounded-lg border border-border p-4">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-muted rounded-lg animate-pulse"></div>
                      <div className="space-y-2">
                        <div className="h-6 bg-muted rounded animate-pulse w-12"></div>
                        <div className="h-3 bg-muted rounded animate-pulse w-20"></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-card rounded-lg border border-border p-6">
                  <div className="h-6 bg-muted rounded animate-pulse w-48 mb-4"></div>
                  <div className="h-64 bg-muted rounded animate-pulse"></div>
                </div>
                <div className="bg-card rounded-lg border border-border p-6">
                  <div className="h-6 bg-muted rounded animate-pulse w-48 mb-4"></div>
                  <div className="h-64 bg-muted rounded animate-pulse"></div>
                </div>
              </div>
            </div>
          ) : (
            <StatisticsPanel statistics={statistics} />
          )}
        </div>
        
        {/* Footer Info */}
        <div className="bg-card rounded-lg border border-border p-6 text-center">
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">
              EyeDisease Detector uses advanced machine learning algorithms to assist in early detection of eye conditions.
            </p>
            <p className="text-xs text-muted-foreground">
              This tool is designed to support healthcare professionals and should not replace professional medical diagnosis.
            </p>
            <div className="flex items-center justify-center space-x-4 mt-4 text-xs text-muted-foreground">
              <span>© {new Date()?.getFullYear()} EyeDisease Detector</span>
              <span>•</span>
              <span>HIPAA Compliant</span>
              <span>•</span>
              <span>FDA Approved Algorithm</span>
            </div>
          </div>
        </div>
      </main>
      
      {/* Dashboard Loading Overlay */}
      {isDashboardLoading && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-card rounded-2xl p-8 shadow-2xl max-w-sm w-full mx-4">
            <div className="text-center space-y-4">
              <div className="flex justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary border-t-transparent"></div>
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-semibold text-foreground">Loading Dashboard</h3>
                <p className="text-sm text-muted-foreground">
                  Fetching your latest analyses and statistics...
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Error Message */}
      {error && !isDashboardLoading && (
        <div className="fixed bottom-4 right-4 z-50">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 shadow-lg max-w-sm">
            <div className="flex items-start space-x-3">
              <div className="w-5 h-5 bg-red-500 rounded-full flex-shrink-0 mt-0.5"></div>
              <div className="flex-1">
                <p className="text-sm font-medium text-red-800">Dashboard Error</p>
                <p className="text-xs text-red-600 mt-1">{error}</p>
                <button
                  onClick={() => setError(null)}
                  className="text-xs text-red-700 hover:text-red-800 mt-2 underline"
                >
                  Dismiss
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HomeDashboard;