// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api/v1';

// Base API service class
class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  // Helper method to make HTTP requests
  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    // Remove authentication token logic - application is now public

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      return { success: true, data };
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      return { 
        success: false, 
        error: error.message,
        message: 'Network error. Please check your connection and try again.'
      };
    }
  }
}

// Authentication Service removed - application is now public

// User Service
class UserService extends ApiService {
  async getUserData() {
    return this.request('/user/profile');
  }

  async updateUserProfile(userData) {
    return this.request('/user/profile', {
      method: 'PUT',
      body: JSON.stringify(userData),
    });
  }
}

// Analysis Service
class AnalysisService extends ApiService {
  async analyzeImage(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    return this.request('/predict', {
      method: 'POST',
      body: formData,
      headers: {}, // Remove Content-Type to let browser set it for FormData
    });
  }

  async getAnalysisHistory(limit = 50, userId = null) {
    const params = new URLSearchParams();
    if (limit) params.append('limit', limit);
    if (userId) params.append('user_id', userId);
    
    const endpoint = `/predictions/history${params.toString() ? '?' + params.toString() : ''}`;
    return this.request(endpoint);
  }

  async getAnalysisById(predictionId) {
    return this.request(`/predictions/${predictionId}`);
  }

  async saveAnalysisToHistory(analysisData) {
    return this.request('/predictions', {
      method: 'POST',
      body: JSON.stringify(analysisData),
    });
  }

  async getModelStatus() {
    return this.request('/model-status');
  }

  async getModelInfo() {
    return this.request('/model-info');
  }

  async getStatistics() {
    return this.request('/statistics');
  }

  // Helper method to construct image URL for a prediction
  getImageUrl(predictionId) {
    if (!predictionId) return null;
    return `${this.baseURL}/images/${predictionId}`;
  }
}

// Report Service
class ReportService extends ApiService {
  async generatePDF(analysisResults) {
    const response = await this.request('/reports/generate-pdf', {
      method: 'POST',
      body: JSON.stringify(analysisResults),
    });
    
    if (response.success) {
      return {
        ...response,
        downloadUrl: `${this.baseURL}/reports/download/${response.data.reportId}`,
        filename: response.data.filename,
      };
    }
    
    return response;
  }

  async downloadReport(reportId) {
    return this.request(`/reports/download/${reportId}`);
  }
}

// Share Service
class ShareService extends ApiService {
  async createShareableLink(analysisResults) {
    const response = await this.request('/share/create', {
      method: 'POST',
      body: JSON.stringify(analysisResults),
    });
    
    if (response.success) {
      return {
        ...response,
        shareUrl: `${window.location.origin}/shared/${response.data.shareId}`,
      };
    }
    
    return response;
  }

  async getSharedAnalysis(shareId) {
    return this.request(`/share/${shareId}`);
  }
}

// Export Service
class ExportService extends ApiService {
  async exportHistoryData(historyData, format = 'csv') {
    try {
      const url = `${this.baseURL}/export/history`;
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data: historyData, format }),
      });
      
      if (!response.ok) {
        throw new Error(`Export failed: ${response.statusText}`);
      }
      
      // Get filename from Content-Disposition header
      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = `patient_history_${new Date().toISOString().slice(0, 10)}.${format}`;
      
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename=([^;]+)/);
        if (filenameMatch) {
          filename = filenameMatch[1].replace(/"/g, '');
        }
      }
      
      // Get the file blob
      const blob = await response.blob();
      
      // Create download link
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename;
      
      // Trigger download
      document.body.appendChild(link);
      link.click();
      
      // Cleanup
      document.body.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);
      
      return {
        success: true,
        filename: filename,
        message: 'Export completed successfully'
      };
      
    } catch (error) {
      console.error('Export failed:', error);
      return {
        success: false,
        error: error.message,
        message: 'Export failed. Please try again.'
      };
    }
  }
}

// Statistics Service
class StatisticsService extends ApiService {
  async getStatistics() {
    return this.request('/statistics/overview');
  }

  async getUserStatistics() {
    return this.request('/statistics/user');
  }
}

// Health Service
class HealthService extends ApiService {
  async checkHealth() {
    return this.request('/health');
  }

  async checkModelStatus() {
    return this.request('/model-status');
  }
}

// Admin Service
class AdminService extends ApiService {
  async getOverview() {
    return this.request('/admin/overview');
  }
}

// Create service instances
export const userService = new UserService();
export const analysisService = new AnalysisService();
export const reportService = new ReportService();
export const shareService = new ShareService();
export const exportService = new ExportService();
export const statisticsService = new StatisticsService();
export const healthService = new HealthService();
export const adminService = new AdminService();

// Export default API service for custom requests
export default ApiService;
