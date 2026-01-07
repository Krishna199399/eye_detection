/**
 * Authentication Service
 * Handles all authentication-related API calls
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

class AuthService {
  /**
   * Register a new user
   * @param {Object} userData - User registration data
   * @returns {Promise<Object>} Registration response
   */
  async register(userData) {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(userData),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Registration failed');
      }

      // Store token and user data in localStorage
      if (data.token) {
        localStorage.setItem('authToken', data.token);
        localStorage.setItem('userData', JSON.stringify(data.user));
      }

      return {
        success: true,
        data: data,
      };
    } catch (error) {
      console.error('Registration error:', error);
      return {
        success: false,
        error: error.message || 'Registration failed. Please try again.',
      };
    }
  }

  /**
   * Login user
   * @param {Object} credentials - Login credentials (email, password)
   * @returns {Promise<Object>} Login response
   */
  async login(credentials) {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Login failed');
      }

      // Store token and user data in localStorage
      if (data.token) {
        localStorage.setItem('authToken', data.token);
        localStorage.setItem('userData', JSON.stringify(data.user));
      }

      return {
        success: true,
        data: data,
      };
    } catch (error) {
      console.error('Login error:', error);
      return {
        success: false,
        error: error.message || 'Login failed. Please check your credentials.',
      };
    }
  }

  /**
   * Logout user
   * @returns {Promise<Object>} Logout response
   */
  async logout() {
    try {
      const token = this.getToken();
      
      if (token) {
        // Call backend logout endpoint
        await fetch(`${API_BASE_URL}/auth/logout`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
          },
        });
      }

      // Clear local storage
      localStorage.removeItem('authToken');
      localStorage.removeItem('userData');

      return {
        success: true,
        message: 'Logged out successfully',
      };
    } catch (error) {
      console.error('Logout error:', error);
      // Still clear local storage even if API call fails
      localStorage.removeItem('authToken');
      localStorage.removeItem('userData');
      
      return {
        success: true,
        message: 'Logged out successfully',
      };
    }
  }

  /**
   * Get stored auth token
   * @returns {string|null} Auth token
   */
  getToken() {
    return localStorage.getItem('authToken');
  }

  /**
   * Get stored user data
   * @returns {Object|null} User data
   */
  getUser() {
    const userData = localStorage.getItem('userData');
    return userData ? JSON.parse(userData) : null;
  }

  /**
   * Check if user is authenticated
   * @returns {boolean} Authentication status
   */
  isAuthenticated() {
    const token = this.getToken();
    const user = this.getUser();
    return !!(token && user);
  }

  /**
   * Check if user exists (for validation during registration)
   * @param {string} email - Email to check
   * @returns {Promise<Object>} Check result
   */
  async checkUserExists(email) {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/check-user/${encodeURIComponent(email)}`);
      const data = await response.json();
      
      return {
        success: true,
        exists: data.exists || false,
      };
    } catch (error) {
      console.error('Check user error:', error);
      return {
        success: false,
        error: error.message,
      };
    }
  }

  /**
   * Get authorization headers for API requests
   * @returns {Object} Headers with authorization token
   */
  getAuthHeaders() {
    const token = this.getToken();
    return {
      'Content-Type': 'application/json',
      ...(token && { 'Authorization': `Bearer ${token}` }),
    };
  }

  /**
   * Update user data in localStorage
   * @param {Object} userData - Updated user data
   */
  updateUser(userData) {
    localStorage.setItem('userData', JSON.stringify(userData));
  }

  /**
   * Verify token validity
   * @returns {Promise<boolean>} Token validity status
   */
  async verifyToken() {
    const token = this.getToken();
    
    if (!token) {
      return false;
    }

    try {
      // Try to make an authenticated request to verify token
      const response = await fetch(`${API_BASE_URL}/health`, {
        headers: this.getAuthHeaders(),
      });

      return response.ok;
    } catch (error) {
      console.error('Token verification error:', error);
      return false;
    }
  }
}

// Export singleton instance
const authService = new AuthService();
export default authService;