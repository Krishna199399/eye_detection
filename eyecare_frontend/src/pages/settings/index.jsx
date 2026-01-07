import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import Header from '../../components/ui/Header';
import Button from '../../components/ui/Button';
import Icon from '../../components/AppIcon';

const Settings = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  
  const [settings, setSettings] = useState({
    // Notifications
    emailNotifications: true,
    analysisNotifications: true,
    marketingEmails: false,
    
    // Privacy
    shareAnalysisData: false,
    anonymousUsage: true,
    
    // Display
    darkMode: false,
    language: 'en',
  });

  const [passwordData, setPasswordData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  });

  const [showPasswordForm, setShowPasswordForm] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  useEffect(() => {
    document.title = 'Settings - EyeCenter';
    
    // Load settings from localStorage
    const savedSettings = localStorage.getItem('userSettings');
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings));
    }
  }, []);

  const handleToggle = (setting) => {
    const newSettings = {
      ...settings,
      [setting]: !settings[setting],
    };
    setSettings(newSettings);
    localStorage.setItem('userSettings', JSON.stringify(newSettings));
    
    setSuccessMessage('Settings updated successfully');
    setTimeout(() => setSuccessMessage(''), 2000);
  };

  const handlePasswordChange = (e) => {
    const { name, value } = e.target;
    setPasswordData(prev => ({
      ...prev,
      [name]: value,
    }));
  };

  const handlePasswordSubmit = async (e) => {
    e.preventDefault();
    setErrorMessage('');
    setSuccessMessage('');

    // Validation
    if (passwordData.newPassword.length < 6) {
      setErrorMessage('New password must be at least 6 characters');
      return;
    }

    if (passwordData.newPassword !== passwordData.confirmPassword) {
      setErrorMessage('Passwords do not match');
      return;
    }

    setIsLoading(true);

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setSuccessMessage('Password changed successfully');
      setPasswordData({
        currentPassword: '',
        newPassword: '',
        confirmPassword: '',
      });
      setShowPasswordForm(false);
      
      setTimeout(() => setSuccessMessage(''), 3000);
    } catch (error) {
      setErrorMessage('Failed to change password. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteAccount = async () => {
    setIsLoading(true);
    setErrorMessage('');

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Logout and redirect
      await logout();
      navigate('/login');
    } catch (error) {
      setErrorMessage('Failed to delete account. Please try again.');
      setIsLoading(false);
    }
  };

  const ToggleSwitch = ({ checked, onChange, label, description }) => (
    <div className="flex items-center justify-between py-4">
      <div className="flex-1">
        <p className="font-medium text-foreground">{label}</p>
        {description && (
          <p className="text-sm text-muted-foreground mt-0.5">{description}</p>
        )}
      </div>
      <button
        onClick={onChange}
        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 ${
          checked ? 'bg-primary' : 'bg-gray-300'
        }`}
      >
        <span
          className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
            checked ? 'translate-x-6' : 'translate-x-1'
          }`}
        />
      </button>
    </div>
  );

  return (
    <div className="min-h-screen bg-background">
      <Header />
      
      <main className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Page Header */}
        <div className="mb-8">
          <button
            onClick={() => navigate(-1)}
            className="flex items-center text-muted-foreground hover:text-foreground mb-4 transition-colors"
          >
            <Icon name="ArrowLeft" size={20} className="mr-2" />
            <span className="text-sm font-medium">Back</span>
          </button>
          
          <h1 className="text-3xl font-heading font-bold text-foreground mb-2">
            Settings
          </h1>
          <p className="text-muted-foreground">
            Manage your account preferences and security
          </p>
        </div>

        {/* Success Message */}
        {successMessage && (
          <div className="mb-6 bg-success/10 border border-success/20 rounded-lg p-4 flex items-start space-x-3">
            <Icon name="CheckCircle" size={20} className="text-success mt-0.5 flex-shrink-0" />
            <p className="text-sm font-medium text-success">{successMessage}</p>
          </div>
        )}

        {/* Error Message */}
        {errorMessage && (
          <div className="mb-6 bg-destructive/10 border border-destructive/20 rounded-lg p-4 flex items-start space-x-3">
            <Icon name="AlertCircle" size={20} className="text-destructive mt-0.5 flex-shrink-0" />
            <p className="text-sm font-medium text-destructive">{errorMessage}</p>
          </div>
        )}

        {/* Notifications Section */}
        <div className="bg-card rounded-xl shadow-lg border border-border p-6 mb-6">
          <div className="flex items-center mb-6">
            <Icon name="Bell" size={24} className="text-primary mr-3" />
            <div>
              <h2 className="text-xl font-heading font-semibold text-foreground">
                Notifications
              </h2>
              <p className="text-sm text-muted-foreground">
                Manage how you receive notifications
              </p>
            </div>
          </div>

          <div className="space-y-1 divide-y divide-border">
            <ToggleSwitch
              checked={settings.emailNotifications}
              onChange={() => handleToggle('emailNotifications')}
              label="Email Notifications"
              description="Receive notifications via email"
            />
            <ToggleSwitch
              checked={settings.analysisNotifications}
              onChange={() => handleToggle('analysisNotifications')}
              label="Analysis Notifications"
              description="Get notified when analysis is complete"
            />
            <ToggleSwitch
              checked={settings.marketingEmails}
              onChange={() => handleToggle('marketingEmails')}
              label="Marketing Emails"
              description="Receive updates about new features and offers"
            />
          </div>
        </div>

        {/* Privacy Section */}
        <div className="bg-card rounded-xl shadow-lg border border-border p-6 mb-6">
          <div className="flex items-center mb-6">
            <Icon name="Shield" size={24} className="text-primary mr-3" />
            <div>
              <h2 className="text-xl font-heading font-semibold text-foreground">
                Privacy & Data
              </h2>
              <p className="text-sm text-muted-foreground">
                Control your data and privacy settings
              </p>
            </div>
          </div>

          <div className="space-y-1 divide-y divide-border">
            <ToggleSwitch
              checked={settings.shareAnalysisData}
              onChange={() => handleToggle('shareAnalysisData')}
              label="Share Analysis Data"
              description="Help improve our AI by sharing anonymized analysis data"
            />
            <ToggleSwitch
              checked={settings.anonymousUsage}
              onChange={() => handleToggle('anonymousUsage')}
              label="Anonymous Usage Statistics"
              description="Allow collection of anonymous usage data"
            />
          </div>
        </div>

        {/* Security Section */}
        <div className="bg-card rounded-xl shadow-lg border border-border p-6 mb-6">
          <div className="flex items-center mb-6">
            <Icon name="Lock" size={24} className="text-primary mr-3" />
            <div>
              <h2 className="text-xl font-heading font-semibold text-foreground">
                Security
              </h2>
              <p className="text-sm text-muted-foreground">
                Manage your account security
              </p>
            </div>
          </div>

          {/* Change Password */}
          <div className="space-y-4">
            {!showPasswordForm ? (
              <div className="flex items-center justify-between py-4 border-b border-border">
                <div>
                  <p className="font-medium text-foreground">Password</p>
                  <p className="text-sm text-muted-foreground">
                    Last changed: Never
                  </p>
                </div>
                <Button
                  variant="outline"
                  onClick={() => setShowPasswordForm(true)}
                  iconName="Edit"
                  size="sm"
                >
                  Change
                </Button>
              </div>
            ) : (
              <form onSubmit={handlePasswordSubmit} className="space-y-4 py-4 border-b border-border">
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    Current Password
                  </label>
                  <input
                    type="password"
                    name="currentPassword"
                    value={passwordData.currentPassword}
                    onChange={handlePasswordChange}
                    className="w-full px-4 py-2.5 border border-input rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors"
                    required
                    disabled={isLoading}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    New Password
                  </label>
                  <input
                    type="password"
                    name="newPassword"
                    value={passwordData.newPassword}
                    onChange={handlePasswordChange}
                    className="w-full px-4 py-2.5 border border-input rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors"
                    required
                    disabled={isLoading}
                    minLength={6}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    Confirm New Password
                  </label>
                  <input
                    type="password"
                    name="confirmPassword"
                    value={passwordData.confirmPassword}
                    onChange={handlePasswordChange}
                    className="w-full px-4 py-2.5 border border-input rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors"
                    required
                    disabled={isLoading}
                  />
                </div>

                <div className="flex items-center space-x-3">
                  <Button
                    type="submit"
                    variant="default"
                    loading={isLoading}
                    disabled={isLoading}
                    size="sm"
                  >
                    Update Password
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => {
                      setShowPasswordForm(false);
                      setPasswordData({
                        currentPassword: '',
                        newPassword: '',
                        confirmPassword: '',
                      });
                      setErrorMessage('');
                    }}
                    disabled={isLoading}
                    size="sm"
                  >
                    Cancel
                  </Button>
                </div>
              </form>
            )}

            {/* Two-Factor Authentication */}
            <div className="flex items-center justify-between py-4">
              <div>
                <p className="font-medium text-foreground">Two-Factor Authentication</p>
                <p className="text-sm text-muted-foreground">
                  Add an extra layer of security
                </p>
              </div>
              <Button
                variant="outline"
                iconName="Shield"
                size="sm"
                onClick={() => setSuccessMessage('Two-factor authentication coming soon!')}
              >
                Enable
              </Button>
            </div>
          </div>
        </div>

        {/* Danger Zone */}
        <div className="bg-card rounded-xl shadow-lg border-2 border-destructive/20 p-6">
          <div className="flex items-center mb-6">
            <Icon name="AlertTriangle" size={24} className="text-destructive mr-3" />
            <div>
              <h2 className="text-xl font-heading font-semibold text-destructive">
                Danger Zone
              </h2>
              <p className="text-sm text-muted-foreground">
                Irreversible actions
              </p>
            </div>
          </div>

          {!showDeleteConfirm ? (
            <div className="flex items-center justify-between py-4">
              <div>
                <p className="font-medium text-foreground">Delete Account</p>
                <p className="text-sm text-muted-foreground">
                  Permanently delete your account and all data
                </p>
              </div>
              <Button
                variant="destructive"
                onClick={() => setShowDeleteConfirm(true)}
                iconName="Trash2"
                size="sm"
              >
                Delete
              </Button>
            </div>
          ) : (
            <div className="p-4 bg-destructive/10 rounded-lg border border-destructive/30">
              <p className="font-medium text-destructive mb-4">
                Are you sure you want to delete your account?
              </p>
              <p className="text-sm text-foreground mb-4">
                This action cannot be undone. All your data, analysis history, and settings will be permanently deleted.
              </p>
              <div className="flex items-center space-x-3">
                <Button
                  variant="destructive"
                  onClick={handleDeleteAccount}
                  loading={isLoading}
                  disabled={isLoading}
                  size="sm"
                >
                  Yes, Delete My Account
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setShowDeleteConfirm(false)}
                  disabled={isLoading}
                  size="sm"
                >
                  Cancel
                </Button>
              </div>
            </div>
          )}
        </div>

        {/* Quick Links */}
        <div className="mt-8 flex flex-wrap gap-4 justify-center text-sm text-muted-foreground">
          <button className="hover:text-primary transition-colors">Privacy Policy</button>
          <span>•</span>
          <button className="hover:text-primary transition-colors">Terms of Service</button>
          <span>•</span>
          <button className="hover:text-primary transition-colors">Help Center</button>
        </div>
      </main>
    </div>
  );
};

export default Settings;