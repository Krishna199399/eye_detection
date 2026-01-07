import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import Header from '../../components/ui/Header';
import Button from '../../components/ui/Button';
import Input from '../../components/ui/Input';
import Icon from '../../components/AppIcon';

const Profile = () => {
  const navigate = useNavigate();
  const { user, updateUser } = useAuth();
  
  const [isEditing, setIsEditing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    phone: '',
    medicalLicense: '',
    facilityName: '',
    facilityAddress: '',
  });

  useEffect(() => {
    document.title = 'My Profile - EyeCenter';
    
    if (user) {
      setFormData({
        firstName: user.first_name || '',
        lastName: user.last_name || '',
        email: user.email || '',
        phone: user.phone || '',
        medicalLicense: user.medical_license || '',
        facilityName: user.facility_name || '',
        facilityAddress: user.facility_address || '',
      });
    }
  }, [user]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleEdit = () => {
    setIsEditing(true);
    setSuccessMessage('');
    setErrorMessage('');
  };

  const handleCancel = () => {
    setIsEditing(false);
    // Reset form data to original user data
    if (user) {
      setFormData({
        firstName: user.first_name || '',
        lastName: user.last_name || '',
        email: user.email || '',
        phone: user.phone || '',
        medicalLicense: user.medical_license || '',
        facilityName: user.facility_name || '',
        facilityAddress: user.facility_address || '',
      });
    }
    setErrorMessage('');
  };

  const handleSave = async () => {
    setIsLoading(true);
    setErrorMessage('');
    setSuccessMessage('');

    try {
      // Simulate API call - In production, call actual API
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Update user context with new data
      const updatedUser = {
        ...user,
        first_name: formData.firstName,
        last_name: formData.lastName,
        phone: formData.phone,
        medical_license: formData.medicalLicense,
        facility_name: formData.facilityName,
        facility_address: formData.facilityAddress,
        full_name: `${formData.firstName} ${formData.lastName}`,
      };
      
      updateUser(updatedUser);
      
      setIsEditing(false);
      setSuccessMessage('Profile updated successfully!');
      
      // Clear success message after 3 seconds
      setTimeout(() => setSuccessMessage(''), 3000);
    } catch (error) {
      console.error('Update profile error:', error);
      setErrorMessage('Failed to update profile. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const getUserInitials = () => {
    const firstName = formData.firstName || '';
    const lastName = formData.lastName || '';
    return `${firstName.charAt(0)}${lastName.charAt(0)}`.toUpperCase() || 'U';
  };

  const isHealthcareProfessional = user?.user_type === 'healthcare_professional';

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
          
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-heading font-bold text-foreground mb-2">
                My Profile
              </h1>
              <p className="text-muted-foreground">
                Manage your account information
              </p>
            </div>
            
            {!isEditing && (
              <Button
                variant="default"
                onClick={handleEdit}
                iconName="Edit"
                iconPosition="left"
              >
                Edit Profile
              </Button>
            )}
          </div>
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

        {/* Profile Card */}
        <div className="bg-card rounded-xl shadow-lg border border-border overflow-hidden">
          {/* Profile Header */}
          <div className="bg-gradient-to-r from-primary to-primary/80 p-8">
            <div className="flex items-center space-x-6">
              <div className="w-24 h-24 bg-white rounded-full flex items-center justify-center text-primary text-3xl font-bold shadow-lg">
                {getUserInitials()}
              </div>
              <div className="text-white">
                <h2 className="text-2xl font-heading font-bold mb-1">
                  {formData.firstName} {formData.lastName}
                </h2>
                <p className="text-primary-foreground/80 mb-2">{formData.email}</p>
                <span className="inline-block px-3 py-1 bg-white/20 rounded-full text-sm font-medium">
                  {isHealthcareProfessional ? 'Healthcare Professional' : 'Patient'}
                </span>
              </div>
            </div>
          </div>

          {/* Profile Content */}
          <div className="p-8 space-y-8">
            {/* Personal Information */}
            <div>
              <h3 className="text-lg font-heading font-semibold text-foreground mb-4 flex items-center">
                <Icon name="User" size={20} className="mr-2 text-primary" />
                Personal Information
              </h3>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    First Name
                  </label>
                  {isEditing ? (
                    <input
                      type="text"
                      name="firstName"
                      value={formData.firstName}
                      onChange={handleChange}
                      className="w-full px-4 py-2.5 border border-input rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors"
                      disabled={isLoading}
                    />
                  ) : (
                    <p className="text-foreground bg-muted px-4 py-2.5 rounded-lg">
                      {formData.firstName || 'Not provided'}
                    </p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    Last Name
                  </label>
                  {isEditing ? (
                    <input
                      type="text"
                      name="lastName"
                      value={formData.lastName}
                      onChange={handleChange}
                      className="w-full px-4 py-2.5 border border-input rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors"
                      disabled={isLoading}
                    />
                  ) : (
                    <p className="text-foreground bg-muted px-4 py-2.5 rounded-lg">
                      {formData.lastName || 'Not provided'}
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Contact Information */}
            <div>
              <h3 className="text-lg font-heading font-semibold text-foreground mb-4 flex items-center">
                <Icon name="Mail" size={20} className="mr-2 text-primary" />
                Contact Information
              </h3>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    Email Address
                  </label>
                  <div className="flex items-center space-x-2">
                    <p className="flex-1 text-foreground bg-muted px-4 py-2.5 rounded-lg">
                      {formData.email}
                    </p>
                    <div className="text-muted-foreground" title="Email cannot be changed">
                      <Icon name="Lock" size={18} />
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Email cannot be changed
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    Phone Number
                  </label>
                  {isEditing ? (
                    <input
                      type="tel"
                      name="phone"
                      value={formData.phone}
                      onChange={handleChange}
                      className="w-full px-4 py-2.5 border border-input rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors"
                      disabled={isLoading}
                      placeholder="+1 (555) 123-4567"
                    />
                  ) : (
                    <p className="text-foreground bg-muted px-4 py-2.5 rounded-lg">
                      {formData.phone || 'Not provided'}
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Professional Information (Healthcare Professionals Only) */}
            {isHealthcareProfessional && (
              <div>
                <h3 className="text-lg font-heading font-semibold text-foreground mb-4 flex items-center">
                  <Icon name="Stethoscope" size={20} className="mr-2 text-primary" />
                  Professional Information
                </h3>
                
                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">
                      Medical License Number
                    </label>
                    {isEditing ? (
                      <input
                        type="text"
                        name="medicalLicense"
                        value={formData.medicalLicense}
                        onChange={handleChange}
                        className="w-full px-4 py-2.5 border border-input rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors"
                        disabled={isLoading}
                      />
                    ) : (
                      <p className="text-foreground bg-muted px-4 py-2.5 rounded-lg">
                        {formData.medicalLicense || 'Not provided'}
                      </p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">
                      Facility/Clinic Name
                    </label>
                    {isEditing ? (
                      <input
                        type="text"
                        name="facilityName"
                        value={formData.facilityName}
                        onChange={handleChange}
                        className="w-full px-4 py-2.5 border border-input rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors"
                        disabled={isLoading}
                      />
                    ) : (
                      <p className="text-foreground bg-muted px-4 py-2.5 rounded-lg">
                        {formData.facilityName || 'Not provided'}
                      </p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-foreground mb-2">
                      Facility Address
                    </label>
                    {isEditing ? (
                      <textarea
                        name="facilityAddress"
                        value={formData.facilityAddress}
                        onChange={handleChange}
                        rows={3}
                        className="w-full px-4 py-2.5 border border-input rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors"
                        disabled={isLoading}
                      />
                    ) : (
                      <p className="text-foreground bg-muted px-4 py-2.5 rounded-lg">
                        {formData.facilityAddress || 'Not provided'}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Account Information */}
            <div>
              <h3 className="text-lg font-heading font-semibold text-foreground mb-4 flex items-center">
                <Icon name="Shield" size={20} className="mr-2 text-primary" />
                Account Information
              </h3>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-2">
                    Account Type
                  </label>
                  <p className="text-foreground bg-muted px-4 py-2.5 rounded-lg capitalize">
                    {user?.user_type?.replace('_', ' ') || 'Not specified'}
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-muted-foreground mb-2">
                    Member Since
                  </label>
                  <p className="text-foreground bg-muted px-4 py-2.5 rounded-lg">
                    {user?.created_at ? new Date(user.created_at).toLocaleDateString('en-US', {
                      year: 'numeric',
                      month: 'long',
                      day: 'numeric'
                    }) : 'Unknown'}
                  </p>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            {isEditing && (
              <div className="flex items-center space-x-4 pt-4 border-t border-border">
                <Button
                  variant="default"
                  onClick={handleSave}
                  loading={isLoading}
                  disabled={isLoading}
                  iconName="Save"
                  iconPosition="left"
                  className="flex-1 md:flex-none"
                >
                  {isLoading ? 'Saving...' : 'Save Changes'}
                </Button>
                <Button
                  variant="outline"
                  onClick={handleCancel}
                  disabled={isLoading}
                  className="flex-1 md:flex-none"
                >
                  Cancel
                </Button>
              </div>
            )}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="mt-8 grid md:grid-cols-2 gap-4">
          <button
            onClick={() => navigate('/settings')}
            className="p-4 bg-card border border-border rounded-lg hover:border-primary transition-colors flex items-center space-x-3"
          >
            <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
              <Icon name="Settings" size={20} className="text-primary" />
            </div>
            <div className="text-left">
              <p className="font-medium text-foreground">Settings</p>
              <p className="text-sm text-muted-foreground">Manage your preferences</p>
            </div>
            <Icon name="ChevronRight" size={20} className="text-muted-foreground ml-auto" />
          </button>

          <button
            onClick={() => navigate('/patient-history')}
            className="p-4 bg-card border border-border rounded-lg hover:border-primary transition-colors flex items-center space-x-3"
          >
            <div className="w-10 h-10 bg-secondary/10 rounded-lg flex items-center justify-center">
              <Icon name="History" size={20} className="text-secondary" />
            </div>
            <div className="text-left">
              <p className="font-medium text-foreground">Analysis History</p>
              <p className="text-sm text-muted-foreground">View past analyses</p>
            </div>
            <Icon name="ChevronRight" size={20} className="text-muted-foreground ml-auto" />
          </button>
        </div>
      </main>
    </div>
  );
};

export default Profile;