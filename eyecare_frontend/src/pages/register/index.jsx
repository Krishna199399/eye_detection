import React, { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import Button from '../../components/ui/Button';
import Icon from '../../components/AppIcon';

const Register = () => {
  const navigate = useNavigate();
  const { register, isAuthenticated } = useAuth();
  
  const [step, setStep] = useState(1); // 1: User Type, 2: Registration Form
  const [formData, setFormData] = useState({
    userType: '', // 'patient' or 'healthcare_professional'
    firstName: '',
    lastName: '',
    email: '',
    phone: '',
    password: '',
    confirmPassword: '',
    medicalLicense: '',
    facilityName: '',
    facilityAddress: '',
  });
  
  const [errors, setErrors] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [apiError, setApiError] = useState('');

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/home-dashboard');
    }
  }, [isAuthenticated, navigate]);

  // Set page title
  useEffect(() => {
    document.title = 'Register - EyeCenter';
  }, []);

  const selectUserType = (type) => {
    setFormData(prev => ({
      ...prev,
      userType: type,
    }));
    setStep(2);
  };

  const validateForm = () => {
    const newErrors = {};

    // Basic fields validation
    if (!formData.firstName.trim()) {
      newErrors.firstName = 'First name is required';
    }

    if (!formData.lastName.trim()) {
      newErrors.lastName = 'Last name is required';
    }

    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid';
    }

    if (!formData.phone.trim()) {
      newErrors.phone = 'Phone number is required';
    } else if (!/^\+?[\d\s-()]+$/.test(formData.phone)) {
      newErrors.phone = 'Phone number is invalid';
    }

    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (formData.password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters';
    }

    if (!formData.confirmPassword) {
      newErrors.confirmPassword = 'Please confirm your password';
    } else if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }

    // Healthcare professional specific validation
    if (formData.userType === 'healthcare_professional') {
      if (!formData.medicalLicense.trim()) {
        newErrors.medicalLicense = 'Medical license is required';
      }
      if (!formData.facilityName.trim()) {
        newErrors.facilityName = 'Facility name is required';
      }
      if (!formData.facilityAddress.trim()) {
        newErrors.facilityAddress = 'Facility address is required';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value,
    }));
    
    // Clear error for this field
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: '',
      }));
    }
    
    // Clear API error
    if (apiError) {
      setApiError('');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setApiError('');

    if (!validateForm()) {
      return;
    }

    setIsLoading(true);

    try {
      // Prepare data for API
      const registrationData = {
        firstName: formData.firstName,
        lastName: formData.lastName,
        email: formData.email,
        phone: formData.phone,
        password: formData.password,
        userType: formData.userType,
        ...(formData.userType === 'healthcare_professional' && {
          medicalLicense: formData.medicalLicense,
          facilityName: formData.facilityName,
          facilityAddress: formData.facilityAddress,
        }),
      };

      const response = await register(registrationData);

      if (response.success) {
        console.log('Registration successful!');
        navigate('/home-dashboard');
      } else {
        setApiError(response.error || 'Registration failed. Please try again.');
      }
    } catch (error) {
      console.error('Registration error:', error);
      setApiError('An unexpected error occurred. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Step 1: User Type Selection
  if (step === 1) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-background to-blue-50 flex items-center justify-center px-4 py-8">
        <div className="w-full max-w-4xl">
          {/* Logo and Title */}
          <div className="text-center mb-12">
            <div className="flex items-center justify-center mb-4">
              <img 
                src="/assets/eyecenter-logo.png" 
                alt="EyeCenter Logo" 
                className="h-20 w-auto"
              />
            </div>
            <h1 className="text-3xl font-heading font-bold text-foreground mb-2">
              Join EyeCenter
            </h1>
            <p className="text-muted-foreground font-body">
              Choose your account type to get started
            </p>
          </div>

          {/* User Type Cards */}
          <div className="grid md:grid-cols-2 gap-6 mb-8">
            {/* Patient Card */}
            <button
              onClick={() => selectUserType('patient')}
              className="bg-card rounded-xl shadow-lg border-2 border-border hover:border-primary p-8 text-left transition-all hover:shadow-xl group"
            >
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-primary transition-colors">
                <Icon name="User" size={24} className="text-primary group-hover:text-white transition-colors" />
              </div>
              <h3 className="text-xl font-heading font-bold text-foreground mb-2">
                Patient Account
              </h3>
              <p className="text-muted-foreground font-body mb-4">
                For individuals seeking eye disease screening and analysis
              </p>
              <ul className="space-y-2 mb-6">
                <li className="flex items-start text-sm text-foreground">
                  <Icon name="Check" size={16} className="text-success mr-2 mt-0.5 flex-shrink-0" />
                  <span>Upload and analyze eye images</span>
                </li>
                <li className="flex items-start text-sm text-foreground">
                  <Icon name="Check" size={16} className="text-success mr-2 mt-0.5 flex-shrink-0" />
                  <span>Access your analysis history</span>
                </li>
                <li className="flex items-start text-sm text-foreground">
                  <Icon name="Check" size={16} className="text-success mr-2 mt-0.5 flex-shrink-0" />
                  <span>Get AI-powered recommendations</span>
                </li>
              </ul>
              <div className="flex items-center text-primary font-medium">
                <span>Select Patient</span>
                <Icon name="ArrowRight" size={16} className="ml-2" />
              </div>
            </button>

            {/* Healthcare Professional Card */}
            <button
              onClick={() => selectUserType('healthcare_professional')}
              className="bg-card rounded-xl shadow-lg border-2 border-border hover:border-secondary p-8 text-left transition-all hover:shadow-xl group"
            >
              <div className="w-12 h-12 bg-emerald-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-secondary transition-colors">
                <Icon name="Stethoscope" size={24} className="text-secondary group-hover:text-white transition-colors" />
              </div>
              <h3 className="text-xl font-heading font-bold text-foreground mb-2">
                Healthcare Professional
              </h3>
              <p className="text-muted-foreground font-body mb-4">
                For doctors, ophthalmologists, and medical practitioners
              </p>
              <ul className="space-y-2 mb-6">
                <li className="flex items-start text-sm text-foreground">
                  <Icon name="Check" size={16} className="text-success mr-2 mt-0.5 flex-shrink-0" />
                  <span>Manage multiple patient analyses</span>
                </li>
                <li className="flex items-start text-sm text-foreground">
                  <Icon name="Check" size={16} className="text-success mr-2 mt-0.5 flex-shrink-0" />
                  <span>Advanced reporting features</span>
                </li>
                <li className="flex items-start text-sm text-foreground">
                  <Icon name="Check" size={16} className="text-success mr-2 mt-0.5 flex-shrink-0" />
                  <span>Professional dashboard access</span>
                </li>
              </ul>
              <div className="flex items-center text-secondary font-medium">
                <span>Select Professional</span>
                <Icon name="ArrowRight" size={16} className="ml-2" />
              </div>
            </button>
          </div>

          {/* Back to Login */}
          <div className="text-center">
            <p className="text-sm text-muted-foreground">
              Already have an account?{' '}
              <Link to="/login" className="text-primary hover:underline font-medium">
                Sign in instead
              </Link>
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Step 2: Registration Form
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-background to-blue-50 flex items-center justify-center px-4 py-8">
      <div className="w-full max-w-2xl">
        {/* Logo and Title */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <img 
              src="/assets/eyecenter-logo.png" 
              alt="EyeCenter Logo" 
              className="h-20 w-auto"
            />
          </div>
          <h1 className="text-3xl font-heading font-bold text-foreground mb-2">
            Create Your Account
          </h1>
          <p className="text-muted-foreground font-body">
            {formData.userType === 'patient' ? 'Patient Registration' : 'Healthcare Professional Registration'}
          </p>
          <button
            onClick={() => setStep(1)}
            className="mt-2 text-sm text-primary hover:underline font-medium inline-flex items-center"
          >
            <Icon name="ArrowLeft" size={14} className="mr-1" />
            Change account type
          </button>
        </div>

        {/* Registration Card */}
        <div className="bg-card rounded-xl shadow-lg border border-border p-8">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* API Error Alert */}
            {apiError && (
              <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4 flex items-start space-x-3">
                <Icon name="AlertCircle" size={20} className="text-destructive mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-destructive">
                    {apiError}
                  </p>
                </div>
              </div>
            )}

            {/* Personal Information */}
            <div>
              <h3 className="text-lg font-heading font-semibold text-foreground mb-4">
                Personal Information
              </h3>
              <div className="grid md:grid-cols-2 gap-4">
                {/* First Name */}
                <div>
                  <label htmlFor="firstName" className="block text-sm font-medium text-foreground mb-2">
                    First Name <span className="text-destructive">*</span>
                  </label>
                  <input
                    type="text"
                    id="firstName"
                    name="firstName"
                    value={formData.firstName}
                    onChange={handleChange}
                    placeholder="John"
                    className={`w-full px-4 py-2.5 border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors ${
                      errors.firstName ? 'border-destructive' : 'border-input'
                    }`}
                    disabled={isLoading}
                  />
                  {errors.firstName && (
                    <p className="mt-1.5 text-sm text-destructive flex items-center">
                      <Icon name="AlertCircle" size={14} className="mr-1" />
                      {errors.firstName}
                    </p>
                  )}
                </div>

                {/* Last Name */}
                <div>
                  <label htmlFor="lastName" className="block text-sm font-medium text-foreground mb-2">
                    Last Name <span className="text-destructive">*</span>
                  </label>
                  <input
                    type="text"
                    id="lastName"
                    name="lastName"
                    value={formData.lastName}
                    onChange={handleChange}
                    placeholder="Doe"
                    className={`w-full px-4 py-2.5 border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors ${
                      errors.lastName ? 'border-destructive' : 'border-input'
                    }`}
                    disabled={isLoading}
                  />
                  {errors.lastName && (
                    <p className="mt-1.5 text-sm text-destructive flex items-center">
                      <Icon name="AlertCircle" size={14} className="mr-1" />
                      {errors.lastName}
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Contact Information */}
            <div>
              <h3 className="text-lg font-heading font-semibold text-foreground mb-4">
                Contact Information
              </h3>
              <div className="space-y-4">
                {/* Email */}
                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-foreground mb-2">
                    Email Address <span className="text-destructive">*</span>
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <Icon name="Mail" size={18} className="text-muted-foreground" />
                    </div>
                    <input
                      type="email"
                      id="email"
                      name="email"
                      value={formData.email}
                      onChange={handleChange}
                      placeholder="john.doe@example.com"
                      className={`w-full pl-10 pr-4 py-2.5 border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors ${
                        errors.email ? 'border-destructive' : 'border-input'
                      }`}
                      disabled={isLoading}
                    />
                  </div>
                  {errors.email && (
                    <p className="mt-1.5 text-sm text-destructive flex items-center">
                      <Icon name="AlertCircle" size={14} className="mr-1" />
                      {errors.email}
                    </p>
                  )}
                </div>

                {/* Phone */}
                <div>
                  <label htmlFor="phone" className="block text-sm font-medium text-foreground mb-2">
                    Phone Number <span className="text-destructive">*</span>
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <Icon name="Phone" size={18} className="text-muted-foreground" />
                    </div>
                    <input
                      type="tel"
                      id="phone"
                      name="phone"
                      value={formData.phone}
                      onChange={handleChange}
                      placeholder="+1 (555) 123-4567"
                      className={`w-full pl-10 pr-4 py-2.5 border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors ${
                        errors.phone ? 'border-destructive' : 'border-input'
                      }`}
                      disabled={isLoading}
                    />
                  </div>
                  {errors.phone && (
                    <p className="mt-1.5 text-sm text-destructive flex items-center">
                      <Icon name="AlertCircle" size={14} className="mr-1" />
                      {errors.phone}
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Healthcare Professional Fields */}
            {formData.userType === 'healthcare_professional' && (
              <div>
                <h3 className="text-lg font-heading font-semibold text-foreground mb-4">
                  Professional Information
                </h3>
                <div className="space-y-4">
                  {/* Medical License */}
                  <div>
                    <label htmlFor="medicalLicense" className="block text-sm font-medium text-foreground mb-2">
                      Medical License Number <span className="text-destructive">*</span>
                    </label>
                    <input
                      type="text"
                      id="medicalLicense"
                      name="medicalLicense"
                      value={formData.medicalLicense}
                      onChange={handleChange}
                      placeholder="Enter your medical license number"
                      className={`w-full px-4 py-2.5 border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors ${
                        errors.medicalLicense ? 'border-destructive' : 'border-input'
                      }`}
                      disabled={isLoading}
                    />
                    {errors.medicalLicense && (
                      <p className="mt-1.5 text-sm text-destructive flex items-center">
                        <Icon name="AlertCircle" size={14} className="mr-1" />
                        {errors.medicalLicense}
                      </p>
                    )}
                  </div>

                  {/* Facility Name */}
                  <div>
                    <label htmlFor="facilityName" className="block text-sm font-medium text-foreground mb-2">
                      Facility/Clinic Name <span className="text-destructive">*</span>
                    </label>
                    <input
                      type="text"
                      id="facilityName"
                      name="facilityName"
                      value={formData.facilityName}
                      onChange={handleChange}
                      placeholder="Enter your facility name"
                      className={`w-full px-4 py-2.5 border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors ${
                        errors.facilityName ? 'border-destructive' : 'border-input'
                      }`}
                      disabled={isLoading}
                    />
                    {errors.facilityName && (
                      <p className="mt-1.5 text-sm text-destructive flex items-center">
                        <Icon name="AlertCircle" size={14} className="mr-1" />
                        {errors.facilityName}
                      </p>
                    )}
                  </div>

                  {/* Facility Address */}
                  <div>
                    <label htmlFor="facilityAddress" className="block text-sm font-medium text-foreground mb-2">
                      Facility Address <span className="text-destructive">*</span>
                    </label>
                    <textarea
                      id="facilityAddress"
                      name="facilityAddress"
                      value={formData.facilityAddress}
                      onChange={handleChange}
                      placeholder="Enter your facility address"
                      rows={3}
                      className={`w-full px-4 py-2.5 border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors ${
                        errors.facilityAddress ? 'border-destructive' : 'border-input'
                      }`}
                      disabled={isLoading}
                    />
                    {errors.facilityAddress && (
                      <p className="mt-1.5 text-sm text-destructive flex items-center">
                        <Icon name="AlertCircle" size={14} className="mr-1" />
                        {errors.facilityAddress}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Security */}
            <div>
              <h3 className="text-lg font-heading font-semibold text-foreground mb-4">
                Security
              </h3>
              <div className="space-y-4">
                {/* Password */}
                <div>
                  <label htmlFor="password" className="block text-sm font-medium text-foreground mb-2">
                    Password <span className="text-destructive">*</span>
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <Icon name="Lock" size={18} className="text-muted-foreground" />
                    </div>
                    <input
                      type={showPassword ? 'text' : 'password'}
                      id="password"
                      name="password"
                      value={formData.password}
                      onChange={handleChange}
                      placeholder="Create a secure password"
                      className={`w-full pl-10 pr-12 py-2.5 border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors ${
                        errors.password ? 'border-destructive' : 'border-input'
                      }`}
                      disabled={isLoading}
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute inset-y-0 right-0 pr-3 flex items-center"
                      disabled={isLoading}
                    >
                      <Icon
                        name={showPassword ? 'EyeOff' : 'Eye'}
                        size={18}
                        className="text-muted-foreground hover:text-foreground transition-colors"
                      />
                    </button>
                  </div>
                  {errors.password && (
                    <p className="mt-1.5 text-sm text-destructive flex items-center">
                      <Icon name="AlertCircle" size={14} className="mr-1" />
                      {errors.password}
                    </p>
                  )}
                </div>

                {/* Confirm Password */}
                <div>
                  <label htmlFor="confirmPassword" className="block text-sm font-medium text-foreground mb-2">
                    Confirm Password <span className="text-destructive">*</span>
                  </label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                      <Icon name="Lock" size={18} className="text-muted-foreground" />
                    </div>
                    <input
                      type={showConfirmPassword ? 'text' : 'password'}
                      id="confirmPassword"
                      name="confirmPassword"
                      value={formData.confirmPassword}
                      onChange={handleChange}
                      placeholder="Confirm your password"
                      className={`w-full pl-10 pr-12 py-2.5 border rounded-lg focus:ring-2 focus:ring-primary focus:border-primary transition-colors ${
                        errors.confirmPassword ? 'border-destructive' : 'border-input'
                      }`}
                      disabled={isLoading}
                    />
                    <button
                      type="button"
                      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                      className="absolute inset-y-0 right-0 pr-3 flex items-center"
                      disabled={isLoading}
                    >
                      <Icon
                        name={showConfirmPassword ? 'EyeOff' : 'Eye'}
                        size={18}
                        className="text-muted-foreground hover:text-foreground transition-colors"
                      />
                    </button>
                  </div>
                  {errors.confirmPassword && (
                    <p className="mt-1.5 text-sm text-destructive flex items-center">
                      <Icon name="AlertCircle" size={14} className="mr-1" />
                      {errors.confirmPassword}
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Terms and Conditions */}
            <div className="bg-muted/30 rounded-lg p-4">
              <label className="flex items-start space-x-3 cursor-pointer">
                <input
                  type="checkbox"
                  required
                  className="w-4 h-4 mt-0.5 rounded border-input text-primary focus:ring-2 focus:ring-primary"
                />
                <span className="text-sm text-foreground">
                  I agree to the{' '}
                  <Link to="/terms" className="text-primary hover:underline font-medium">
                    Terms of Service
                  </Link>{' '}
                  and{' '}
                  <Link to="/privacy" className="text-primary hover:underline font-medium">
                    Privacy Policy
                  </Link>
                </span>
              </label>
            </div>

            {/* Submit Button */}
            <Button
              type="submit"
              variant="default"
              size="lg"
              fullWidth
              loading={isLoading}
              disabled={isLoading}
              className="font-semibold"
            >
              {isLoading ? 'Creating Account...' : 'Create Account'}
            </Button>
          </form>

          {/* Login Link */}
          <div className="mt-6 text-center">
            <p className="text-sm text-muted-foreground">
              Already have an account?{' '}
              <Link to="/login" className="text-primary hover:underline font-medium">
                Sign in instead
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Register;