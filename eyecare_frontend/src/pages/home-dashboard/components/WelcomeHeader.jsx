import React from 'react';
import Icon from '../../../components/AppIcon';

const WelcomeHeader = ({ userName = "Dr. Sarah Johnson" }) => {
  const getCurrentGreeting = () => {
    const hour = new Date()?.getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 17) return 'Good afternoon';
    return 'Good evening';
  };

  const getCurrentDate = () => {
    const options = { 
      weekday: 'long', 
      year: 'numeric', 
      month: 'long', 
      day: 'numeric' 
    };
    return new Date()?.toLocaleDateString('en-US', options);
  };

  return (
    <div className="bg-gradient-to-r from-primary to-primary/80 rounded-lg p-6 text-white">
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <h1 className="text-2xl font-semibold">
            {getCurrentGreeting()}, {userName}
          </h1>
          <p className="text-primary-foreground/80 text-sm">
            {getCurrentDate()}
          </p>
          <p className="text-primary-foreground/90 text-sm max-w-md">
            Welcome to your EyeDisease Detector dashboard. Ready to help patients with advanced AI-powered eye disease detection.
          </p>
        </div>
        
        <div className="hidden md:flex items-center space-x-4">
          <div className="bg-white/10 rounded-lg p-3">
            <Icon name="Eye" size={32} color="white" />
          </div>
        </div>
      </div>
      
      <div className="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white/10 rounded-lg p-3">
          <div className="flex items-center space-x-2">
            <Icon name="Clock" size={16} color="white" />
            <span className="text-sm font-medium">24/7 Available</span>
          </div>
          <p className="text-xs text-primary-foreground/80 mt-1">
            Instant analysis anytime
          </p>
        </div>
        
        <div className="bg-white/10 rounded-lg p-3">
          <div className="flex items-center space-x-2">
            <Icon name="Shield" size={16} color="white" />
            <span className="text-sm font-medium">Secure & Private</span>
          </div>
          <p className="text-xs text-primary-foreground/80 mt-1">
            HIPAA compliant platform
          </p>
        </div>
        
        <div className="bg-white/10 rounded-lg p-3">
          <div className="flex items-center space-x-2">
            <Icon name="Zap" size={16} color="white" />
            <span className="text-sm font-medium">AI Powered</span>
          </div>
          <p className="text-xs text-primary-foreground/80 mt-1">
            Advanced ML detection
          </p>
        </div>
      </div>
    </div>
  );
};

export default WelcomeHeader;