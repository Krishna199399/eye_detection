import React from 'react';
import { useNavigate } from 'react-router-dom';
import Button from '../../../components/ui/Button';
import Icon from '../../../components/AppIcon';

const QuickActions = () => {
  const navigate = useNavigate();

  const actions = [
    {
      id: 'upload',
      title: 'New Analysis',
      description: 'Upload and analyze eye images',
      icon: 'Upload',
      color: 'bg-primary',
      route: '/image-upload-analysis'
    },
    {
      id: 'history',
      title: 'Patient History',
      description: 'View past analyses and reports',
      icon: 'History',
      color: 'bg-secondary',
      route: '/patient-history'
    }
  ];

  const handleActionClick = (route) => {
    navigate(route);
  };

  return (
    <div className="bg-card rounded-lg border border-border p-6">
      <h3 className="text-lg font-semibold text-foreground mb-4">Quick Actions</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {actions?.map((action) => (
          <div
            key={action?.id}
            className="group cursor-pointer bg-muted/50 rounded-lg p-4 hover:bg-muted transition-all duration-200 hover:shadow-card"
            onClick={() => handleActionClick(action?.route)}
          >
            <div className="flex items-start space-x-3">
              <div className={`w-10 h-10 ${action?.color} rounded-lg flex items-center justify-center group-hover:scale-105 transition-transform duration-200`}>
                <Icon name={action?.icon} size={20} color="white" />
              </div>
              
              <div className="flex-1 min-w-0">
                <h4 className="text-sm font-medium text-foreground group-hover:text-primary transition-colors duration-200">
                  {action?.title}
                </h4>
                <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                  {action?.description}
                </p>
              </div>
              
              <Icon 
                name="ChevronRight" 
                size={16} 
                className="text-muted-foreground group-hover:text-primary group-hover:translate-x-1 transition-all duration-200" 
              />
            </div>
          </div>
        ))}
      </div>
      <div className="mt-6 pt-4 border-t border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Icon name="Shield" size={16} color="var(--color-success)" />
            <span className="text-xs text-muted-foreground">HIPAA Compliant Platform</span>
          </div>
          
          <Button
            variant="ghost"
            size="sm"
            iconName="HelpCircle"
            iconPosition="left"
            onClick={() => navigate('/help')}
            className="text-xs"
          >
            Help & Support
          </Button>
        </div>
      </div>
    </div>
  );
};

export default QuickActions;