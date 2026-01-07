import React from 'react';
import Icon from '../../../components/AppIcon';
import Image from '../../../components/AppImage';
import Button from '../../../components/ui/Button';

const RecentAnalysisCard = ({ analysis, onViewDetails }) => {
  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy':
        return 'text-success bg-success/10';
      case 'diseased':
        return 'text-destructive bg-destructive/10';
      default:
        return 'text-warning bg-warning/10';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy':
        return 'CheckCircle';
      case 'diseased':
        return 'AlertTriangle';
      default:
        return 'Clock';
    }
  };

  return (
    <div className="bg-card rounded-lg border border-border p-4 hover:shadow-elevated transition-all duration-200">
      <div className="flex items-start space-x-4">
        <div className="w-16 h-16 rounded-lg overflow-hidden bg-muted flex-shrink-0">
          <Image
            src={analysis?.imageUrl}
            alt="Eye scan"
            className="w-full h-full object-cover"
          />
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between">
            <div className="space-y-1">
              <h4 className="text-sm font-medium text-foreground truncate">
                {analysis?.patientName}
              </h4>
              <p className="text-xs text-muted-foreground">
                {analysis?.timestamp}
              </p>
            </div>
            
            <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(analysis?.status)}`}>
              <Icon name={getStatusIcon(analysis?.status)} size={12} className="mr-1" />
              {analysis?.status === 'healthy' ? 'Healthy' : analysis?.condition}
            </div>
          </div>
          
          <div className="mt-2 space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">Confidence</span>
              <span className="font-medium text-foreground">{analysis?.confidence}%</span>
            </div>
            <div className="w-full bg-muted rounded-full h-1.5">
              <div
                className="bg-primary h-1.5 rounded-full transition-all duration-300"
                style={{ width: `${analysis?.confidence}%` }}
              />
            </div>
          </div>
          
          <div className="mt-3 flex items-center justify-between">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onViewDetails(analysis?.id)}
              iconName="Eye"
              iconPosition="left"
              className="text-xs"
            >
              View Details
            </Button>
            
            <Button
              variant="ghost"
              size="sm"
              iconName="Download"
              iconSize={14}
              className="text-xs"
            >
              Report
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecentAnalysisCard;