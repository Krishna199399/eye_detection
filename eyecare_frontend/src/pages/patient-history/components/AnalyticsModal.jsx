import React from 'react';
import Icon from '../../../components/AppIcon';
import Image from '../../../components/AppImage';
import Button from '../../../components/ui/Button';

const AnalyticsModal = ({ isOpen, onClose, historyItem }) => {
  if (!isOpen || !historyItem) return null;

  // Mock data for analytics - in real app, this would come from the API
  const analyticsData = {
    modelPredictions: {
      [historyItem.condition?.toLowerCase() || 'normal']: historyItem.confidence,
      'normal': historyItem.condition?.toLowerCase() === 'normal' ? historyItem.confidence : (100 - historyItem.confidence),
      'cataract': historyItem.condition?.toLowerCase() === 'cataract' ? historyItem.confidence : Math.max(0, 25 - (historyItem.confidence - 75)),
      'glaucoma': historyItem.condition?.toLowerCase() === 'glaucoma' ? historyItem.confidence : Math.max(0, 15 - (historyItem.confidence - 85)),
      'diabetic_retinopathy': historyItem.condition?.toLowerCase() === 'diabetic retinopathy' ? historyItem.confidence : Math.max(0, 10 - (historyItem.confidence - 90)),
    },
    processingMetrics: {
      imageQuality: 'Excellent',
      processingTime: '2.4s',
      modelVersion: 'v2.1.0',
      imageResolution: '1024x1024',
      enhancementApplied: 'Yes'
    },
    riskAssessment: {
      immediate: historyItem.confidence >= 85 && historyItem.condition?.toLowerCase() !== 'normal' ? 'High' : 'Low',
      shortTerm: historyItem.confidence >= 70 ? 'Monitor' : 'Routine',
      longTerm: historyItem.condition?.toLowerCase() !== 'normal' ? 'Follow-up Required' : 'Normal'
    }
  };

  const getConditionColor = (condition) => {
    const colors = {
      'normal': 'text-success',
      'cataract': 'text-warning',
      'glaucoma': 'text-destructive',
      'diabetic retinopathy': 'text-accent',
      'macular_degeneration': 'text-secondary'
    };
    return colors?.[condition?.toLowerCase()] || 'text-muted-foreground';
  };

  const getRiskColor = (risk) => {
    const colors = {
      'High': 'text-destructive bg-destructive/10 border-destructive/20',
      'Medium': 'text-warning bg-warning/10 border-warning/20',
      'Low': 'text-success bg-success/10 border-success/20',
      'Monitor': 'text-warning bg-warning/10 border-warning/20',
      'Routine': 'text-success bg-success/10 border-success/20',
      'Follow-up Required': 'text-warning bg-warning/10 border-warning/20',
      'Normal': 'text-success bg-success/10 border-success/20'
    };
    return colors[risk] || 'text-muted-foreground bg-muted/10 border-muted/20';
  };

  const PredictionBar = ({ label, percentage, isHighlighted }) => (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className={`text-sm font-medium ${isHighlighted ? 'text-primary' : 'text-muted-foreground'}`}>
          {label.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
        </span>
        <span className={`text-sm font-semibold ${isHighlighted ? 'text-primary' : 'text-foreground'}`}>
          {percentage.toFixed(1)}%
        </span>
      </div>
      <div className="w-full bg-muted rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all duration-500 ${
            isHighlighted ? 'bg-primary' : 'bg-muted-foreground/30'
          }`}
          style={{ width: `${Math.max(percentage, 2)}%` }}
        />
      </div>
    </div>
  );

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-card border border-border rounded-lg shadow-modal max-w-5xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
          <div className="flex items-center space-x-3">
            <Icon name="BarChart3" size={24} className="text-primary" />
            <h2 className="text-xl font-semibold text-foreground">Analysis Analytics</h2>
          </div>
          <Button
            variant="ghost"
            size="sm"
            iconName="X"
            onClick={onClose}
          />
        </div>

        {/* Content */}
        <div className="p-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Image and Basic Info */}
            <div className="lg:col-span-1 space-y-4">
              <h3 className="text-lg font-medium text-foreground">Scan Overview</h3>
              
              {/* Image */}
              <div className="aspect-square bg-muted rounded-lg overflow-hidden">
                <Image
                  src={historyItem?.imageUrl}
                  alt={`Eye scan analysis ${historyItem?.id}`}
                  className="w-full h-full object-cover"
                />
              </div>

              {/* Basic Info */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Analysis ID</span>
                  <span className="text-sm font-mono text-foreground">{historyItem?.id?.slice(-8)}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Date</span>
                  <span className="text-sm text-foreground">
                    {new Date(historyItem.date)?.toLocaleDateString()}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Time</span>
                  <span className="text-sm text-foreground">
                    {new Date(historyItem.date)?.toLocaleTimeString()}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Primary Condition</span>
                  <span className={`text-sm font-semibold ${getConditionColor(historyItem?.condition)}`}>
                    {historyItem?.condition}
                  </span>
                </div>
              </div>
            </div>

            {/* Model Predictions */}
            <div className="lg:col-span-1 space-y-4">
              <h3 className="text-lg font-medium text-foreground">Model Predictions</h3>
              <div className="space-y-4">
                {Object.entries(analyticsData.modelPredictions).map(([condition, confidence]) => (
                  <PredictionBar
                    key={condition}
                    label={condition}
                    percentage={confidence}
                    isHighlighted={condition.toLowerCase() === historyItem.condition?.toLowerCase()}
                  />
                ))}
              </div>

              {/* Confidence Meter */}
              <div className="mt-6 p-4 bg-muted/20 rounded-lg">
                <div className="text-center">
                  <div className="text-3xl font-bold text-primary mb-2">
                    {historyItem?.confidence}%
                  </div>
                  <div className="text-sm text-muted-foreground">Overall Confidence</div>
                  <div className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium mt-2 border ${
                    historyItem?.confidence >= 90 ? 'text-success bg-success/10 border-success/20' :
                    historyItem?.confidence >= 70 ? 'text-warning bg-warning/10 border-warning/20' :
                    'text-destructive bg-destructive/10 border-destructive/20'
                  }`}>
                    {historyItem?.confidence >= 90 ? 'High Confidence' :
                     historyItem?.confidence >= 70 ? 'Medium Confidence' : 'Low Confidence'}
                  </div>
                </div>
              </div>
            </div>

            {/* Analytics & Risk Assessment */}
            <div className="lg:col-span-1 space-y-6">
              <div>
                <h3 className="text-lg font-medium text-foreground mb-4">Risk Assessment</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Immediate Risk</span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getRiskColor(analyticsData.riskAssessment.immediate)}`}>
                      {analyticsData.riskAssessment.immediate}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Short-term Action</span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getRiskColor(analyticsData.riskAssessment.shortTerm)}`}>
                      {analyticsData.riskAssessment.shortTerm}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Long-term Plan</span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getRiskColor(analyticsData.riskAssessment.longTerm)}`}>
                      {analyticsData.riskAssessment.longTerm}
                    </span>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-medium text-foreground mb-4">Processing Metrics</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Image Quality</span>
                    <span className="text-sm font-medium text-success">{analyticsData.processingMetrics.imageQuality}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Processing Time</span>
                    <span className="text-sm font-medium text-foreground">{analyticsData.processingMetrics.processingTime}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Model Version</span>
                    <span className="text-sm font-mono text-foreground">{analyticsData.processingMetrics.modelVersion}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Resolution</span>
                    <span className="text-sm font-mono text-foreground">{analyticsData.processingMetrics.imageResolution}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Enhancement</span>
                    <span className={`text-sm font-medium ${analyticsData.processingMetrics.enhancementApplied === 'Yes' ? 'text-success' : 'text-muted-foreground'}`}>
                      {analyticsData.processingMetrics.enhancementApplied}
                    </span>
                  </div>
                </div>
              </div>

              {/* Quick Actions */}
              <div className="space-y-3">
                <h3 className="text-lg font-medium text-foreground">Quick Actions</h3>
                <div className="grid grid-cols-1 gap-2">
                  <Button variant="outline" iconName="Download" iconPosition="left" size="sm" className="w-full justify-start">
                    Download Full Report
                  </Button>
                  <Button variant="outline" iconName="Share" iconPosition="left" size="sm" className="w-full justify-start">
                    Share Analysis
                  </Button>
                  <Button variant="outline" iconName="Eye" iconPosition="left" size="sm" className="w-full justify-start">
                    View Details
                  </Button>
                </div>
              </div>
            </div>
          </div>

          {/* Additional Insights */}
          {historyItem.condition?.toLowerCase() !== 'normal' && (
            <div className="mt-8 p-4 bg-warning/5 border border-warning/20 rounded-lg">
              <div className="flex items-start space-x-3">
                <Icon name="AlertTriangle" size={20} className="text-warning mt-0.5 flex-shrink-0" />
                <div>
                  <h4 className="text-sm font-medium text-warning mb-1">Medical Attention Recommended</h4>
                  <p className="text-sm text-muted-foreground">
                    This analysis detected {historyItem.condition?.toLowerCase()} with {historyItem.confidence}% confidence. 
                    Consider scheduling an appointment with an ophthalmologist for professional evaluation and treatment options.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t border-border bg-muted/20">
          <div className="text-sm text-muted-foreground">
            Generated on {new Date().toLocaleDateString()} • AI-Powered Analysis
          </div>
          <div className="flex items-center space-x-3">
            <Button variant="outline" onClick={onClose}>
              Close
            </Button>
            <Button variant="default" iconName="Download" iconPosition="left">
              Export Analytics
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsModal;
