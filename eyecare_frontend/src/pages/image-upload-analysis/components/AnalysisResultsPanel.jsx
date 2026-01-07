import React from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const AnalysisResultsPanel = ({ 
  analysisResults, 
  isLoading, 
  onGeneratePDF, 
  onShareResults, 
  onSaveToHistory 
}) => {
  if (isLoading) {
    return (
      <div className="bg-card border border-border rounded-lg p-8 h-full flex flex-col items-center justify-center">
        <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4"></div>
        <h3 className="text-lg font-heading font-semibold text-foreground mb-2">
          Analyzing Image...
        </h3>
        <p className="text-sm font-body text-muted-foreground text-center mb-4">
          Our AI is examining your eye image for potential conditions
        </p>
        <div className="w-full max-w-xs bg-muted rounded-full h-2">
          <div className="bg-primary h-2 rounded-full animate-pulse" style={{ width: '65%' }}></div>
        </div>
        <p className="text-xs font-body text-muted-foreground mt-2">
          Estimated time: 15-30 seconds
        </p>
      </div>
    );
  }

  if (!analysisResults) {
    return (
      <div className="bg-card border border-border rounded-lg p-8 h-full flex flex-col items-center justify-center">
        <div className="w-24 h-24 bg-muted rounded-full flex items-center justify-center mb-4">
          <Icon name="Brain" size={48} color="var(--color-muted-foreground)" />
        </div>
        <h3 className="text-lg font-heading font-semibold text-foreground mb-2">
          Ready for Analysis
        </h3>
        <p className="text-sm font-body text-muted-foreground text-center">
          Upload an image to see AI-powered diagnostic results
        </p>
      </div>
    );
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 90) return 'text-success';
    if (confidence >= 70) return 'text-warning';
    return 'text-error';
  };

  const getConfidenceBgColor = (confidence) => {
    if (confidence >= 90) return 'bg-success';
    if (confidence >= 70) return 'bg-warning';
    return 'bg-error';
  };

  // Extract data from backend response format
  const results = analysisResults?.results || {};
  const predictedClass = results.predicted_class || 'Unknown';
  const confidence = results.confidence || 0;
  const allPredictions = results.all_predictions || {};
  const recommendations = results.recommendations || [];
  const riskLevel = results.risk_level || 'Unknown';
  const timestamp = analysisResults?.timestamp || new Date().toLocaleString();

  // Format predicted class for display
  const formatClassName = (className) => {
    if (!className) return 'Unknown';
    return className
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const displayClassName = formatClassName(predictedClass);
  const isHealthy = predictedClass === 'normal';

  return (
    <div className="bg-card border border-border rounded-lg overflow-hidden h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-heading font-semibold text-foreground">
              Analysis Results
            </h3>
            <p className="text-sm font-body text-muted-foreground">
              AI Model v2.1.3 • {new Date(timestamp).toLocaleString()}
            </p>
          </div>
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
            isHealthy ? 'bg-success bg-opacity-10 text-success' : 'bg-error bg-opacity-10 text-error'
          }`}>
            {displayClassName}
          </div>
        </div>
      </div>
      {/* Results Content */}
      <div className="flex-1 p-4 overflow-y-auto">
        {/* Overall Classification */}
        <div className="mb-6">
          <h4 className="text-md font-heading font-semibold text-foreground mb-3">
            Overall Assessment
          </h4>
          <div className="bg-muted rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-body text-muted-foreground">Predicted Condition</span>
              <span className="text-sm font-medium text-foreground">
                {displayClassName}
              </span>
            </div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-body text-muted-foreground">Confidence Level</span>
              <span className={`text-sm font-medium ${getConfidenceColor(confidence)}`}>
                {Math.round(confidence)}%
              </span>
            </div>
            <div className="w-full bg-border rounded-full h-2 mb-3">
              <div 
                className={`h-2 rounded-full ${getConfidenceBgColor(confidence)}`}
                style={{ width: `${confidence}%` }}
              ></div>
            </div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-body text-muted-foreground">Risk Level</span>
              <span className={`text-sm font-medium ${
                riskLevel === 'Low' ? 'text-success' : 
                riskLevel === 'Medium' ? 'text-warning' : 'text-error'
              }`}>
                {riskLevel}
              </span>
            </div>
          </div>
        </div>

        {/* Specific Conditions */}
        <div className="mb-6">
          <h4 className="text-md font-heading font-semibold text-foreground mb-3">
            Condition Analysis
          </h4>
          <div className="space-y-3">
            {Object.entries(allPredictions).map(([conditionKey, probability], index) => {
              const conditionName = formatClassName(conditionKey);
              const isMainCondition = conditionKey === predictedClass;
              const probabilityPercent = Math.round(probability);
              
              return (
                <div key={index} className={`rounded-lg p-4 ${
                  isMainCondition ? 'bg-primary bg-opacity-10 border border-primary border-opacity-20' : 'bg-muted'
                }`}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <Icon 
                        name={isMainCondition ? "AlertTriangle" : "Info"} 
                        size={16} 
                        color={isMainCondition ? "var(--color-primary)" : "var(--color-muted-foreground)"}
                      />
                      <span className={`text-sm font-medium ${
                        isMainCondition ? 'text-primary' : 'text-foreground'
                      }`}>
                        {conditionName}
                        {isMainCondition && ' (Predicted)'}
                      </span>
                    </div>
                    <span className={`text-sm font-medium ${
                      isMainCondition ? 'text-primary' : 'text-muted-foreground'
                    }`}>
                      {probabilityPercent}%
                    </span>
                  </div>
                  <div className="w-full bg-border rounded-full h-1.5">
                    <div 
                      className={`h-1.5 rounded-full ${
                        isMainCondition ? 'bg-primary' : 'bg-muted-foreground'
                      }`}
                      style={{ width: `${probability}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Recommendations */}
        <div className="mb-6">
          <h4 className="text-md font-heading font-semibold text-foreground mb-3">
            Recommendations
          </h4>
          <div className="bg-muted rounded-lg p-4">
            {recommendations.length > 0 ? (
              <ul className="space-y-2">
                {recommendations.map((recommendation, index) => (
                  <li key={index} className="flex items-start space-x-2">
                    <Icon name="ArrowRight" size={14} color="var(--color-primary)" className="mt-0.5" />
                    <span className="text-sm font-body text-foreground">{recommendation}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm font-body text-muted-foreground text-center py-4">
                No specific recommendations available.
              </p>
            )}
          </div>
        </div>
      </div>
      {/* Action Buttons */}
      <div className="p-4 border-t border-border bg-muted">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <Button
            variant="default"
            onClick={onGeneratePDF}
            iconName="Download"
            iconPosition="left"
            iconSize={16}
            className="w-full"
          >
            Generate PDF
          </Button>
          <Button
            variant="outline"
            onClick={onShareResults}
            iconName="Share"
            iconPosition="left"
            iconSize={16}
            className="w-full"
          >
            Share Results
          </Button>
          <Button
            variant="secondary"
            onClick={onSaveToHistory}
            iconName="Save"
            iconPosition="left"
            iconSize={16}
            className="w-full"
          >
            Save to History
          </Button>
        </div>
      </div>
    </div>
  );
};

export default AnalysisResultsPanel;