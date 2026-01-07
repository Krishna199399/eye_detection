import React, { useState } from 'react';
import Icon from '../../../components/AppIcon';
import Image from '../../../components/AppImage';
import Button from '../../../components/ui/Button';
import jsPDF from 'jspdf';

const DetailsModal = ({ isOpen, onClose, historyItem, onCompareAnalysis }) => {
  const [isDownloading, setIsDownloading] = useState(false);
  const [isSharing, setIsSharing] = useState(false);
  
  if (!isOpen || !historyItem) return null;

  const getConditionColor = (condition) => {
    const colors = {
      'healthy': 'text-success',
      'cataracts': 'text-warning',
      'glaucoma': 'text-destructive',
      'diabetic_retinopathy': 'text-accent',
      'macular_degeneration': 'text-secondary'
    };
    return colors?.[condition] || 'text-muted-foreground';
  };

  const getConfidenceLevel = (confidence) => {
    if (confidence >= 90) return { level: 'High', color: 'text-success' };
    if (confidence >= 70) return { level: 'Medium', color: 'text-warning' };
    return { level: 'Low', color: 'text-destructive' };
  };

  const confidenceInfo = getConfidenceLevel(historyItem?.confidence);

  // PDF Report Generation Function
  const generatePDFReport = (item) => {
    try {
      const doc = new jsPDF();
      const pageWidth = doc.internal.pageSize.getWidth();
      const pageHeight = doc.internal.pageSize.getHeight();
      let currentY = 25;
      const margin = 20;
      const lineHeight = 7;
      const sectionSpacing = 18;

      // Helper function for rounded rectangles
      const addRoundedRect = (x, y, width, height, radius = 3) => {
        doc.roundedRect(x, y, width, height, radius, radius);
      };

      // Set default font
      doc.setFont('helvetica');

      // Header with gradient effect
      const addGradientHeader = (x, y, width, height) => {
        const steps = 20;
        const stepHeight = height / steps;
        for (let i = 0; i < steps; i++) {
          const ratio = i / steps;
          const blue = Math.floor(255 * (1 - ratio * 0.3));
          const green = Math.floor(123 * (1 + ratio * 0.5));
          doc.setFillColor(0, green, blue);
          doc.rect(x, y + i * stepHeight, width, stepHeight, 'F');
        }
      };

      // Header Background
      addGradientHeader(0, 0, pageWidth, 65);
      
      // Company Logo
      doc.setFontSize(28);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(255, 255, 255);
      doc.text('EyeDisease', pageWidth / 2, currentY, { align: 'center' });
      
      currentY += 10;
      doc.setFontSize(22);
      doc.setTextColor(240, 248, 255);
      doc.text('DETECTOR', pageWidth / 2, currentY, { align: 'center' });
      
      currentY += 12;
      doc.setFontSize(16);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(255, 255, 255);
      doc.text('Eye Disease Analysis Report', pageWidth / 2, currentY, { align: 'center' });
      
      currentY += 6;
      doc.setFontSize(11);
      doc.setTextColor(220, 230, 255);
      doc.text('AI-Powered Diagnostic Analysis System', pageWidth / 2, currentY, { align: 'center' });
      
      currentY = 80;

      // Patient Information Section
      doc.setFillColor(248, 249, 250);
      addRoundedRect(margin - 5, currentY - 3, pageWidth - 2 * margin + 10, 45);
      doc.rect(margin - 5, currentY - 3, pageWidth - 2 * margin + 10, 45, 'F');
      
      doc.setFontSize(14);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(33, 37, 41);
      doc.text('📋 PATIENT INFORMATION', margin, currentY + 5);
      
      currentY += 15;
      
      const cardWidth = (pageWidth - 2 * margin - 10) / 2;
      const cardHeight = 25;
      
      // Left card - Analysis Info
      doc.setFillColor(255, 255, 255);
      doc.setDrawColor(220, 223, 230);
      doc.setLineWidth(0.5);
      addRoundedRect(margin, currentY, cardWidth, cardHeight);
      doc.rect(margin, currentY, cardWidth, cardHeight, 'FD');
      
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(108, 117, 125);
      doc.text('ANALYSIS ID', margin + 5, currentY + 8);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(33, 37, 41);
      doc.text(item?.id?.slice(-12) || 'N/A', margin + 5, currentY + 15);
      
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(108, 117, 125);
      doc.text('FILENAME', margin + 5, currentY + 20);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(33, 37, 41);
      const filename = item?.filename || 'N/A';
      const truncatedFilename = filename.length > 15 ? filename.substring(0, 15) + '...' : filename;
      doc.text(truncatedFilename, margin + 5, currentY + 27);
      
      // Right card - Date Info
      doc.setFillColor(255, 255, 255);
      addRoundedRect(margin + cardWidth + 5, currentY, cardWidth, cardHeight);
      doc.rect(margin + cardWidth + 5, currentY, cardWidth, cardHeight, 'FD');
      
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(108, 117, 125);
      doc.text('ANALYSIS DATE', margin + cardWidth + 10, currentY + 8);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(33, 37, 41);
      const analysisDate = new Date(item?.date).toLocaleDateString('en-US', { 
        year: 'numeric', month: 'short', day: 'numeric', 
        hour: '2-digit', minute: '2-digit'
      });
      doc.text(analysisDate, margin + cardWidth + 10, currentY + 15);
      
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(108, 117, 125);
      doc.text('REPORT GENERATED', margin + cardWidth + 10, currentY + 20);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(33, 37, 41);
      const reportDate = new Date().toLocaleDateString('en-US', { 
        year: 'numeric', month: 'short', day: 'numeric', 
        hour: '2-digit', minute: '2-digit'
      });
      doc.text(reportDate, margin + cardWidth + 10, currentY + 27);
      
      currentY += cardHeight + sectionSpacing;

      // Analysis Results Section
      const condition = item?.condition || 'Unknown';
      const confidence = item?.confidence || 0;
      
      doc.setFontSize(14);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(33, 37, 41);
      doc.text('🎯 ANALYSIS RESULTS', margin, currentY);
      currentY += 12;
      
      const resultBoxHeight = 50;
      let borderColor, fillColor, textColor;
      
      if (condition.toLowerCase() === 'normal' || condition.toLowerCase() === 'healthy') {
        borderColor = [40, 167, 69];
        fillColor = [212, 237, 218];
        textColor = [21, 87, 36];
      } else if (confidence >= 85) {
        borderColor = [220, 53, 69];
        fillColor = [248, 215, 218];
        textColor = [114, 28, 36];
      } else if (confidence >= 65) {
        borderColor = [255, 193, 7];
        fillColor = [255, 243, 205];
        textColor = [133, 100, 4];
      } else {
        borderColor = [108, 117, 125];
        fillColor = [248, 249, 250];
        textColor = [73, 80, 87];
      }
      
      doc.setFillColor(...fillColor);
      doc.setDrawColor(...borderColor);
      doc.setLineWidth(2);
      addRoundedRect(margin, currentY, pageWidth - 2 * margin, resultBoxHeight);
      doc.rect(margin, currentY, pageWidth - 2 * margin, resultBoxHeight, 'FD');
      
      let conditionIcon = '🔍';
      if (condition.toLowerCase().includes('cataract')) conditionIcon = '👁️';
      else if (condition.toLowerCase().includes('glaucoma')) conditionIcon = '⚠️';
      else if (condition.toLowerCase().includes('diabetic')) conditionIcon = '🩺';
      else if (condition.toLowerCase() === 'normal' || condition.toLowerCase() === 'healthy') conditionIcon = '✅';
      
      currentY += 12;
      doc.setFontSize(16);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(...textColor);
      doc.text(`${conditionIcon} ${condition.toUpperCase()}`, margin + 10, currentY);
      
      currentY += 10;
      doc.setFontSize(12);
      doc.setFont('helvetica', 'normal');
      doc.text(`Confidence Level: ${confidence}%`, margin + 10, currentY);
      
      const barWidth = 80;
      const barHeight = 6;
      const barX = margin + 10;
      const barY = currentY + 3;
      
      doc.setFillColor(233, 236, 239);
      doc.rect(barX, barY, barWidth, barHeight, 'F');
      
      const fillWidth = (confidence / 100) * barWidth;
      doc.setFillColor(...borderColor);
      doc.rect(barX, barY, fillWidth, barHeight, 'F');
      
      currentY += 15;
      const riskAssessment = confidence >= 90 ? 'HIGH CONFIDENCE' : 
                            confidence >= 70 ? 'MEDIUM CONFIDENCE' : 
                            confidence >= 50 ? 'LOW CONFIDENCE' : 'VERY LOW CONFIDENCE';
      
      doc.setFontSize(10);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(108, 117, 125);
      doc.text('RISK ASSESSMENT:', margin + 10, currentY);
      
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(...textColor);
      doc.text(riskAssessment, margin + 65, currentY);
      
      currentY += 20;

      // Medical Disclaimer
      if (currentY > pageHeight - 60) {
        doc.addPage();
        currentY = 25;
      }
      
      doc.setFillColor(255, 243, 205);
      doc.setDrawColor(255, 193, 7);
      doc.setLineWidth(2);
      const disclaimerHeight = 35;
      addRoundedRect(margin, currentY, pageWidth - 2 * margin, disclaimerHeight);
      doc.rect(margin, currentY, pageWidth - 2 * margin, disclaimerHeight, 'FD');
      
      currentY += 10;
      doc.setFontSize(11);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(133, 100, 4);
      doc.text('⚠️ IMPORTANT MEDICAL DISCLAIMER', margin + 8, currentY);
      
      currentY += 6;
      doc.setFontSize(9);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(114, 28, 36);
      
      const disclaimerText = 'This AI-generated analysis is designed to assist healthcare professionals and should never replace professional medical diagnosis, examination, or treatment. Always consult with a qualified ophthalmologist or healthcare provider for proper medical evaluation and treatment decisions.';
      const splitDisclaimer = doc.splitTextToSize(disclaimerText, pageWidth - 2 * margin - 16);
      
      splitDisclaimer.forEach((line) => {
        doc.text(line, margin + 8, currentY);
        currentY += 4;
      });
      
      // Footer
      const footerY = pageHeight - 20;
      doc.setFontSize(8);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(108, 117, 125);
      doc.text('EyeDisease Detector™ - Advanced AI-Powered Eye Disease Detection System', pageWidth / 2, footerY, { align: 'center' });
      
      // Save PDF
      const timestamp = new Date().toISOString().slice(0, 19).replace(/[:.]/g, '-');
      const pdfFilename = `Eye_Analysis_Report_${(item?.id?.slice(-8)) || 'Unknown'}_${timestamp}.pdf`;
      
      doc.save(pdfFilename);
      
      console.log('PDF report generated successfully:', pdfFilename);
      
    } catch (error) {
      console.error('Failed to generate PDF report:', error);
      throw error;
    }
  };

  // Handle Download Report
  const handleDownloadReport = async () => {
    setIsDownloading(true);
    try {
      await generatePDFReport(historyItem);
    } catch (error) {
      console.error('Download failed:', error);
      alert('Failed to download report. Please try again.');
    } finally {
      setIsDownloading(false);
    }
  };

  // Handle Share functionality
  const handleShare = async () => {
    setIsSharing(true);
    try {
      const shareData = {
        title: 'Eye Disease Analysis Report',
        text: `Analysis Report for ${historyItem?.condition?.replace('_', ' ')?.replace(/\b\w/g, l => l?.toUpperCase())} - Confidence: ${historyItem?.confidence}%`,
        url: window.location.href
      };

      if (navigator.share && navigator.canShare && navigator.canShare(shareData)) {
        await navigator.share(shareData);
      } else {
        // Fallback: Copy to clipboard
        const shareText = `Eye Disease Analysis Report\n\nCondition: ${historyItem?.condition?.replace('_', ' ')?.replace(/\b\w/g, l => l?.toUpperCase())}\nConfidence: ${historyItem?.confidence}%\nDate: ${new Date(historyItem?.date).toLocaleDateString()}\nAnalysis ID: ${historyItem?.id}`;
        
        await navigator.clipboard.writeText(shareText);
        alert('Analysis details copied to clipboard!');
      }
    } catch (error) {
      console.error('Share failed:', error);
      // Final fallback - show share modal with copy option
      const shareText = `Eye Disease Analysis Report\n\nCondition: ${historyItem?.condition?.replace('_', ' ')?.replace(/\b\w/g, l => l?.toUpperCase())}\nConfidence: ${historyItem?.confidence}%\nDate: ${new Date(historyItem?.date).toLocaleDateString()}\nAnalysis ID: ${historyItem?.id}`;
      
      if (confirm('Share feature unavailable. Copy analysis details to clipboard?')) {
        try {
          await navigator.clipboard.writeText(shareText);
          alert('Analysis details copied to clipboard!');
        } catch (clipboardError) {
          // Show the text in a prompt for manual copying
          prompt('Please copy this text manually:', shareText);
        }
      }
    } finally {
      setIsSharing(false);
    }
  };

  // Handle Compare functionality
  const handleCompare = () => {
    if (onCompareAnalysis) {
      onCompareAnalysis(historyItem);
    } else {
      // Fallback - open analytics modal or comparison view
      console.log('Opening comparison for:', historyItem?.id);
      alert(`Comparison feature will open analytics for analysis ID: ${historyItem?.id}`);
    }
    onClose(); // Close the details modal
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-card border border-border rounded-lg shadow-modal max-w-4xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-border">
          <h2 className="text-xl font-semibold text-foreground">Analysis Details</h2>
          <Button
            variant="ghost"
            size="sm"
            iconName="X"
            onClick={onClose}
          />
        </div>

        {/* Content */}
        <div className="p-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Image Section */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium text-foreground">Original Image</h3>
              <div className="aspect-square bg-muted rounded-lg overflow-hidden">
                <Image
                  src={historyItem?.imageUrl}
                  alt={`Eye scan analysis ${historyItem?.id}`}
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="flex items-center justify-between text-sm text-muted-foreground">
                <span>Image ID: {historyItem?.id}</span>
                <span>Size: {historyItem?.imageSize || '1024x1024'}</span>
              </div>
            </div>

            {/* Analysis Results */}
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-foreground mb-4">Analysis Results</h3>
                
                {/* Primary Diagnosis */}
                <div className="bg-muted/20 rounded-lg p-4 mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-muted-foreground">Primary Diagnosis</span>
                    <span className={`text-sm font-medium ${confidenceInfo?.color}`}>
                      {confidenceInfo?.level} Confidence
                    </span>
                  </div>
                  <div className="flex items-center space-x-3">
                    <span className={`text-xl font-semibold ${getConditionColor(historyItem?.condition)}`}>
                      {historyItem?.condition?.replace('_', ' ')?.replace(/\b\w/g, l => l?.toUpperCase())}
                    </span>
                    <span className="text-2xl font-bold text-foreground">
                      {historyItem?.confidence}%
                    </span>
                  </div>
                </div>

                {/* Analysis Metadata */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between py-2 border-b border-border">
                    <span className="text-sm font-medium text-muted-foreground">Analysis Date</span>
                    <span className="text-sm text-foreground">
                      {new Date(historyItem.date)?.toLocaleDateString('en-US', {
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </span>
                  </div>
                  
                  <div className="flex items-center justify-between py-2 border-b border-border">
                    <span className="text-sm font-medium text-muted-foreground">Processing Time</span>
                    <span className="text-sm text-foreground">{historyItem?.processingTime || '2.3s'}</span>
                  </div>
                  
                  <div className="flex items-center justify-between py-2 border-b border-border">
                    <span className="text-sm font-medium text-muted-foreground">Model Version</span>
                    <span className="text-sm text-foreground">{historyItem?.modelVersion || 'v2.1.0'}</span>
                  </div>
                  
                  <div className="flex items-center justify-between py-2 border-b border-border">
                    <span className="text-sm font-medium text-muted-foreground">Image Quality</span>
                    <span className="text-sm text-success">{historyItem?.imageQuality || 'Excellent'}</span>
                  </div>
                </div>

                {/* Additional Findings */}
                {historyItem?.additionalFindings && (
                  <div className="mt-6">
                    <h4 className="text-md font-medium text-foreground mb-3">Additional Findings</h4>
                    <div className="bg-muted/20 rounded-lg p-4">
                      <ul className="space-y-2">
                        {historyItem?.additionalFindings?.map((finding, index) => (
                          <li key={index} className="flex items-start space-x-2">
                            <Icon name="ChevronRight" size={16} className="text-muted-foreground mt-0.5 flex-shrink-0" />
                            <span className="text-sm text-foreground">{finding}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}

                {/* Notes */}
                {historyItem?.notes && (
                  <div className="mt-6">
                    <h4 className="text-md font-medium text-foreground mb-3">Notes</h4>
                    <div className="bg-muted/20 rounded-lg p-4">
                      <p className="text-sm text-foreground">{historyItem?.notes}</p>
                    </div>
                  </div>
                )}

                {/* Recommendations */}
                {historyItem?.recommendations && (
                  <div className="mt-6">
                    <h4 className="text-md font-medium text-foreground mb-3">Recommendations</h4>
                    <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
                      <ul className="space-y-2">
                        {historyItem?.recommendations?.map((recommendation, index) => (
                          <li key={index} className="flex items-start space-x-2">
                            <Icon name="AlertCircle" size={16} className="text-primary mt-0.5 flex-shrink-0" />
                            <span className="text-sm text-foreground">{recommendation}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between p-6 border-t border-border bg-muted/20">
          <div className="text-sm text-muted-foreground">
            Last updated: {new Date(historyItem.date)?.toLocaleDateString()}
          </div>
          <div className="flex items-center space-x-3">
            <Button
              variant="outline"
              iconName="Download"
              iconPosition="left"
              onClick={handleDownloadReport}
              disabled={isDownloading}
            >
              {isDownloading ? 'Generating...' : 'Download Report'}
            </Button>
            <Button
              variant="outline"
              iconName="Share"
              iconPosition="left"
              onClick={handleShare}
              disabled={isSharing}
            >
              {isSharing ? 'Sharing...' : 'Share'}
            </Button>
            <Button
              variant="default"
              iconName="BarChart3"
              iconPosition="left"
              onClick={handleCompare}
            >
              Compare
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DetailsModal;