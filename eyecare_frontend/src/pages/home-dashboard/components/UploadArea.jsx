import React, { useState, useRef } from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const UploadArea = ({ onFileSelect }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e?.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e?.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e?.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e?.dataTransfer?.files);
    const validFiles = files?.filter(file => 
      file?.type === 'image/jpeg' || file?.type === 'image/png'
    );
    
    if (validFiles?.length > 0) {
      onFileSelect(validFiles?.[0]);
    }
  };

  const handleFileInput = (e) => {
    const file = e?.target?.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  };

  const handleBrowseClick = () => {
    fileInputRef?.current?.click();
  };

  return (
    <div className="bg-card rounded-lg border-2 border-dashed border-border p-8 text-center transition-all duration-200 hover:border-primary/50">
      <div
        className={`relative ${isDragOver ? 'border-primary bg-primary/5' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png"
          onChange={handleFileInput}
          className="hidden"
        />
        
        <div className="flex flex-col items-center space-y-4">
          <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center">
            <Icon name="Upload" size={32} color="var(--color-primary)" />
          </div>
          
          <div className="space-y-2">
            <h3 className="text-lg font-semibold text-foreground">
              Upload Eye Image for Analysis
            </h3>
            <p className="text-sm text-muted-foreground max-w-md">
              Drag and drop your eye image here, or click to browse. 
              Supported formats: JPEG, PNG (Max size: 10MB)
            </p>
          </div>
          
          <Button
            variant="outline"
            onClick={handleBrowseClick}
            iconName="FolderOpen"
            iconPosition="left"
            className="mt-4"
          >
            Browse Files
          </Button>
          
          <div className="flex items-center space-x-4 text-xs text-muted-foreground">
            <div className="flex items-center space-x-1">
              <Icon name="Shield" size={14} />
              <span>HIPAA Compliant</span>
            </div>
            <div className="flex items-center space-x-1">
              <Icon name="Lock" size={14} />
              <span>Secure Upload</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UploadArea;