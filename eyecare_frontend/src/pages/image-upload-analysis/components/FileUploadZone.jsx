import React, { useCallback, useState } from 'react';
import Icon from '../../../components/AppIcon';
import Button from '../../../components/ui/Button';

const FileUploadZone = ({ onFileUpload, isUploading }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadError, setUploadError] = useState('');

  const validateFile = (file) => {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp'];
    const maxSize = 10 * 1024 * 1024; // 10MB

    if (!allowedTypes?.includes(file?.type)) {
      return 'Please upload a valid image file (JPEG, PNG, or BMP)';
    }

    if (file?.size > maxSize) {
      return 'File size must be less than 10MB';
    }

    return null;
  };

  const processFile = (file) => {
    const error = validateFile(file);
    if (error) {
      setUploadError(error);
      return;
    }

    setUploadError('');
    
    // Create file preview and detect image dimensions
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const fileData = {
          file,
          name: file?.name,
          size: (file?.size / 1024 / 1024)?.toFixed(2) + ' MB',
          format: file?.type?.split('/')?.[1]?.toUpperCase(),
          dimensions: `${img.width} × ${img.height}`,
          uploadTime: new Date()?.toLocaleString(),
          preview: e?.target?.result
        };
        onFileUpload(fileData);
      };
      img.onerror = () => {
        // Fallback if image dimension detection fails
        const fileData = {
          file,
          name: file?.name,
          size: (file?.size / 1024 / 1024)?.toFixed(2) + ' MB',
          format: file?.type?.split('/')?.[1]?.toUpperCase(),
          dimensions: 'Unknown',
          uploadTime: new Date()?.toLocaleString(),
          preview: e?.target?.result
        };
        onFileUpload(fileData);
      };
      img.src = e?.target?.result;
    };
    reader?.readAsDataURL(file);
  };

  const handleDrop = useCallback((e) => {
    e?.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e?.dataTransfer?.files);
    if (files?.length > 0) {
      processFile(files?.[0]);
    }
  }, []);

  const handleDragOver = useCallback((e) => {
    e?.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e?.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleFileSelect = (e) => {
    const files = Array.from(e?.target?.files);
    if (files?.length > 0) {
      processFile(files?.[0]);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div
        className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 ${
          isDragOver
            ? 'border-primary bg-primary bg-opacity-5' :'border-border hover:border-primary hover:bg-muted'
        } ${isUploading ? 'pointer-events-none opacity-50' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        {isUploading ? (
          <div className="flex flex-col items-center">
            <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4"></div>
            <h3 className="text-lg font-heading font-semibold text-foreground mb-2">
              Uploading Image...
            </h3>
            <p className="text-sm font-body text-muted-foreground">
              Please wait while we process your file
            </p>
          </div>
        ) : (
          <>
            <div className="w-20 h-20 flex items-center justify-center mx-auto mb-4">
              <img 
                src="/assets/eyecenter-logo.png" 
                alt="EyeCenter Logo" 
                className="w-full h-full object-contain"
                onError={(e) => {
                  // Fallback to icon if image fails to load
                  e.target.style.display = 'none';
                  e.target.nextElementSibling.style.display = 'block';
                }}
              />
              <div className="w-16 h-16 bg-primary bg-opacity-10 rounded-full flex items-center justify-center" style={{display: 'none'}}>
                <Icon name="Upload" size={32} color="var(--color-primary)" />
              </div>
            </div>
            
            <h3 className="text-lg font-heading font-semibold text-foreground mb-2">
              Upload Eye Image for Analysis
            </h3>
            
            <p className="text-sm font-body text-muted-foreground mb-6">
              Drag and drop your eye image here, or click to browse files
            </p>
            
            <div className="space-y-4">
              <Button
                variant="default"
                onClick={() => document.getElementById('file-input')?.click()}
                iconName="FolderOpen"
                iconPosition="left"
                iconSize={16}
              >
                Choose File
              </Button>
              
              <input
                id="file-input"
                type="file"
                accept="image/jpeg,image/jpg,image/png,image/bmp"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>
            
            <div className="mt-6 pt-6 border-t border-border">
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-body text-muted-foreground">
                <div className="flex items-center justify-center space-x-1">
                  <Icon name="FileImage" size={14} />
                  <span>JPEG, PNG, BMP</span>
                </div>
                <div className="flex items-center justify-center space-x-1">
                  <Icon name="HardDrive" size={14} />
                  <span>Max 10MB</span>
                </div>
                <div className="flex items-center justify-center space-x-1">
                  <Icon name="Shield" size={14} />
                  <span>Secure Upload</span>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
      {uploadError && (
        <div className="mt-4 p-3 bg-error bg-opacity-10 border border-error border-opacity-20 rounded-lg">
          <div className="flex items-center space-x-2">
            <Icon name="AlertCircle" size={16} color="var(--color-error)" />
            <span className="text-sm font-body text-error">{uploadError}</span>
          </div>
        </div>
      )}
      <div className="mt-4 p-4 bg-muted rounded-lg">
        <h4 className="text-sm font-heading font-semibold text-foreground mb-2">
          Image Guidelines:
        </h4>
        <ul className="text-xs font-body text-muted-foreground space-y-1">
          <li>• Ensure the eye is well-lit and clearly visible</li>
          <li>• Avoid blurry or low-resolution images</li>
          <li>• Remove any obstructions like glasses or contact lenses</li>
          <li>• Center the eye in the frame for best results</li>
        </ul>
      </div>
    </div>
  );
};

export default FileUploadZone;