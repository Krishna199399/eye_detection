import React, { useState } from 'react';
import Icon from '../../../components/AppIcon';
import Image from '../../../components/AppImage';
import Button from '../../../components/ui/Button';

const ImagePreviewPanel = ({ uploadedImage, onRotate, onZoom, zoomLevel, rotation }) => {
  const [isZoomed, setIsZoomed] = useState(false);

  const handleZoomToggle = () => {
    setIsZoomed(!isZoomed);
    onZoom(!isZoomed);
  };

  const handleRotateLeft = () => {
    onRotate(rotation - 90);
  };

  const handleRotateRight = () => {
    onRotate(rotation + 90);
  };

  if (!uploadedImage) {
    return (
      <div className="bg-card border border-border rounded-lg p-8 h-full flex flex-col items-center justify-center">
        <div className="w-24 h-24 bg-muted rounded-full flex items-center justify-center mb-4">
          <Icon name="Image" size={48} color="var(--color-muted-foreground)" />
        </div>
        <h3 className="text-lg font-heading font-semibold text-foreground mb-2">
          No Image Selected
        </h3>
        <p className="text-sm font-body text-muted-foreground text-center">
          Upload an eye image to begin analysis
        </p>
      </div>
    );
  }

  return (
    <div className="bg-card border border-border rounded-lg overflow-hidden h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-heading font-semibold text-foreground">
              Uploaded Image
            </h3>
            <p className="text-sm font-body text-muted-foreground">
              {uploadedImage?.name} • {uploadedImage?.size}
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleRotateLeft}
              iconName="RotateCcw"
              iconSize={16}
            >
              Rotate
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleRotateRight}
              iconName="RotateCw"
              iconSize={16}
            />
            <Button
              variant="outline"
              size="sm"
              onClick={handleZoomToggle}
              iconName={isZoomed ? "ZoomOut" : "ZoomIn"}
              iconSize={16}
            >
              {isZoomed ? "Zoom Out" : "Zoom In"}
            </Button>
          </div>
        </div>
      </div>
      {/* Image Display */}
      <div className="flex-1 p-4 overflow-hidden">
        <div className="relative w-full h-full bg-muted rounded-lg overflow-hidden">
          <div 
            className={`w-full h-full flex items-center justify-center transition-transform duration-300 ${
              isZoomed ? 'scale-150 cursor-move' : 'scale-100'
            }`}
            style={{ transform: `rotate(${rotation}deg) ${isZoomed ? 'scale(1.5)' : 'scale(1)'}` }}
          >
            <Image
              src={uploadedImage?.preview}
              alt="Eye scan for analysis"
              className="max-w-full max-h-full object-contain"
            />
          </div>
          
          {/* Zoom indicator */}
          {isZoomed && (
            <div className="absolute top-2 right-2 bg-black bg-opacity-75 text-white px-2 py-1 rounded text-xs font-mono">
              150%
            </div>
          )}
        </div>
      </div>
      {/* Image Info */}
      <div className="p-4 border-t border-border bg-muted">
        <div className="grid grid-cols-2 gap-4 text-sm font-body">
          <div>
            <span className="text-muted-foreground">Upload Time:</span>
            <p className="text-foreground font-medium">{uploadedImage?.uploadTime}</p>
          </div>
          <div>
            <span className="text-muted-foreground">File Size:</span>
            <p className="text-foreground font-medium">{uploadedImage?.size}</p>
          </div>
          <div>
            <span className="text-muted-foreground">Dimensions:</span>
            <p className="text-foreground font-medium">{uploadedImage?.dimensions}</p>
          </div>
          <div>
            <span className="text-muted-foreground">Format:</span>
            <p className="text-foreground font-medium">{uploadedImage?.format}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImagePreviewPanel;