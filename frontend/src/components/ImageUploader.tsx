import React, { useState, useRef } from 'react';
import { UploadCloud, ArrowRight, Loader2, FileCheck, X, Scan } from 'lucide-react';

interface ImageUploaderProps {
  onAnalyze: (file: File) => void;
  isLoading: boolean;
  onReset: () => void;
  selectedFile: File | null;
  setSelectedFile: (file: File | null) => void;
  previewUrl: string | null;
  setPreviewUrl: (url: string | null) => void;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({
  onAnalyze,
  isLoading,
  onReset,
  selectedFile,
  setSelectedFile,
  previewUrl,
  setPreviewUrl,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please select a valid image file (PNG, JPG, WEBP, BMP).');
      return;
    }
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    onReset();
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <section id="analyze" className="bg-white border border-[#E6E1DA] rounded-2xl p-6 sm:p-8 shadow-xs space-y-6">
      
      {/* Header section with Eyebrow and Step pill */}
      <div className="flex items-start justify-between gap-4 border-b border-[#E6E1DA] pb-5">
        <div className="space-y-1">
          <p className="text-xs font-mono tracking-widest text-[#6B7280] uppercase font-semibold">
            ANALYZE AN IMAGE
          </p>
          <h2 className="text-xl sm:text-2xl font-extrabold text-[#111827] tracking-tight">
            Evaluate a prediction, then its reliability
          </h2>
          <p className="text-xs sm:text-sm text-[#6B7280]">
            Evaluate a computer-vision prediction and estimate whether it should be trusted.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          {selectedFile && (
            <button
              onClick={handleClear}
              disabled={isLoading}
              className="text-xs text-[#6B7280] hover:text-[#111827] flex items-center gap-1 bg-[#FAF8F5] hover:bg-[#F3EFEA] px-3 py-1.5 rounded-lg border border-[#E6E1DA] transition font-medium"
            >
              <X className="h-3.5 w-3.5" />
              Reset
            </button>
          )}
          <span className="text-xs font-mono font-medium px-3 py-1.5 rounded-full bg-[#FAF8F5] text-[#6B7280] border border-[#E6E1DA] shrink-0">
            STEP 01
          </span>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/png, image/jpeg, image/webp, image/bmp"
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
          }
        }}
      />

      {/* Main Upload Box */}
      {!selectedFile ? (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`bg-grid-pattern border-2 border-dashed rounded-xl p-8 sm:p-12 text-center transition-all duration-200 ${
            isDragOver
              ? 'border-[#1B4332] bg-[#E8F5E9]/30 scale-[1.002]'
              : 'border-[#D8D2C7] bg-[#FAF8F5]/40 hover:bg-[#FAF8F5]/80'
          }`}
        >
          <div className="mx-auto h-12 w-12 rounded-xl bg-white border border-[#E6E1DA] flex items-center justify-center mb-4 text-[#1B4332] shadow-xs">
            <UploadCloud className="h-6 w-6" />
          </div>

          <p className="text-base font-bold text-[#111827]">
            Drag & drop an image here
          </p>
          <p className="text-xs text-[#6B7280] mt-1 mb-5">
            PNG, JPG or WEBP — up to 10 MB
          </p>

          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="inline-flex items-center space-x-2 bg-white hover:bg-[#FAF8F5] text-[#111827] border border-[#E6E1DA] px-4 py-2 rounded-lg text-xs font-semibold shadow-xs transition"
          >
            <Scan className="h-3.5 w-3.5 text-[#6B7280]" />
            <span>Browse files</span>
          </button>
        </div>
      ) : (
        <div className="bg-[#FAF8F5] rounded-xl border border-[#E6E1DA] p-4 flex flex-col sm:flex-row items-center gap-4">
          {previewUrl && (
            <div className="relative h-28 w-28 rounded-lg overflow-hidden border border-[#E6E1DA] bg-white shrink-0 shadow-xs">
              <img
                src={previewUrl}
                alt="Selected Preview"
                className="h-full w-full object-cover"
              />
            </div>
          )}
          <div className="flex-1 min-w-0 space-y-1 text-center sm:text-left">
            <div className="flex items-center justify-center sm:justify-start gap-2">
              <FileCheck className="h-4 w-4 text-[#1B4332] shrink-0" />
              <p className="text-sm font-bold text-[#111827] truncate">{selectedFile.name}</p>
            </div>
            <p className="text-xs text-[#6B7280]">{formatFileSize(selectedFile.size)}</p>
            <p className="text-xs text-[#9CA3AF] font-mono">Format: {selectedFile.type || 'image/png'}</p>
          </div>
        </div>
      )}

      {/* Action Footer */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2">
        <button
          onClick={() => selectedFile && onAnalyze(selectedFile)}
          disabled={!selectedFile || isLoading}
          className={`w-full sm:w-auto px-8 py-3 rounded-xl font-bold text-sm flex items-center justify-center space-x-2 transition-all shadow-xs ${
            !selectedFile || isLoading
              ? 'bg-[#F3EFEA] text-[#9CA3AF] border border-[#E6E1DA] cursor-not-allowed'
              : 'bg-[#1B4332] hover:bg-[#143326] text-white active:scale-[0.995]'
          }`}
        >
          {isLoading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin text-white" />
              <span>Analyzing image...</span>
            </>
          ) : (
            <>
              <span>Analyze Image</span>
              <ArrowRight className="h-4 w-4" />
            </>
          )}
        </button>

        <p className="text-xs font-mono text-[#9CA3AF]">
          images are evaluated, never stored
        </p>
      </div>

    </section>
  );
};


