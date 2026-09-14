import React from 'react';
import { ArrowDown, Layers, ShieldCheck, Image as ImageIcon } from 'lucide-react';
import type { PredictResponse } from '../types/api';

interface PredictionPathDiagramProps {
  result: PredictResponse | null;
  previewUrl: string | null;
}

export const PredictionPathDiagram: React.FC<PredictionPathDiagramProps> = ({
  result,
  previewUrl,
}) => {
  const isAnalyzed = result !== null;
  
  const confidencePercent = isAnalyzed ? Math.round(result.vision.confidence * 1000) / 10 : 0;
  const trustPercent = isAnalyzed ? Math.round(result.trust.trust_probability * 1000) / 10 : 0;
  
  // Normalized signal progress bars for visualization (scale 0-100%)
  const entropyWidth = isAnalyzed ? Math.min(100, Math.max(10, Math.round((1 - Math.min(1, result.vision.entropy)) * 100))) : 40;
  const distWidth = isAnalyzed ? Math.min(100, Math.max(10, Math.round((1 - Math.min(1, result.reliability.feature_distance / 20)) * 100))) : 60;
  const oodWidth = isAnalyzed ? Math.min(100, Math.max(10, Math.round((1 - Math.min(1, result.reliability.ood_score / 10)) * 100))) : 75;
  const consistencyWidth = isAnalyzed ? Math.min(100, Math.max(10, Math.round(result.reliability.augmentation_consistency * 100))) : 85;

  return (
    <section className="bg-white border border-[#E6E1DA] rounded-2xl p-6 sm:p-8 shadow-xs space-y-6 relative overflow-hidden">
      
      {/* Top Header */}
      <div className="flex items-center justify-between border-b border-[#E6E1DA] pb-4">
        <div className="space-y-1">
          <p className="text-xs font-mono tracking-widest text-[#6B7280] uppercase font-semibold">
            ANALYSIS VIEW
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-[11px] font-mono text-[#9CA3AF] uppercase hidden sm:inline">
            {isAnalyzed ? 'REAL MODEL OUTPUT' : 'ILLUSTRATIVE PIPELINE SCHEMA'}
          </span>
          <span className={`text-xs font-mono px-2.5 py-0.5 rounded-full border ${
            isAnalyzed 
              ? 'bg-[#E8F5E9] text-[#1B4332] border-[#C8E6C9]' 
              : 'bg-[#FAF8F5] text-[#6B7280] border-[#E6E1DA]'
          }`}>
            {isAnalyzed ? 'evaluated' : 'awaiting analysis'}
          </span>
        </div>
      </div>

      {/* Dual Path Columns Container */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 relative">
        
        {/* Left Column: PREDICTION PATH */}
        <div className="space-y-4">
          <p className="text-xs font-mono uppercase tracking-widest text-[#6B7280] font-semibold">
            PREDICTION PATH
          </p>

          {/* Image Input Container */}
          <div className="h-40 rounded-xl bg-dot-pattern border border-[#E6E1DA] flex items-center justify-center relative overflow-hidden bg-[#FAF8F5]">
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Prediction Input"
                className="h-full w-full object-contain p-2"
              />
            ) : (
              <div className="text-[#9CA3AF] flex flex-col items-center space-y-1">
                <ImageIcon className="h-8 w-8" />
                <span className="text-xs font-mono">input image</span>
              </div>
            )}
          </div>

          <div className="flex justify-center text-[#9CA3AF]">
            <ArrowDown className="h-4 w-4" />
          </div>

          {/* Vision Prediction Card */}
          <div className="bg-[#FAF8F5] border border-[#E6E1DA] rounded-xl p-4 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="h-8 w-8 rounded-lg bg-white border border-[#E6E1DA] flex items-center justify-center text-[#111827]">
                <Layers className="h-4 w-4" />
              </div>
              <span className="text-sm font-bold text-[#111827]">Vision Prediction</span>
            </div>
            <span className="text-xs font-mono text-[#6B7280]">softmax</span>
          </div>

          <div className="flex justify-center text-[#9CA3AF]">
            <ArrowDown className="h-4 w-4" />
          </div>

          {/* Softmax Confidence Box */}
          <div className="bg-[#FAF8F5] border border-[#E6E1DA] rounded-xl p-4 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-[#111827]">Confidence</span>
              <span className="font-mono font-bold text-[#111827]">
                {isAnalyzed ? `${confidencePercent}%` : 'awaiting image'}
              </span>
            </div>
            <div className="h-2 w-full bg-[#E6E1DA] rounded-full overflow-hidden">
              <div
                className="h-full bg-[#374151] rounded-full transition-all duration-500"
                style={{ width: `${isAnalyzed ? confidencePercent : 0}%` }}
              />
            </div>
            {!isAnalyzed && (
              <p className="text-[11px] font-mono text-[#9CA3AF]">softmax score</p>
            )}
          </div>
        </div>

        {/* Right Column: RELIABILITY PATH */}
        <div className="space-y-4">
          <p className="text-xs font-mono uppercase tracking-widest text-[#6B7280] font-semibold">
            RELIABILITY PATH
          </p>

          {/* Signals Box */}
          <div className="h-40 rounded-xl bg-[#FAF8F5] border border-[#E6E1DA] p-4 flex flex-col justify-center space-y-2.5">
            {/* Signal 1: Entropy */}
            <div className="space-y-1">
              <div className="flex justify-between text-[11px] font-mono text-[#6B7280]">
                <span>entropy</span>
                <span>{isAnalyzed ? result.vision.entropy.toFixed(3) : '-'}</span>
              </div>
              <div className="h-1.5 w-full bg-[#E6E1DA] rounded-full overflow-hidden">
                <div className="h-full bg-[#2D6A4F] rounded-full transition-all duration-500" style={{ width: `${entropyWidth}%` }} />
              </div>
            </div>

            {/* Signal 2: Feature Dist */}
            <div className="space-y-1">
              <div className="flex justify-between text-[11px] font-mono text-[#6B7280]">
                <span>feature dist.</span>
                <span>{isAnalyzed ? result.reliability.feature_distance.toFixed(2) : '-'}</span>
              </div>
              <div className="h-1.5 w-full bg-[#E6E1DA] rounded-full overflow-hidden">
                <div className="h-full bg-[#2D6A4F] rounded-full transition-all duration-500" style={{ width: `${distWidth}%` }} />
              </div>
            </div>

            {/* Signal 3: OOD Score */}
            <div className="space-y-1">
              <div className="flex justify-between text-[11px] font-mono text-[#6B7280]">
                <span>ood score</span>
                <span>{isAnalyzed ? result.reliability.ood_score.toFixed(2) : '-'}</span>
              </div>
              <div className="h-1.5 w-full bg-[#E6E1DA] rounded-full overflow-hidden">
                <div className="h-full bg-[#2D6A4F] rounded-full transition-all duration-500" style={{ width: `${oodWidth}%` }} />
              </div>
            </div>

            {/* Signal 4: Consistency */}
            <div className="space-y-1">
              <div className="flex justify-between text-[11px] font-mono text-[#6B7280]">
                <span>consistency</span>
                <span>{isAnalyzed ? `${Math.round(result.reliability.augmentation_consistency * 100)}%` : '-'}</span>
              </div>
              <div className="h-1.5 w-full bg-[#E6E1DA] rounded-full overflow-hidden">
                <div className="h-full bg-[#2D6A4F] rounded-full transition-all duration-500" style={{ width: `${consistencyWidth}%` }} />
              </div>
            </div>
          </div>

          <div className="flex justify-center text-[#9CA3AF]">
            <ArrowDown className="h-4 w-4" />
          </div>

          {/* Trust Model Box */}
          <div className="bg-[#FAF8F5] border border-[#E6E1DA] rounded-xl p-4 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="h-8 w-8 rounded-lg bg-white border border-[#E6E1DA] flex items-center justify-center text-[#1B4332]">
                <ShieldCheck className="h-4 w-4" />
              </div>
              <span className="text-sm font-bold text-[#111827]">Trust Model</span>
            </div>
            <span className="text-xs font-mono text-[#6B7280]">Random Forest</span>
          </div>

          <div className="flex justify-center text-[#9CA3AF]">
            <ArrowDown className="h-4 w-4" />
          </div>

          {/* Trust Score Box */}
          <div className={`border rounded-xl p-4 space-y-2 transition-all duration-300 ${
            isAnalyzed && result.trust.status === 'TRUSTED'
              ? 'bg-[#E8F5E9]/60 border-[#C8E6C9]'
              : isAnalyzed && result.trust.status === 'UNTRUSTED'
              ? 'bg-[#FFEBEE]/60 border-[#FFCDD2]'
              : 'bg-[#FAF8F5] border-[#E6E1DA]'
          }`}>
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-[#111827]">Trust Score</span>
              <span className="font-mono font-bold text-[#111827]">
                {isAnalyzed ? `${trustPercent}%` : 'awaiting image'}
              </span>
            </div>
            <div className="h-2 w-full bg-[#E6E1DA] rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  isAnalyzed && result.trust.status === 'TRUSTED'
                    ? 'bg-[#1B4332]'
                    : isAnalyzed && result.trust.status === 'UNTRUSTED'
                    ? 'bg-[#C53030]'
                    : 'bg-[#2D6A4F]'
                }`}
                style={{ width: `${isAnalyzed ? trustPercent : 0}%` }}
              />
            </div>
            {!isAnalyzed && (
              <p className="text-[11px] font-mono text-[#9CA3AF]">trust probability</p>
            )}

          </div>

        </div>

      </div>

      {/* Output Badge Callout at bottom */}
      <div className="pt-2 flex justify-center">
        <div className="bg-white border border-[#E6E1DA] shadow-xs px-6 py-2.5 rounded-xl flex items-center space-x-3 text-xs font-mono">
          <span className="text-[#9CA3AF] uppercase">OUTPUT</span>
          <span className="text-[#111827] font-bold">
            {isAnalyzed ? (
              <span className={`px-2.5 py-1 rounded-lg font-extrabold ${
                result.trust.status === 'TRUSTED'
                  ? 'bg-[#E8F5E9] text-[#1B4332] border border-[#C8E6C9]'
                  : 'bg-[#FFEBEE] text-[#C53030] border border-[#FFCDD2]'
              }`}>
                {result.trust.status}
              </span>
            ) : (
              'Trusted / Untrusted'
            )}
          </span>
        </div>
      </div>

    </section>
  );
};
