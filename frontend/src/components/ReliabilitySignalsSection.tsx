import React from 'react';
import { Gauge, Waves, Compass, Zap, RefreshCw, Sun, Database, ShieldCheck, BarChart2, ArrowRight, TreeDeciduous } from 'lucide-react';
import type { PredictResponse } from '../types/api';

interface ReliabilitySignalsSectionProps {
  result?: PredictResponse | null;
}

export const ReliabilitySignalsSection: React.FC<ReliabilitySignalsSectionProps> = () => {
  const signalDefinitions = [
    {
      id: 'confidence',
      title: 'Confidence',
      body: 'How strongly the classifier favors its top predicted class.',
      badge: 'HIGHER IS SAFER',
      badgeStyle: 'bg-[#D8F3DC] text-[#1B4332] border-[#B7E4C7]',
      icon: Gauge,
      iconBg: 'bg-[#E8F5E9] border-[#C8E6C9] text-[#1B4332]',
      secondary: 'Uses classifier output probabilities (e.g., top-1 confidence, probability distribution).',
    },
    {
      id: 'entropy',
      title: 'Prediction Entropy',
      body: 'How spread out probability mass is across all classes.',
      badge: 'LOWER IS SAFER',
      badgeStyle: 'bg-[#FEF3C7] text-[#92400E] border-[#FDE68A]',
      icon: Waves,
      iconBg: 'bg-[#FEF3C7] border-[#FDE68A] text-[#92400E]',
      secondary: 'Measures uncertainty in the predicted class distribution (e.g., entropy of softmax probabilities).',
    },
    {
      id: 'distance',
      title: 'Feature Distance',
      body: 'Distance from training feature distribution centroid.',
      badge: 'LOWER IS SAFER',
      badgeStyle: 'bg-[#FAF8F5] text-[#374151] border-[#E6E1DA]',
      icon: Compass,
      iconBg: 'bg-[#E8F5E9] border-[#C8E6C9] text-[#1B4332]',
      secondary: 'Measures how different the input features are from training data (e.g., feature space distance).',
    },
    {
      id: 'ood',
      title: 'OOD Energy Score',
      body: 'Energy-based evidence about whether the input differs from the reference distribution.',
      badge: 'LOWER IS SAFER',
      badgeStyle: 'bg-[#FEF3C7] text-[#92400E] border-[#FDE68A]',
      icon: Zap,
      iconBg: 'bg-[#FEF3C7] border-[#FDE68A] text-[#92400E]',
      secondary: 'Uses the classifier\'s log-sum-exp energy score.',
    },
    {
      id: 'aug',
      title: 'Augmentation Consistency',
      body: 'Stability of prediction under test-time image perturbations.',
      badge: 'HIGHER IS SAFER',
      badgeStyle: 'bg-[#D8F3DC] text-[#1B4332] border-[#B7E4C7]',
      icon: RefreshCw,
      iconBg: 'bg-[#E8F5E9] border-[#C8E6C9] text-[#1B4332]',
      secondary: 'Compares predictions across K=3 test-time augmentations (flip, center-crop/resize, and 5° rotation).',
    },
    {
      id: 'quality',
      title: 'Composite Quality',
      body: 'Laplacian sharpness, brightness, and contrast characteristics.',
      badge: 'DIAGNOSTIC METRIC',
      badgeStyle: 'bg-[#FAF8F5] text-[#374151] border-[#E6E1DA]',
      icon: Sun,
      iconBg: 'bg-[#FEF3C7] border-[#FDE68A] text-[#92400E]',
      secondary: 'Aggregates image-quality indicators (e.g., sharpness, brightness, contrast).',
    },
  ];

  return (
    <section id="signals" className="bg-white border border-[#E6E1DA] rounded-3xl p-6 sm:p-9 shadow-xs space-y-8">
      
      {/* Header with Title and Upper Right Callout */}
      <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-6 border-b border-[#E6E1DA] pb-6">
        <div className="space-y-3 max-w-3xl">
          <p className="text-xs font-mono tracking-widest text-[#6B7280] uppercase font-bold">
            RELIABILITY LAYER
          </p>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-[#111827] tracking-tight">
            Six reliability signal categories.
          </h2>
          <p className="text-base text-[#4B5563] leading-relaxed">
            Nine features grouped into six signal categories provide complementary evidence about prediction reliability beyond raw softmax confidence.
          </p>
        </div>

        {/* Upper Right Callout Box */}
        <div className="bg-[#E8F5E9]/60 border border-[#C8E6C9] rounded-2xl p-4 flex items-start space-x-3.5 max-w-sm shrink-0">
          <div className="h-9 w-9 rounded-xl bg-white border border-[#C8E6C9] flex items-center justify-center text-[#1B4332] shrink-0 mt-0.5 shadow-xs">
            <Database className="h-4 w-4" />
          </div>
          <div>
            <p className="text-xs font-bold text-[#111827]">From multiple perspectives</p>
            <p className="text-[11px] text-[#4B5563] leading-normal mt-0.5">
              Each signal captures a different aspect of input reliability. Together, they help the model detect when a prediction may fail.
            </p>
          </div>
        </div>
      </div>

      {/* 6 Signal Cards Grid (3 Columns × 2 Rows on Desktop) */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {signalDefinitions.map((sig) => {
          const Icon = sig.icon;
          return (
            <div
              key={sig.id}
              className="bg-[#FAF8F5] border border-[#E6E1DA] rounded-2xl p-5 flex flex-col justify-between space-y-4 shadow-xs hover:border-[#D8D2C7] transition-all"
            >
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2.5">
                    <div className={`p-2 rounded-xl border ${sig.iconBg} shadow-xs`}>
                      <Icon className="h-4 w-4" />
                    </div>
                    <h3 className="text-base font-bold text-[#111827]">
                      {sig.title}
                    </h3>
                  </div>
                  <span className={`text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full border uppercase ${sig.badgeStyle}`}>
                    {sig.badge}
                  </span>
                </div>

                <p className="text-xs text-[#374151] leading-relaxed">
                  {sig.body}
                </p>
              </div>

              {/* Secondary Explanation Box */}
              <div className="bg-white border border-[#E6E1DA] rounded-xl p-3 text-xs text-[#6B7280] leading-relaxed">
                {sig.secondary}
              </div>
            </div>
          );
        })}
      </div>

      {/* Signal Aggregation Flow Pipeline Card */}
      <div className="bg-white border border-[#E6E1DA] rounded-2xl p-4 sm:p-5 shadow-xs space-y-4">
        
        {/* Top Pipeline Flow Row (Single Horizontal Row on Desktop) */}
        <div className="flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-3 lg:gap-3">
          
          {/* Block 1: 6 Signal Categories List (flex-[1.2]) */}
          <div className="flex-[1.2] bg-[#FAF8F5] border border-[#E6E1DA] rounded-xl p-3 space-y-1.5">
            <span className="text-[10px] font-mono font-bold text-[#6B7280] uppercase tracking-wider block">
              6 RELIABILITY SIGNAL CATEGORIES
            </span>
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-1.5 text-center">
              <div className="bg-white border border-[#E6E1DA] rounded-lg p-1 flex flex-col items-center">
                <Gauge className="h-3 w-3 text-[#1B4332] mb-0.5" />
                <span className="text-[9px] font-mono text-[#374151] truncate w-full">Confidence</span>
              </div>
              <div className="bg-white border border-[#E6E1DA] rounded-lg p-1 flex flex-col items-center">
                <Waves className="h-3 w-3 text-[#92400E] mb-0.5" />
                <span className="text-[9px] font-mono text-[#374151] truncate w-full">Entropy</span>
              </div>
              <div className="bg-white border border-[#E6E1DA] rounded-lg p-1 flex flex-col items-center">
                <Compass className="h-3 w-3 text-[#1B4332] mb-0.5" />
                <span className="text-[9px] font-mono text-[#374151] truncate w-full">Distance</span>
              </div>
              <div className="bg-white border border-[#E6E1DA] rounded-lg p-1 flex flex-col items-center">
                <Zap className="h-3 w-3 text-[#92400E] mb-0.5" />
                <span className="text-[9px] font-mono text-[#374151] truncate w-full">OOD Energy</span>
              </div>
              <div className="bg-white border border-[#E6E1DA] rounded-lg p-1 flex flex-col items-center">
                <RefreshCw className="h-3 w-3 text-[#1B4332] mb-0.5" />
                <span className="text-[9px] font-mono text-[#374151] truncate w-full">Consistency</span>
              </div>
              <div className="bg-white border border-[#E6E1DA] rounded-lg p-1 flex flex-col items-center">
                <Sun className="h-3 w-3 text-[#92400E] mb-0.5" />
                <span className="text-[9px] font-mono text-[#374151] truncate w-full">Quality</span>
              </div>
            </div>
          </div>

          {/* Arrow Connector 1 */}
          <div className="flex justify-center shrink-0 text-[#9CA3AF]">
            <ArrowRight className="h-4 w-4 rotate-90 lg:rotate-0" />
          </div>

          {/* Block 2: Combined Signal Evidence (flex-1) */}
          <div className="flex-1 bg-[#FAF8F5] border border-[#E6E1DA] rounded-xl p-3 flex items-center space-x-3 min-h-[76px]">
            <div className="h-8 w-8 rounded-lg bg-white border border-[#E6E1DA] flex items-center justify-center text-[#1B4332] shrink-0 shadow-xs">
              <BarChart2 className="h-4 w-4" />
            </div>
            <div>
              <p className="text-[10px] font-mono font-bold text-[#111827] uppercase tracking-wider">
                COMBINED SIGNAL EVIDENCE
              </p>
              <p className="text-[10px] text-[#6B7280] leading-snug mt-0.5">
                Nine features from six categories are combined as input to the trust model.
              </p>
            </div>
          </div>

          {/* Arrow Connector 2 */}
          <div className="flex justify-center shrink-0 text-[#9CA3AF]">
            <ArrowRight className="h-4 w-4 rotate-90 lg:rotate-0" />
          </div>

          {/* Block 3: Random Forest Trust Model (flex-1) */}
          <div className="flex-1 bg-[#E8F5E9] border border-[#A3E635] rounded-xl p-3 flex items-center space-x-3 min-h-[76px] shadow-xs">
            <div className="h-8 w-8 rounded-lg bg-[#1B4332] text-white flex items-center justify-center shrink-0 shadow-xs">
              <TreeDeciduous className="h-4 w-4 text-emerald-300" />
            </div>
            <div>
              <p className="text-[10px] font-mono font-bold text-[#1B4332] uppercase tracking-wider">
                RANDOM FOREST TRUST MODEL
              </p>
              <p className="text-[10px] text-[#1B4332] font-medium leading-snug mt-0.5">
                Estimates the probability that the prediction is correct.
              </p>
            </div>
          </div>

        </div>

        {/* Bottom Supporting Statements Row (Single Horizontal Row) */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-3 border-t border-[#E6E1DA]">
          <div className="flex items-start space-x-2.5">
            <div className="h-7 w-7 rounded-lg bg-[#FAF8F5] border border-[#E6E1DA] flex items-center justify-center text-[#1B4332] shrink-0 mt-0.5">
              <ShieldCheck className="h-3.5 w-3.5" />
            </div>
            <div>
              <p className="text-xs font-bold text-[#111827]">Complementary evidence</p>
              <p className="text-[10px] text-[#6B7280]">Signals capture different aspects of reliability.</p>
            </div>
          </div>

          <div className="flex items-start space-x-2.5">
            <div className="h-7 w-7 rounded-lg bg-[#FAF8F5] border border-[#E6E1DA] flex items-center justify-center text-[#1B4332] shrink-0 mt-0.5">
              <Database className="h-3.5 w-3.5" />
            </div>
            <div>
              <p className="text-xs font-bold text-[#111827]">More robust decisions</p>
              <p className="text-[10px] text-[#6B7280]">Helps detect when predictions may fail.</p>
            </div>
          </div>

          <div className="flex items-start space-x-2.5">
            <div className="h-7 w-7 rounded-lg bg-[#FAF8F5] border border-[#E6E1DA] flex items-center justify-center text-[#1B4332] shrink-0 mt-0.5">
              <BarChart2 className="h-3.5 w-3.5" />
            </div>
            <div>
              <p className="text-xs font-bold text-[#111827]">Built from real data</p>
              <p className="text-[10px] text-[#6B7280]">Fitted on validation signals with zero test-set leakage.</p>
            </div>
          </div>
        </div>

      </div>

    </section>
  );
};
