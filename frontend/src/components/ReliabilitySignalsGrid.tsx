import React from 'react';
import { Layers, Activity, Compass, Zap, Repeat, SunMedium, Gauge } from 'lucide-react';
import type { ReliabilityResponse, VisionResponse } from '../types/api';

interface ReliabilitySignalsGridProps {
  reliability: ReliabilityResponse;
  vision: VisionResponse;
}

export const ReliabilitySignalsGrid: React.FC<ReliabilitySignalsGridProps> = ({
  reliability,
  vision,
}) => {
  const conf = vision.confidence;
  const entropy = vision.entropy;
  const dist = reliability.feature_distance;
  const ood = reliability.ood_score;
  const aug = reliability.augmentation_consistency;
  const qual = reliability.composite_quality;

  const signals = [
    {
      id: 'confidence',
      name: 'Softmax Confidence',
      value: `${(conf * 100).toFixed(1)}%`,
      raw: conf,
      icon: Activity,
      direction: 'Higher = More Reliable',
      directionType: 'higher',
      progressPercent: Math.min(conf * 100, 100),
      description: 'Maximum class probability max_c p_c output by the softmax layer.',
    },
    {
      id: 'entropy',
      name: 'Prediction Entropy',
      value: entropy.toFixed(4),
      raw: entropy,
      icon: Layers,
      direction: 'Lower = More Reliable',
      directionType: 'lower',
      progressPercent: Math.min((entropy / 2.3026) * 100, 100),
      description: 'Shannon entropy measuring uncertainty across the 10-class probability distribution.',
    },
    {
      id: 'distance',
      name: 'Feature Centroid Distance',
      value: dist.toFixed(2),
      raw: dist,
      icon: Compass,
      direction: 'Lower = More Reliable',
      directionType: 'lower',
      progressPercent: Math.min((dist / 25.0) * 100, 100),
      description: 'Euclidean distance in 512-D bottleneck feature space to predicted class centroid.',
    },
    {
      id: 'ood',
      name: 'OOD / Logit Energy Score',
      value: ood.toFixed(2),
      raw: ood,
      icon: Zap,
      direction: 'Higher = In-Distribution',
      directionType: 'higher',
      progressPercent: Math.min(Math.max((ood / 10.0) * 100, 5), 100),
      description: 'Logit free-energy score at T=1.0. Higher energy indicates in-distribution samples.',
    },
    {
      id: 'aug',
      name: 'Augmentation Consistency',
      value: `${(aug * 100).toFixed(0)}%`,
      raw: aug,
      icon: Repeat,
      direction: 'Higher = More Reliable',
      directionType: 'higher',
      progressPercent: aug * 100,
      description: 'Agreement rate across K=3 light deterministic test-time image augmentations.',
    },
    {
      id: 'quality',
      name: 'Composite Image Quality',
      value: qual.toFixed(2),
      raw: qual,
      icon: SunMedium,
      direction: 'Diagnostic Metric',
      directionType: 'neutral',
      progressPercent: Math.min((qual / 5.0) * 100, 100),
      description: `Laplacian sharpness (${reliability.sharpness.toFixed(3)}), brightness (${reliability.brightness.toFixed(2)}), contrast (${reliability.contrast.toFixed(2)}).`,
    },
  ];

  return (
    <div className="bg-white border border-[#E6E1DA] rounded-2xl p-6 sm:p-8 shadow-xs space-y-6">
      
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-[#E6E1DA] pb-4 gap-2">
        <div className="flex items-center space-x-2">
          <Gauge className="h-5 w-5 text-[#111827]" />
          <h3 className="text-base font-bold text-[#111827]">Per-Prediction Reliability Signals</h3>
        </div>
        <span className="text-xs text-[#6B7280] font-mono">6 Categories / 9 Raw Signal Metrics</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
        {signals.map((sig) => {
          const Icon = sig.icon;
          return (
            <div
              key={sig.id}
              className="bg-[#FAF8F5] rounded-xl p-5 border border-[#E6E1DA] flex flex-col justify-between space-y-4 hover:border-[#D8D2C7] transition"
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center space-x-3">
                  <div className="p-2 rounded-lg bg-white border border-[#E6E1DA] text-[#1B4332] shadow-xs">
                    <Icon className="h-4 w-4" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-[#111827]">{sig.name}</h4>
                    <span
                      className={`text-[10px] font-mono px-2 py-0.5 rounded border inline-block mt-0.5 ${
                        sig.directionType === 'higher'
                          ? 'bg-[#E8F5E9] text-[#1B4332] border-[#C8E6C9]'
                          : sig.directionType === 'lower'
                          ? 'bg-[#FAF8F5] text-[#374151] border-[#E6E1DA]'
                          : 'bg-[#FAF8F5] text-[#6B7280] border-[#E6E1DA]'
                      }`}
                    >
                      {sig.direction}
                    </span>
                  </div>
                </div>
              </div>

              <div>
                <div className="flex items-baseline justify-between mb-2">
                  <span className="text-xs text-[#6B7280] font-mono">Metric Value</span>
                  <span className="text-xl font-bold font-mono text-[#111827]">{sig.value}</span>
                </div>

                {/* Progress bar */}
                <div className="w-full bg-[#E6E1DA] rounded-full h-2 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      sig.directionType === 'lower'
                        ? sig.progressPercent > 60
                          ? 'bg-[#D97706]'
                          : 'bg-[#374151]'
                        : sig.progressPercent < 40
                        ? 'bg-[#D97706]'
                        : 'bg-[#1B4332]'
                    }`}
                    style={{ width: `${Math.max(sig.progressPercent, 4)}%` }}
                  />
                </div>
              </div>

              <p className="text-[11px] text-[#6B7280] leading-tight border-t border-[#E6E1DA] pt-3">
                {sig.description}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
};


