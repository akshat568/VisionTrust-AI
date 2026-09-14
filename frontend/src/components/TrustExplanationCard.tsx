import React from 'react';
import { HelpCircle, Check, Info } from 'lucide-react';
import type { PredictResponse } from '../types/api';

interface TrustExplanationCardProps {
  result: PredictResponse;
}

export const TrustExplanationCard: React.FC<TrustExplanationCardProps> = ({ result }) => {
  const { vision, reliability, trust, prediction } = result;

  const conf = vision.confidence;
  const entropy = vision.entropy;
  const dist = reliability.feature_distance;
  const aug = reliability.augmentation_consistency;

  const insights: { text: string; positive: boolean }[] = [];

  // 1. Augmentation consistency insight
  if (aug >= 0.99) {
    insights.push({
      text: 'Prediction remains 100% consistent across test-time augmentations (flip, center-crop/resize, and 5° rotation).',
      positive: true,
    });
  } else if (aug >= 0.66) {
    insights.push({
      text: `Prediction exhibits moderate sensitivity under test-time augmentations (${(aug * 100).toFixed(0)}% agreement).`,
      positive: false,
    });
  } else {
    insights.push({
      text: `High perturbation sensitivity detected under test-time augmentations (${(aug * 100).toFixed(0)}% agreement).`,
      positive: false,
    });
  }

  // 2. Entropy / Uncertainty insight
  if (entropy < 0.30) {
    insights.push({
      text: `Shannon entropy is low (${entropy.toFixed(3)}), indicating strong probability concentration on '${prediction.class_name}'.`,
      positive: true,
    });
  } else if (entropy > 0.80) {
    insights.push({
      text: `Elevated entropy (${entropy.toFixed(3)}) exposes secondary class probability uncertainty.`,
      positive: false,
    });
  }

  // 3. Feature centroid distance insight
  if (dist < 10.5) {
    insights.push({
      text: `Bottleneck feature representation has a centroid distance of ${dist.toFixed(2)}.`,
      positive: true,
    });
  } else {
    insights.push({
      text: `Penultimate feature representation lies far (distance: ${dist.toFixed(2)}) from class reference centroids.`,
      positive: false,
    });
  }

  // 4. Overconfidence warning
  if (conf >= 0.85 && trust.trust_probability < 0.50) {
    insights.push({
      text: `CAUTION: Softmax confidence is high (${(conf * 100).toFixed(1)}%), but overall signal vector indicates elevated risk of misclassification.`,
      positive: false,
    });
  }

  return (
    <div className="bg-white border border-[#E6E1DA] rounded-2xl p-6 sm:p-8 shadow-xs space-y-5">
      <div className="flex items-center space-x-2 border-b border-[#E6E1DA] pb-4">
        <HelpCircle className="h-5 w-5 text-[#111827]" />
        <h3 className="text-base font-bold text-[#111827]">
          Why this prediction received this trust score
        </h3>
      </div>

      <p className="text-xs text-[#6B7280] leading-relaxed">
        The Trust Layer evaluates 6 independent reliability signal categories extracted from the vision model and input image rather than relying solely on softmax confidence:
      </p>

      <div className="space-y-3">
        {insights.map((item, idx) => (
          <div
            key={idx}
            className={`flex items-start space-x-3 p-3.5 rounded-xl border text-xs leading-relaxed ${
              item.positive
                ? 'bg-[#E8F5E9]/80 border-[#C8E6C9] text-[#1B4332]'
                : 'bg-[#FAF8F5] border-[#E6E1DA] text-[#374151]'
            }`}
          >
            {item.positive ? (
              <Check className="h-4 w-4 text-[#1B4332] shrink-0 mt-0.5" />
            ) : (
              <Info className="h-4 w-4 text-[#6B7280] shrink-0 mt-0.5" />
            )}
            <span>{item.text}</span>
          </div>
        ))}
      </div>

      <div className="border-t border-[#E6E1DA] pt-3 text-[11px] text-[#6B7280] flex items-center justify-between font-mono">
        <span>Decision Threshold: {result.trust.status === 'TRUSTED' ? '≥ 0.50' : '< 0.50'}</span>
        <span>Random Forest Trust Model</span>
      </div>
    </div>
  );
};


