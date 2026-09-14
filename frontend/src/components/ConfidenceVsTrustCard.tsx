import React from 'react';
import { ArrowLeftRight, Info, AlertOctagon, CheckCircle2 } from 'lucide-react';
import type { PredictResponse } from '../types/api';

interface ConfidenceVsTrustCardProps {
  result: PredictResponse;
}

export const ConfidenceVsTrustCard: React.FC<ConfidenceVsTrustCardProps> = ({ result }) => {
  const conf = result.vision.confidence;
  const trust = result.trust.trust_probability;
  const isOverconfident = conf >= 0.70 && trust < 0.50;

  const confPercent = (conf * 100).toFixed(1);
  const trustPercent = (trust * 100).toFixed(1);

  return (
    <div className="bg-white border border-[#E6E1DA] rounded-2xl p-6 sm:p-8 shadow-xs space-y-6">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-[#E6E1DA] pb-4 gap-2">
        <div className="flex items-center space-x-2">
          <ArrowLeftRight className="h-5 w-5 text-[#111827]" />
          <h3 className="text-base font-bold text-[#111827]">Confidence vs. Trust Score</h3>
        </div>
        <span className="text-xs text-[#6B7280] font-mono">Vision Softmax vs. Independent Trust Model</span>
      </div>

      {/* Rationale Callout */}
      <p className="text-xs text-[#4B5563] leading-relaxed bg-[#FAF8F5] border border-[#E6E1DA] p-4 rounded-xl">
        <strong className="text-[#111827]">Softmax Confidence</strong> measures how strongly the classifier favors its top predicted class.{' '}
        <strong className="text-[#111827]">Trust Score</strong> estimates the true probability that the prediction is correct, computed by our independent Random Forest model using per-prediction reliability signals.
      </p>

      {/* Side-by-Side Horizontal Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        
        {/* Softmax Confidence */}
        <div className="bg-[#FAF8F5] rounded-xl p-5 border border-[#E6E1DA] space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono uppercase tracking-wider text-[#6B7280] font-semibold">Vision Model Confidence</span>
            <span className="text-lg font-bold font-mono text-[#111827]">{confPercent}%</span>
          </div>
          <div className="w-full bg-[#E6E1DA] rounded-full h-2.5 overflow-hidden">
            <div
              className="bg-[#374151] h-full rounded-full transition-all duration-500"
              style={{ width: `${Math.min(conf * 100, 100)}%` }}
            />
          </div>
          <p className="text-[11px] text-[#6B7280]">
            Raw max softmax probability output from ResNet-18.
          </p>
        </div>

        {/* Trust Model Probability */}
        <div className="bg-[#FAF8F5] rounded-xl p-5 border border-[#E6E1DA] space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-mono uppercase tracking-wider text-[#6B7280] font-semibold">Random Forest Trust Score</span>
            <span
              className={`text-lg font-bold font-mono ${
                trust >= 0.50 ? 'text-[#1B4332]' : 'text-[#C53030]'
              }`}
            >
              {trustPercent}%
            </span>
          </div>
          <div className="w-full bg-[#E6E1DA] rounded-full h-2.5 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                trust >= 0.50 ? 'bg-[#1B4332]' : 'bg-[#C53030]'
              }`}
              style={{ width: `${Math.min(trust * 100, 100)}%` }}
            />
          </div>
          <p className="text-[11px] text-[#6B7280]">
            Estimated probability P(correct) from 6 reliability signals.
          </p>
        </div>

      </div>

      {/* Conceptual Insight Banner */}
      <div
        className={`rounded-xl p-4 border flex items-start space-x-3 text-xs ${
          isOverconfident
            ? 'bg-[#FFEBEE] border-[#FFCDD2] text-[#9B2C2C]'
            : trust >= 0.50
            ? 'bg-[#E8F5E9] border-[#C8E6C9] text-[#1B4332]'
            : 'bg-[#FEF3C7] border-[#FDE68A] text-[#92400E]'
        }`}
      >
        {isOverconfident ? (
          <AlertOctagon className="h-5 w-5 text-[#C53030] shrink-0 mt-0.5" />
        ) : trust >= 0.50 ? (
          <CheckCircle2 className="h-5 w-5 text-[#1B4332] shrink-0 mt-0.5" />
        ) : (
          <Info className="h-5 w-5 text-[#D97706] shrink-0 mt-0.5" />
        )}
        <div className="space-y-1">
          <p className="font-bold uppercase tracking-wider">
            {isOverconfident
              ? 'Overconfidence Discrepancy Detected'
              : trust >= 0.50
              ? 'Reliable Prediction Alignment'
              : 'Elevated Uncertainty Warning'}
          </p>
          <p className="text-[#374151] leading-relaxed">
            {isOverconfident
              ? `The vision model outputs high confidence (${confPercent}%), but the Trust Model identifies significant reliability risk (Trust: ${trustPercent}%), indicating potential distribution shift or perturbation sensitivity.`
              : trust >= 0.50
              ? `The vision prediction confidence (${confPercent}%) aligns with the downstream reliability score (${trustPercent}%), supported by strong signal consistency across test-time augmentations.`
              : `Both confidence (${confPercent}%) and trust score (${trustPercent}%) reflect uncertainty. The prediction should be treated with caution in safety-critical automated pipelines.`}
          </p>
        </div>
      </div>

    </div>
  );
};


