import React from 'react';
import { Eye, ShieldCheck, CheckCircle2, AlertTriangle, Clock } from 'lucide-react';
import type { PredictResponse } from '../types/api';

interface ResultSummaryCardProps {
  result: PredictResponse;
}

export const ResultSummaryCard: React.FC<ResultSummaryCardProps> = ({ result }) => {
  const { prediction, vision, trust, latency_ms } = result;
  const isTrusted = trust.status === 'TRUSTED';

  const trustPercent = `${(trust.trust_probability * 100).toFixed(1)}%`;
  const failurePercent = `${(trust.failure_probability * 100).toFixed(1)}%`;
  const confPercent = `${(vision.confidence * 100).toFixed(1)}%`;

  return (
    <div className="bg-white border border-[#E6E1DA] rounded-2xl p-6 sm:p-8 shadow-xs space-y-6">
      
      {/* Top Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E6E1DA] pb-4">
        <div>
          <p className="text-xs font-mono tracking-widest text-[#6B7280] uppercase font-semibold">
            ANALYSIS RESULT
          </p>
          <h3 className="text-lg font-bold text-[#111827]">
            Model Verdict & Metrics Overview
          </h3>
        </div>

        {latency_ms && (
          <div className="flex items-center space-x-1.5 text-xs font-mono text-[#6B7280] bg-[#FAF8F5] border border-[#E6E1DA] px-3 py-1.5 rounded-lg shrink-0">
            <Clock className="h-3.5 w-3.5 text-[#9CA3AF]" />
            <span>{latency_ms} ms</span>
          </div>
        )}
      </div>

      {/* Main 2-Column Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        
        {/* Vision Prediction Metric Card */}
        <div className="bg-[#FAF8F5] rounded-xl p-5 border border-[#E6E1DA] flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center space-x-2 text-xs font-semibold text-[#6B7280] tracking-wider uppercase mb-1">
              <Eye className="h-4 w-4 text-[#1B4332]" />
              <span>Vision Model Prediction</span>
            </div>
            <h4 className="text-3xl font-extrabold text-[#111827] capitalize tracking-tight mt-1">
              {prediction.class_name}
            </h4>
            <p className="text-xs text-[#6B7280] mt-1 font-mono">
              CIFAR-10 Class #{prediction.class_id}
            </p>
          </div>

          <div className="border-t border-[#E6E1DA] pt-3 flex items-center justify-between">
            <span className="text-xs font-medium text-[#6B7280]">Softmax Confidence:</span>
            <span className="text-xl font-bold font-mono text-[#111827]">{confPercent}</span>
          </div>
        </div>

        {/* Trust Model Status & Metrics */}
        <div
          className={`rounded-xl p-5 border flex flex-col justify-between space-y-4 transition-colors ${
            isTrusted
              ? 'bg-[#E8F5E9]/70 border-[#C8E6C9]'
              : 'bg-[#FFEBEE]/70 border-[#FFCDD2]'
          }`}
        >
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <div className="flex items-center space-x-1.5 text-xs font-semibold tracking-wider uppercase">
                <ShieldCheck className={`h-4 w-4 ${isTrusted ? 'text-[#1B4332]' : 'text-[#C53030]'}`} />
                <span className={isTrusted ? 'text-[#1B4332]' : 'text-[#C53030]'}>
                  Trust Assessment
                </span>
              </div>
              <div>
                <span
                  className={`inline-flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg text-xs font-extrabold uppercase tracking-wider ${
                    isTrusted
                      ? 'bg-[#1B4332] text-white'
                      : 'bg-[#C53030] text-white'
                  }`}
                >
                  {isTrusted ? (
                    <CheckCircle2 className="h-3.5 w-3.5" />
                  ) : (
                    <AlertTriangle className="h-3.5 w-3.5" />
                  )}
                  <span>{trust.status}</span>
                </span>
              </div>
            </div>

            <div className="text-right">
              <span className="text-xs font-mono uppercase text-[#6B7280] block">
                Trust Score
              </span>
              <span
                className={`text-3xl font-extrabold font-mono ${
                  isTrusted ? 'text-[#1B4332]' : 'text-[#C53030]'
                }`}
              >
                {trustPercent}
              </span>
            </div>
          </div>

          <div className={`border-t pt-3 flex items-center justify-between ${isTrusted ? 'border-[#C8E6C9]' : 'border-[#FFCDD2]'}`}>
            <span className="text-xs font-medium text-[#6B7280]">
              Estimated Failure Probability:
            </span>
            <span className={`text-sm font-bold font-mono ${isTrusted ? 'text-[#1B4332]' : 'text-[#C53030]'}`}>
              {failurePercent}
            </span>
          </div>
        </div>

      </div>

    </div>
  );
};


