import React from 'react';
import { Image as ImageIcon, Layers, ShieldCheck, CheckCircle2, AlertTriangle, BarChart2, Activity, ArrowRight, Clock } from 'lucide-react';
import type { PredictResponse } from '../types/api';

interface ArchitectureSectionProps {
  result?: PredictResponse | null;
  previewUrl?: string | null;
}

export const ArchitectureSection: React.FC<ArchitectureSectionProps> = ({ result, previewUrl }) => {
  const hasResult = !!result;
  const isTrusted = result ? result.trust.status === 'TRUSTED' : false;

  return (
    <section id="architecture" className="bg-white border border-[#E6E1DA] rounded-3xl p-6 sm:p-9 shadow-xs space-y-8">
      
      {/* Header with Title and Subtitle */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-[#E6E1DA] pb-6">
        <div className="space-y-2">
          <p className="text-xs font-mono tracking-widest text-[#6B7280] uppercase font-bold">
            SYSTEM ARCHITECTURE
          </p>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-extrabold text-[#111827] tracking-tight">
            Two independent paths. One reliability decision.
          </h2>
        </div>
        <p className="text-xs sm:text-sm text-[#4B5563] max-w-md md:text-right leading-relaxed font-medium">
          The vision predictor and reliability layer operate independently, then combine their outputs to decide whether the prediction should be trusted.
        </p>
      </div>

      {/* Main Architecture Diagram Container */}
      <div className="bg-[#FAF8F5] border border-[#E6E1DA] rounded-3xl p-6 sm:p-8 space-y-8 shadow-xs relative overflow-hidden">
        
        {/* Directional Flow Diagram Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-center relative">
          
          {/* 1. INPUT IMAGE CARD (Left) */}
          <div className="lg:col-span-3 bg-white border border-[#E6E1DA] rounded-2xl p-4 flex flex-col items-center justify-between text-center shadow-xs min-h-[230px] relative">
            <span className="text-[10px] font-mono font-bold text-[#6B7280] uppercase tracking-wider">
              INPUT IMAGE
            </span>

            {previewUrl ? (
              <div className="h-28 w-full rounded-xl overflow-hidden border border-[#E6E1DA] bg-[#FAF8F5] flex items-center justify-center p-1.5 my-1">
                <img
                  src={previewUrl}
                  alt="Input Image Preview"
                  className="max-h-full max-w-full object-contain rounded-lg"
                />
              </div>
            ) : (
              <div className="h-28 w-full rounded-xl border border-dashed border-[#D1D5DB] bg-[#FAF8F5] flex flex-col items-center justify-center p-2 my-1">
                <ImageIcon className="h-6 w-6 text-[#9CA3AF] mb-1" />
                <p className="text-[11px] font-bold text-[#374151]">Image (224 × 224)</p>
                <p className="text-[9px] text-[#9CA3AF] text-center">PNG, JPG or WEBP<br />(up to 10 MB)</p>
              </div>
            )}

            {/* Explicit Directional Branching Connector Labels */}
            <div className="w-full pt-2 border-t border-[#E6E1DA]/80 flex items-center justify-between text-[9px] font-mono">
              <span className="inline-flex items-center gap-1 text-[#1B4332] font-bold">
                <ArrowRight className="h-3 w-3 text-[#1B4332]" /> Prediction
              </span>
              <span className="inline-flex items-center gap-1 text-[#D97706] font-bold">
                <ArrowRight className="h-3 w-3 text-[#D97706]" /> Reliability
              </span>
            </div>
          </div>

          {/* 2. MIDDLE DUAL INDEPENDENT PATHWAYS (Green Top & Amber Bottom) */}
          <div className="lg:col-span-6 space-y-4 relative">
            
            {/* TOP GREEN PREDICTION PATH */}
            <div className="bg-white border border-[#E6E1DA] border-l-4 border-l-[#1B4332] rounded-2xl p-3.5 space-y-2 shadow-xs">
              <div className="flex items-center justify-between">
                <span className="inline-flex items-center gap-1 bg-[#D8F3DC] text-[#1B4332] border border-[#B7E4C7] px-2.5 py-0.5 text-[10px] font-mono font-bold rounded-md uppercase">
                  <Layers className="h-3 w-3" /> PREDICTION PATH
                </span>
                <span className="text-[10px] font-mono text-[#1B4332] font-bold inline-flex items-center gap-1">
                  ResNet-18 <ArrowRight className="h-3 w-3 text-[#1B4332]" />
                </span>
              </div>

              <div className="flex items-center space-x-2">
                <div className="flex-1 bg-[#FAF8F5] border border-[#E6E1DA] rounded-xl p-2.5 space-y-0.5">
                  <p className="font-bold text-[#111827] text-xs">Vision Predictor</p>
                  <p className="text-[10px] text-[#6B7280]">ResNet-18</p>
                </div>

                <ArrowRight className="h-4 w-4 text-[#1B4332] shrink-0" />

                <div className="flex-1 bg-[#FAF8F5] border border-[#E6E1DA] rounded-xl p-2.5 space-y-0.5">
                  <p className="font-bold text-[#111827] text-xs">Class + Confidence</p>
                  <p className="text-[10px] text-[#6B7280]">Class probabilities & confidence</p>
                </div>
              </div>
            </div>

            {/* BOTTOM AMBER RELIABILITY PATH */}
            <div className="bg-white border border-[#E6E1DA] border-l-4 border-l-[#D97706] rounded-2xl p-3.5 space-y-2 shadow-xs">
              <div className="flex items-center justify-between">
                <span className="inline-flex items-center gap-1 bg-[#FEF3C7] text-[#92400E] border border-[#FDE68A] px-2.5 py-0.5 text-[10px] font-mono font-bold rounded-md uppercase">
                  <Activity className="h-3 w-3" /> RELIABILITY PATH
                </span>
                <span className="text-[10px] font-mono text-[#D97706] font-bold inline-flex items-center gap-1">
                  Independent Model <ArrowRight className="h-3 w-3 text-[#D97706]" />
                </span>
              </div>

              <div className="flex items-center space-x-1.5">
                <div className="flex-1 bg-[#FAF8F5] border border-[#E6E1DA] rounded-xl p-2 space-y-0.5">
                  <p className="font-bold text-[#111827] text-[11px]">Reliability Signals</p>
                  <p className="text-[9px] text-[#6B7280]">6 categories</p>
                </div>

                <ArrowRight className="h-3.5 w-3.5 text-[#D97706] shrink-0" />

                <div className="flex-1 bg-[#FAF8F5] border border-[#E6E1DA] rounded-xl p-2 space-y-0.5">
                  <p className="font-bold text-[#111827] text-[11px]">Random Forest</p>
                  <p className="text-[9px] text-[#6B7280]">Trust Model</p>
                </div>

                <ArrowRight className="h-3.5 w-3.5 text-[#D97706] shrink-0" />

                <div className="flex-1 bg-[#FAF8F5] border border-[#E6E1DA] rounded-xl p-2 space-y-0.5">
                  <p className="font-bold text-[#111827] text-[11px]">Trust Probability</p>
                  <p className="text-[9px] text-[#6B7280]">P(correct)</p>
                </div>
              </div>
            </div>

          </div>

          {/* 3. FINAL DECISION CONVERGENCE CARD (Right) */}
          <div className="lg:col-span-3">
            {!hasResult ? (
              /* EMPTY STATE: Neutral card */
              <div className="bg-white border border-[#E6E1DA] rounded-2xl p-5 flex flex-col items-center justify-center text-center shadow-xs min-h-[230px]">
                <div className="h-10 w-10 rounded-full bg-[#FAF8F5] border border-[#E6E1DA] flex items-center justify-center text-[#9CA3AF] mb-2.5">
                  <Clock className="h-5 w-5" />
                </div>
                <h3 className="text-base font-extrabold text-[#111827] tracking-tight">
                  AWAITING ANALYSIS
                </h3>
                <p className="text-xs text-[#6B7280] mt-1 font-medium">
                  Upload an image to evaluate trust.
                </p>
                <div className="mt-3 pt-2.5 border-t border-[#E6E1DA] w-full text-[10px] font-mono text-[#9CA3AF] tracking-wider uppercase">
                  TRUSTED / UNTRUSTED
                </div>
              </div>
            ) : isTrusted ? (
              /* TRUSTED STATE: Green card */
              <div className="bg-[#E8F5E9] border border-[#A3E635] rounded-2xl p-5 flex flex-col items-center justify-center text-center shadow-xs min-h-[230px]">
                <div className="h-10 w-10 rounded-full bg-[#1B4332] text-white flex items-center justify-center mb-2.5 shadow-xs">
                  <CheckCircle2 className="h-6 w-6 text-emerald-300" />
                </div>
                <h3 className="text-base font-extrabold text-[#111827] tracking-tight">
                  TRUSTED
                </h3>
                <p className="text-xs font-semibold text-[#1B4332] mt-1">
                  Reliable prediction
                </p>
                <p className="text-[11px] font-mono text-[#1B4332] mt-2 pt-2 border-t border-[#A3E635]/60 w-full font-bold">
                  Trust P: {(result.trust.trust_probability * 100).toFixed(1)}%
                </p>
              </div>
            ) : (
              /* UNTRUSTED STATE: Coral card */
              <div className="bg-[#FFEBEE] border border-[#FFCDD2] rounded-2xl p-5 flex flex-col items-center justify-center text-center shadow-xs min-h-[230px]">
                <div className="h-10 w-10 rounded-full bg-[#C53030] text-white flex items-center justify-center mb-2.5 shadow-xs">
                  <AlertTriangle className="h-6 w-6 text-red-200" />
                </div>
                <h3 className="text-base font-extrabold text-[#9B2C2C] tracking-tight">
                  UNTRUSTED
                </h3>
                <p className="text-xs font-semibold text-[#C53030] mt-1">
                  Prediction may be unreliable
                </p>
                <p className="text-[11px] font-mono text-[#9B2C2C] mt-2 pt-2 border-t border-[#FFCDD2]/80 w-full font-bold">
                  Failure Risk: {(result.trust.failure_probability * 100).toFixed(1)}%
                </p>
              </div>
            )}
          </div>

        </div>

        {/* BOTTOM SUPPORTING POINTS (3 Columns spanning width inside architecture card) */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-6 border-t border-[#E6E1DA]">
          
          <div className="flex items-start space-x-3 bg-white p-3.5 rounded-xl border border-[#E6E1DA] shadow-xs">
            <div className="h-8 w-8 rounded-lg bg-[#FAF8F5] border border-[#E6E1DA] flex items-center justify-center text-[#1B4332] shrink-0">
              <Layers className="h-4 w-4" />
            </div>
            <div>
              <p className="text-xs font-bold text-[#111827]">Separation of concerns</p>
              <p className="text-[11px] text-[#6B7280]">Prediction and reliability are independent.</p>
            </div>
          </div>

          <div className="flex items-start space-x-3 bg-white p-3.5 rounded-xl border border-[#E6E1DA] shadow-xs">
            <div className="h-8 w-8 rounded-lg bg-[#FAF8F5] border border-[#E6E1DA] flex items-center justify-center text-[#1B4332] shrink-0">
              <ShieldCheck className="h-4 w-4" />
            </div>
            <div>
              <p className="text-xs font-bold text-[#111827]">Robust to distribution shift</p>
              <p className="text-[11px] text-[#6B7280]">Detects when inputs may be unreliable.</p>
            </div>
          </div>

          <div className="flex items-start space-x-3 bg-white p-3.5 rounded-xl border border-[#E6E1DA] shadow-xs">
            <div className="h-8 w-8 rounded-lg bg-[#FAF8F5] border border-[#E6E1DA] flex items-center justify-center text-[#1B4332] shrink-0">
              <BarChart2 className="h-4 w-4" />
            </div>
            <div>
              <p className="text-xs font-bold text-[#111827]">More trustworthy outputs</p>
              <p className="text-[11px] text-[#6B7280]">Know when to trust, not just what the model predicts.</p>
            </div>
          </div>

        </div>

      </div>

    </section>
  );
};
