import React from 'react';
import { Layers, ShieldCheck, ArrowRight, ArrowLeft } from 'lucide-react';

export const CoreDistinctionSection: React.FC = () => {
  return (
    <section id="core-distinction" className="bg-white border border-[#E6E1DA] rounded-3xl p-6 sm:p-9 shadow-xs space-y-8">
      
      {/* Section Header */}
      <div className="space-y-3">
        <p className="text-xs font-mono tracking-widest text-[#6B7280] uppercase font-bold">
          CORE DISTINCTION
        </p>
        <h2 className="text-3xl sm:text-4xl font-extrabold text-[#111827] tracking-tight leading-tight">
          Confidence <span className="text-[#C53030] font-black">≠</span> Trust
        </h2>
        <p className="text-base text-[#4B5563] max-w-3xl leading-relaxed">
          A classifier can be highly confident and still be wrong. VisionTrust AI uses an independent reliability layer to estimate whether the prediction is likely to be correct.
        </p>
      </div>

      {/* Side by Side Two Large Conceptual Cards */}
      <div className="space-y-4">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          
          {/* LEFT CARD — MODEL CONFIDENCE */}
          <div className="lg:col-span-6 bg-[#FAF8F5] border border-[#E6E1DA] rounded-2xl p-6 sm:p-7 flex flex-col justify-between space-y-6 shadow-xs">
            
            <div className="space-y-5">
              {/* Header Row */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2.5">
                  <div className="h-8 w-8 rounded-lg bg-[#E8F5E9] border border-[#C8E6C9] flex items-center justify-center text-[#1B4332]">
                    <Layers className="h-4 w-4" />
                  </div>
                  <span className="text-xs font-mono tracking-widest text-[#111827] uppercase font-bold">
                    MODEL CONFIDENCE
                  </span>
                </div>
                <span className="text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full bg-[#E8F5E9] text-[#1B4332] border border-[#C8E6C9] uppercase">
                  CLASSIFIER
                </span>
              </div>

              <p className="text-sm font-semibold text-[#374151]">
                How strongly the vision model favors its prediction.
              </p>

              {/* Visual Conceptual Representation + 3 Points */}
              <div className="grid grid-cols-1 sm:grid-cols-12 gap-4 items-center pt-2">
                
                {/* Left: Non-numerical Probability Distribution Bars */}
                <div className="sm:col-span-5 bg-white border border-[#E6E1DA] rounded-xl p-3 space-y-2 text-center">
                  <div className="flex items-end justify-center space-x-1.5 h-20 pt-2">
                    <div className="flex flex-col items-center space-y-1">
                      <div className="w-3 bg-[#1B4332] h-16 rounded-t-xs" />
                      <span className="text-[9px] font-mono text-[#6B7280]">plane</span>
                    </div>
                    <div className="flex flex-col items-center space-y-1">
                      <div className="w-3 bg-[#9CA3AF] h-9 rounded-t-xs" />
                      <span className="text-[9px] font-mono text-[#6B7280]">car</span>
                    </div>
                    <div className="flex flex-col items-center space-y-1">
                      <div className="w-3 bg-[#D1D5DB] h-5 rounded-t-xs" />
                      <span className="text-[9px] font-mono text-[#6B7280]">bird</span>
                    </div>
                    <div className="flex flex-col items-center space-y-1">
                      <div className="w-3 bg-[#E6E1DA] h-3 rounded-t-xs" />
                      <span className="text-[9px] font-mono text-[#6B7280]">cat</span>
                    </div>
                    <div className="flex flex-col items-center space-y-1 justify-end h-full pb-3">
                      <span className="text-[9px] font-mono text-[#9CA3AF]">...</span>
                    </div>
                  </div>
                </div>

                {/* Right: 3 Concise Bullet Points */}
                <div className="sm:col-span-7 space-y-2.5 text-xs text-[#374151]">
                  <div className="flex items-start space-x-2.5">
                    <span className="h-5 w-5 rounded-full bg-[#E8F5E9] text-[#1B4332] font-mono font-bold text-[10px] flex items-center justify-center shrink-0 mt-0.5">
                      1
                    </span>
                    <span>Produced by Vision Predictor (ResNet-18)</span>
                  </div>
                  <div className="flex items-start space-x-2.5">
                    <span className="h-5 w-5 rounded-full bg-[#E8F5E9] text-[#1B4332] font-mono font-bold text-[10px] flex items-center justify-center shrink-0 mt-0.5">
                      2
                    </span>
                    <span>Based on classifier output probabilities</span>
                  </div>
                  <div className="flex items-start space-x-2.5">
                    <span className="h-5 w-5 rounded-full bg-[#E8F5E9] text-[#1B4332] font-mono font-bold text-[10px] flex items-center justify-center shrink-0 mt-0.5">
                      3
                    </span>
                    <span>Does not measure whether the input is reliable</span>
                  </div>
                </div>

              </div>
            </div>

            {/* Bottom Banner */}
            <div className="pt-3 border-t border-[#E6E1DA] flex items-center justify-between text-[11px] font-mono">
              <span className="text-[#6B7280] font-bold tracking-wider uppercase">ANSWERS:</span>
              <span className="text-[#111827] font-semibold">"WHICH CLASS DOES THE MODEL PREFER?"</span>
            </div>

          </div>

          {/* RIGHT CARD — TRUST PROBABILITY */}
          <div className="lg:col-span-6 bg-[#FAF8F5] border border-[#E6E1DA] rounded-2xl p-6 sm:p-7 flex flex-col justify-between space-y-6 shadow-xs">
            
            <div className="space-y-5">
              {/* Header Row */}
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2.5">
                  <div className="h-8 w-8 rounded-lg bg-[#FEF3C7] border border-[#FDE68A] flex items-center justify-center text-[#92400E]">
                    <ShieldCheck className="h-4 w-4" />
                  </div>
                  <span className="text-xs font-mono tracking-widest text-[#111827] uppercase font-bold">
                    TRUST PROBABILITY
                  </span>
                </div>
                <span className="text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full bg-[#FEF3C7] text-[#92400E] border border-[#FDE68A] uppercase">
                  RELIABILITY LAYER
                </span>
              </div>

              <p className="text-sm font-semibold text-[#374151]">
                Estimated probability that the prediction is correct.
              </p>

              {/* Visual Conceptual Representation + 3 Points */}
              <div className="grid grid-cols-1 sm:grid-cols-12 gap-4 items-center pt-2">
                
                {/* Left: Conceptual Trust Scale Bar */}
                <div className="sm:col-span-5 bg-white border border-[#E6E1DA] rounded-xl p-3 space-y-2 text-center">
                  <div className="space-y-2 py-2">
                    <div className="flex h-3 rounded-md overflow-hidden space-x-1">
                      <div className="flex-1 bg-[#FDE68A] rounded-xs" />
                      <div className="flex-1 bg-[#F59E0B] rounded-xs" />
                      <div className="flex-1 bg-[#D97706] rounded-xs" />
                      <div className="flex-1 bg-[#2D6A4F] rounded-xs" />
                      <div className="flex-1 bg-[#1B4332] rounded-xs" />
                    </div>
                    <div className="flex justify-between text-[9px] font-mono text-[#6B7280]">
                      <span>Lower trust<br />(higher risk)</span>
                      <span className="text-right">Higher trust<br />(more reliable)</span>
                    </div>
                  </div>
                </div>

                {/* Right: 3 Concise Bullet Points */}
                <div className="sm:col-span-7 space-y-2.5 text-xs text-[#374151]">
                  <div className="flex items-start space-x-2.5">
                    <span className="h-5 w-5 rounded-full bg-[#FEF3C7] text-[#92400E] font-mono font-bold text-[10px] flex items-center justify-center shrink-0 mt-0.5">
                      1
                    </span>
                    <span>Produced by the independent Random Forest Trust Model</span>
                  </div>
                  <div className="flex items-start space-x-2.5">
                    <span className="h-5 w-5 rounded-full bg-[#FEF3C7] text-[#92400E] font-mono font-bold text-[10px] flex items-center justify-center shrink-0 mt-0.5">
                      2
                    </span>
                    <span>Uses six reliability signal categories</span>
                  </div>
                  <div className="flex items-start space-x-2.5">
                    <span className="h-5 w-5 rounded-full bg-[#FEF3C7] text-[#92400E] font-mono font-bold text-[10px] flex items-center justify-center shrink-0 mt-0.5">
                      3
                    </span>
                    <span>Estimates whether the prediction may fail</span>
                  </div>
                </div>

              </div>
            </div>

            {/* Bottom Banner */}
            <div className="pt-3 border-t border-[#E6E1DA] flex items-center justify-between text-[11px] font-mono">
              <span className="text-[#6B7280] font-bold tracking-wider uppercase">ANSWERS:</span>
              <span className="text-[#111827] font-semibold">"IS THIS PREDICTION LIKELY TO BE CORRECT?"</span>
            </div>

          </div>

        </div>

        {/* Center Integrated Separator Label */}
        <div className="flex items-center justify-center pt-2">
          <div className="bg-white border border-[#E6E1DA] rounded-full px-4 py-1.5 text-xs font-mono font-bold text-[#6B7280] shadow-xs flex items-center space-x-2">
            <ArrowLeft className="h-3.5 w-3.5 text-[#9CA3AF]" />
            <span>DIFFERENT QUESTIONS</span>
            <ArrowRight className="h-3.5 w-3.5 text-[#9CA3AF]" />
          </div>
        </div>
      </div>

    </section>
  );
};
