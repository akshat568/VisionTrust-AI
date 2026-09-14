import React from 'react';
import { Image as ImageIcon, Layers, Target, ShieldCheck, Gauge, ArrowDown } from 'lucide-react';

export const PipelineSection: React.FC = () => {
  return (
    <section id="pipeline" className="bg-white border border-[#E6E1DA] rounded-3xl p-6 sm:p-10 shadow-xs space-y-8">
      
      {/* Header */}
      <div className="space-y-2 border-b border-[#E6E1DA] pb-6">
        <p className="text-xs font-mono tracking-widest text-[#6B7280] uppercase font-semibold">
          END-TO-END WORKFLOW
        </p>
        <h2 className="text-2xl sm:text-3xl font-extrabold text-[#111827] tracking-tight">
          The evaluation pipeline
        </h2>
        <p className="text-sm sm:text-base text-[#4B5563] max-w-3xl leading-relaxed">
          From raw pixel input to independent trust assessment — how VisionTrust AI evaluates each prediction.
        </p>
      </div>

      {/* Pipeline Visual Container */}
      <div className="bg-[#FAF8F5] border border-[#E6E1DA] rounded-2xl p-6 sm:p-8 space-y-8 shadow-xs relative">
        
        {/* Step 01: Input Image */}
        <div className="flex justify-center">
          <div className="bg-white border border-[#E6E1DA] rounded-2xl px-6 py-3.5 flex items-center space-x-3 shadow-xs">
            <div className="p-2 rounded-lg bg-[#FAF8F5] text-[#1B4332]">
              <ImageIcon className="h-4 w-4" />
            </div>
            <div>
              <span className="text-[10px] font-mono text-[#9CA3AF] block font-bold">01. INPUT</span>
              <span className="text-xs font-extrabold text-[#111827]">Normalized Image Tensor (32×32)</span>
            </div>
          </div>
        </div>

        <div className="flex justify-center text-[#9CA3AF]">
          <ArrowDown className="h-5 w-5" />
        </div>

        {/* Parallel Split Pipeline */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-stretch">
          
          {/* Left: Classifier Branch */}
          <div className="bg-white border border-[#E6E1DA] rounded-2xl p-6 space-y-4 shadow-xs flex flex-col justify-between">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-[#9CA3AF] font-bold">02. CLASSIFIER</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#FAF8F5] text-[#374151] border border-[#E6E1DA]">
                  VISION PATH
                </span>
              </div>
              <div className="flex items-center space-x-2.5">
                <div className="p-2 rounded-lg bg-[#FAF8F5] text-[#111827]">
                  <Layers className="h-4 w-4" />
                </div>
                <h3 className="text-sm font-bold text-[#111827]">Vision Predictor (ResNet-18)</h3>
              </div>
              <p className="text-xs text-[#6B7280]">
                Computes class probability logits and softmax confidence.
              </p>
            </div>

            <div className="pt-3 border-t border-[#E6E1DA] text-xs font-mono font-bold text-[#111827]">
              Output: Predicted Class + Softmax Confidence
            </div>
          </div>

          {/* Right: Reliability Layer Branch */}
          <div className="bg-white border border-[#E6E1DA] rounded-2xl p-6 space-y-4 shadow-xs flex flex-col justify-between">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono text-[#1B4332] font-bold">03 & 04. RELIABILITY</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-[#E8F5E9] text-[#1B4332] border border-[#C8E6C9]">
                  TRUST PATH
                </span>
              </div>

              <div className="space-y-3">
                <div className="flex items-center space-x-2.5">
                  <div className="p-2 rounded-lg bg-[#E8F5E9] text-[#1B4332]">
                    <Target className="h-4 w-4" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-[#111827]">03. Reliability Signals</h3>
                    <p className="text-xs text-[#6B7280]">Extracts 9 reliability features across 6 signal categories</p>
                  </div>
                </div>

                <div className="flex items-center space-x-2.5">
                  <div className="p-2 rounded-lg bg-[#E8F5E9] text-[#1B4332]">
                    <ShieldCheck className="h-4 w-4" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-[#111827]">04. Random Forest Trust Model</h3>
                    <p className="text-xs text-[#6B7280]">Estimates whether the original prediction is likely correct</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="pt-3 border-t border-[#E6E1DA] text-xs font-mono font-bold text-[#1B4332]">
              Output: Trust Probability P(correct)
            </div>
          </div>

        </div>

        <div className="flex justify-center text-[#9CA3AF]">
          <ArrowDown className="h-5 w-5" />
        </div>

        {/* Step 05: Final Verdict */}
        <div className="flex justify-center">
          <div className="bg-[#1B4332] text-white border border-[#143326] rounded-2xl px-8 py-4 flex items-center space-x-3 shadow-md font-mono text-xs sm:text-sm font-bold">
            <Gauge className="h-5 w-5 text-emerald-300" />
            <span>05. TRUSTED / UNTRUSTED VERDICT</span>
          </div>
        </div>

      </div>

    </section>
  );
};

