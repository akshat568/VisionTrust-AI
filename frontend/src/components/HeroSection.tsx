import React from 'react';
import { ArrowRight, ArrowDown, Cpu, Database, ShieldCheck, CheckCircle2, AlertTriangle, Layers, Sparkles, BarChart2, Clock, Image as ImageIcon } from 'lucide-react';
import type { PredictResponse } from '../types/api';

interface HeroSectionProps {
  result?: PredictResponse | null;
  previewUrl?: string | null;
}

const CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];

export const HeroSection: React.FC<HeroSectionProps> = ({ result, previewUrl }) => {
  const hasResult = !!result;
  const isTrusted = result ? result.trust.status === 'TRUSTED' : false;
  const trustProbability = result ? result.trust.trust_probability : null;
  const trustPercent = trustProbability !== null ? Math.round(trustProbability * 100) : 0;

  // Classes list: real top probabilities from vision response if result present, or neutral placeholders if awaiting analysis
  const topClasses = hasResult && result?.vision?.probabilities
    ? result.vision.probabilities
        .map((prob: number, idx: number) => ({
          label: CIFAR10_CLASSES[idx] || `class_${idx}`,
          prob: prob.toFixed(2),
          widthPercent: Math.max(5, Math.round(prob * 100)),
        }))
        .sort((a: { prob: string }, b: { prob: string }) => parseFloat(b.prob) - parseFloat(a.prob))
        .slice(0, 4)
    : [
        { label: 'Class 1', prob: '--', widthPercent: 0 },
        { label: 'Class 2', prob: '--', widthPercent: 0 },
        { label: 'Class 3', prob: '--', widthPercent: 0 },
        { label: 'Class 4', prob: '--', widthPercent: 0 },
      ];

  return (
    <section className="bg-grid-pattern rounded-3xl border border-[#E6E1DA] p-5 sm:p-7 lg:p-9 bg-white/70 relative overflow-hidden space-y-8 shadow-xs">
      
      {/* Main Two-Column Desktop Hero Layout (~50% / ~50% split) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-10 items-center">
        
        {/* LEFT COLUMN: Editorial Copy & Value Propositions */}
        <div className="lg:col-span-6 space-y-6">
          
          {/* Eyebrow Tag */}
          <p className="text-xs font-mono tracking-widest text-[#6B7280] uppercase font-bold">
            RELIABILITY LAYER FOR VISION MODELS
          </p>

          {/* Headline with Soft Green Highlight Pill */}
          <h1 className="text-3xl sm:text-4xl lg:text-[46px] font-extrabold text-[#111827] tracking-tight leading-[1.12]">
            Know when a{' '}
            <span className="block mt-1">
              computer-vision{' '}
              <span className="bg-[#D8F3DC] text-[#111827] px-2 py-0.5 rounded-lg inline-block">
                prediction should be trusted.
              </span>
            </span>
          </h1>

          {/* Supporting Narrative */}
          <p className="text-base text-[#4B5563] leading-relaxed max-w-xl">
            VisionTrust AI separates prediction confidence from prediction reliability, using an independent reliability layer to estimate when a model output may fail.
          </p>

          {/* Action Buttons */}
          <div className="flex flex-wrap items-center gap-3 pt-1">
            <a
              href="#analyze"
              className="inline-flex items-center space-x-2 bg-[#1B4332] hover:bg-[#143326] text-white px-6 py-3.5 rounded-xl font-medium text-sm transition-all shadow-xs"
            >
              <span>Analyze an image</span>
              <ArrowRight className="h-4 w-4" />
            </a>
            <a
              href="#architecture"
              className="inline-flex items-center space-x-2 bg-white hover:bg-[#FAF8F5] text-[#111827] border border-[#E6E1DA] px-6 py-3.5 rounded-xl font-medium text-sm transition-all shadow-xs"
            >
              <span>How it works</span>
            </a>
          </div>

          {/* Three Value Points Below Buttons */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-4 border-t border-[#E6E1DA]/60">
            <div className="flex items-start space-x-2.5">
              <div className="h-7 w-7 rounded-lg bg-[#E8F5E9] border border-[#C8E6C9] flex items-center justify-center text-[#1B4332] shrink-0 mt-0.5">
                <ShieldCheck className="h-4 w-4" />
              </div>
              <div>
                <p className="text-xs font-bold text-[#111827]">More reliable AI</p>
                <p className="text-[11px] text-[#6B7280]">Detect uncertain predictions</p>
              </div>
            </div>

            <div className="flex items-start space-x-2.5">
              <div className="h-7 w-7 rounded-lg bg-[#E8F5E9] border border-[#C8E6C9] flex items-center justify-center text-[#1B4332] shrink-0 mt-0.5">
                <BarChart2 className="h-4 w-4" />
              </div>
              <div>
                <p className="text-xs font-bold text-[#111827]">Built for real-world data</p>
                <p className="text-[11px] text-[#6B7280]">Handles distribution shifts</p>
              </div>
            </div>

            <div className="flex items-start space-x-2.5">
              <div className="h-7 w-7 rounded-lg bg-[#E8F5E9] border border-[#C8E6C9] flex items-center justify-center text-[#1B4332] shrink-0 mt-0.5">
                <Cpu className="h-4 w-4" />
              </div>
              <div>
                <p className="text-xs font-bold text-[#111827]">Transparent decisions</p>
                <p className="text-[11px] text-[#6B7280]">Understand why to trust</p>
              </div>
            </div>
          </div>

        </div>

        {/* RIGHT COLUMN: Polished Architecture Visualization Card */}
        <div className="lg:col-span-6 bg-[#FAF8F5]/90 border border-[#E6E1DA] rounded-3xl p-5 sm:p-6 space-y-5 shadow-xs relative">
          
          {/* 3 Architecture Pathway Columns */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 items-stretch relative">
            
            {/* Column 1: Prediction Path */}
            <div className="bg-white border border-[#E6E1DA] rounded-2xl p-3 flex flex-col justify-between space-y-2.5 shadow-xs">
              <div className="text-center">
                <span className="inline-flex items-center gap-1 bg-[#D8F3DC] text-[#1B4332] border border-[#B7E4C7] px-2.5 py-1 text-[10px] font-mono font-bold rounded-md uppercase">
                  <Layers className="h-3 w-3" /> PREDICTION PATH
                </span>
              </div>

              <div className="bg-[#FAF8F5] border border-[#E6E1DA] rounded-xl p-2.5 text-center flex items-center justify-center gap-1.5">
                <Layers className="h-3.5 w-3.5 text-[#1B4332]" />
                <div>
                  <p className="text-xs font-bold text-[#111827]">Vision Model</p>
                  <p className="text-[10px] text-[#6B7280]">(ResNet-18)</p>
                </div>
              </div>

              <div className="flex justify-center text-[#9CA3AF]">
                <ArrowDown className="h-3.5 w-3.5" />
              </div>

              <div className="bg-[#FAF8F5] border border-[#E6E1DA] rounded-xl p-2 text-center">
                <p className="text-xs font-bold text-[#111827]">Class Prediction</p>
                <p className="text-[10px] text-[#6B7280]">+ Confidence</p>
              </div>

              {/* Class Probabilities Bars Representation */}
              <div className="space-y-1.5 pt-1 bg-[#FAF8F5] border border-[#E6E1DA] rounded-xl p-2">
                {topClasses.map((item, idx) => (
                  <div key={idx} className="flex items-center justify-between text-[10px] font-mono">
                    <span className="text-[#374151] truncate max-w-[55px]">{item.label}</span>
                    <div className="flex-1 mx-1.5 h-1.5 bg-[#E6E1DA] rounded-full overflow-hidden">
                      <div
                        className="h-full bg-[#1B4332] rounded-full transition-all duration-500"
                        style={{ width: `${item.widthPercent}%` }}
                      />
                    </div>
                    <span className="text-[#6B7280] shrink-0">{item.prob}</span>
                  </div>
                ))}
                {!hasResult && (
                  <p className="text-[9px] text-[#9CA3AF] text-center font-mono pt-0.5">...</p>
                )}
              </div>
            </div>

            {/* Column 2: Input Image Card */}
            <div className="bg-white border border-[#E6E1DA] rounded-2xl p-3 flex flex-col items-center justify-between text-center shadow-xs">
              <span className="text-[10px] font-mono font-bold text-[#6B7280] uppercase tracking-wider">
                INPUT IMAGE
              </span>
              
              {previewUrl ? (
                <div className="h-32 sm:h-36 w-full rounded-xl overflow-hidden border border-[#E6E1DA] bg-[#FAF8F5] flex items-center justify-center p-2 my-2">
                  <img
                    src={previewUrl}
                    alt="Input Image Preview"
                    className="max-h-full max-w-full object-contain rounded-lg"
                  />
                </div>
              ) : (
                <div className="h-32 sm:h-36 w-full rounded-xl overflow-hidden border border-dashed border-[#D1D5DB] bg-[#FAF8F5] flex flex-col items-center justify-center p-3 my-2 text-center">
                  <div className="h-9 w-9 rounded-full bg-white border border-[#E6E1DA] flex items-center justify-center text-[#9CA3AF] mb-1.5">
                    <ImageIcon className="h-4 w-4" />
                  </div>
                  <p className="text-xs font-semibold text-[#374151]">Upload an image to analyze</p>
                  <p className="text-[10px] text-[#9CA3AF] mt-0.5">PNG, JPG or WEBP (up to 10 MB)</p>
                </div>
              )}

              <div>
                <p className="text-xs font-bold text-[#111827]">Image (224 × 224)</p>
                <p className="text-[10px] text-[#6B7280]">
                  {previewUrl ? 'Analyzed image' : 'Upload any image to analyze'}
                </p>
              </div>
            </div>

            {/* Column 3: Reliability Path */}
            <div className="bg-white border border-[#E6E1DA] rounded-2xl p-3 flex flex-col justify-between space-y-2.5 shadow-xs">
              <div className="text-center">
                <span className="inline-flex items-center gap-1 bg-[#FEF3C7] text-[#92400E] border border-[#FDE68A] px-2.5 py-1 text-[10px] font-mono font-bold rounded-md uppercase">
                  <Sparkles className="h-3 w-3" /> RELIABILITY PATH
                </span>
              </div>

              <div className="bg-[#FAF8F5] border border-[#E6E1DA] rounded-xl p-2 text-center">
                <p className="text-xs font-bold text-[#111827]">Reliability Signals</p>
                <p className="text-[10px] text-[#6B7280]">(6 categories)</p>
              </div>

              <div className="flex justify-center text-[#9CA3AF]">
                <ArrowDown className="h-3.5 w-3.5" />
              </div>

              <div className="bg-[#FAF8F5] border border-[#E6E1DA] rounded-xl p-2 text-center">
                <p className="text-xs font-bold text-[#111827]">Random Forest</p>
                <p className="text-[10px] text-[#6B7280]">Trust Model</p>
              </div>

              <div className="flex justify-center text-[#9CA3AF]">
                <ArrowDown className="h-3.5 w-3.5" />
              </div>

              {/* Trust Probability Score Card */}
              <div className="bg-[#FAF8F5] border border-[#E6E1DA] rounded-xl p-2 space-y-1.5">
                <div className="flex justify-between items-center text-[10px] font-bold text-[#111827] font-mono">
                  <span>Trust Probability</span>
                  <span>{trustProbability !== null ? trustProbability.toFixed(2) : '--'}</span>
                </div>
                <div className="h-2 w-full bg-[#E6E1DA] rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      !hasResult ? 'bg-transparent' : isTrusted ? 'bg-[#1B4332]' : 'bg-[#C53030]'
                    }`}
                    style={{ width: `${trustPercent}%` }}
                  />
                </div>
              </div>
            </div>

          </div>

          {/* Converging Bottom Verdict Banner */}
          <div className="pt-2 flex flex-col items-center relative w-full">
            <div
              className={`w-full py-3.5 px-6 rounded-2xl font-mono text-xs sm:text-sm flex items-center justify-start sm:justify-center space-x-3.5 shadow-xs transition-all border ${
                !hasResult
                  ? 'bg-white text-[#111827] border-[#E6E1DA]'
                  : isTrusted
                  ? 'bg-[#1B4332] text-white border-[#2D6A4F]'
                  : 'bg-[#C53030] text-white border-[#E53E3E]'
              }`}
            >
              {!hasResult ? (
                <>
                  <div className="h-9 w-9 rounded-full bg-[#FAF8F5] border border-[#E6E1DA] flex items-center justify-center text-[#6B7280] shrink-0">
                    <Clock className="h-5 w-5" />
                  </div>
                  <div className="text-left font-sans">
                    <p className="text-sm font-bold text-[#111827]">Awaiting Analysis</p>
                    <p className="text-xs text-[#6B7280]">Upload an image to evaluate trust</p>
                  </div>
                </>
              ) : isTrusted ? (
                <>
                  <CheckCircle2 className="h-5 w-5 text-emerald-300 shrink-0" />
                  <span className="font-bold">TRUSTED — Reliable prediction</span>
                </>
              ) : (
                <>
                  <AlertTriangle className="h-5 w-5 text-red-200 shrink-0" />
                  <span className="font-bold">UNTRUSTED — High failure risk</span>
                </>
              )}
            </div>
          </div>

        </div>

      </div>

      {/* Bottom Information Strip (Full Width across Hero) */}
      <div className="grid grid-cols-1 sm:grid-cols-3 bg-white rounded-2xl border border-[#E6E1DA] divide-y sm:divide-y-0 sm:divide-x divide-[#E6E1DA] shadow-xs p-1">
        
        {/* Column 1: Backbone */}
        <div className="p-4 flex items-start space-x-3.5">
          <div className="h-10 w-10 rounded-xl bg-[#FAF8F5] border border-[#E6E1DA] flex items-center justify-center text-[#1B4332] shrink-0">
            <Cpu className="h-5 w-5" />
          </div>
          <div className="space-y-0.5">
            <p className="text-[10px] font-mono uppercase tracking-wider text-[#9CA3AF] font-bold">
              BACKBONE
            </p>
            <p className="text-sm font-extrabold text-[#111827]">
              ResNet-18
            </p>
            <p className="text-xs text-[#6B7280]">
              Pre-trained computer vision model
            </p>
          </div>
        </div>

        {/* Column 2: Benchmark */}
        <div className="p-4 flex items-start space-x-3.5">
          <div className="h-10 w-10 rounded-xl bg-[#FAF8F5] border border-[#E6E1DA] flex items-center justify-center text-[#1B4332] shrink-0">
            <Database className="h-5 w-5" />
          </div>
          <div className="space-y-0.5">
            <p className="text-[10px] font-mono uppercase tracking-wider text-[#9CA3AF] font-bold">
              BENCHMARK
            </p>
            <p className="text-sm font-extrabold text-[#111827]">
              CIFAR-10
            </p>
            <p className="text-xs text-[#6B7280]">
              10 classes, 32 × 32 images
            </p>
          </div>
        </div>

        {/* Column 3: Trust Layer */}
        <div className="p-4 flex items-start space-x-3.5">
          <div className="h-10 w-10 rounded-xl bg-[#FAF8F5] border border-[#E6E1DA] flex items-center justify-center text-[#1B4332] shrink-0">
            <ShieldCheck className="h-5 w-5" />
          </div>
          <div className="space-y-0.5">
            <p className="text-[10px] font-mono uppercase tracking-wider text-[#9CA3AF] font-bold">
              TRUST LAYER
            </p>
            <p className="text-sm font-extrabold text-[#111827]">
              Independent
            </p>
            <p className="text-xs text-[#6B7280]">
              Random Forest Trust Model
            </p>
          </div>
        </div>

      </div>

    </section>
  );
};
