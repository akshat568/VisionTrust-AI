import React from 'react';
import { BarChart2, ShieldCheck, Target, Filter, CheckCircle2, Trophy, Sliders } from 'lucide-react';

export const BenchmarkSection: React.FC = () => {
  return (
    <section id="benchmarks" className="bg-white border border-[#E6E1DA] rounded-3xl p-6 sm:p-9 shadow-xs space-y-8">
      
      {/* 1. Header with Verification Badge Card */}
      <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-6 border-b border-[#E6E1DA] pb-6">
        <div className="space-y-3 max-w-3xl">
          <p className="text-xs font-mono tracking-widest text-[#6B7280] uppercase font-bold">
            EVALUATION
          </p>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-[#111827] tracking-tight">
            Benchmark preview
          </h2>
          <p className="text-base text-[#4B5563] leading-relaxed">
            Reliability is measured, not claimed. Verified empirical metrics from the CIFAR-10 evaluation run.
          </p>
        </div>

        {/* Upper Right Verification Badge Card */}
        <div className="bg-[#E8F5E9]/60 border border-[#C8E6C9] rounded-2xl p-4 flex items-start space-x-3.5 max-w-sm shrink-0">
          <div className="h-9 w-9 rounded-xl bg-white border border-[#C8E6C9] flex items-center justify-center text-[#1B4332] shrink-0 mt-0.5 shadow-xs">
            <Trophy className="h-4 w-4" />
          </div>
          <div>
            <p className="text-xs font-mono font-bold text-[#111827] uppercase">cifar-10 benchmark verified</p>
            <p className="text-[11px] text-[#4B5563] leading-normal mt-0.5">
              All metrics computed on the standard CIFAR-10 test set (10,000 images).
            </p>
          </div>
        </div>
      </div>

      {/* 2. Main Two-Column Evaluation Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        
        {/* LEFT COLUMN: Dataset + Core Metrics Card (6 cols) */}
        <div className="lg:col-span-6 bg-[#FAF8F5] border border-[#E6E1DA] rounded-2xl p-6 flex flex-col justify-between space-y-3.5 shadow-xs">
          
          {/* Top Header Row */}
          <div className="flex items-center justify-between border-b border-[#E6E1DA] pb-3">
            <div>
              <p className="text-xs font-mono font-bold text-[#111827] uppercase tracking-wider">
                Dataset
              </p>
              <p className="text-xs text-[#6B7280]">10 classes · 32×32 resolution</p>
            </div>
            <span className="text-xs font-mono font-bold text-[#1B4332] bg-white border border-[#E6E1DA] px-3 py-1 rounded-full shadow-xs">
              CIFAR-10 (10,000 Test Split)
            </span>
          </div>

          {/* Metric Row 1: Clean Accuracy */}
          <div className="bg-white border border-[#E6E1DA] rounded-xl p-3.5 flex items-center justify-between">
            <div className="flex items-start space-x-3">
              <div className="h-8 w-8 rounded-lg bg-[#FAF8F5] border border-[#E6E1DA] flex items-center justify-center text-[#1B4332] shrink-0 mt-0.5">
                <BarChart2 className="h-4 w-4" />
              </div>
              <div>
                <p className="text-xs font-bold text-[#111827]">Clean accuracy</p>
                <p className="text-[11px] text-[#6B7280]">Top-1 accuracy on clean test set</p>
              </div>
            </div>
            <span className="text-base font-extrabold text-[#111827] font-mono">81.25%</span>
          </div>

          {/* Metric Row 2: Failure Detection AUROC */}
          <div className="bg-white border border-[#E6E1DA] rounded-xl p-3.5 flex items-center justify-between">
            <div className="flex items-start space-x-3">
              <div className="h-8 w-8 rounded-lg bg-[#FAF8F5] border border-[#E6E1DA] flex items-center justify-center text-[#1B4332] shrink-0 mt-0.5">
                <ShieldCheck className="h-4 w-4" />
              </div>
              <div>
                <p className="text-xs font-bold text-[#111827]">Failure detection AUROC</p>
                <p className="text-[11px] text-[#6B7280]">Separating correct from incorrect predictions</p>
              </div>
            </div>
            <div className="text-right">
              <span className="text-xs font-extrabold text-[#1B4332] font-mono block">86.69% (Trust Model)</span>
              <span className="text-[10px] text-[#6B7280] font-mono">vs 85.61% (Softmax)</span>
            </div>
          </div>

          {/* Metric Row 3: Failure Detection AUPRC */}
          <div className="bg-white border border-[#E6E1DA] rounded-xl p-3.5 flex items-center justify-between">
            <div className="flex items-start space-x-3">
              <div className="h-8 w-8 rounded-lg bg-[#FAF8F5] border border-[#E6E1DA] flex items-center justify-center text-[#1B4332] shrink-0 mt-0.5">
                <Target className="h-4 w-4" />
              </div>
              <div>
                <p className="text-xs font-bold text-[#111827]">Failure detection AUPRC</p>
                <p className="text-[11px] text-[#6B7280]">Precision-recall AUC for misclassifications</p>
              </div>
            </div>
            <div className="text-right">
              <span className="text-xs font-extrabold text-[#1B4332] font-mono block">0.5708</span>
              <span className="text-[10px] text-[#1B4332] font-mono font-bold">+0.0357 absolute gain vs Softmax</span>
            </div>
          </div>

          {/* Metric Row 4: Selective Prediction */}
          <div className="bg-white border border-[#E6E1DA] rounded-xl p-3.5 flex items-center justify-between">
            <div className="flex items-start space-x-3">
              <div className="h-8 w-8 rounded-lg bg-[#FAF8F5] border border-[#E6E1DA] flex items-center justify-center text-[#1B4332] shrink-0 mt-0.5">
                <Filter className="h-4 w-4" />
              </div>
              <div>
                <p className="text-xs font-bold text-[#111827]">Selective prediction (80% coverage)</p>
                <p className="text-[11px] text-[#6B7280]">Retained accuracy filtering lowest 20% trust</p>
              </div>
            </div>
            <div className="text-right">
              <span className="text-xs font-extrabold text-[#1B4332] font-mono block">90.33% Accuracy</span>
              <span className="text-[10px] text-[#C53030] font-mono font-bold">9.68% Risk</span>
            </div>
          </div>

        </div>

        {/* RIGHT COLUMN: Failure Detection Comparison & Selective Prediction (6 cols) */}
        <div className="lg:col-span-6 space-y-6 flex flex-col justify-between">
          
          {/* Top Card: Failure Detection Comparison */}
          <div className="bg-[#FAF8F5] border border-[#E6E1DA] rounded-2xl p-5 sm:p-6 space-y-4 shadow-xs">
            <div className="flex items-center justify-between border-b border-[#E6E1DA] pb-3">
              <div>
                <h3 className="text-xs font-mono font-bold text-[#111827] uppercase tracking-wider">
                  FAILURE DETECTION COMPARISON
                </h3>
                <p className="text-[11px] text-[#6B7280] mt-0.5">
                  The trust model better separates correct and incorrect predictions.
                </p>
              </div>
              <div className="flex items-center space-x-3 text-[10px] font-mono shrink-0">
                <span className="flex items-center gap-1 text-[#1B4332] font-bold">
                  <span className="h-2 w-2 rounded-full bg-[#1B4332]"></span> Trust Model
                </span>
                <span className="flex items-center gap-1 text-[#D97706] font-bold">
                  <span className="h-2 w-2 rounded-full bg-[#D97706]"></span> Softmax
                </span>
              </div>
            </div>

            {/* Comparison Charts */}
            <div className="grid grid-cols-2 gap-4 pt-1">
              
              {/* AUROC Chart */}
              <div className="bg-white border border-[#E6E1DA] rounded-xl p-3.5 space-y-3 text-center">
                <div className="flex justify-between text-[10px] font-mono text-[#6B7280]">
                  <span className="font-bold text-[#111827]">AUROC</span>
                  <span>Higher is better</span>
                </div>
                <div className="flex items-end justify-center space-x-3 h-28 pt-2">
                  <div className="flex flex-col items-center flex-1 h-full justify-end">
                    <span className="text-[10px] font-mono font-extrabold text-[#1B4332] mb-1">86.69%</span>
                    <div className="w-full bg-[#1B4332] rounded-t-sm h-[86%]" />
                    <span className="text-[9px] font-mono text-[#6B7280] mt-1.5">Trust</span>
                  </div>
                  <div className="flex flex-col items-center flex-1 h-full justify-end">
                    <span className="text-[10px] font-mono font-bold text-[#D97706] mb-1">85.61%</span>
                    <div className="w-full bg-[#D97706] rounded-t-sm h-[85%]" />
                    <span className="text-[9px] font-mono text-[#6B7280] mt-1.5">Softmax</span>
                  </div>
                </div>
              </div>

              {/* AUPRC Chart */}
              <div className="bg-white border border-[#E6E1DA] rounded-xl p-3.5 space-y-3 text-center">
                <div className="flex justify-between text-[10px] font-mono text-[#6B7280]">
                  <span className="font-bold text-[#111827]">AUPRC</span>
                  <span>Higher is better</span>
                </div>
                <div className="flex items-end justify-center space-x-3 h-28 pt-2">
                  <div className="flex flex-col items-center flex-1 h-full justify-end">
                    <span className="text-[10px] font-mono font-extrabold text-[#1B4332] mb-1">0.5708</span>
                    <div className="w-full bg-[#1B4332] rounded-t-sm h-[80%]" />
                    <span className="text-[9px] font-mono text-[#6B7280] mt-1.5">Trust</span>
                  </div>
                  <div className="flex flex-col items-center flex-1 h-full justify-end">
                    <span className="text-[10px] font-mono font-bold text-[#D97706] mb-1">0.5351</span>
                    <div className="w-full bg-[#D97706] rounded-t-sm h-[72%]" />
                    <span className="text-[9px] font-mono text-[#6B7280] mt-1.5">Softmax</span>
                  </div>
                </div>
              </div>

            </div>
          </div>

          {/* Bottom Card: Selective Prediction Performance */}
          <div className="bg-[#FAF8F5] border border-[#E6E1DA] rounded-2xl p-5 sm:p-6 space-y-3 shadow-xs">
            <div>
              <h3 className="text-xs font-mono font-bold text-[#111827] uppercase tracking-wider">
                SELECTIVE PREDICTION PERFORMANCE
              </h3>
              <p className="text-[11px] text-[#6B7280] mt-0.5">
                Filtering low-trust predictions retains high accuracy.
              </p>
            </div>

            <div className="bg-white border border-[#E6E1DA] rounded-xl p-3.5 flex items-center justify-between gap-3">
              <div className="flex-1 space-y-1.5">
                <div className="flex justify-between text-[10px] font-mono text-[#6B7280]">
                  <span>Retained Coverage (80%)</span>
                  <span className="font-bold text-[#C53030]">9.68% Risk</span>
                </div>
                <div className="h-3 w-full bg-[#E6E1DA] rounded-full overflow-hidden">
                  <div className="h-full bg-[#1B4332] rounded-full w-[80%]" />
                </div>
              </div>
              <div className="text-right pl-2 border-l border-[#E6E1DA]">
                <span className="text-xl font-extrabold text-[#111827] font-mono block">90.33%</span>
                <span className="text-[9px] text-[#6B7280] font-mono">Accuracy</span>
              </div>
            </div>

            <div className="bg-[#E8F5E9]/60 border border-[#C8E6C9] rounded-xl p-2.5 flex items-center space-x-2 text-xs text-[#1B4332]">
              <CheckCircle2 className="h-4 w-4 text-[#1B4332] shrink-0" />
              <p className="text-[11px] leading-snug">
                At 80% coverage, the Trust Model retains 90.33% accuracy while identifying higher-risk predictions.
              </p>
            </div>
          </div>

        </div>

      </div>

      {/* 3. Calibration Metrics Strip */}
      <div className="bg-[#FAF8F5] border border-[#E6E1DA] rounded-2xl p-4 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs font-mono shadow-xs">
        <div className="flex items-center space-x-2 text-[#111827] font-bold">
          <Sliders className="h-4 w-4 text-[#1B4332]" />
          <span>CALIBRATION METRICS</span>
        </div>
        <div className="flex items-center space-x-6 text-[#374151]">
          <div>
            <span className="text-[#6B7280] mr-2">Expected Calibration Error (ECE):</span>
            <span className="font-bold text-[#1B4332]">1.21%</span>
          </div>
          <div>
            <span className="text-[#6B7280] mr-2">Brier Score:</span>
            <span className="font-bold text-[#1B4332]">0.1077</span>
          </div>
        </div>
      </div>

      {/* 4. Key Empirical Insight Banner */}
      <div className="bg-[#E8F5E9]/60 border border-[#C8E6C9] rounded-2xl p-4 sm:p-5 flex items-start space-x-3.5 shadow-xs">
        <Trophy className="h-5 w-5 text-[#1B4332] shrink-0 mt-0.5" />
        <div>
          <h4 className="text-sm font-bold text-[#111827] leading-tight mb-0.5">
            Key Empirical Insight
          </h4>
          <p className="text-xs sm:text-sm text-[#374151] leading-relaxed">
            The Random Forest Trust Model achieves an AUROC of <strong className="text-[#111827]">86.69%</strong> and a Failure AUPRC of <strong className="text-[#111827]">0.5708</strong>, outperforming raw softmax confidence for prediction-failure detection on the CIFAR-10 evaluation.
          </p>
        </div>
      </div>

    </section>
  );
};
