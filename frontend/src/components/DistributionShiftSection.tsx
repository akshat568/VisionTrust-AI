import React from 'react';
import { AlertTriangle, CheckCircle2, RefreshCw, ArrowRight, BarChart2, ShieldCheck, Database, Lightbulb } from 'lucide-react';

export const DistributionShiftSection: React.FC = () => {
  const shifts = [
    'Blur',
    'Noise',
    'Brightness',
    'Contrast',
    'Rotation',
    'Out-of-distribution',
  ];

  // Exact benchmark empirical results: clean baseline test accuracy = 81.25% vs severe shift (s5)
  const benchmarkResults = [
    { name: 'Blur', clean: 81.25, shifted: 30.33 },
    { name: 'Noise', clean: 81.25, shifted: 17.25 },
    { name: 'Brightness', clean: 81.25, shifted: 49.56 },
    { name: 'Contrast', clean: 81.25, shifted: 45.15 },
    { name: 'Rotation', clean: 81.25, shifted: 30.88 },
  ];

  return (
    <section id="robustness" className="bg-white border border-[#E6E1DA] rounded-3xl p-6 sm:p-9 shadow-xs space-y-8">
      
      {/* 1. Header with Upper Right Insight Card */}
      <div className="flex flex-col lg:flex-row lg:items-start justify-between gap-6 border-b border-[#E6E1DA] pb-6">
        <div className="space-y-3 max-w-3xl">
          <p className="text-xs font-mono tracking-widest text-[#6B7280] uppercase font-bold">
            DISTRIBUTION SHIFT
          </p>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-[#111827] tracking-tight">
            Designed for predictions that can fail.
          </h2>
          <p className="text-base text-[#4B5563] leading-relaxed">
            Real images rarely match training conditions. As inputs drift, accuracy can degrade while model confidence remains deceptively high — which is exactly where a reliability estimate matters.
          </p>

          {/* Shift Filter Chips */}
          <div className="flex flex-wrap gap-2 pt-2">
            {shifts.map((shift) => (
              <span
                key={shift}
                className="text-xs font-mono px-3 py-1 rounded-full bg-[#FAF8F5] text-[#374151] border border-[#E6E1DA] font-semibold"
              >
                {shift}
              </span>
            ))}
          </div>
        </div>

        {/* Upper Right Pale Green Insight Card */}
        <div className="bg-[#E8F5E9]/60 border border-[#C8E6C9] rounded-2xl p-4 flex items-start space-x-3.5 max-w-sm shrink-0">
          <div className="h-9 w-9 rounded-xl bg-white border border-[#C8E6C9] flex items-center justify-center text-[#1B4332] shrink-0 mt-0.5 shadow-xs">
            <BarChart2 className="h-4 w-4" />
          </div>
          <div>
            <p className="text-xs font-bold text-[#111827]">Same confidence. Different reality.</p>
            <p className="text-[11px] text-[#4B5563] leading-normal mt-0.5">
              Vision models can be highly confident even when inputs deviate from the training distribution. VisionTrust uses reliability signals to detect these failure cases.
            </p>
          </div>
        </div>
      </div>

      {/* 2. Three-Stage Flow Cards */}
      <div className="flex flex-col lg:flex-row items-stretch gap-4 relative">
        
        {/* Stage 01: Clean Image */}
        <div className="flex-1 bg-[#FAF8F5] border border-[#C8E6C9] rounded-2xl p-5 flex flex-col justify-between space-y-4 shadow-xs">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono tracking-wider text-[#6B7280] uppercase font-bold">
                STAGE 01
              </span>
              <span className="text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full bg-[#D8F3DC] text-[#1B4332] border border-[#B7E4C7] flex items-center gap-1">
                <CheckCircle2 className="h-3 w-3" />
                <span>STABLE</span>
              </span>
            </div>

            <h3 className="text-base font-bold text-[#111827]">Clean Image</h3>
            <p className="text-xs text-[#4B5563] leading-relaxed">
              Input matches reference training domain distribution. Model accuracy is high and confidence is generally high.
            </p>
          </div>

          <div className="pt-3 border-t border-[#E6E1DA] text-[11px] font-mono text-[#1B4332] font-bold">
            High Accuracy • High Confidence
          </div>
        </div>

        {/* Arrow Connector 1 */}
        <div className="hidden lg:flex items-center justify-center text-[#9CA3AF]">
          <ArrowRight className="h-4 w-4" />
        </div>

        {/* Stage 02: Shifted Input */}
        <div className="flex-1 bg-[#FAF8F5] border border-[#FDE68A] rounded-2xl p-5 flex flex-col justify-between space-y-4 shadow-xs">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono tracking-wider text-[#6B7280] uppercase font-bold">
                STAGE 02
              </span>
              <span className="text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full bg-[#FEF3C7] text-[#92400E] border border-[#FDE68A] flex items-center gap-1">
                <RefreshCw className="h-3 w-3" />
                <span>DRIFT</span>
              </span>
            </div>

            <h3 className="text-base font-bold text-[#111827]">Shifted Input</h3>
            <p className="text-xs text-[#4B5563] leading-relaxed">
              Image undergoes natural corruption (blur, noise, exposure, or rotation shift) in real-world deployment.
            </p>
          </div>

          <div className="pt-3 border-t border-[#E6E1DA] text-[11px] font-mono text-[#92400E] font-bold">
            Accuracy Drops • Confidence Can Remain High
          </div>
        </div>

        {/* Arrow Connector 2 */}
        <div className="hidden lg:flex items-center justify-center text-[#9CA3AF]">
          <ArrowRight className="h-4 w-4" />
        </div>

        {/* Stage 03: Reliability Risk */}
        <div className="flex-1 bg-[#FAF8F5] border border-[#FFCDD2] rounded-2xl p-5 flex flex-col justify-between space-y-4 shadow-xs">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono tracking-wider text-[#6B7280] uppercase font-bold">
                STAGE 03
              </span>
              <span className="text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full bg-[#FFEBEE] text-[#C53030] border border-[#FFCDD2] flex items-center gap-1">
                <AlertTriangle className="h-3 w-3" />
                <span>RISK</span>
              </span>
            </div>

            <h3 className="text-base font-bold text-[#111827]">Reliability Risk</h3>
            <p className="text-xs text-[#4B5563] leading-relaxed">
              The Random Forest Trust Model combines reliability signals to estimate whether the prediction may fail.
            </p>
          </div>

          <div className="pt-3 border-t border-[#E6E1DA] text-[11px] font-mono text-[#C53030] font-bold">
            Reliability Layer Estimates Prediction Risk
          </div>
        </div>

      </div>

      {/* 3. Real-World Robustness Results Panel (Chart + Key Insight Card) */}
      <div className="bg-[#FAF8F5] border border-[#E6E1DA] rounded-3xl p-6 sm:p-7 shadow-xs">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          
          {/* Grouped Bar Chart Area (Left ~8 cols) */}
          <div className="lg:col-span-8 bg-white border border-[#E6E1DA] rounded-2xl p-5 flex flex-col justify-between space-y-6 shadow-xs">
            
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[#E6E1DA] pb-4">
              <div>
                <h3 className="text-xs font-mono font-bold text-[#111827] uppercase tracking-wider">
                  REAL-WORLD ROBUSTNESS RESULTS
                </h3>
                <p className="text-xs text-[#6B7280] mt-0.5">
                  Model accuracy drops significantly under common distribution shifts, while confidence often remains high.
                </p>
              </div>

              {/* Legend */}
              <div className="flex items-center space-x-3 text-[10px] font-mono shrink-0 pt-1 sm:pt-0">
                <div className="flex items-center space-x-1.5">
                  <div className="h-2.5 w-2.5 rounded-full bg-[#1B4332]" />
                  <span className="text-[#374151] font-semibold">Clean (In-dist.)</span>
                </div>
                <div className="flex items-center space-x-1.5">
                  <div className="h-2.5 w-2.5 rounded-full bg-[#D97706]" />
                  <span className="text-[#374151] font-semibold">Severe Shift (s5)</span>
                </div>
              </div>
            </div>

            {/* Custom Grouped Bar Chart Visualization */}
            <div className="space-y-4 pt-2">
              <div className="flex items-end justify-between space-x-2 sm:space-x-4 h-48 border-b border-[#E6E1DA] pb-2 px-2">
                {benchmarkResults.map((res) => (
                  <div key={res.name} className="flex-1 flex flex-col items-center h-full justify-end group">
                    <div className="w-full flex items-end justify-center space-x-1 h-36">
                      
                      {/* Clean Bar */}
                      <div className="flex-1 max-w-[28px] flex flex-col items-center h-full justify-end">
                        <span className="text-[9px] font-mono font-bold text-[#1B4332] mb-1">
                          {res.clean.toFixed(1)}%
                        </span>
                        <div
                          className="w-full bg-[#1B4332] rounded-t-sm transition-all duration-500"
                          style={{ height: `${res.clean}%` }}
                        />
                      </div>

                      {/* Severe Shift Bar */}
                      <div className="flex-1 max-w-[28px] flex flex-col items-center h-full justify-end">
                        <span className="text-[9px] font-mono font-bold text-[#D97706] mb-1">
                          {res.shifted.toFixed(1)}%
                        </span>
                        <div
                          className="w-full bg-[#D97706] rounded-t-sm transition-all duration-500"
                          style={{ height: `${res.shifted}%` }}
                        />
                      </div>

                    </div>

                    <span className="text-[11px] font-mono font-bold text-[#374151] mt-3">
                      {res.name}
                    </span>
                  </div>
                ))}
              </div>
            </div>

          </div>

          {/* Key Insight Card (Right ~4 cols) */}
          <div className="lg:col-span-4 bg-white border border-[#E6E1DA] rounded-2xl p-5 sm:p-6 flex flex-col justify-between space-y-4 shadow-xs">
            
            <div className="space-y-3">
              <div className="flex items-center space-x-2">
                <div className="p-1.5 rounded-lg bg-[#FAF8F5] border border-[#E6E1DA] text-[#111827]">
                  <Lightbulb className="h-4 w-4" />
                </div>
                <span className="text-[10px] font-mono font-bold text-[#6B7280] uppercase tracking-wider">
                  KEY INSIGHT
                </span>
              </div>

              <h4 className="text-base font-extrabold text-[#111827] leading-snug">
                Accuracy can drop dramatically while confidence remains high.
              </h4>

              <p className="text-xs text-[#4B5563] leading-relaxed">
                This mismatch makes reliability estimation essential for safe real-world deployment.
              </p>
            </div>

            {/* Bottom Warning Strip */}
            <div className="bg-[#FFEBEE] border border-[#FFCDD2] rounded-xl p-3 text-center">
              <p className="text-xs font-bold text-[#C53030]">
                High confidence does not mean correct.
              </p>
            </div>

          </div>

        </div>
      </div>

      {/* 4. Bottom Takeaway Row */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4 border-t border-[#E6E1DA]">
        <div className="flex items-start space-x-3">
          <div className="h-8 w-8 rounded-lg bg-[#FAF8F5] border border-[#E6E1DA] flex items-center justify-center text-[#1B4332] shrink-0 mt-0.5">
            <ShieldCheck className="h-4 w-4" />
          </div>
          <div>
            <p className="text-xs font-bold text-[#111827]">Evaluated under distribution shift</p>
            <p className="text-[11px] text-[#6B7280]">More reliable under real-world conditions.</p>
          </div>
        </div>

        <div className="flex items-start space-x-3">
          <div className="h-8 w-8 rounded-lg bg-[#FAF8F5] border border-[#E6E1DA] flex items-center justify-center text-[#1B4332] shrink-0 mt-0.5">
            <BarChart2 className="h-4 w-4" />
          </div>
          <div>
            <p className="text-xs font-bold text-[#111827]">Complements model confidence</p>
            <p className="text-[11px] text-[#6B7280]">Catches failures that confidence may miss.</p>
          </div>
        </div>

        <div className="flex items-start space-x-3">
          <div className="h-8 w-8 rounded-lg bg-[#FAF8F5] border border-[#E6E1DA] flex items-center justify-center text-[#1B4332] shrink-0 mt-0.5">
            <Database className="h-4 w-4" />
          </div>
          <div>
            <p className="text-xs font-bold text-[#111827]">Validated with real data</p>
            <p className="text-[11px] text-[#6B7280]">Evaluated on in-distribution and shifted examples.</p>
          </div>
        </div>
      </div>

    </section>
  );
};
