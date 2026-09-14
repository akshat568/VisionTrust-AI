import React from 'react';
import type { HealthResponse } from '../types/api';

interface HeaderProps {
  health: HealthResponse | null;
}

export const Header: React.FC<HeaderProps> = ({ health }) => {
  const isOnline = health?.status === 'ok' && health?.model_loaded;

  return (
    <header className="border-b border-[#E6E1DA] bg-[#FAF8F5]/90 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-[95%] w-full mx-auto px-3 sm:px-6 lg:px-8 py-3.5">
        <div className="flex items-center justify-between">
          
          {/* Logo & Brand Info (Moved to left edge) */}
          <div className="flex items-center space-x-3 shrink-0">
            <div className="h-9 w-9 rounded-lg bg-white border border-[#E6E1DA] flex items-center justify-center shrink-0 shadow-xs font-mono font-bold text-xs text-[#1B4332]">
              VT
            </div>
            <div>
              <h1 className="text-[15px] font-bold tracking-tight text-[#111827]">VisionTrust AI</h1>
              <p className="text-xs text-[#6B7280] font-normal">
                Reliability-Aware Computer Vision
              </p>
            </div>
          </div>

          {/* Center Navigation Links (Centered, 15px text, medium weight) */}
          <nav className="hidden lg:flex items-center space-x-8 text-[15px] font-medium text-[#374151]">
            <a href="#analyze" className="hover:text-[#111827] transition-colors">Analyze</a>
            <a href="#core-distinction" className="hover:text-[#111827] transition-colors">Distinction</a>
            <a href="#architecture" className="hover:text-[#111827] transition-colors">How It Works</a>
            <a href="#signals" className="hover:text-[#111827] transition-colors">Signals</a>
            <a href="#robustness" className="hover:text-[#111827] transition-colors">Shift</a>
            <a href="#benchmarks" className="hover:text-[#111827] transition-colors">Benchmarks</a>
          </nav>

          {/* System Health Badge (Moved to right edge) */}
          <div className="flex items-center shrink-0">
            <div className="flex items-center space-x-2 bg-white px-3.5 py-1.5 rounded-full border border-[#E6E1DA] text-xs shadow-xs font-medium text-[#111827]">
              <span className="relative flex h-2 w-2">
                {isOnline ? (
                  <>
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-[#2D6A4F]"></span>
                  </>
                ) : (
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500"></span>
                )}
              </span>
              <span>
                {isOnline ? 'Backend Healthy' : 'Backend Offline'}
              </span>
            </div>
          </div>

        </div>
      </div>
    </header>
  );
};



