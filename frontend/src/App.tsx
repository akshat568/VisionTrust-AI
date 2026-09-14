import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { HeroSection } from './components/HeroSection';
import { CoreDistinctionSection } from './components/CoreDistinctionSection';
import { ArchitectureSection } from './components/ArchitectureSection';
import { ImageUploader } from './components/ImageUploader';
import { PredictionPathDiagram } from './components/PredictionPathDiagram';
import { ResultSummaryCard } from './components/ResultSummaryCard';
import { ConfidenceVsTrustCard } from './components/ConfidenceVsTrustCard';
import { ReliabilitySignalsGrid } from './components/ReliabilitySignalsGrid';
import { TrustExplanationCard } from './components/TrustExplanationCard';
import { ClassProbabilitiesChart } from './components/ClassProbabilitiesChart';
import { ReliabilitySignalsSection } from './components/ReliabilitySignalsSection';
import { DistributionShiftSection } from './components/DistributionShiftSection';
import { BenchmarkSection } from './components/BenchmarkSection';
import { PipelineSection } from './components/PipelineSection';
import { fetchHealth, predictImage } from './services/api';
import type { HealthResponse, PredictResponse } from './types/api';
import { AlertCircle, RefreshCw } from 'lucide-react';

export function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const checkHealthStatus = async () => {
    const res = await fetchHealth();
    setHealth(res);
  };

  useEffect(() => {
    checkHealthStatus();
  }, []);

  const handleAnalyze = async (file: File) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await predictImage(file);
      setResult(res);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred during prediction.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const isApiOffline = health !== null && (!health.status || health.status !== 'ok' || !health.model_loaded);

  return (
    <div className="min-h-screen flex flex-col bg-[#FAF8F5] text-[#111827] font-sans selection:bg-[#E8F5E9] selection:text-[#1B4332]">
      {/* Navigation Header */}
      <Header health={health} />

      {/* Main Content Area */}
      <main className="flex-1 max-w-[1500px] w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-12">
        
        {/* Offline API Warning Banner */}
        {isApiOffline && (
          <div className="bg-[#FEF3C7] border border-[#FDE68A] rounded-2xl p-4 flex items-center justify-between text-[#92400E] text-xs shadow-xs">
            <div className="flex items-center space-x-3">
              <AlertCircle className="h-5 w-5 text-[#D97706] shrink-0" />
              <div>
                <p className="font-bold">VisionTrust FastAPI Backend Server is Offline</p>
                <p className="text-[#B45309] mt-0.5">
                  Launch the backend using: <code className="font-mono bg-[#FDE68A]/60 px-2 py-0.5 rounded text-[#78350F] font-semibold">uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000</code>
                </p>
              </div>
            </div>
            <button
              onClick={checkHealthStatus}
              className="px-3.5 py-1.5 rounded-lg bg-white hover:bg-[#FAF8F5] text-[#78350F] font-semibold flex items-center gap-1.5 transition shrink-0 border border-[#FDE68A] shadow-xs"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              <span>Retry</span>
            </button>
          </div>
        )}

        {/* Error Banner */}
        {error && (
          <div className="bg-[#FFEBEE] border border-[#FFCDD2] rounded-2xl p-4 flex items-center space-x-3 text-[#9B2C2C] text-xs shadow-xs">
            <AlertCircle className="h-5 w-5 text-[#C53030] shrink-0" />
            <div className="flex-1">
              <p className="font-bold">Analysis Error</p>
              <p className="text-[#C53030] mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {/* 1. Hero Section (Wide 2-Part Composition matching reference image) */}
        <HeroSection result={result} previewUrl={previewUrl} />

        {/* 2. Core Distinction: Confidence ≠ Trust (Screenshot 7) */}
        <CoreDistinctionSection />

        {/* 3. Architecture Overview Section (Screenshot 3) */}
        <ArchitectureSection result={result} previewUrl={previewUrl} />

        {/* 4. Interactive Upload & Dual-Path Analysis View Grid (Screenshot 4 & 1/2) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <div className="lg:col-span-6">
            <ImageUploader
              onAnalyze={handleAnalyze}
              isLoading={isLoading}
              onReset={handleReset}
              selectedFile={selectedFile}
              setSelectedFile={setSelectedFile}
              previewUrl={previewUrl}
              setPreviewUrl={setPreviewUrl}
            />
          </div>

          <div className="lg:col-span-6">
            <PredictionPathDiagram
              result={result}
              previewUrl={previewUrl}
            />
          </div>
        </div>

        {/* 5. Real Results Dashboard (Rendered when API result is returned) */}
        {result && (
          <div className="space-y-8 animate-in fade-in duration-300">
            {/* Primary Result Summary */}
            <ResultSummaryCard result={result} />

            {/* Confidence vs Trust Comparison */}
            <ConfidenceVsTrustCard result={result} />

            {/* Per-Prediction Reliability Signals Grid */}
            <ReliabilitySignalsGrid reliability={result.reliability} vision={result.vision} />

            {/* Explanation & Probabilities Side-by-Side */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
              <TrustExplanationCard result={result} />
              <ClassProbabilitiesChart result={result} />
            </div>
          </div>
        )}

        {/* 6. Reliability Signals Overview (Screenshot 6 & 8) */}
        <ReliabilitySignalsSection result={result} />

        {/* 7. Distribution Shift / Robustness (Screenshot 9) */}
        <DistributionShiftSection />

        {/* 8. Benchmark Evaluation Section (Screenshot 10) */}
        <BenchmarkSection />

        {/* 9. Evaluation Pipeline Stepper (Screenshot 5) */}
        <PipelineSection />

      </main>

      {/* Footer */}
      <footer className="border-t border-[#E6E1DA] bg-white py-8 mt-12 text-center text-xs text-[#6B7280]">
        <div className="max-w-[1500px] w-full mx-auto px-4 sm:px-6 lg:px-8 space-y-1 font-mono">
          <p className="font-bold text-[#111827]">VisionTrust AI — Reliability-Aware Computer Vision Framework</p>
          <p className="text-[#9CA3AF]">
            Frozen ResNet-18 Vision Model • Random Forest Trust Model • 6 Reliability Signal Categories • CIFAR-10
          </p>
        </div>
      </footer>

    </div>
  );
}

export default App;



