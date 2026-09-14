import React from 'react';
import { BarChart3 } from 'lucide-react';
import type { PredictResponse } from '../types/api';

const CIFAR10_NAMES = [
  'airplane',
  'automobile',
  'bird',
  'cat',
  'deer',
  'dog',
  'frog',
  'horse',
  'ship',
  'truck',
];

interface ClassProbabilitiesChartProps {
  result: PredictResponse;
}

export const ClassProbabilitiesChart: React.FC<ClassProbabilitiesChartProps> = ({ result }) => {
  const { probabilities } = result.vision;
  const predictedClassId = result.prediction.class_id;

  // Build sorted list of classes by probability descending
  const classData = CIFAR10_NAMES.map((name, id) => ({
    id,
    name,
    prob: probabilities[id] || 0.0,
    isPredicted: id === predictedClassId,
  })).sort((a, b) => b.prob - a.prob);

  const maxProb = Math.max(...probabilities, 0.001);

  return (
    <div className="bg-white border border-[#E6E1DA] rounded-2xl p-6 sm:p-8 shadow-xs space-y-5">
      <div className="flex items-center justify-between border-b border-[#E6E1DA] pb-4">
        <div className="flex items-center space-x-2">
          <BarChart3 className="h-5 w-5 text-[#111827]" />
          <h3 className="text-base font-bold text-[#111827]">Class Probabilities</h3>
        </div>
        <span className="text-xs font-mono text-[#6B7280]">Softmax Output (10 Classes)</span>
      </div>

      <div className="space-y-2.5 pt-1">
        {classData.map((item) => {
          const percent = (item.prob * 100).toFixed(1);
          const widthPercent = (item.prob / maxProb) * 100;

          return (
            <div key={item.id} className="flex items-center space-x-3 text-xs">
              <span
                className={`w-24 truncate font-mono text-right capitalize ${
                  item.isPredicted ? 'font-extrabold text-[#1B4332]' : 'text-[#6B7280]'
                }`}
              >
                {item.name}
              </span>

              <div className="flex-1 bg-[#FAF8F5] rounded-full h-4.5 p-0.5 border border-[#E6E1DA] relative overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    item.isPredicted
                      ? 'bg-[#1B4332]'
                      : 'bg-[#D8D2C7]'
                  }`}
                  style={{ width: `${Math.max(widthPercent, 2)}%` }}
                />
              </div>

              <span
                className={`w-14 text-right font-mono text-xs ${
                  item.isPredicted ? 'font-bold text-[#1B4332]' : 'text-[#9CA3AF]'
                }`}
              >
                {percent}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};


