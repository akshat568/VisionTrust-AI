import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ResultSummaryCard } from '../components/ResultSummaryCard';
import type { PredictResponse } from '../types/api';

const mockResult: PredictResponse = {
  prediction: { class_id: 6, class_name: 'frog' },
  vision: {
    confidence: 0.9254,
    entropy: 0.2415,
    probabilities: [0.01, 0.02, 0.9254, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0],
  },
  reliability: {
    feature_distance: 9.85,
    ood_score: 3.41,
    augmentation_consistency: 1.0,
    sharpness: 0.08,
    brightness: 0.49,
    contrast: 0.24,
    composite_quality: 0.33,
  },
  trust: {
    trust_probability: 0.9421,
    failure_probability: 0.0579,
    status: 'TRUSTED',
  },
  latency_ms: 125.4,
};

describe('ResultSummaryCard Component', () => {
  it('renders predicted class name, confidence, trust score, and status badge', () => {
    render(<ResultSummaryCard result={mockResult} />);

    expect(screen.getByText('frog')).toBeInTheDocument();
    expect(screen.getByText('92.5%')).toBeInTheDocument();
    expect(screen.getByText('94.2%')).toBeInTheDocument();
    expect(screen.getByText('TRUSTED')).toBeInTheDocument();
    expect(screen.getByText('5.8%')).toBeInTheDocument();
  });
});
