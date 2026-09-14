export interface PredictionResponse {
  class_id: number;
  class_name: string;
}

export interface VisionResponse {
  confidence: number;
  entropy: number;
  probabilities: number[];
}

export interface ReliabilityResponse {
  feature_distance: number;
  ood_score: number;
  augmentation_consistency: number;
  sharpness: number;
  brightness: number;
  contrast: number;
  composite_quality: number;
}

export interface TrustResponse {
  trust_probability: number;
  failure_probability: number;
  status: 'TRUSTED' | 'UNTRUSTED';
}

export interface PredictResponse {
  prediction: PredictionResponse;
  vision: VisionResponse;
  reliability: ReliabilityResponse;
  trust: TrustResponse;
  latency_ms?: number | null;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
}

export interface ErrorResponse {
  detail: string;
  error_code?: string;
}
