import type { PredictResponse, HealthResponse, ErrorResponse } from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export async function fetchHealth(): Promise<HealthResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error(`Health check failed with status ${response.status}`);
    }
    return await response.json();
  } catch {
    return {
      status: 'offline',
      model_loaded: false,
    };
  }
}

export async function predictImage(file: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append('image', file);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });
  } catch {
    throw new Error('Unable to connect to VisionTrust API backend. Please ensure the server is running at ' + API_BASE_URL);
  }

  if (!response.ok) {
    let errorDetail = `Prediction request failed (HTTP ${response.status})`;
    try {
      const errData: ErrorResponse = await response.json();
      if (errData.detail) {
        errorDetail = errData.detail;
      }
    } catch {
      // Fallback to generic message
    }
    throw new Error(errorDetail);
  }

  return await response.json();
}
