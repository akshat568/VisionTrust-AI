from dataclasses import dataclass, asdict
from typing import Dict, List, Any


@dataclass
class InferenceResult:
    """Encapsulates prediction output, reliability signals, and trust scores for an image."""

    predicted_class_id: int
    predicted_class_name: str
    confidence: float
    predicted_logits: List[float]
    probabilities: List[float]
    reliability_signals: Dict[str, float]
    trust_score: float
    reliability_status: str
    failure_probability: float
    decision_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert InferenceResult to a plain Python dictionary suitable for JSON serialization."""
        data = asdict(self)
        # Ensure all floating point values are plain python float
        data["confidence"] = float(data["confidence"])
        data["trust_score"] = float(data["trust_score"])
        data["failure_probability"] = float(data["failure_probability"])
        data["decision_threshold"] = float(data["decision_threshold"])
        data["predicted_logits"] = [float(x) for x in data["predicted_logits"]]
        data["probabilities"] = [float(x) for x in data["probabilities"]]
        data["reliability_signals"] = {
            k: float(v) for k, v in data["reliability_signals"].items()
        }
        return data
