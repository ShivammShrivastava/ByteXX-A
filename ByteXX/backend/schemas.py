"""
SatQuery AI — Pydantic Schemas
================================
"""
from typing import Optional, Any
from pydantic import BaseModel, Field


# ─── VQA ─────────────────────────────────────
class VQARequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64-encoded image (PNG/JPEG)")
    question: str = Field(..., description="Question about the image")

class VQAResponse(BaseModel):
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning_trace: str
    query_id: Optional[str] = None
    model: str = "satquery-ai-vqa"
    task: str = "vqa"


# ─── Caption ─────────────────────────────────
class CaptionRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64-encoded image (PNG/JPEG)")

class CaptionResponse(BaseModel):
    caption: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning_trace: str
    query_id: Optional[str] = None
    model: str = "satquery-ai-vqa"
    task: str = "caption"


# ─── Referring ───────────────────────────────
class ReferringRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64-encoded image (PNG/JPEG)")
    expression: str = Field(..., description="Object description to locate")

class ReferringResponse(BaseModel):
    raw_output: str
    bbox: Optional[list[float]] = Field(None, description="[x1, y1, x2, y2] normalized 0-1")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning_trace: str
    query_id: Optional[str] = None
    model: str = "satquery-ai-vqa"
    task: str = "referring"


# ─── Firebase direct submit ───────────────────
class FirebaseQueryRequest(BaseModel):
    image_url: str = Field(..., description="Firebase Storage download URL of the image")
    question: str  = Field(..., description="User's question")
    task: str      = Field("vqa", description="vqa | caption | refer")

class FirebaseQueryResponse(BaseModel):
    query_id: str
    status: str = "pending"
    message: str = "Query submitted. Listen to /results/{query_id} in Firebase."


# ─── Health ──────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_name: str
    gpu_name: Optional[str] = None
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    queries_processed: int = 0
    errors: int = 0
    firebase_listener: bool = False


# ─── Query status ────────────────────────────
class QueryStatusResponse(BaseModel):
    query_id: str
    status: str       # pending | processing | done | error
    question: Optional[str] = None
    task: Optional[str] = None
    timestamp: Optional[int] = None

class QueryResultResponse(BaseModel):
    query_id: str
    answer: str
    confidence: float
    task: str
    processed_at: Optional[int] = None


# ─── Stats ───────────────────────────────────
class StatsResponse(BaseModel):
    total_queries: int
    errors: int
    model_loaded: bool
    firebase_connected: bool
    uptime_seconds: float


# ─── CNN Detection ────────────────────────────
class CNNDetectRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64-encoded image (PNG/JPEG)")
    conf_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Minimum detection confidence")
    target_classes: Optional[list[str]] = Field(None, description="Object classes to filter (e.g. ['vehicle', 'aircraft'])")

class CNNDetectResponse(BaseModel):
    object_counts: dict[str, int]
    total_detections: int
    bboxes: list[list[float]]
    bbox_labels: list[str]
    bbox_confidences: list[float]
    annotated_image_b64: Optional[str] = None
    confidence: float
    duration_ms: float
    model_used: str = "yolov8n"
    errors: list[str] = []


# ─── CNN Segmentation ─────────────────────────
class CNNSegmentRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64-encoded image (PNG/JPEG)")

class CNNSegmentResponse(BaseModel):
    land_cover_percentages: dict[str, float]
    land_cover_display: dict[str, float]
    dominant_land_cover: str
    mask_image_b64: Optional[str] = None
    confidence: float
    duration_ms: float
    model_used: str = "resnet50-fcn-landcover"
    error: Optional[str] = None


# ─── CNN Full Pipeline ─────────────────────────
class CNNFullRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64-encoded image (PNG/JPEG)")
    tasks: list[str] = Field(["detect", "segment"], description="Tasks to run: detect | segment")
    conf_threshold: float = Field(0.25, ge=0.0, le=1.0)

class CNNFullResponse(BaseModel):
    object_counts: dict[str, int]
    total_detections: int
    bboxes: list[list[float]]
    bbox_labels: list[str]
    bbox_confidences: list[float]
    annotated_image_b64: Optional[str] = None
    land_cover_percentages: dict[str, float]
    land_cover_display: dict[str, float]
    dominant_land_cover: str
    mask_image_b64: Optional[str] = None
    tasks_run: list[str]
    overall_confidence: float
    total_duration_ms: float
    errors: list[str] = []


# ─── AI Agent ─────────────────────────────────
class AgentAnalyzeRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64-encoded image (PNG/JPEG)")
    question: str = Field(..., description="User's question about the image")
    conf_threshold: float = Field(0.25, ge=0.0, le=1.0, description="YOLO confidence threshold")

class AgentAnalyzeResponse(BaseModel):
    summary: str
    vlm_answer: Optional[str] = None
    caption: Optional[str] = None
    object_counts: dict[str, int]
    total_detections: int
    bboxes: list[list[float]]
    bbox_labels: list[str]
    bbox_confidences: list[float]
    annotated_image_b64: Optional[str] = None
    land_cover: dict[str, float]
    land_cover_display: dict[str, float]
    dominant_land_cover: str
    mask_image_b64: Optional[str] = None
    confidence: float
    tools_used: list[str]
    routing_mode: str
    execution_steps: list[dict[str, Any]]
    verification_log: list[dict[str, Any]]
    reasoning_trace: str
    errors: list[str]
    duration_ms: float

class AgentStatusResponse(BaseModel):
    agent_ready: bool
    cnn_detect_ready: bool
    cnn_segment_ready: bool
    vlm_ready: bool
    message: str
