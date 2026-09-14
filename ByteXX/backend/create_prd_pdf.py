import sys
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "SatQuery AI — Product Requirement Document (PRD)")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
        
        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)
        
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 30, page_text)
        self.drawString(54, 30, "CONFIDENTIAL — SatQuery AI Engineering & Product Team")
        self.restoreState()

def create_prd_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    COLOR_PRIMARY = colors.HexColor("#1E3A8A")    # Deep Navy/Blue
    COLOR_SECONDARY = colors.HexColor("#0284C7")  # Bright Sky Blue
    COLOR_DARK = colors.HexColor("#0F172A")       # Dark Slate Body
    COLOR_LIGHT = colors.HexColor("#F8FAFC")      # Light Slate BG
    COLOR_BORDER = colors.HexColor("#E2E8F0")     # Border Slate
    COLOR_ACCENT = colors.HexColor("#0D9488")     # Teal Accent
    
    # Custom Styles
    style_title = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=COLOR_PRIMARY,
        spaceAfter=6
    )
    
    style_subtitle = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=COLOR_SECONDARY,
        spaceAfter=15
    )
    
    style_meta = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569")
    )
    
    style_h1 = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=COLOR_PRIMARY,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    style_h2 = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=COLOR_SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    style_body = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=COLOR_DARK,
        spaceAfter=6,
        alignment=TA_LEFT
    )

    style_bullet = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=COLOR_DARK,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    style_code = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0F172A"),
        backColor=colors.HexColor("#F1F5F9"),
        borderColor=COLOR_BORDER,
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=8
    )
    
    style_th = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.white,
        alignment=TA_LEFT
    )
    
    style_td = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=COLOR_DARK,
        alignment=TA_LEFT
    )

    story = []
    
    # ── Header Banner ──
    story.append(Paragraph("SatQuery AI — Product Requirement Document", style_title))
    story.append(Paragraph("Geospatial Vision-Language Intelligence Platform", style_subtitle))
    
    # Metadata Table
    meta_data = [
        [
            Paragraph("<b>Version:</b> 1.0.0", style_meta),
            Paragraph("<b>Status:</b> Approved / Active", style_meta),
            Paragraph("<b>Date:</b> September 2026", style_meta),
            Paragraph("<b>Author:</b> Antigravity AI & Suryansh", style_meta)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[120, 130, 120, 134])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), COLOR_LIGHT),
        ('BOX', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 14))
    
    # ── Section 1: Executive Overview ──
    story.append(Paragraph("1. Executive Summary & Vision", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceBefore=0, spaceAfter=8))
    
    story.append(Paragraph(
        "<b>SatQuery AI</b> is a next-generation geospatial vision-language intelligence platform designed to extract actionable insights from satellite and high-resolution aerial imagery. Powered by a fine-tuned vision-language model (<b>Qwen2.5-VL-3B-Instruct</b>) combined with a domain-specific LoRA adapter (<b>satquery-ai-vqa-lora</b>), SatQuery AI transforms complex satellite imagery into natural-language answers, detailed descriptions, and localized object bounding boxes.",
        style_body
    ))
    story.append(Paragraph(
        "The primary goal of SatQuery AI is to eliminate manual visual inspection overhead for defense analysts, urban planners, environmental researchers, and disaster recovery teams by enabling conversational visual query processing with sub-3-second inference latencies and zero-cloud dependency options.",
        style_body
    ))
    
    # ── Section 2: Core Objectives & Metrics ──
    story.append(Spacer(1, 4))
    story.append(Paragraph("2. Strategic Objectives & Success Metrics", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceBefore=0, spaceAfter=8))
    
    metrics_data = [
        [Paragraph("Objective", style_th), Paragraph("Target Metric", style_th), Paragraph("Validation Method", style_th)],
        [
            Paragraph("High VQA Accuracy", style_td),
            Paragraph("Top-1 Accuracy > 88% on remote sensing benchmarks", style_td),
            Paragraph("Automated benchmark evaluation on RSVQA & AID datasets", style_td)
        ],
        [
            Paragraph("Low-Latency Inference", style_td),
            Paragraph("< 2.5s inference time on standard GPU (4-bit NF4)", style_td),
            Paragraph("Uvicorn response timer logging & /api/health telemetry", style_td)
        ],
        [
            Paragraph("Low Hardware Footprint", style_td),
            Paragraph("< 6.0 GB VRAM requirement for execution", style_td),
            Paragraph("bitsandbytes 4-bit NF4 quantization validation", style_td)
        ],
        [
            Paragraph("Realtime Synchronization", style_td),
            Paragraph("Sub-500ms sync delay over Firebase Realtime DB", style_td),
            Paragraph("End-to-end event stream benchmark", style_td)
        ]
    ]
    t_metrics = Table(metrics_data, colWidths=[130, 180, 194])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COLOR_PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, COLOR_LIGHT]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_metrics)
    
    # ── Section 3: Feature Specifications ──
    story.append(Spacer(1, 10))
    story.append(Paragraph("3. Functional Feature Specifications", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceBefore=0, spaceAfter=8))
    
    story.append(Paragraph("Feature 1: Visual Question Answering (VQA)", style_h2))
    story.append(Paragraph("• <b>Capability:</b> Accepts a satellite/aerial image (base64 or multipart upload) alongside a natural language question (e.g., <i>'How many industrial tanks are located in the north sector?'</i>).", style_bullet))
    story.append(Paragraph("• <b>Endpoints:</b> <code>POST /api/vqa</code>, <code>POST /api/vqa/upload</code>", style_bullet))
    story.append(Paragraph("• <b>Output:</b> Concise, context-aware natural language answer + confidence score + processing timing.", style_bullet))
    
    story.append(Paragraph("Feature 2: Automated Satellite Image Captioning", style_h2))
    story.append(Paragraph("• <b>Capability:</b> Generates comprehensive textual descriptions covering terrain type, land usage, building density, water bodies, and vegetation index.", style_bullet))
    story.append(Paragraph("• <b>Endpoints:</b> <code>POST /api/caption</code>, <code>POST /api/caption/upload</code>", style_bullet))
    story.append(Paragraph("• <b>Output:</b> Detailed paragraph describing key visual elements in remote sensing terminology.", style_bullet))
    
    story.append(Paragraph("Feature 3: Referring Expression Grounding (Object Localization)", style_h2))
    story.append(Paragraph("• <b>Capability:</b> Locates specific objects requested via text (e.g., <i>'The dark green solar panel array near the road'</i>) and predicts spatial coordinates.", style_bullet))
    story.append(Paragraph("• <b>Endpoints:</b> <code>POST /api/refer</code>, <code>POST /api/refer/upload</code>", style_bullet))
    story.append(Paragraph("• <b>Output:</b> Normalized bounding box <code>[x1, y1, x2, y2]</code> scaled from <code>[0.0, 1.0]</code>.", style_bullet))

    story.append(Paragraph("Feature 4: Multi-Channel Realtime Engine (Firebase + REST)", style_h2))
    story.append(Paragraph("• <b>Capability:</b> Integrated dual-pipeline supporting direct REST HTTP requests and asynchronous web application sync via Firebase Realtime Database and Firebase Storage.", style_bullet))
    story.append(Paragraph("• <b>Firebase Nodes:</b> <code>/queries/{id}</code> (incoming queue) & <code>/results/{id}</code> (processed output stream).", style_bullet))

    # ── Section 4: Architecture & Stack ──
    story.append(Spacer(1, 10))
    story.append(Paragraph("4. Technical System Architecture", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceBefore=0, spaceAfter=8))
    
    arch_data = [
        [Paragraph("Layer", style_th), Paragraph("Technology Stack", style_th), Paragraph("Role & Key Configuration", style_th)],
        [
            Paragraph("AI / LLM Model", style_td),
            Paragraph("Qwen2.5-VL-3B + LoRA<br/>PyTorch 2.14 · PEFT 0.20", style_td),
            Paragraph("Fine-tuned vision-language model with 4-bit NF4 quantization for low-memory execution.", style_td)
        ],
        [
            Paragraph("Backend Framework", style_td),
            Paragraph("FastAPI 0.115+<br/>Uvicorn ASGI Server", style_td),
            Paragraph("Asynchronous REST API, CORS middleware, lifespan events, and openapi documentation.", style_td)
        ],
        [
            Paragraph("Realtime DB & Storage", style_td),
            Paragraph("Firebase Admin SDK<br/>Realtime DB + Cloud Storage", style_td),
            Paragraph("Background listener service for async client processing and cloud image uploads.", style_td)
        ],
        [
            Paragraph("Image Processing", style_td),
            Paragraph("Pillow 12.3<br/>qwen-vl-utils 0.0.14", style_td),
            Paragraph("Image format decoding, base64 extraction, aspect-ratio patch division for Qwen vision encoder.", style_td)
        ]
    ]
    t_arch = Table(arch_data, colWidths=[110, 160, 234])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), COLOR_PRIMARY),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, COLOR_LIGHT]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_arch)
    
    # ── Section 5: API Specification & Endpoint Schema ──
    story.append(Spacer(1, 10))
    story.append(Paragraph("5. REST API & Endpoint Contracts", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceBefore=0, spaceAfter=8))
    
    story.append(Paragraph("Sample Health Check Payload (GET /api/health):", style_h2))
    code_health = """{
  "status": "ok",
  "model_loaded": true,
  "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
  "firebase_listener": true,
  "queries_processed": 142,
  "errors": 0
}"""
    story.append(Paragraph(code_health, style_code))

    story.append(Paragraph("Sample VQA Request & Response (POST /api/vqa):", style_h2))
    code_vqa = """// Request
{
  "image_base64": "data:image/jpeg;base64,...",
  "question": "What is the primary land use in this region?"
}

// Response
{
  "answer": "The image shows predominantly agricultural farmland with circular center-pivot irrigation fields.",
  "confidence": 0.94,
  "task": "vqa",
  "inference_time_ms": 1420
}"""
    story.append(Paragraph(code_vqa, style_code))

    # ── Section 6: Non-Functional Requirements ──
    story.append(Spacer(1, 10))
    story.append(Paragraph("6. Non-Functional & Security Requirements", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceBefore=0, spaceAfter=8))
    
    story.append(Paragraph("• <b>Data Privacy & Security:</b> All core inference operations execute locally on-device. No telemetry or imagery data is sent to 3rd-party LLM providers.", style_bullet))
    story.append(Paragraph("• <b>Scalability:</b> Stateless FastAPI handlers allow multi-worker horizontal scaling behind load balancers.", style_bullet))
    story.append(Paragraph("• <b>Reliability & Fallbacks:</b> Includes stub fallback execution mode (<code>SATQUERY_MODEL_MODE=stub</code>) for cloud UI testing when GPU hardware is offline.", style_bullet))
    story.append(Paragraph("• <b>Maximum File Constraints:</b> Enforces a hard limit of 20 MB per uploaded satellite image to prevent buffer exhaustion.", style_bullet))

    # ── Section 7: Future Roadmap ──
    story.append(Spacer(1, 10))
    story.append(Paragraph("7. Product Roadmap & Future Enhancements", style_h1))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_PRIMARY, spaceBefore=0, spaceAfter=8))
    
    story.append(Paragraph("• <b>Phase 1 (Current):</b> Local Qwen2.5-VL-3B + LoRA deployment, REST API, Firebase Realtime Sync.", style_bullet))
    story.append(Paragraph("• <b>Phase 2 (Q4 2026):</b> Multi-temporal change detection (comparing satellite passes across different dates).", style_bullet))
    story.append(Paragraph("• <b>Phase 3 (Q1 2027):</b> GeoTIFF metadata parsing (extracting exact lat/long bounding boxes and spatial resolution metadata).", style_bullet))
    story.append(Paragraph("• <b>Phase 4 (Q2 2027):</b> Interactive bounding box rendering directly in Web UI canvas.", style_bullet))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PRD PDF successfully generated at: {filename}")

if __name__ == "__main__":
    output_path = r"C:\Users\Suryansh\ByteXX\SatQuery_AI_PRD.pdf"
    create_prd_pdf(output_path)
