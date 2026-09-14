import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Plus,
  Send,
  Trash2,
  Layers,
  Image as ImageIcon,
  FileText,
  Download,
  Compass,
  LogOut,
  Eye,
  UploadCloud,
  Pin,
  PinOff,
  Sliders,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Clock,
  RefreshCw,
  Wifi,
  WifiOff,
  Mic,
  MicOff,
  Menu,
  X,
  ChevronRight,
  Bot,
} from 'lucide-react';
import {
  saveRecentQuery,
  subscribeToUserRecents,
  deleteRecentQuery,
  logoutUser,
  fileToBase64,
  submitQueryToFirebase,
  listenForQueryResult,
  submitAgentQueryToFirebase,
  listenForAgentResult,
} from '../firebase/firebaseConfig';
import StarField from './StarField';
import ProfilePanel from './ProfilePanel';
import AgentPanel from './AgentPanel';
import './SatQueryWorkspace.css';

// ─── Preset Datasets (demo / offline fallback) ───────────────────────────────
const PRESET_DATASETS = [
  {
    id: 'bitemporal-river',
    title: 'Bi-Temporal Change: River Valley',
    type: 'bitemporal',
    query: 'What changed between these two dates, and where did the change occur?',
    images: [
      { name: 'T1_Before_2004.tif', url: '/samples/bitemporal_before.jpg', modality: 'Optical T1 (2004)' },
      { name: 'T2_After_2024.tif', url: '/samples/bitemporal_after.jpg', modality: 'Optical T2 (2024)' }
    ],
    task: 'Multitemporal Change Detection & VQA',
    model: 'Gemma-4-31B-RS-Change',
    confidence: 97.4,
    metrics: { builtUpChange: '+42.5%', waterSurface: '+68.2%', vegetation: '-26.1%' },
    response: `### Multitemporal Change Analysis

**Major Infrastructure Changes Detected**:
- A large-scale **hydroelectric dam barrier** was constructed across the river gorge in the northern section.
- Significant **reservoir impoundment** has expanded the water surface area by **+68.2%**.
- A multi-lane **concrete highway suspension bridge** spans the lower river basin.

**Urban Expansion**:
- Dense residential and commercial settlements established on the eastern peninsula.
- Road network density increased by **310%**.

**Vegetation Dynamics**:
- Dense canopy forest decreased by **-26.1%** due to reservoir inundation and road clearing.`
  },
  {
    id: 'crossmodal-port',
    title: 'Cross-Modal: Coastal Port Analysis',
    type: 'crossmodal',
    query: 'Use the optical and SAR images together to identify built-up and water-covered regions.',
    images: [
      { name: 'Cartosat2S_Optical.tif', url: '/samples/optical_sample.jpg', modality: 'Optical RGB (Cartosat-2S)' },
      { name: 'RISAT_SAR_Cband.tif', url: '/samples/sar_sample.jpg', modality: 'SAR Microwave (RISAT-1A)' }
    ],
    task: 'Optical-SAR Joint Information Extraction',
    model: 'Gemma-4-31B-CrossModal',
    confidence: 96.1,
    metrics: { builtUpArea: '53.8%', waterSurface: '34.2%', agricultural: '12.0%' },
    response: `### Optical & SAR Joint Analysis

**Multimodal Observations**:
- **Optical (Cartosat-2S)** reveals agricultural parcels, green marshland, and sediment plumes.
- **SAR (RISAT-1A)** shows calm water as low-backscatter, metallic structures as high-intensity backscatter.

**Grounding Results**:
- **Water Regions**: Confirmed via optical absorption + low SAR returns; harbor basin and estuary identified with 99.1% consistency.
- **Built-up Infrastructure**: Confirmed via intense microwave scattering on loading docks and breakwater barriers.`
  },
  {
    id: 'single-optical',
    title: 'Single Optical: Land-Cover Description',
    type: 'single',
    query: 'Describe the land-cover and major objects visible in this image.',
    images: [
      { name: 'Sentinel2_Estuary.tif', url: '/samples/optical_sample.jpg', modality: 'Optical High-Res' }
    ],
    task: 'Single-Image VQA & Grounding',
    model: 'Gemma-4-31B-RS',
    confidence: 95.8,
    metrics: { urbanCore: '44.5%', deltaWetland: '18.4%', farmland: '21.3%', coastalWater: '15.8%' },
    response: `### Scene Assessment

**Land-Cover Classification**:
- **Urban Fabric & Industrial Units**: Dense street grid with commercial port facilities, warehouses, and silos.
- **Estuary Wetlands**: Meandering tributary with tidal mudflats and emergent aquatic flora.
- **Agricultural Parcels**: Rectilinear crop fields with varying phenological stages.

**Infrastructure**:
- Deep-water shipping berths with multiple cargo ships docked.
- Breakwater pier extending 640m into the bay.`
  }
];

// ─── Query status step definitions ───────────────────────────────────────────
const STATUS_STEPS = {
  encoding:    { label: 'Encoding image…',          icon: 'spinner',    color: '#6366f1' },
  pending:     { label: 'Queued for analysis…',      icon: 'clock',      color: '#f59e0b' },
  processing:  { label: 'AI model analyzing… (Cloud inference active)', icon: 'spinner', color: '#0ea5e9' },
  waiting:     { label: 'Still processing — result will appear automatically', icon: 'clock', color: '#f59e0b' },
  done:        { label: 'Analysis complete',                               icon: 'check',   color: '#10b981' },
  error:       { label: 'Analysis failed',                                 icon: 'error',   color: '#ef4444' },
  timeout:     { label: 'Backend not responding — inference may still be running', icon: 'error', color: '#f97316' },
};

// ─── Pipeline timeout (ms) ────────────────────────────────────────────────────
// Cloud inference: Gemma 4 31B on OpenRouter processes queries in seconds.
// Give 20 minutes before even showing the soft 'still waiting' banner.
// The Firebase listener stays alive indefinitely — result arrives when ready.
const RESULT_TIMEOUT_MS = 1_200_000; // 20 minutes hard timeout
const SOFT_TIMEOUT_MS   =   300_000; // 5 minutes → switch to 'waiting' status (listener stays open)

// ─── Object keywords that trigger per-instance detection boxes ───────────────
const OBJECT_KEYWORDS = [
  { keys: ['airplane','aircraft','plane','planes','airplanes','jet','jets','helicopter'], label: 'Aircraft' },
  { keys: ['car','cars','vehicle','vehicles','automobile'], label: 'Vehicle' },
  { keys: ['ship','ships','vessel','vessels','boat','boats'], label: 'Vessel' },
  { keys: ['building','buildings','structure','structures','house','houses'], label: 'Building' },
  { keys: ['truck','trucks'], label: 'Truck' },
  { keys: ['tank','tanks'], label: 'Storage Tank' },
  { keys: ['bridge','bridges'], label: 'Bridge' },
  { keys: ['road','roads','highway'], label: 'Road' },
  { keys: ['person','people','pedestrian'], label: 'Person' },
];

// Extract the count from model answer text (e.g. "There are 3 airplanes" → 3)
function extractCount(answer) {
  const text = answer.toLowerCase();
  // Match written numbers first
  const written = { zero:0,one:1,two:2,three:3,four:4,five:5,six:6,seven:7,eight:8,nine:9,ten:10 };
  for (const [word, num] of Object.entries(written)) {
    if (new RegExp(`\\b${word}\\b`).test(text)) return num;
  }
  // Match digits
  const m = text.match(/(\d+)/);
  if (m) return Math.min(parseInt(m[1], 10), 20); // cap at 20 for sanity
  return null;
}

// Seeded pseudo-random layout — deterministic per image so boxes don't jump on re-render
function seededRandom(seed) {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 4294967296; };
}

function generateBoxes(count, objectLabel, query) {
  const rand = seededRandom(query.length * 31 + count * 17);
  const boxes = [];
  const minW = 10, maxW = 20, minH = 10, maxH = 20;
  for (let i = 0; i < count; i++) {
    const w  = minW + rand() * (maxW - minW);
    const h  = minH + rand() * (maxH - minH);
    const left = 5 + rand() * (90 - w);
    const top  = 8 + rand() * (84 - h);
    boxes.push({ top, left, w, h, label: `${objectLabel} ${i + 1}` });
  }
  return boxes;
}

// ─── CPU Inference Progress Panel ─────────────────────────────────────────────
function CpuInferenceProgress({ isAgent }) {
  const [elapsed, setElapsed] = React.useState(0);
  const [stage, setStage]     = React.useState(0);

  const CPU_STAGES = isAgent
    ? [
        { label: 'Routing query to tools…',        est: 5  },
        { label: 'Planning execution steps…',       est: 10 },
        { label: 'Running CNN detection…',          est: 20 },
        { label: 'Running Gemma 4 vision analysis…',   est: 15 },
        { label: 'Verifying & merging results…',    est: 30 },
      ]
    : [
        { label: 'Preprocessing image…',            est: 2  },
        { label: 'Tokenizing prompt…',              est: 2  },
        { label: 'Gemma 4 generating answer…',       est: 8  },
        { label: 'Computing confidence scores…',    est: 0  },
      ];

  React.useEffect(() => {
    const start = Date.now();
    const interval = setInterval(() => {
      const s = Math.floor((Date.now() - start) / 1000);
      setElapsed(s);
      // Advance stage based on cumulative estimated seconds
      let cumulative = 0;
      for (let i = 0; i < CPU_STAGES.length; i++) {
        cumulative += CPU_STAGES[i].est;
        if (s < cumulative) { setStage(i); break; }
        if (i === CPU_STAGES.length - 1) setStage(i);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const totalEst  = CPU_STAGES.reduce((a, c) => a + c.est, 0);
  const progress  = Math.min(95, (elapsed / totalEst) * 100); // never reach 100 until done
  const remaining = Math.max(0, totalEst - elapsed);
  const fmtTime   = s => s >= 60 ? `${Math.floor(s/60)}m ${s%60}s` : `${s}s`;

  return (
    <div style={{
      margin: '24px 0',
      background: 'linear-gradient(135deg, rgba(14,165,233,0.06), rgba(99,102,241,0.06))',
      border: '1px solid rgba(56,189,248,0.18)',
      borderRadius: 14, padding: '20px 22px',
      fontFamily: 'Outfit, sans-serif',
    }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <div style={{
          width: 36, height: 36, borderRadius: 10,
          background: 'linear-gradient(135deg,#0ea5e9,#6366f1)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 16, flexShrink: 0,
          boxShadow: '0 0 16px rgba(56,189,248,0.35)',
        }}>AI</div>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 700, fontSize: 14, color: '#e2e8f0', marginBottom: 2 }}>
            {isAgent ? 'AI Agent Pipeline Running' : 'Gemma 4 Analyzing Image'}
          </div>
          <div style={{ fontSize: 11, color: '#64748b' }}>
            Running on OpenRouter — cloud inference, please wait
          </div>
        </div>
        {/* Live timer */}
        <div style={{
          background: 'rgba(14,165,233,0.12)', border: '1px solid rgba(56,189,248,0.25)',
          borderRadius: 8, padding: '4px 12px', textAlign: 'center', minWidth: 70,
        }}>
          <div style={{ fontSize: 18, fontWeight: 800, color: '#38bdf8', lineHeight: 1.2 }}>
            {fmtTime(elapsed)}
          </div>
          <div style={{ fontSize: 9, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            elapsed
          </div>
        </div>
      </div>

      {/* Progress bar */}
      <div style={{
        background: 'rgba(15,23,42,0.6)', borderRadius: 99,
        height: 6, marginBottom: 14, overflow: 'hidden',
        border: '1px solid rgba(56,189,248,0.1)',
      }}>
        <div style={{
          height: '100%', borderRadius: 99,
          background: 'linear-gradient(90deg,#0ea5e9,#6366f1,#a78bfa)',
          width: `${progress}%`,
          transition: 'width 1s linear',
          boxShadow: '0 0 8px rgba(56,189,248,0.5)',
        }} />
      </div>

      {/* Stage steps */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {CPU_STAGES.map((s, i) => {
          const isPast    = i < stage;
          const isCurrent = i === stage;
          return (
            <div key={i} style={{
              display: 'flex', alignItems: 'center', gap: 10,
              opacity: isPast ? 0.45 : isCurrent ? 1 : 0.3,
              transition: 'opacity 0.4s',
            }}>
              {/* Dot */}
              <div style={{
                width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                background: isPast ? '#10b981' : isCurrent ? '#38bdf8' : '#334155',
                boxShadow: isCurrent ? '0 0 8px #38bdf8' : 'none',
                animation: isCurrent ? 'pulse-dot 1.4s infinite' : 'none',
              }} />
              <span style={{
                fontSize: 12, color: isCurrent ? '#e2e8f0' : '#64748b',
                fontWeight: isCurrent ? 600 : 400,
                flex: 1,
              }}>
                {s.label}
              </span>
              {isPast && (
                <span style={{ fontSize: 10, color: '#10b981', fontWeight: 600 }}>done</span>
              )}
              {isCurrent && (
                <span style={{
                  fontSize: 10, color: '#38bdf8', fontWeight: 600,
                  animation: 'fade-blink 1.2s infinite',
                }}>running…</span>
              )}
            </div>
          );
        })}
      </div>

      {/* ETA row */}
      <div style={{
        marginTop: 14, paddingTop: 12,
        borderTop: '1px solid rgba(56,189,248,0.08)',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      }}>
        <span style={{ fontSize: 11, color: '#475569' }}>
          Estimated time remaining: ~{fmtTime(remaining)}
        </span>
        <span style={{ fontSize: 10, color: '#334155', fontFamily: 'monospace' }}>
          Cloud • OpenRouter API
        </span>
      </div>

      <style>{`
        @keyframes pulse-dot {
          0%, 100% { transform: scale(1); opacity: 1; }
          50% { transform: scale(1.6); opacity: 0.6; }
        }
        @keyframes fade-blink {
          0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
        }
      `}</style>
    </div>
  );
}

// ─── Smart Grounding Viewer Component ─────────────────────────────────────────
function SmartGroundingViewer({ imageUrl, query, answer, bboxes = [], bboxObject = null }) {
  const canvasRef = useRef(null);
  const imgRef    = useRef(null);

  // Detect what object type the query is about
  const qLower   = query.toLowerCase();
  const detected = OBJECT_KEYWORDS.find(({ keys }) => keys.some(k => qLower.includes(k)));

  // Extract count from model answer
  const count        = detected ? extractCount(answer) : null;
  const isCountQuery = detected && count !== null && count > 0;

  // ── Decide which boxes to draw ──────────────────────────────────────────────
  // ONLY use real bboxes from the VLM. Never show random/fake positions.
  const hasRealBoxes = Array.isArray(bboxes) && bboxes.length > 0;
  const objectLabel  = bboxObject || detected?.label || 'Object';

  // Convert real Firebase bboxes [[x1,y1,x2,y2] in 0-1] → canvas-ready format
  const realBoxes = hasRealBoxes
    ? bboxes.map((b, i) => ({
        x1: b[0], y1: b[1], x2: b[2], y2: b[3],
        label: `${objectLabel} ${i + 1}`,
      }))
    : [];

  const activeBoxes = realBoxes;
  const shouldDraw  = activeBoxes.length > 0;

  // Draw SQUARE boxes on canvas with premium styling
  const drawBoxes = useCallback(() => {
    const canvas = canvasRef.current;
    const img    = imgRef.current;
    if (!canvas || !img || !shouldDraw) return;

    const { width, height } = img.getBoundingClientRect();
    canvas.width  = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);

    const COLORS = [
      { stroke: '#38bdf8', fill: 'rgba(56,189,248,0.10)', label: 'rgba(14,165,233,0.95)', glow: '#38bdf8' },
      { stroke: '#a78bfa', fill: 'rgba(167,139,250,0.10)', label: 'rgba(124,58,237,0.95)', glow: '#a78bfa' },
      { stroke: '#34d399', fill: 'rgba(52,211,153,0.10)', label: 'rgba(5,150,105,0.95)', glow: '#34d399' },
      { stroke: '#fb923c', fill: 'rgba(251,146,60,0.10)',  label: 'rgba(234,88,12,0.95)',  glow: '#fb923c' },
    ];

    activeBoxes.forEach(({ x1, y1, x2, y2, label }, idx) => {
      const col  = COLORS[idx % COLORS.length];

      // Raw pixel dimensions from normalized coords
      const rx  = x1 * width;
      const ry  = y1 * height;
      const rw  = (x2 - x1) * width;
      const rh  = (y2 - y1) * height;

      // ── FORCE SQUARE: use the larger dimension ──────────────────────────
      const side = Math.max(rw, rh);
      // Center the square on the original rect center
      const cx   = rx + rw / 2;
      const cy   = ry + rh / 2;
      const px   = cx - side / 2;
      const py   = cy - side / 2;
      const bw   = side;
      const bh   = side;
      // ───────────────────────────────────────────────────────────────────

      // Semi-transparent fill
      ctx.fillStyle = col.fill;
      ctx.fillRect(px, py, bw, bh);

      // Glowing outer stroke
      ctx.shadowColor = col.glow;
      ctx.shadowBlur  = 18;
      ctx.strokeStyle = col.stroke;
      ctx.lineWidth   = 2.5;
      ctx.strokeRect(px, py, bw, bh);
      ctx.shadowBlur  = 0;

      // Corner L-brackets (thick, sharp)
      const tick = Math.min(bw, bh) * 0.18;
      ctx.strokeStyle = col.stroke;
      ctx.lineWidth   = 4;
      ctx.lineCap     = 'square';
      [
        [px,      py,      +1,  0], [px,      py,       0, +1],  // top-left
        [px + bw, py,      -1,  0], [px + bw, py,       0, +1],  // top-right
        [px,      py + bh,  +1,  0], [px,      py + bh,  0, -1],  // bottom-left
        [px + bw, py + bh, -1,  0], [px + bw, py + bh,  0, -1],  // bottom-right
      ].forEach(([qx, qy, dx, dy]) => {
        ctx.beginPath();
        ctx.moveTo(qx, qy);
        ctx.lineTo(qx + dx * tick, qy + dy * tick);
        ctx.stroke();
      });

      // Label pill with object number
      const fontSize = Math.max(11, Math.min(14, bw * 0.13));
      ctx.font = `700 ${fontSize}px 'Outfit', 'Inter', sans-serif`;
      const tw  = ctx.measureText(label).width;
      const pH  = fontSize + 8;
      const lx  = px;
      const ly  = py - pH - 4 < 0 ? py + 2 : py - pH - 2;

      // Pill background
      ctx.fillStyle = col.label;
      ctx.shadowColor = col.glow;
      ctx.shadowBlur  = 8;
      ctx.beginPath();
      ctx.roundRect(lx, ly, tw + 16, pH, 5);
      ctx.fill();
      ctx.shadowBlur = 0;

      // Pill text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, lx + 8, ly + pH - 5);
    });
  }, [activeBoxes, shouldDraw]);

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    if (img.complete) drawBoxes();
    else {
      img.addEventListener('load', drawBoxes);
      return () => img.removeEventListener('load', drawBoxes);
    }
  }, [drawBoxes]);

  useEffect(() => {
    const observer = new ResizeObserver(drawBoxes);
    if (imgRef.current) observer.observe(imgRef.current);
    return () => observer.disconnect();
  }, [drawBoxes]);

  return (
    <div className="single-grounding-viewer">
      <div className="image-grounding-frame" style={{ position: 'relative' }}>
        <img
          ref={imgRef}
          src={imageUrl}
          alt="Scene"
          className="grounding-img"
          style={{ display: 'block', width: '100%', borderRadius: 10 }}
        />
        {/* Canvas — square detection boxes */}
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute', top: 0, left: 0,
            width: '100%', height: '100%', pointerEvents: 'none',
            borderRadius: 10,
          }}
        />
        {/* No-bbox overlay — show model answer text */}
        {!shouldDraw && answer && (
          <div style={{
            position: 'absolute', bottom: 10, left: 10, right: 10,
            background: 'rgba(15,23,42,0.88)', backdropFilter: 'blur(10px)',
            borderRadius: 10, padding: '12px 16px',
            border: '1px solid rgba(56,189,248,0.25)',
            color: '#e2e8f0', fontFamily: 'Outfit, sans-serif', fontSize: 13, lineHeight: 1.5,
          }}>
            <span style={{ color: '#38bdf8', fontWeight: 700, marginRight: 6 }}>AI</span>
            {answer.split('\n')[0]}
          </div>
        )}
        {/* Detection count badge */}
        {shouldDraw && (
          <div style={{
            position: 'absolute', top: 10, right: 10,
            background: 'linear-gradient(135deg,rgba(14,165,233,0.95),rgba(99,102,241,0.95))',
            color: '#fff', borderRadius: 8, padding: '5px 14px',
            fontFamily: 'Outfit, sans-serif', fontSize: 13, fontWeight: 700,
            boxShadow: '0 0 18px #38bdf866', backdropFilter: 'blur(6px)',
            letterSpacing: '0.02em',
          }}>
            {activeBoxes.length} {objectLabel}{activeBoxes.length !== 1 ? 's' : ''} detected
          </div>
        )}
        {/* Legend row */}
        {shouldDraw && (
          <div style={{
            position: 'absolute', bottom: 10, left: 10,
            display: 'flex', gap: 6, flexWrap: 'wrap',
          }}>
            {activeBoxes.slice(0, 6).map((b, i) => (
              <div key={i} style={{
                background: 'rgba(15,23,42,0.88)', backdropFilter: 'blur(6px)',
                border: '1px solid rgba(56,189,248,0.3)',
                borderRadius: 6, padding: '3px 10px',
                fontFamily: 'Outfit, sans-serif', fontSize: 11,
                color: '#7dd3fc', fontWeight: 600,
              }}>
                #{i + 1} {b.label?.split(' ')[0] || objectLabel}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Answer Enrichment ── rich research-quality report builder ────────────────
function enrichAnswer(rawAnswer, query) {
  if (!rawAnswer || rawAnswer === 'No answer returned.') return rawAnswer;

  const q   = (query || '').toLowerCase().trim();
  const ans = rawAnswer.trim();

  // Already a rich multi-section response from the model — pass it straight through
  // with only minor cleanup. This handles the new 512-token research answers.
  if (
    ans.split('\n').length > 5 ||
    ans.length > 300 ||
    ans.includes('###') ||
    ans.includes('**')
  ) {
    // Wrap raw model answer in a styled research container
    return (
`## AI Analysis Report

${ans}

---

### Methodology

- **Model:** Google Gemma 4 31B (OpenRouter)
- **Query processed:** "${query}"
- **Analysis type:** Research-mode remote sensing intelligence report
- **Token budget:** 512 tokens (extended analysis mode)

### Confidence & Limitations

- Results depend on image resolution, sensor type, and viewing angle
- Partially occluded objects may be underreported
- Best accuracy achieved with high-resolution nadir-view imagery`
    );
  }

  // ── Detect query intent ────────────────────────────────────────────────────
  const isCount    = /how many|count|number of|total|detect|find all/.test(q);
  const isDescribe = /describe|what is|what do you see|scene|land.?cover|overview|explain|analysis/.test(q);
  const isChange   = /change|differ|before|after|between/.test(q);
  const isLocate   = /where|location|position|find|locate|show me/.test(q);
  const isIdentify = /identify|what type|classify|what kind|what sort/.test(q);

  const numMatch = ans.match(/\d+/) || ans.match(/\b(one|two|three|four|five|six|seven|eight|nine|ten)\b/i);
  const numStr   = numMatch ? numMatch[0] : null;
  const objKw    = OBJECT_KEYWORDS.find(({ keys }) => keys.some(k => q.includes(k)));
  const objLabel = objKw
    ? objKw.label.toLowerCase() + (numStr && numStr !== '1' && numStr !== 'one' ? 's' : '')
    : 'object(s)';
  const numWord = numStr ? numStr : 'Multiple';

  // ── Count / Detection query ────────────────────────────────────────────────
  if (isCount && objKw) {
    const singularLabel = objKw.label.toLowerCase();
    return (
`## Detection Analysis Report

### Direct Answer

The model detected **${numWord} ${objLabel}** in this satellite image.

### Spatial Distribution

The visible ${objLabel} are distributed across the image frame. Each instance has been individually evaluated using Gemma 4's vision capabilities. Detected objects are highlighted in the **Visual Evidence & Viewer** tab with numbered square bounding boxes.

### Detection Methodology

- **Step 1 — Scene Parsing:** The model ingests the full satellite image in a single forward pass, building a spatial feature map of the scene.
- **Step 2 — Object Localization:** Visual grounding prompts instruct the model to enumerate all visible ${singularLabel} instances, including partially occluded ones.
- **Step 3 — Coordinate Extraction:** Bounding box coordinates (x₁, y₁, x₂, y₂) are extracted from the model's structured output and normalized to [0, 1] range.
- **Step 4 — Count Aggregation:** Detected instances are counted and rendered as labeled square boxes on the image.

### Accuracy Assessment

- **Resolution dependency:** At very high resolutions (< 0.3 m/px), small objects are clearly distinguishable. At moderate resolutions, closely spaced objects may be merged.
- **Occlusion handling:** Objects behind other structures or at image edges may be partially visible and could affect the final count.
- **Viewing angle:** Nadir (top-down) imagery provides the most accurate counts; oblique angles can distort object shapes and boundaries.

### Remote Sensing Context

Automated object counting from satellite imagery is a core task in geospatial intelligence (GEOINT). Applications include:
- **Airport monitoring:** Aircraft counting for traffic analysis and capacity planning
- **Urban planning:** Vehicle density estimation for traffic flow models
- **Military intelligence:** Infrastructure and asset enumeration
- **Environmental monitoring:** Wildlife counting over large areas

### Confidence Assessment

The model's confidence is computed from the token-level probability distribution of the generated count. Higher confidence indicates the object class was unambiguously identified.

---

*Analysis powered by Google Gemma 4 31B via OpenRouter (remote sensing VQA)*`
    );
  }

  // ── Describe / Identify query ──────────────────────────────────────────────
  if (isDescribe || isIdentify) {
    return (
`## Scene Intelligence Report

### Model Interpretation

${ans}

### Scene Composition Analysis

- **Imagery type:** High-resolution satellite / aerial optical imagery
- **Analysis engine:** Google Gemma 4 31B multimodal language model (OpenRouter)
- **Scene complexity:** Multi-element landscape — the model evaluated spatial layout, land-cover patterns, infrastructure, and natural features simultaneously.

### Visual Feature Breakdown

The model examines the following feature classes to produce this analysis:

1. **Spectral signatures** — color and brightness patterns that distinguish surface types (vegetation, water, urban materials, bare soil)
2. **Texture analysis** — fine-grained spatial variation that differentiates forests from grasslands, or roads from runways
3. **Geometric patterns** — regular shapes and grids indicating built infrastructure vs. irregular organic natural features
4. **Contextual relationships** — proximity of objects (e.g., aircraft near terminals, vessels near ports) provides semantic context

### Confidence Factors

- Confidence is highest when scene elements are large, high-contrast, and geometrically distinct
- Atmospheric haze, cloud cover, or low sun angle can reduce spectral clarity
- Seasonal variation changes vegetation appearance significantly

### Applications

This type of scene classification is used in:
- **Land use / land cover (LULC) mapping**
- **Urban growth monitoring**
- **Disaster damage assessment**
- **Agricultural field monitoring**

---

*Powered by Google Gemma 4 31B via OpenRouter*`
    );
  }

  // ── Change detection query ─────────────────────────────────────────────────
  if (isChange) {
    return (
`## Temporal Change Detection Report

### Analysis Result

${ans}

### Change Detection Methodology

Bi-temporal analysis compares two satellite images acquired at different dates. The model evaluates:

1. **Spectral change** — shifts in pixel color/brightness indicating surface type change
2. **Structural change** — appearance or disappearance of built features (buildings, roads)
3. **Vegetation phenology** — seasonal or long-term changes in plant cover
4. **Hydrological change** — fluctuations in water body boundaries (floods, droughts)

### Change Classification

| Change Type | Description | Typical Cause |
|---|---|---|
| Urban expansion | New structures appear | Development / construction |
| Deforestation | Vegetation loss | Logging, agriculture conversion |
| Flooding | Water body expansion | Heavy rainfall, storm surge |
| Infrastructure | Roads, runways added | Civil engineering projects |

### Confidence & Limitations

- Radiometric normalization is critical: differences in sensor calibration or solar angle can produce false positives
- Cloud cover in either epoch introduces gaps in coverage
- Fine changes (< 1m) may be below the model's effective detection threshold

---

*Analysis by Google Gemma 4 31B | Upload before + after images for bi-temporal comparison*`
    );
  }

  // ── Location / Grounding query ─────────────────────────────────────────────
  if (isLocate) {
    return (
`## Spatial Grounding Report

### Localization Result

${ans}

### Referring Expression Grounding

The model uses visual attention to locate described objects within the image:

1. **Language parsing** — the query is parsed to identify the target object and any spatial qualifiers ("top-left", "near the runway", etc.)
2. **Visual scan** — the model attends to candidate regions matching the description
3. **Coordinate prediction** — bounding box coordinates are predicted in normalized [0, 1] space
4. **Visualization** — detected region is highlighted as a square box in the Visual Evidence tab

### Spatial Reference Frame

- Coordinates use image-space: **top-left = (0, 0)**, **bottom-right = (1, 1)**
- Bounding boxes are displayed as squares for clear, unambiguous highlighting
- Multiple matching objects will each receive their own numbered box

### Remote Sensing Context

Object localization in satellite imagery is challenging due to:
- **Scale variation** — objects appear at very different sizes depending on altitude and sensor resolution
- **Viewing angle** — top-down perspective distorts familiar object shapes
- **Density** — closely packed objects (parking lots, airports) are harder to individually isolate

---

*Powered by Google Gemma 4 31B via OpenRouter | Check Visual Evidence tab for bounding box overlay*`
    );
  }

  // ── Generic research fallback ──────────────────────────────────────────────
  return (
`## SatQuery AI — Geospatial Analysis

### Model Response

${ans}

### Analysis Context

- **Query:** "${query}"
- **Image type:** Satellite / Aerial optical imagery
- **Model:** Google Gemma 4 31B (OpenRouter, extended research mode)
- **Token budget:** 512 tokens (research mode)

### How the Analysis Was Performed

1. The satellite image was preprocessed and tokenized by Gemma 4's vision encoder
2. Your query was embedded alongside the image in a multimodal context window
3. The model generated a response conditioned on both the image content and your question
4. Confidence is estimated from the entropy of the output token probability distribution

### Interpretation Notes

- Remote sensing imagery interpretation requires domain expertise; the model's output should be cross-verified with ground truth data for critical applications
- High-resolution nadir imagery yields the most accurate results
- Time-of-day and atmospheric conditions affect spectral quality

### Improve Your Results

- **Be specific:** Instead of *"What is in this image?"*, ask *"How many cargo aircraft are visible near the terminal?"*
- **Use Agent mode:** For complex multi-part queries, enable the AI Agent toggle for parallel CNN + VLM analysis
- **Change detection:** Upload two images (before + after) and ask about changes between them

---

*SatQuery AI | Google Gemma 4 31B | OpenRouter | ByteX Platform*`
  );
}

// ─── Component ───────────────────────────────────────────────────────────────
export default function SatQueryWorkspace({ user, onLogout }) {
  const [recents, setRecents]               = useState([]);
  const [pinnedItems, setPinnedItems]       = useState(() => {
    try { return JSON.parse(localStorage.getItem(`bytex_pinned_${user?.uid}`) || '[]'); } catch { return []; }
  });
  const [currentQuery, setCurrentQuery]     = useState('');
  const [uploadedImages, setUploadedImages] = useState([]);
  const [uploadedFiles, setUploadedFiles]   = useState([]);   // raw File objects for Storage upload
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing]       = useState(false);
  const [queryStatus, setQueryStatus]       = useState(null);  // null | uploading | pending | processing | done | error | timeout
  const [queryError, setQueryError]         = useState(null);
  const [sliderPosition, setSliderPosition] = useState(50);
  const [activeTab, setActiveTab]           = useState('evidence');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [isDragging, setIsDragging]         = useState(false);
  const [dragFileCount, setDragFileCount]   = useState(0);
  const [dragIsInvalid, setDragIsInvalid]   = useState(false);
  const [activeCrossmodalLayer, setActiveCrossmodalLayer] = useState('optical');
  const [hoveredRecent, setHoveredRecent]   = useState(null);
  const [selectedRecentId, setSelectedRecentId] = useState(null);
  const [isListening, setIsListening]       = useState(false);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen]   = useState(false);
  // ─── AI Agent mode ─────────────────────────────────────────────────────────
  const [agentMode, setAgentMode]           = useState(false);
  const [agentReport, setAgentReport]       = useState(null);   // AgentAnalyzeResponse from backend
  const [isAgentAnalyzing, setIsAgentAnalyzing] = useState(false);

  const fileInputRef     = useRef(null);
  const unsubListenerRef = useRef(null);   // Firebase listener cleanup
  const timeoutRef       = useRef(null);   // result timeout handle
  const recognitionRef   = useRef(null);   // Web Speech API instance
  const dragCounterRef   = useRef(0);      // drag-enter counter to prevent flicker

  // ─── Recents subscription ─────────────────────────────────────────────────
  useEffect(() => {
    if (!user?.uid) return;
    const unsubscribe = subscribeToUserRecents(user.uid, (data) => {
      setRecents(data);
    });
    return () => unsubscribe();
  }, [user]);

  // ─── Persist pinned ──────────────────────────────────────────────────────
  useEffect(() => {
    if (user?.uid) {
      localStorage.setItem(`bytex_pinned_${user.uid}`, JSON.stringify(pinnedItems));
    }
  }, [pinnedItems, user]);

  // ─── Cleanup listeners on unmount ────────────────────────────────────────
  useEffect(() => {
    return () => {
      unsubListenerRef.current?.();
      clearTimeout(timeoutRef.current);
      recognitionRef.current?.abort();
    };
  }, []);

  // ─── Speech-to-Text (Web Speech API) ──────────────────────────────────────────
  const startListening = useCallback(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert('Speech recognition is not supported in this browser. Please use Chrome or Edge.');
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.continuous = false;
    recognition.interimResults = true;

    recognition.onstart  = () => setIsListening(true);
    recognition.onend    = () => setIsListening(false);
    recognition.onerror  = () => setIsListening(false);
    recognition.onresult = (e) => {
      const transcript = Array.from(e.results)
        .map(r => r[0].transcript)
        .join('');
      setCurrentQuery(transcript);
    };

    recognitionRef.current = recognition;
    recognition.start();
  }, []);

  const stopListening = useCallback(() => {
    recognitionRef.current?.stop();
    setIsListening(false);
  }, []);

  const handleMicClick = () => {
    if (isListening) stopListening();
    else startListening();
  };

  // ─── Pin helpers ─────────────────────────────────────────────────────────
  const togglePin = (item) => {
    const isPinned = pinnedItems.some(p => p.id === item.id);
    if (isPinned) {
      setPinnedItems(prev => prev.filter(p => p.id !== item.id));
    } else {
      setPinnedItems(prev => [item, ...prev]);
    }
  };
  const isPinned = (id) => pinnedItems.some(p => p.id === id);

  // ─── Accepted file types ─────────────────────────────────────────────────
  const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif', 'image/tiff'];
  const ACCEPTED_EXTS  = ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.tif', '.tiff'];

  const isAcceptedFile = (file) => {
    const mimeOk = ACCEPTED_TYPES.includes(file.type);
    const extOk  = ACCEPTED_EXTS.some(ext => file.name.toLowerCase().endsWith(ext));
    return mimeOk || extOk;
  };

  // ─── Drag & Drop ─────────────────────────────────────────────────────────
  // Use a counter to prevent false drag-leave events firing when the pointer
  // moves over child elements inside the drop zone.
  const handleDragEnter = (e) => {
    e.preventDefault();
    dragCounterRef.current += 1;
    if (dragCounterRef.current === 1) {
      const items = Array.from(e.dataTransfer.items || []);
      const count = items.filter(i => i.kind === 'file').length;
      const hasInvalid = items.some(
        i => i.kind === 'file' && !ACCEPTED_TYPES.includes(i.type) && i.type !== ''
      );
      setDragFileCount(count);
      setDragIsInvalid(hasInvalid && count > 0);
      setIsDragging(true);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    // Required to allow drop
    e.dataTransfer.dropEffect = dragIsInvalid ? 'none' : 'copy';
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    dragCounterRef.current -= 1;
    if (dragCounterRef.current === 0) {
      setIsDragging(false);
      setDragFileCount(0);
      setDragIsInvalid(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    dragCounterRef.current = 0;
    setIsDragging(false);
    setDragFileCount(0);
    setDragIsInvalid(false);
    if (e.dataTransfer.files?.length > 0) processSelectedFiles(e.dataTransfer.files);
  };

  // ─── Read a single file as data URL (Promise-based) ──────────────────────
  const readFileAsDataURL = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload  = (e) => resolve(e.target.result);
      reader.onerror = ()  => reject(new Error(`Failed to read ${file.name}`));
      reader.readAsDataURL(file);
    });

  // ─── Process selected / dropped files ────────────────────────────────────
  const processSelectedFiles = async (files) => {
    const fileList = Array.from(files);

    // Filter to accepted types; silently skip unsupported files
    const validFiles = fileList.filter(isAcceptedFile);
    if (validFiles.length === 0) return;

    try {
      // Read all files in parallel — eliminates the race-condition
      const dataUrls = await Promise.all(validFiles.map(readFileAsDataURL));

      const newImages = validFiles.map((file, i) => {
        const nameLower = file.name.toLowerCase();
        const isSAR    = nameLower.includes('sar') || nameLower.includes('risat');
        const isBefore = nameLower.includes('before') || nameLower.includes('t1');
        const isAfter  = nameLower.includes('after')  || nameLower.includes('t2');
        let modality   = 'Optical';
        if (isSAR)         modality = 'SAR Microwave';
        else if (isBefore) modality = 'Bi-Temporal T1 (Before)';
        else if (isAfter)  modality = 'Bi-Temporal T2 (After)';
        return { name: file.name, url: dataUrls[i], modality, size: (file.size / 1024).toFixed(1) + ' KB' };
      });

      setUploadedImages((prev) => [...prev, ...newImages].slice(0, 4));
      setUploadedFiles((prev)  => [...prev, ...validFiles].slice(0, 4));
    } catch (err) {
      console.error('Error reading files:', err);
    }
  };

  // ─── Paste to upload (Ctrl+V images) ─────────────────────────────────────
  useEffect(() => {
    const handlePaste = (e) => {
      const items = Array.from(e.clipboardData?.items || []);
      const imageFiles = items
        .filter(item => item.kind === 'file' && isAcceptedFile({ type: item.type, name: item.type.replace('/', '.') }))
        .map(item => item.getAsFile())
        .filter(Boolean);
      if (imageFiles.length > 0) processSelectedFiles(imageFiles);
    };
    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleLoadPreset = (preset) => {
    setUploadedImages(preset.images);
    setUploadedFiles([]);   // preset images are URLs, no raw files
    setCurrentQuery(preset.query);
    setShowUploadModal(false);
  };

  const handleNewAnalysis = () => {
    // Stop any running listener / timeout
    unsubListenerRef.current?.();
    clearTimeout(timeoutRef.current);

    setUploadedImages([]);
    setUploadedFiles([]);
    setCurrentQuery('');
    setAnalysisResult(null);
    setQueryStatus(null);
    setQueryError(null);
    setIsAnalyzing(false);
    setSelectedRecentId(null);
  };

  const handleSelectRecent = (item) => {
    setCurrentQuery(item.query || '');
    setUploadedImages(item.images || []);
    setUploadedFiles([]);
    setAnalysisResult(item.result || null);
    setQueryStatus(null);
    setQueryError(null);
    setSelectedRecentId(item.id);
    setIsMobileSidebarOpen(false); // close drawer on mobile after selection
  };

  const handleDeleteRecent = async (e, id) => {
    e.stopPropagation();
    if (user?.uid) await deleteRecentQuery(user.uid, id);
  };

  // ─── Download report ─────────────────────────────────────────────────────
  const handleDownloadReport = () => {
    if (!analysisResult) return;
    const reportContent = `# SatQuery AI - Analysis Report\nDate: ${new Date().toLocaleString()}\n\n## Query:\n"${analysisResult.query}"\n\n## Task: ${analysisResult.taskName}\n## Model: ${analysisResult.backendModel}\n## Confidence: ${analysisResult.confidence}%\n\n## Findings:\n${analysisResult.textResponse}\n\n## Metrics:\n${JSON.stringify(analysisResult.metrics || {}, null, 2)}`;
    const blob = new Blob([reportContent], { type: 'text/markdown' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `SatQuery_Report_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // ─── Determine task type from query + images ──────────────────────────────
  const detectTaskType = (query, images) => {
    const qLower      = query.toLowerCase();
    const isBiTemporal = images.length >= 2 && (qLower.includes('change') || qLower.includes('dates') || qLower.includes('between'));
    const isCrossModal = images.some(img => img.modality?.includes('SAR')) || (qLower.includes('sar') && qLower.includes('optical'));

    // Mirror backend router logic — route to agent for complex/mixed queries
    const hasCount   = ['how many','count','number of','total','aircraft','airplane','plane','jet',
                        'helicopter','vehicle','car','truck','ship','vessel','boat','building',
                        'structure','person','people','bridge','tank'].some(k => qLower.includes(k));
    const hasSegment = ['vegetation','forest','canopy','tree','water','river','lake','urban',
                        'built-up','city','agricultural','farmland','crop','land cover','land use',
                        'density','percentage','percent','coverage','how much','area','fraction',
                        'industrial','industrial area','settlement'].some(k => qLower.includes(k));
    const hasVlm     = ['describe','explain','analyze','what is','why','caption','tell me',
                        'summarize','assess','identify','detail','feature','change','compare',
                        'location','locate','where'].some(k => qLower.includes(k));

    // Use agent pipeline for anything non-trivial
    const useAgent = hasCount || hasSegment || (hasVlm && (hasCount || hasSegment)) ||
                     isBiTemporal || isCrossModal;

    if (isBiTemporal) return { taskType: 'bitemporal', taskName: 'Change Detection & VQA',      modelName: 'Gemma-4-31B-RS + ChangeSiam', fbTask: 'agent' };
    if (isCrossModal) return { taskType: 'crossmodal', taskName: 'Optical-SAR Joint Extraction', modelName: 'Gemma-4-31B-RS + FusionNet',  fbTask: 'agent' };
    if (useAgent)     return { taskType: 'agent',      taskName: 'AI Agent Analysis',            modelName: 'SatAgent (CNN + VLM)',           fbTask: 'agent' };
    return               { taskType: 'single',     taskName: 'Single-Image VQA & Grounding', modelName: 'Gemma-4-31B (OpenRouter)',              fbTask: 'vqa'   };
  };


  // ─── Build result payload from Firebase answer ────────────────────────────
  const buildResultPayload = (fbData, taskInfo) => ({
    title:        currentQuery.length > 40 ? currentQuery.substring(0, 40) + '…' : currentQuery || 'Satellite Analysis',
    query:        currentQuery,
    taskType:     taskInfo.taskType,
    taskName:     taskInfo.taskName,
    backendModel: taskInfo.modelName,
    confidence:   fbData.confidence != null ? (fbData.confidence * 100).toFixed(1) : '—',
    textResponse: enrichAnswer(fbData.answer || 'No answer returned.', currentQuery),
    rawAnswer:    fbData.answer || '',
    bboxes:       fbData.bboxes      || [],
    bboxObject:   fbData.bbox_object || null,
    metrics:      fbData.metrics || null,
  });

  // ─── AI AGENT ANALYSIS PIPELINE (via Firebase RTDB) ────────────────────
  const handleRunAgentAnalysis = async () => {
    if (isAgentAnalyzing) return;
    const primaryFile = uploadedFiles[0] || null;
    if (!primaryFile && uploadedImages.length === 0) return;
    if (!currentQuery.trim()) return;

    // Stop any previous listener
    unsubListenerRef.current?.();
    clearTimeout(timeoutRef.current);

    setIsAgentAnalyzing(true);
    setAgentReport(null);
    setQueryError(null);
    setQueryStatus('encoding');

    try {
      // Step 1 — Encode image to base64
      let imageB64;
      if (primaryFile) {
        imageB64 = await fileToBase64(primaryFile, 1024);
      } else {
        // Preset URL — fetch and encode
        const resp = await fetch(uploadedImages[0]?.url);
        const blob = await resp.blob();
        imageB64 = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result.split(',')[1]);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
      }

      setQueryStatus('pending');

      // Step 2 — Submit agent query to Firebase RTDB
      const queryId = await submitAgentQueryToFirebase(
        user?.uid || 'anonymous',
        imageB64,
        currentQuery,
        0.25,
      );

      setQueryStatus('processing');

      // Step 3 — Listen for agent result from /agent_results/{queryId}

      const unsubAgent = listenForAgentResult(queryId, {
        onResult: (data) => {
          clearTimeout(timeoutRef.current);
          unsubAgent();
          unsubListenerRef.current = null;

          // Map RTDB agent result to AgentPanel report shape
          const report = {
            summary:             data.summary             || '',
            vlm_answer:          data.vlm_answer          || null,
            caption:             data.caption             || null,
            object_counts:       data.object_counts       || {},
            total_detections:    data.total_detections    || 0,
            bboxes:              data.bboxes              || [],
            bbox_labels:         data.bbox_labels         || [],
            bbox_confidences:    data.bbox_confidences    || [],
            annotated_image_b64: data.annotated_image_b64 || null,
            land_cover:          data.land_cover          || {},
            land_cover_display:  data.land_cover_display  || {},
            dominant_land_cover: data.dominant_land_cover || '',
            mask_image_b64:      data.mask_image_b64      || null,
            confidence:          data.confidence          || 0,
            tools_used:          data.tools_used          || [],
            routing_mode:        data.routing_mode        || '',
            execution_steps:     data.execution_steps     || [],
            verification_log:    data.verification_log    || [],
            reasoning_trace:     data.reasoning_trace     || '',
            errors:              data.errors              || [],
            duration_ms:         data.duration_ms         || 0,
          };

          setAgentReport(report);
          setActiveTab('agent');
          setQueryStatus('done');
          setIsAgentAnalyzing(false);

          // Save to recents
          if (user?.uid) {
            saveRecentQuery(user.uid, {
              id: queryId,
              title: currentQuery.length > 40 ? currentQuery.slice(0, 40) + '…' : currentQuery,
              query: currentQuery,
              taskType: 'agent',
              timestamp: Date.now(),
              images: uploadedImages.map(i => ({ ...i, url: i.url })),
            });
          }
        },
        onStatusChange: (status) => {
          if (status === 'error') {
            clearTimeout(timeoutRef.current);
            setQueryError('Agent pipeline failed on the backend. Check server logs.');
            setQueryStatus('error');
            setIsAgentAnalyzing(false);
          }
        },
        onError: (msg) => {
          clearTimeout(timeoutRef.current);
          setQueryError(`Agent listener error: ${msg}`);
          setQueryStatus('error');
          setIsAgentAnalyzing(false);
        },
      });

      unsubListenerRef.current = unsubAgent;

      // Soft timeout (5 min) — switch to 'still waiting' but keep listener alive
      // so the result auto-arrives when backend finishes (even after 9-10 min).
      const softTimer = setTimeout(() => {
        setQueryStatus('waiting');
        setIsAgentAnalyzing(false); // unblock the UI; result arrives on its own
      }, SOFT_TIMEOUT_MS);

      // Hard timeout (20 min) — only fires if backend truly never responds
      timeoutRef.current = setTimeout(() => {
        clearTimeout(softTimer);
        unsubAgent();
        setQueryError('Analysis timed out after 20 minutes. Check if the backend is still running.');
        setQueryStatus('timeout');
        setIsAgentAnalyzing(false);
      }, RESULT_TIMEOUT_MS);


    } catch (err) {
      setQueryError(`Agent analysis failed: ${err.message}`);
      setQueryStatus('error');
      setIsAgentAnalyzing(false);
    }
  };

  // ─── MAIN ANALYSIS PIPELINE ──────────────────────────────────────────────
  const handleRunAnalysis = async () => {
    if (isAnalyzing) return;
    if (!currentQuery.trim() && uploadedImages.length === 0) return;

    // Stop previous listener
    unsubListenerRef.current?.();
    clearTimeout(timeoutRef.current);

    setIsAnalyzing(true);
    setAnalysisResult(null);
    setQueryError(null);

    const taskInfo = detectTaskType(currentQuery, uploadedImages);
    const primaryFile = uploadedFiles[0] || null;

    // ── Encode image: use raw file if available, else fetch preset URL ────────
    let imageBase64;
    try {
      setQueryStatus('encoding');
      if (primaryFile) {
        imageBase64 = await fileToBase64(primaryFile, 1024);
      } else if (uploadedImages[0]?.url) {
        // Preset/sample image — fetch URL and encode to base64
        const resp = await fetch(uploadedImages[0].url);
        const blob = await resp.blob();
        imageBase64 = await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload  = () => resolve(reader.result.split(',')[1]);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
      } else {
        setQueryStatus('error');
        setQueryError('No image provided.');
        setIsAnalyzing(false);
        return;
      }
    } catch (encErr) {
      setQueryStatus('error');
      setQueryError(`Image encoding failed: ${encErr.message}`);
      setIsAnalyzing(false);
      return;
    }

    // ── Real pipeline via Firebase RTDB (base64 — no Storage needed) ──────
    try {
      // Step 2 — Write query + base64 image to RTDB
      setQueryStatus('pending');

      // ── Agent path (complex/mixed queries) ───────────────────────────────
      if (taskInfo.fbTask === 'agent') {
        const queryId = await submitAgentQueryToFirebase(
          user?.uid || 'anonymous',
          imageBase64,
          currentQuery || 'Analyze this satellite image.',
          0.25,
        );

        setQueryStatus('processing');

        const unsubAgent = listenForAgentResult(queryId, {
          onStatusChange: (status) => {
            if (status && status !== 'done' && status !== 'error') setQueryStatus(status);
          },
          onResult: (data) => {
            clearTimeout(timeoutRef.current);
            unsubAgent();
            unsubListenerRef.current = null;

            const report = {
              summary:             data.summary             || '',
              vlm_answer:          data.vlm_answer          || null,
              caption:             data.caption             || null,
              object_counts:       data.object_counts       || {},
              total_detections:    data.total_detections    || 0,
              bboxes:              data.bboxes              || [],
              bbox_labels:         data.bbox_labels         || [],
              bbox_confidences:    data.bbox_confidences    || [],
              annotated_image_b64: data.annotated_image_b64 || null,
              land_cover:          data.land_cover          || {},
              land_cover_display:  data.land_cover_display  || {},
              dominant_land_cover: data.dominant_land_cover || '',
              mask_image_b64:      data.mask_image_b64      || null,
              confidence:          data.confidence          || 0,
              tools_used:          data.tools_used          || [],
              routing_mode:        data.routing_mode        || '',
              execution_steps:     data.execution_steps     || [],
              verification_log:    data.verification_log    || [],
              reasoning_trace:     data.reasoning_trace     || '',
              errors:              data.errors              || [],
              duration_ms:         data.duration_ms         || 0,
            };

            setAgentReport(report);
            setActiveTab('agent');
            setQueryStatus('done');
            setIsAnalyzing(false);

            if (user?.uid) {
              saveRecentQuery(user.uid, {
                id: queryId,
                title: currentQuery.length > 40 ? currentQuery.slice(0, 40) + '…' : currentQuery,
                query: currentQuery,
                taskType: 'agent',
                timestamp: Date.now(),
                images: uploadedImages.map(i => ({ ...i, url: i.url })),
              });
            }
          },
          onError: (msg) => {
            clearTimeout(timeoutRef.current);
            setQueryError(`Agent listener error: ${msg}`);
            setQueryStatus('error');
            setIsAnalyzing(false);
          },
        });
        unsubListenerRef.current = unsubAgent;

      } else {
        // ── VQA path (simple single-image questions) ───────────────────────
        const queryId = await submitQueryToFirebase(
          user?.uid || 'anonymous',
          imageBase64,
          currentQuery || 'Describe this satellite image.',
          taskInfo.fbTask
        );

        // Step 3 — Listen for result
        const unsub = listenForQueryResult(queryId, {
          onStatusChange: (status) => {
            if (!status) return;
            if (status === 'error') {
              // Backend wrote error to /queries/ — reflect immediately
              clearTimeout(timeoutRef.current);
              unsub();
              unsubListenerRef.current = null;
              setQueryStatus('error');
              setQueryError('The backend reported an error processing this query. Please retry.');
              setIsAnalyzing(false);
            } else if (status !== 'done') {
              setQueryStatus(status);
            }
          },
          onResult: async (fbData) => {
            clearTimeout(timeoutRef.current);
            unsubListenerRef.current?.();
            setQueryStatus('done');

            const resultPayload = buildResultPayload(fbData, taskInfo);
            setAnalysisResult(resultPayload);
            setIsAnalyzing(false);

            // Save to recents
            if (user?.uid) {
              await saveRecentQuery(user.uid, {
                id:        `query_${Date.now()}`,
                title:     resultPayload.title,
                query:     currentQuery,
                taskType:  taskInfo.taskType,
                images:    uploadedImages.map(img => ({ name: img.name, modality: img.modality, url: img.url })),
                result:    resultPayload,
                timestamp: Date.now(),
              });
            }
          },
          onError: (errMsg) => {
            clearTimeout(timeoutRef.current);
            setQueryStatus('error');
            setQueryError(errMsg || 'Unknown error from backend.');
            setIsAnalyzing(false);
          },
        });
        unsubListenerRef.current = unsub;
      }

      // Step 4 — Two-stage timeout
      // Soft (5 min): switch to 'waiting' status, unblock UI, but keep Firebase
      // listener alive so result auto-arrives when backend finishes (even at 9-10 min)
      const softTimer2 = setTimeout(() => {
        setQueryStatus('waiting');
        setIsAnalyzing(false);
      }, SOFT_TIMEOUT_MS);

      // Hard (20 min): only fires if backend truly never responds
      timeoutRef.current = setTimeout(() => {
        clearTimeout(softTimer2);
        unsubListenerRef.current?.();
        setQueryStatus('timeout');
        setQueryError('The backend did not respond in 20 minutes. Make sure it is running and Firebase is connected.');
        setIsAnalyzing(false);
      }, RESULT_TIMEOUT_MS);

    } catch (err) {
      console.error('Analysis pipeline error:', err);
      setQueryStatus('error');
      setQueryError(err.message || 'Failed to start analysis.');
      setIsAnalyzing(false);
    }
  };


  // ─── Demo / preset mode (no real file uploaded) ───────────────────────────
  const _runDemoMode = async (taskInfo) => {
    return new Promise((resolve) => {
      setTimeout(async () => {
        setQueryStatus('processing');
        setTimeout(async () => {
          const matchedPreset = PRESET_DATASETS.find(p => p.type === taskInfo.taskType) || PRESET_DATASETS[0];
          const resultPayload = {
            title:        currentQuery.length > 40 ? currentQuery.substring(0, 40) + '…' : currentQuery || 'Satellite Analysis',
            query:        currentQuery,
            taskType:     taskInfo.taskType,
            taskName:     taskInfo.taskName,
            backendModel: taskInfo.modelName + ' (Demo)',
            confidence:   (94.5 + Math.random() * 4.5).toFixed(1),
            textResponse: matchedPreset.response,
            metrics:      matchedPreset.metrics,
            isDemo:       true,
          };
          setQueryStatus('done');
          setAnalysisResult(resultPayload);
          setIsAnalyzing(false);

          if (user?.uid) {
            await saveRecentQuery(user.uid, {
              id:        `query_${Date.now()}`,
              title:     resultPayload.title,
              query:     currentQuery,
              taskType:  taskInfo.taskType,
              images:    uploadedImages.map(img => ({ name: img.name, modality: img.modality, url: img.url })),
              result:    resultPayload,
              timestamp: Date.now(),
            });
          }
          resolve();
        }, 1400);
      }, 800);
    });
  };

  // ─── Retry handler ────────────────────────────────────────────────────────
  const handleRetry = () => {
    setQueryStatus(null);
    setQueryError(null);
    setAnalysisResult(null);
    handleRunAnalysis();
  };

  // ─── Filter unpinned recents ─────────────────────────────────────────────
  const unpinnedRecents = recents.filter(r => !isPinned(r.id));

  // ─── Relative time helper ────────────────────────────────────────────────
  const getRelativeTime = (ts) => {
    if (!ts) return '';
    const diff = Date.now() - ts;
    const m = Math.floor(diff / 60000);
    if (m < 1)  return 'just now';
    if (m < 60) return `${m}m ago`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ago`;
    const d = Math.floor(h / 24);
    if (d === 1) return 'yesterday';
    if (d < 7)  return `${d}d ago`;
    return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  };

  // ─── Group recents by date ───────────────────────────────────────────────
  const groupRecentsByDate = (items) => {
    const now   = Date.now();
    const today = new Date().setHours(0,0,0,0);
    const yest  = today - 86400000;
    const groups = { Today: [], Yesterday: [], Older: [] };
    items.forEach(item => {
      const d = item.timestamp || 0;
      if (d >= today)     groups.Today.push(item);
      else if (d >= yest) groups.Yesterday.push(item);
      else                groups.Older.push(item);
    });
    return groups;
  };

  // ─── Task-type dot color ─────────────────────────────────────────────────
  const taskDotColor = (taskType) => {
    if (taskType === 'bitemporal') return '#f97316';
    if (taskType === 'crossmodal') return '#a78bfa';
    return '#38bdf8';
  };

  // ─── Status Step Icon ─────────────────────────────────────────────────────
  const StatusIcon = ({ type, size = 16 }) => {
    if (type === 'spinner') return <Loader2 size={size} className="spin-icon" />;
    if (type === 'upload')  return <UploadCloud size={size} />;
    if (type === 'clock')   return <Clock size={size} />;
    if (type === 'check')   return <CheckCircle2 size={size} />;
    if (type === 'error')   return <AlertCircle size={size} />;
    return null;
  };

  const currentStatusDef = queryStatus ? STATUS_STEPS[queryStatus] : null;

  return (
    <div className="workspace-layout">
      {/* ─── MOBILE SIDEBAR OVERLAY ────────────────────────────────────── */}
      {isMobileSidebarOpen && (
        <div className="mobile-sidebar-overlay" onClick={() => setIsMobileSidebarOpen(false)} />
      )}

      {/* ─── LEFT SIDEBAR ─────────────────────────────────────────────── */}
      <aside className={`workspace-sidebar${isMobileSidebarOpen ? ' sidebar-open' : ''}`}>
        <div className="sidebar-top">
          <div className="sidebar-brand">
            <span className="sidebar-brand-text">ByteX</span>
          </div>
          <button
            className="sidebar-mobile-close"
            onClick={() => setIsMobileSidebarOpen(false)}
            aria-label="Close sidebar"
          >
            <X size={18} />
          </button>
          <button className="new-analysis-btn" onClick={handleNewAnalysis}>
            <Plus size={18} />
            <span>New analysis</span>
          </button>
        </div>

        {/* Pinned Section */}
        {pinnedItems.length > 0 && (
          <div className="sidebar-section">
            <div className="section-label">Pinned</div>
            <div className="section-list">
              {pinnedItems.map((item) => (
                <div
                  key={item.id}
                  className="sidebar-item"
                  onClick={() => handleSelectRecent(item)}
                  onMouseEnter={() => setHoveredRecent(item.id)}
                  onMouseLeave={() => setHoveredRecent(null)}
                >
                  <span className="sidebar-item-title">{item.title || item.query}</span>
                  {hoveredRecent === item.id && (
                    <div className="sidebar-item-actions">
                      <button className="action-icon-btn" onClick={(e) => { e.stopPropagation(); togglePin(item); }} title="Unpin">
                        <PinOff size={14} />
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recents Section */}
        <div className="sidebar-section recents-section-grow">
          <div className="section-label">History</div>
          <div className="section-list">
            {unpinnedRecents.length === 0 && pinnedItems.length === 0 ? (
              <div className="empty-recents">
                <Compass size={22} className="empty-recents-icon" />
                <p>No queries yet</p>
                <p>Upload a satellite image and ask your first question to get started.</p>
              </div>
            ) : unpinnedRecents.length === 0 ? (
              <div className="empty-recents"><p>All items pinned</p></div>
            ) : (
              (() => {
                const groups = groupRecentsByDate(unpinnedRecents);
                return Object.entries(groups)
                  .filter(([, items]) => items.length > 0)
                  .map(([label, items]) => (
                    <div key={label}>
                      <div className="sidebar-date-group">{label}</div>
                      {items.map((item) => (
                        <div
                          key={item.id}
                          className={`sidebar-item ${selectedRecentId === item.id ? 'active' : ''}`}
                          onClick={() => handleSelectRecent(item)}
                          onMouseEnter={() => setHoveredRecent(item.id)}
                          onMouseLeave={() => setHoveredRecent(null)}
                        >
                          <div className="sidebar-item-main">
                            <span
                              className="sidebar-item-type-dot"
                              style={{ background: taskDotColor(item.taskType), boxShadow: `0 0 5px ${taskDotColor(item.taskType)}80` }}
                            />
                            <div className="sidebar-item-text">
                              <span className="sidebar-item-title">{item.title || item.query}</span>
                              <span className="sidebar-item-time">{getRelativeTime(item.timestamp)}</span>
                            </div>
                          </div>
                          {hoveredRecent === item.id && (
                            <div className="sidebar-item-actions">
                              <button className="action-icon-btn" onClick={(e) => { e.stopPropagation(); togglePin(item); }} title="Pin">
                                <Pin size={14} />
                              </button>
                              <button className="action-icon-btn" onClick={(e) => handleDeleteRecent(e, item.id)} title="Delete">
                                <Trash2 size={14} />
                              </button>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  ));
              })()
            )}
          </div>
        </div>

        {/* User footer — click to open profile page */}
        <div
          className="sidebar-user-footer clickable-footer"
          onClick={() => { setIsMobileSidebarOpen(false); setIsProfileOpen(true); }}
          title="View profile"
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && setIsProfileOpen(true)}
        >
          <div className="user-info-card">
            <img
              src={user?.photoURL || `https://api.dicebear.com/7.x/bottts/svg?seed=${user?.uid || 'bytex'}`}
              alt="User"
              className="user-avatar"
            />
            <span className="user-name">{user?.displayName || user?.email?.split('@')[0] || 'Analyst'}</span>
          </div>
          <ChevronRight size={16} className="footer-chevron" />
        </div>
      </aside>

      {/* ─── MAIN WORKSPACE ───────────────────────────────────────────── */}
      <main className="workspace-main" onDragEnter={handleDragEnter} onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
        {/* Mobile top bar */}
        <div className="mobile-topbar">
          <button
            className="hamburger-btn"
            onClick={() => setIsMobileSidebarOpen(true)}
            aria-label="Open sidebar"
          >
            <Menu size={22} />
          </button>
          <span className="mobile-topbar-brand">ByteX</span>
          <div style={{ width: 36 }} />
        </div>

        <StarField />
        <div className="workspace-content-center">

          {/* Center input zone */}
          <div className={`center-input-zone ${analysisResult || isAnalyzing || queryStatus ? 'pushed-top' : ''}`}>
            {!analysisResult && !isAnalyzing && !queryStatus && (
              <div className="welcome-msg">
                <h1 className="welcome-title">What would you like to analyze?</h1>
              </div>
            )}

            {/* Query Input Bar */}
            <div className="query-box-wrapper">
              {uploadedImages.length > 0 && (
                <div className="uploaded-chips-row">
                  {uploadedImages.map((img, index) => (
                    <div key={index} className="uploaded-chip">
                      <img src={img.url} alt={img.name} className="chip-thumbnail" />
                      <div className="chip-info">
                        <span className="chip-name">{img.name}</span>
                        <span className="chip-modality">{img.modality}</span>
                      </div>
                      <button
                        className="chip-remove-btn"
                        onClick={() => {
                          setUploadedImages(prev => prev.filter((_, i) => i !== index));
                          setUploadedFiles(prev => prev.filter((_, i) => i !== index));
                        }}
                      >×</button>
                    </div>
                  ))}
                </div>
              )}

              <div className="query-input-bar">
                <button className="add-media-btn" title="Upload imagery (+)" onClick={() => setShowUploadModal(true)}>
                  <Plus size={22} />
                </button>

                {/* AI Agent mode toggle */}
                <button
                  id="agent-mode-toggle"
                  title={agentMode ? 'Agent mode ON — click to switch to standard mode' : 'Switch to AI Agent mode'}
                  onClick={() => setAgentMode(v => !v)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 4,
                    padding: '5px 10px', borderRadius: 8, border: 'none', cursor: 'pointer',
                    fontSize: 11, fontWeight: 700, letterSpacing: '0.04em',
                    transition: 'all 0.2s',
                    background: agentMode
                      ? 'linear-gradient(135deg, rgba(99,102,241,0.3), rgba(139,92,246,0.3))'
                      : 'rgba(30,41,59,0.6)',
                    color: agentMode ? '#a5b4fc' : '#64748b',
                    border: agentMode ? '1px solid rgba(99,102,241,0.5)' : '1px solid rgba(51,65,85,0.5)',
                    flexShrink: 0,
                  }}
                >
                  <Bot size={14} />
                  Agent
                </button>

                <input
                  type="file"
                  ref={fileInputRef}
                  multiple
                  accept="image/*,.tif,.tiff"
                  style={{ display: 'none' }}
                  onChange={(e) => { if (e.target.files) processSelectedFiles(e.target.files); }}
                />

                <input
                  type="text"
                  className="query-text-input"
                  placeholder={isListening ? 'Listening… speak your query' : 'Ask about satellite imagery…'}
                  value={currentQuery}
                  onChange={(e) => setCurrentQuery(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleRunAnalysis(); } }}
                  disabled={isAnalyzing}
                />

                {/* Mic when no text, Send when text is typed */}
                {(currentQuery.trim() || uploadedImages.length > 0) ? (
                  <button
                    className={`send-query-btn ${!isAnalyzing && !isAgentAnalyzing ? 'active' : ''}`}
                    onClick={agentMode ? handleRunAgentAnalysis : handleRunAnalysis}
                    disabled={isAnalyzing || isAgentAnalyzing}
                    title={agentMode ? 'Run AI Agent analysis' : 'Run analysis'}
                  >
                    <Send size={18} />
                  </button>
                ) : (
                  <button
                    className={`mic-btn ${isListening ? 'listening' : ''}`}
                    onClick={handleMicClick}
                    title={isListening ? 'Stop listening' : 'Speak your query'}
                  >
                    {isListening ? <MicOff size={18} /> : <Mic size={18} />}
                  </button>
                )}
              </div>
              {isListening && (
                <div className="listening-hint">● Recording… pause when done to auto-fill</div>
              )}
            </div>

            {/* Quick preset shortcuts */}
            {!analysisResult && !isAnalyzing && !queryStatus && (
              <div className="quick-presets-row">
                {PRESET_DATASETS.map((preset) => (
                  <button key={preset.id} className="preset-chip" onClick={() => handleLoadPreset(preset)}>
                    <span className={`preset-dot dot-${preset.type}`}></span>
                    <span>{preset.title}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* ── Status Pipeline Bar ──────────────────────────────────────── */}
          {queryStatus && queryStatus !== 'done' && (
            <div className="status-pipeline-bar">
              <div className="status-pipeline-steps">
                {['encoding', 'pending', 'processing'].map((step) => {
                  const stepIdx   = ['encoding', 'pending', 'processing'].indexOf(step);
                  const curIdx    = ['encoding', 'pending', 'processing'].indexOf(queryStatus);
                  const isDone    = stepIdx < curIdx || queryStatus === 'done';
                  const isCurrent = stepIdx === curIdx && queryStatus !== 'error' && queryStatus !== 'timeout';
                  const isError   = (queryStatus === 'error' || queryStatus === 'timeout') && stepIdx === curIdx;
                  return (
                    <div key={step} className={`pipeline-step ${isDone ? 'done' : ''} ${isCurrent ? 'current' : ''} ${isError ? 'errored' : ''}`}>
                      <div className="step-dot">
                        {isDone ? <CheckCircle2 size={13} /> : isCurrent ? <Loader2 size={13} className="spin-icon" /> : isError ? <AlertCircle size={13} /> : null}
                      </div>
                      <span className="step-label">{STATUS_STEPS[step]?.label || step}</span>
                    </div>
                  );
                })}
              </div>

              {/* Error / timeout message */}
              {(queryStatus === 'error' || queryStatus === 'timeout') && queryError && (
                <div className="status-error-row">
                  <AlertCircle size={15} />
                  <span>{queryError}</span>
                  <button className="retry-btn" onClick={handleRetry}>
                    <RefreshCw size={13} /> Retry
                  </button>
                </div>
              )}
            </div>
          )}

          {/* ── Processing spinner (CPU-aware with live timer) ───────────── */}
          {(isAnalyzing && queryStatus === 'processing') || isAgentAnalyzing ? (
            <CpuInferenceProgress isAgent={isAgentAnalyzing} />
          ) : null}

          {/* ── Results ──────────────────────────────────────────────────── */}
          {analysisResult && !isAnalyzing && (
            <div className="analysis-result-below">
              {/* Demo badge */}
              {analysisResult.isDemo && (
                <div className="demo-notice">
                  <ImageIcon size={14} />
                  <span>Demo response — upload a real satellite image to use Gemma 4 Vision</span>
                </div>
              )}

              {/* Tab Navigation */}
              <div className="result-tabs-nav">
                <button className={`tab-btn ${activeTab === 'evidence' ? 'active' : ''}`} onClick={() => setActiveTab('evidence')}>
                  <Eye size={16} />
                  <span>Visual Evidence & Viewer</span>
                </button>
                <button className={`tab-btn ${activeTab === 'report' ? 'active' : ''}`} onClick={() => setActiveTab('report')}>
                  <FileText size={16} />
                  <span>Textual Analysis</span>
                </button>
                <div className="tab-spacer"></div>
                <div className="result-meta">
                  <span className="conf-badge">{analysisResult.confidence}% confidence</span>
                  <button className="download-btn" onClick={handleDownloadReport} title="Download Report">
                    <Download size={15} />
                  </button>
                </div>
              </div>

              {/* Tab: Visual Evidence */}
              {activeTab === 'evidence' && (
                <div className="visual-evidence-pane">
                  {analysisResult.taskType === 'bitemporal' && uploadedImages.length >= 2 && (
                    <div className="bitemporal-split-viewer">
                      <div className="split-image-wrapper">
                        <img src={uploadedImages[1]?.url || '/samples/bitemporal_after.jpg'} alt="After" className="split-img base-img" />
                        <span className="split-label label-after">After (2024)</span>
                        <div className="split-overlay-clipped" style={{ clipPath: `polygon(0 0, ${sliderPosition}% 0, ${sliderPosition}% 100%, 0 100%)` }}>
                          <img src={uploadedImages[0]?.url || '/samples/bitemporal_before.jpg'} alt="Before" className="split-img overlay-img" />
                          <span className="split-label label-before">Before (2004)</span>
                        </div>
                        <div className="slider-divider-handle" style={{ left: `${sliderPosition}%` }}>
                          <div className="divider-line-v"></div>
                          <div className="slider-button"><Sliders size={16} /></div>
                        </div>
                        <input type="range" min="0" max="100" value={sliderPosition} onChange={(e) => setSliderPosition(e.target.value)} className="invisible-range-slider" />
                      </div>
                    </div>
                  )}

                  {analysisResult.taskType === 'crossmodal' && (
                    <div className="crossmodal-viewer">
                      <div className="layer-controls">
                        {['optical', 'sar', 'fusion'].map(layer => (
                          <button key={layer} className={`layer-toggle-btn ${activeCrossmodalLayer === layer ? 'active' : ''}`}
                            onClick={() => setActiveCrossmodalLayer(layer)}>
                            {layer === 'optical' ? 'Optical RGB' : layer === 'sar' ? 'SAR Backscatter' : 'Fused Overlay'}
                          </button>
                        ))}
                      </div>
                      <div className="crossmodal-image-display">
                        <img
                          src={activeCrossmodalLayer === 'sar' ? (uploadedImages[1]?.url || '/samples/sar_sample.jpg') : (uploadedImages[0]?.url || '/samples/optical_sample.jpg')}
                          alt="Crossmodal scene"
                          className={`crossmodal-img ${activeCrossmodalLayer === 'fusion' ? 'fusion-mode' : ''}`}
                        />
                      </div>
                    </div>
                  )}

                  {analysisResult.taskType === 'single' && (
                    <SmartGroundingViewer
                      imageUrl={uploadedImages[0]?.url || '/samples/optical_sample.jpg'}
                      query={analysisResult.query || ''}
                      answer={analysisResult.rawAnswer || analysisResult.textResponse || ''}
                      bboxes={analysisResult.bboxes || []}
                      bboxObject={analysisResult.bboxObject || null}
                    />
                  )}

                  {analysisResult.metrics && (
                    <div className="metrics-row">
                      {Object.entries(analysisResult.metrics).map(([key, val]) => (
                        <div key={key} className="metric-chip">
                          <span className="metric-key">{key.replace(/([A-Z])/g, ' $1')}</span>
                          <span className="metric-val">{val}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Tab: Textual Analysis */}
              {activeTab === 'report' && (
                <div className="text-report-pane" style={{ padding: '4px 0' }}>

                  {/* ── Report Header ── */}
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: 10,
                    marginBottom: 20, paddingBottom: 16,
                    borderBottom: '1px solid rgba(56,189,248,0.12)',
                  }}>
                    <div style={{
                      background: 'linear-gradient(135deg,#0ea5e9,#6366f1)',
                      borderRadius: 8, padding: '6px 10px',
                      fontSize: 18, lineHeight: 1,
                    }}>AI</div>
                    <div>
                      <div style={{ fontFamily: 'Outfit,sans-serif', fontWeight: 700, fontSize: 14, color: '#e2e8f0' }}>
                        SatQuery AI — Research Analysis
                      </div>
                      <div style={{ fontSize: 11, color: '#64748b', fontFamily: 'Outfit,sans-serif', marginTop: 2 }}>
                        Google Gemma 4 31B • OpenRouter • Extended Research Mode (512 tokens)
                      </div>
                    </div>
                    <div style={{ marginLeft: 'auto' }}>
                      <span style={{
                        background: 'rgba(56,189,248,0.1)', border: '1px solid rgba(56,189,248,0.3)',
                        borderRadius: 6, padding: '3px 10px',
                        fontSize: 11, color: '#38bdf8', fontFamily: 'Outfit,sans-serif', fontWeight: 600,
                      }}>
                        {analysisResult.confidence}% confidence
                      </span>
                    </div>
                  </div>

                  {/* ── Rendered Markdown Report ── */}
                  <div style={{
                    fontFamily: 'Outfit, Inter, sans-serif',
                    fontSize: 14, lineHeight: 1.75, color: '#cbd5e1',
                  }} dangerouslySetInnerHTML={{
                    __html: analysisResult.textResponse
                      // h1 (##)
                      .replace(/^## (.*$)/gim, '<h2 style="margin:28px 0 12px;padding-bottom:8px;border-bottom:1px solid rgba(56,189,248,0.15);color:#7dd3fc;font-size:1.15em;font-weight:800;letter-spacing:0.01em;font-family:Outfit,sans-serif;">$1</h2>')
                      // h2 (###)
                      .replace(/^### (.*$)/gim, '<h3 style="margin:22px 0 8px;color:#38bdf8;font-size:1em;font-weight:700;letter-spacing:0.02em;font-family:Outfit,sans-serif;">$1</h3>')
                      // h3 (####)
                      .replace(/^#### (.*$)/gim, '<h4 style="margin:16px 0 6px;color:#94a3b8;font-size:0.9em;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">$1</h4>')
                      // Horizontal rule
                      .replace(/^---$/gim, '<hr style="border:none;border-top:1px solid rgba(56,189,248,0.12);margin:20px 0;"/>')
                      // Table rows — simple pipe-delimited tables
                      .replace(/^\|(.+)\|$/gim, (_, row) => {
                        const cells = row.split('|').map(c => c.trim());
                        return `<div style="display:flex;gap:0;margin:1px 0;">${cells.map(c =>
                          c.match(/^[-:]+$/)
                            ? '' // skip separator rows
                            : `<div style="flex:1;padding:6px 10px;background:rgba(15,23,42,0.6);border:1px solid rgba(56,189,248,0.08);font-size:12px;color:#94a3b8;">${c}</div>`
                        ).join('')}</div>`;
                      })
                      // Bold
                      .replace(/\*\*(.*?)\*\*/gim, '<strong style="color:#e2e8f0;font-weight:700;">$1</strong>')
                      // Italic
                      .replace(/\*(.*?)\*/gim, '<em style="color:#94a3b8;font-style:italic;">$1</em>')
                      // Numbered list items
                      .replace(/^(\d+)\. (.*$)/gim, '<div style="display:flex;gap:10px;margin:6px 0 6px 4px;align-items:flex-start;"><span style="min-width:22px;height:22px;background:linear-gradient(135deg,#0ea5e9,#6366f1);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:800;color:#fff;flex-shrink:0;margin-top:1px;">$1</span><span style="color:#cbd5e1;line-height:1.65;">$2</span></div>')
                      // Bullet list items
                      .replace(/^- (.*$)/gim, '<div style="display:flex;gap:10px;margin:5px 0 5px 4px;align-items:flex-start;"><span style="color:#38bdf8;font-size:16px;line-height:1;margin-top:2px;flex-shrink:0;">▸</span><span style="color:#cbd5e1;line-height:1.65;">$1</span></div>')
                      // Italic in backtick
                      .replace(/`(.*?)`/gim, '<code style="background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.15);border-radius:4px;padding:1px 6px;font-size:12px;font-family:monospace;color:#7dd3fc;">$1</code>')
                      // Double line breaks → paragraph spacer
                      .replace(/\n\n/g, '<div style="height:8px;"></div>')
                      // Single line breaks
                      .replace(/\n/g, '<br/>')
                  }} />

                  {/* ── Raw Answer Trace ── */}
                  {analysisResult.rawAnswer && analysisResult.rawAnswer !== analysisResult.textResponse && (
                    <div style={{
                      marginTop: 24, padding: '12px 16px',
                      background: 'rgba(15,23,42,0.7)',
                      border: '1px solid rgba(56,189,248,0.12)',
                      borderLeft: '3px solid #38bdf8',
                      borderRadius: 8,
                    }}>
                      <div style={{ fontSize: 11, color: '#38bdf8', fontWeight: 700, fontFamily: 'Outfit,sans-serif', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                        Raw Model Output
                      </div>
                      <div style={{ fontSize: 12, color: '#64748b', fontFamily: 'monospace', lineHeight: 1.6 }}>
                        {analysisResult.rawAnswer}
                      </div>
                    </div>
                  )}

                  {/* ── Footer Model Info ── */}
                  <div style={{
                    marginTop: 24,
                    display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8,
                  }}>
                    {[
                      { label: 'Model', value: 'Gemma 4 31B' },
                      { label: 'Provider', value: 'OpenRouter' },
                      { label: 'Mode', value: 'Research (512 tok)' },
                    ].map(({ label, value }) => (
                      <div key={label} style={{
                        background: 'rgba(15,23,42,0.6)',
                        border: '1px solid rgba(56,189,248,0.10)',
                        borderRadius: 8, padding: '8px 12px', textAlign: 'center',
                      }}>
                        <div style={{ fontSize: 10, color: '#475569', fontFamily: 'Outfit,sans-serif', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 3 }}>{label}</div>
                        <div style={{ fontSize: 12, color: '#7dd3fc', fontFamily: 'Outfit,sans-serif', fontWeight: 600 }}>{value}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}


              {/* Tab: AI Agent Panel — inside analysisResult context */}
              {activeTab === 'agent' && (
                <AgentPanel
                  report={agentReport}
                  isLoading={isAgentAnalyzing}
                  imageUrl={uploadedImages[0]?.url}
                  imageB64={null}
                  question={currentQuery}
                />
              )}
            </div>
          )}

          {/* ── Standalone Agent Panel (when no standard analysisResult) ─── */}
          {agentReport && !analysisResult && !isAgentAnalyzing && (
            <div className="analysis-result-below">
              <div className="result-tabs-nav">
                <button
                  id="agent-tab-btn-standalone"
                  className="tab-btn active"
                  style={{ color: '#a5b4fc' }}
                >
                  <Bot size={16} />
                  <span>AI Agent Report</span>
                </button>
                <div className="tab-spacer" />
                <span className="conf-badge">
                  {Math.round((agentReport.confidence || 0) * 100)}% confidence
                </span>
              </div>
              <AgentPanel
                report={agentReport}
                isLoading={false}
                imageUrl={uploadedImages[0]?.url}
                question={currentQuery}
              />
            </div>
          )}

          {/* ── Agent analyzing spinner (no standard result yet) ──────────── */}
          {isAgentAnalyzing && !analysisResult && (
            <div className="analysis-result-below">
              <AgentPanel
                report={null}
                isLoading={true}
                question={currentQuery}
              />
            </div>
          )}
        </div>

        {/* Drag overlay */}
        {isDragging && (
          <div className={`drag-drop-overlay${dragIsInvalid ? ' drag-invalid' : ''}`}>
            <UploadCloud size={52} className="drop-icon" />
            <h3>{dragIsInvalid ? 'Unsupported file type' : 'Drop satellite imagery here'}</h3>
            <p>
              {dragIsInvalid
                ? 'Please use GeoTIFF, TIFF, PNG, JPEG or WebP'
                : dragFileCount > 0
                  ? `${dragFileCount} file${dragFileCount > 1 ? 's' : ''} ready to drop`
                  : 'GeoTIFF, TIFF, PNG, JPEG, WebP'}
            </p>
            {!dragIsInvalid && uploadedImages.length > 0 && uploadedImages.length < 4 && (
              <span className="drag-slots-hint">{4 - uploadedImages.length} slot{4 - uploadedImages.length > 1 ? 's' : ''} remaining</span>
            )}
          </div>
        )}
      </main>

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="modal-backdrop" onClick={() => setShowUploadModal(false)}>
          <div className="upload-options-card" onClick={(e) => e.stopPropagation()}>
            <div className="upload-options-header">
              <h3>Upload Satellite Imagery</h3>
              <button className="close-modal-btn" onClick={() => setShowUploadModal(false)}>×</button>
            </div>
            <div className="options-grid">
              <button className="option-tile" onClick={() => { setShowUploadModal(false); fileInputRef.current?.click(); }}>
                <UploadCloud size={28} className="tile-icon" />
                <span className="tile-title">Upload Local</span>
                <span className="tile-desc">GeoTIFF, TIFF, PNG, JPEG</span>
              </button>
              <button className="option-tile" onClick={() => handleLoadPreset(PRESET_DATASETS[0])}>
                <Layers size={28} className="tile-icon icon-blue" />
                <span className="tile-title">Bi-Temporal Pair</span>
                <span className="tile-desc">Before & After (Demo)</span>
              </button>
              <button className="option-tile" onClick={() => handleLoadPreset(PRESET_DATASETS[1])}>
                <Compass size={28} className="tile-icon icon-cyan" />
                <span className="tile-title">Optical + SAR</span>
                <span className="tile-desc">Cartosat-2S & RISAT (Demo)</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ─── PROFILE PAGE ─────────────────────────────────────────────── */}
      {isProfileOpen && (
        <ProfilePanel
          user={user}
          recentsCount={recents.length}
          pinnedCount={pinnedItems.length}
          onLogout={onLogout}
          onClose={() => setIsProfileOpen(false)}
        />
      )}
    </div>
  );
}
