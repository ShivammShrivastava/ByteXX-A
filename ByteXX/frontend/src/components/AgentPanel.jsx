/**
 * AgentPanel.jsx — AI Agent Results Panel
 * =========================================
 * Renders the structured output from the SatAgent pipeline:
 *   - Routing mode badge + reason
 *   - Live execution step indicators (spinner → checkmark)
 *   - Object detection count tiles + annotated image
 *   - Land cover horizontal bar chart
 *   - VLM natural language answer
 *   - Confidence gauge bar
 *   - Reasoning trace (collapsible)
 *   - Verification log notices
 */

import React, { useState } from 'react';
import {
  Bot, Zap, Brain, Layers, Eye, Target, CheckCircle2,
  AlertCircle, XCircle, ChevronRight, Clock, Info,
  BarChart3, Map, Shield
} from 'lucide-react';
import './AgentPanel.css';

// ─── Helpers ──────────────────────────────────────────────────────────────────

const LAND_COVER_COLOURS = {
  vegetation:   { cls: 'lc-vegetation',   dot: '#22c55e' },
  water:        { cls: 'lc-water',        dot: '#3b82f6' },
  urban:        { cls: 'lc-urban',        dot: '#94a3b8' },
  agricultural: { cls: 'lc-agricultural', dot: '#d97706' },
};

const ROUTE_MODE_BADGE = {
  cnn_only:  { cls: 'badge-mode-cnn',    label: '⚡ Fast CNN' },
  vlm_only:  { cls: 'badge-mode-vlm',    label: '🧠 VLM Only' },
  hybrid:    { cls: 'badge-mode-hybrid', label: '⚡🧠 Hybrid' },
  caption:   { cls: 'badge-mode-vlm',    label: '🧠 Caption' },
};

function confClass(c) {
  if (c >= 0.7) return 'conf-high';
  if (c >= 0.45) return 'conf-mid';
  return 'conf-low';
}

function pct(n) {
  return `${(n * 100).toFixed(0)}%`;
}

// ─── Sub-components ────────────────────────────────────────────────────────────

function StepRow({ step }) {
  const statusCls = step.status === 'done' ? 'step-done'
    : step.status === 'running' ? 'step-running'
    : step.status === 'error'   ? 'step-error'
    : '';

  const icon = step.status === 'running' ? (
    <div className="step-status-icon"><div className="step-spinner" /></div>
  ) : step.status === 'done' ? (
    <div className="step-status-icon step-check"><CheckCircle2 size={16} /></div>
  ) : step.status === 'error' ? (
    <div className="step-status-icon step-error-icon"><XCircle size={16} /></div>
  ) : (
    <div className="step-pending-dot" />
  );

  const conf = step.confidence;
  const confCls = conf != null ? confClass(conf) : '';

  return (
    <div className={`agent-step-row ${statusCls}`}>
      {icon}
      <div className="step-info">
        <div className="step-tool">{step.tool}</div>
        <div className="step-desc">{step.description}</div>
      </div>
      <div className="step-meta">
        {conf != null && (
          <span className={`step-conf ${confCls}`}>{pct(conf)}</span>
        )}
        {step.duration_ms > 0 && (
          <span className="step-duration">{step.duration_ms.toFixed(0)}ms</span>
        )}
      </div>
    </div>
  );
}


function LandCoverChart({ landCoverDisplay }) {
  const entries = Object.entries(landCoverDisplay || {});
  if (!entries.length) return null;

  // Map display name back to a key for styling
  const keyFor = (name) => {
    const n = name.toLowerCase();
    if (n.includes('veg') || n.includes('forest')) return 'vegetation';
    if (n.includes('water') || n.includes('river')) return 'water';
    if (n.includes('urban') || n.includes('built')) return 'urban';
    if (n.includes('agri') || n.includes('farm')) return 'agricultural';
    return 'urban';
  };

  return (
    <div className="land-cover-chart">
      {entries.map(([name, pctVal]) => {
        const key   = keyFor(name);
        const style = LAND_COVER_COLOURS[key] || LAND_COVER_COLOURS.urban;
        return (
          <div className="lc-row" key={name}>
            <div className="lc-label-row">
              <span className="lc-name">
                <span className="lc-dot" style={{ background: style.dot }} />
                {name}
              </span>
              <span className="lc-pct">{pctVal.toFixed(1)}%</span>
            </div>
            <div className="lc-bar-track">
              <div
                className={`lc-bar-fill ${style.cls}`}
                style={{ width: `${Math.min(pctVal, 100)}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}


function ObjectCountGrid({ objectCounts, totalDetections }) {
  const entries = Object.entries(objectCounts || {});

  if (!entries.length) {
    return <p className="no-detections">No objects detected above confidence threshold.</p>;
  }

  return (
    <div className="object-count-grid">
      {entries.map(([label, count]) => (
        <div className="object-count-chip" key={label}>
          <div>
            <div className="object-count-num">{count}</div>
            <div className="object-count-label">{label}</div>
          </div>
        </div>
      ))}
    </div>
  );
}


function AnnotatedImagePair({ originalUrl, originalB64, annotatedB64, maskB64 }) {
  // Build src strings
  const origSrc = originalB64
    ? (originalB64.startsWith('data:') ? originalB64 : `data:image/jpeg;base64,${originalB64}`)
    : originalUrl;
  const annotSrc = annotatedB64
    ? `data:image/png;base64,${annotatedB64}`
    : null;
  const maskSrc = maskB64
    ? `data:image/png;base64,${maskB64}`
    : null;

  const frames = [
    origSrc  ? { label: 'Original',          src: origSrc  } : null,
    annotSrc ? { label: 'Detection Boxes',    src: annotSrc } : null,
    maskSrc  ? { label: 'Land Cover Mask',    src: maskSrc  } : null,
  ].filter(Boolean);

  if (!frames.length) return null;

  return (
    <div className="agent-image-pair" style={{ gridTemplateColumns: `repeat(${Math.min(frames.length, 2)}, 1fr)` }}>
      {frames.map(f => (
        <div className="agent-image-container" key={f.label}>
          <span className="agent-image-sublabel">{f.label}</span>
          <div className="agent-image-frame">
            <img src={f.src} alt={f.label} />
          </div>
        </div>
      ))}
    </div>
  );
}


function ConfidenceBar({ confidence }) {
  const pctNum = Math.round((confidence || 0) * 100);
  const cls = confClass(confidence || 0);
  return (
    <div className="agent-confidence-row">
      <div className="conf-bar-track">
        <div
          className={`conf-bar-fill ${cls}`}
          style={{ width: `${pctNum}%` }}
        />
      </div>
      <span className="conf-pct-label">{pctNum}%</span>
    </div>
  );
}


function ReasoningTrace({ trace }) {
  const [open, setOpen] = useState(false);
  if (!trace) return null;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <button
        className={`agent-trace-toggle ${open ? 'open' : ''}`}
        onClick={() => setOpen(v => !v)}
        id="agent-trace-toggle-btn"
      >
        <ChevronRight size={12} />
        Reasoning Trace
      </button>
      {open && <pre className="agent-trace-content">{trace}</pre>}
    </div>
  );
}


function VerificationLog({ verificationLog }) {
  const triggered = (verificationLog || []).filter(v => v.triggered);
  if (!triggered.length) return null;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {triggered.map((v, i) => (
        <div
          key={i}
          className={`verification-notice ${v.final_confidence >= 0.45 ? 'verified-ok' : ''}`}
        >
          <Shield size={13} style={{ flexShrink: 0, marginTop: 1 }} />
          <span>
            <strong>Re-analysis triggered</strong> for <code>{v.original_tool}</code>:&nbsp;
            {v.note || `Confidence improved from ${(v.original_confidence * 100).toFixed(0)}% → ${(v.final_confidence * 100).toFixed(0)}%`}
          </span>
        </div>
      ))}
    </div>
  );
}

// ─── Main Component ────────────────────────────────────────────────────────────

/**
 * AgentPanel
 *
 * Props:
 *   report        — AgentAnalyzeResponse from backend (or null while loading)
 *   isLoading     — true while pipeline is running
 *   executingSteps — live steps array for streaming status (optional)
 *   imageUrl      — original image URL for display
 *   imageB64      — original image base64 (optional)
 *   question      — the user's query
 */
export default function AgentPanel({
  report,
  isLoading,
  executingSteps,
  imageUrl,
  imageB64,
  question,
}) {
  const steps = executingSteps || report?.execution_steps || [];
  const modeBadge = ROUTE_MODE_BADGE[report?.routing_mode] || ROUTE_MODE_BADGE.hybrid;

  // Tools used → badges
  const toolBadges = [];
  if (report?.tools_used?.some(t => t.includes('yolo') || t.includes('cnn_detect'))) {
    toolBadges.push({ cls: 'badge-yolo', label: 'YOLOv8' });
  }
  if (report?.tools_used?.some(t => t.includes('segment') || t.includes('unet') || t.includes('resnet'))) {
    toolBadges.push({ cls: 'badge-unet', label: 'ResNet/FCN' });
  }
  if (report?.tools_used?.some(t => t.includes('vlm') || t.includes('gemma') || t.includes('qwen'))) {
    toolBadges.push({ cls: 'badge-vlm', label: 'Gemma Vision' });
  }

  const hasDetections   = report && Object.keys(report.object_counts || {}).length > 0;
  const hasLandCover    = report && Object.keys(report.land_cover_display || {}).length > 0;
  const hasAnnotated    = report?.annotated_image_b64;
  const hasMask         = report?.mask_image_b64;
  const hasVlmAnswer    = report?.vlm_answer || report?.caption;
  const hasVerification = (report?.verification_log || []).some(v => v.triggered);

  return (
    <div className="agent-panel" id="agent-panel-root">

      {/* ── Header ──────────────────────────────────────────────── */}
      <div className="agent-panel-header">
        <div className="agent-panel-title">
          <Bot size={18} />
          AI Agent Report
        </div>

        <div className="agent-badges">
          {report && (
            <span className={`badge ${modeBadge.cls}`}>
              {modeBadge.label}
            </span>
          )}
          {toolBadges.map(b => (
            <span key={b.label} className={`badge ${b.cls}`}>{b.label}</span>
          ))}
          {report && (
            <span className="badge badge-conf">
              <span style={{ opacity: 0.6, fontSize: 9 }}>CONF</span>
              &nbsp;{Math.round((report.confidence || 0) * 100)}%
            </span>
          )}
          {report && (
            <span className="agent-duration-tag">
              <Clock size={10} />
              {report.duration_ms ? `${(report.duration_ms / 1000).toFixed(1)}s` : '—'}
            </span>
          )}
        </div>
      </div>

      {/* ── Routing Reason ──────────────────────────────────────── */}
      {report?.routing_mode && (
        <div className="agent-routing-reason">
          <Info size={13} />
          <span>
            <strong>Route:</strong> {
              ROUTE_MODE_BADGE[report.routing_mode]?.label || report.routing_mode
            }
            {report.execution_steps?.[0] && ` — ${report.execution_steps.length} step(s) executed`}
          </span>
        </div>
      )}

      {/* ── Errors ──────────────────────────────────────────────── */}
      {report?.errors?.length > 0 && (
        <div className="agent-error-banner">
          <AlertCircle size={15} />
          <span>{report.errors.join(' | ')}</span>
        </div>
      )}

      {/* ── Execution Steps ─────────────────────────────────────── */}
      {steps.length > 0 && (
        <div className="agent-steps-section">
          <div className="agent-steps-label">
            <Zap size={11} style={{ display: 'inline', marginRight: 4 }} />
            Execution Pipeline
          </div>
          <div className="agent-steps-grid">
            {steps.map((step, i) => (
              <StepRow key={step.step_index ?? i} step={step} />
            ))}
          </div>
        </div>
      )}

      {/* Loading placeholder steps */}
      {isLoading && steps.length === 0 && (
        <div className="agent-steps-section">
          <div className="agent-steps-label">Executing pipeline…</div>
          {[0, 1, 2].map(i => (
            <div className="agent-step-row step-running" key={i} style={{ opacity: 0.5 }}>
              <div className="step-spinner" />
              <div className="step-info">
                <div className="step-tool" style={{ color: '#475569' }}>Initializing…</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Results Grid ────────────────────────────────────────── */}
      {report && (
        <div className="agent-results-grid">

          {/* Object Detection Card */}
          {(hasDetections || report.tools_used?.some(t => t.includes('detect'))) && (
            <div className="agent-card">
              <div className="agent-card-label">
                <Target size={13} />
                Object Detection (YOLOv8)
              </div>
              <ObjectCountGrid
                objectCounts={report.object_counts}
                totalDetections={report.total_detections}
              />
              <div style={{ fontSize: 11, color: '#475569' }}>
                {report.total_detections} total detection{report.total_detections !== 1 ? 's' : ''}
              </div>
            </div>
          )}

          {/* Land Cover Card */}
          {hasLandCover && (
            <div className="agent-card">
              <div className="agent-card-label">
                <Map size={13} />
                Land Cover (ResNet/FCN)
              </div>
              <LandCoverChart landCoverDisplay={report.land_cover_display} />
              {report.dominant_land_cover && (
                <div style={{ fontSize: 11, color: '#64748b' }}>
                  Dominant:&nbsp;
                  <strong style={{ color: '#94a3b8' }}>
                    {report.land_cover_display
                      ? Object.keys(report.land_cover_display).find(k =>
                          k.toLowerCase().includes(report.dominant_land_cover)
                        ) || report.dominant_land_cover
                      : report.dominant_land_cover}
                  </strong>
                </div>
              )}
            </div>
          )}

          {/* Annotated Image(s) */}
          {(hasAnnotated || hasMask) && (
            <div className="agent-card full-width">
              <div className="agent-card-label">
                <Eye size={13} />
                Visual Analysis
              </div>
              <AnnotatedImagePair
                originalUrl={imageUrl}
                originalB64={imageB64}
                annotatedB64={report.annotated_image_b64}
                maskB64={report.mask_image_b64}
              />
            </div>
          )}

          {/* VLM Answer */}
          {hasVlmAnswer && (
            <div className="agent-card full-width">
              <div className="agent-card-label">
                <Brain size={13} />
                {report.caption ? 'Image Caption' : 'Scene Analysis'} (Gemma 4)
              </div>
              <div className="agent-summary-text"
                dangerouslySetInnerHTML={{
                  __html: (report.vlm_answer || report.caption || '')
                    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\n/g, '<br/>')
                }}
              />
            </div>
          )}

          {/* Summary */}
          {report.summary && report.summary !== 'Analysis complete.' && (
            <div className="agent-card full-width">
              <div className="agent-card-label">
                <BarChart3 size={13} />
                Structured Summary
              </div>
              <div className="agent-summary-text"
                dangerouslySetInnerHTML={{
                  __html: report.summary
                    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\n\n/g, '<br/><br/>')
                    .replace(/\n/g, '<br/>')
                }}
              />
            </div>
          )}

          {/* Confidence */}
          <div className="agent-card">
            <div className="agent-card-label">
              <Layers size={13} />
              Overall Confidence
            </div>
            <ConfidenceBar confidence={report.confidence} />
            <div style={{ fontSize: 11, color: '#475569' }}>
              Tools: {(report.tools_used || []).join(', ') || 'none'}
            </div>
          </div>

          {/* Verification */}
          {hasVerification && (
            <div className="agent-card">
              <div className="agent-card-label">
                <Shield size={13} />
                Verification Log
              </div>
              <VerificationLog verificationLog={report.verification_log} />
            </div>
          )}

          {/* Reasoning Trace */}
          {report.reasoning_trace && (
            <div className="agent-card full-width">
              <ReasoningTrace trace={report.reasoning_trace} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
