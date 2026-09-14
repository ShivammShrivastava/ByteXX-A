"""
Firebase Realtime Database Listener — Gemma 4 / OpenRouter Edition
===================================================================
Listens for new queries written by the frontend to /queries/{queryId},
dispatches to the correct pipeline (VQA / Agent / CNN),
and writes structured results back to Firebase RTDB.

Flow:
  1. Frontend writes {uid, image_base64, question, task, status: "pending"}
     to /queries/{queryId}
  2. This listener polls /queries for "pending" entries every 2 s
  3. Picks them up, marks "processing", dispatches to the right handler
  4. Handler calls Gemma 4 via OpenRouter (VQA/caption/refer tasks)
     or YOLO CNN pipeline (cnn/agent tasks)
  5. Writes result to /results/{queryId} (or /agent_results/ / /cnn_results/)
  6. Marks query "done" so the frontend real-time listener triggers

Realtime DB structure:
  /queries/{queryId}/
    uid:           str
    image_base64:  str   (base64 JPEG, no data: prefix)
    question:      str
    task:          str   ("vqa" | "caption" | "refer" | "agent" | "cnn")
    status:        str   ("pending" | "processing" | "done" | "error")
    timestamp:     int   (Unix ms)
    source:        str   ("frontend_web")

  /results/{queryId}/         ← VQA / caption / refer answers
    answer:        str
    confidence:    float
    bboxes:        list   (optional, from CNN)
    bbox_object:   str    (optional)
    task:          str
    model:         str    ("google/gemma-4-31b-it:free" or fallback)
    processed_at:  int

  /agent_results/{queryId}/   ← Full AI Agent structured reports
    (see AI Agents pipeline for full schema)

  /cnn_results/{queryId}/     ← CNN-only results
    (see CNN pipeline for full schema)
"""

import asyncio
import base64
import io
import threading
import time
import urllib.request

from PIL import Image

from backend.firebase_config import (
    FIREBASE_CONFIG,
    DB_QUERIES_PATH,
    DB_RESULTS_PATH,
    DB_AGENT_PATH,
    DB_CNN_PATH,
    init_firebase,
    is_initialized,
    db_get,
    db_set,
    db_update,
)


class FirebaseListener:
    """
    Background thread that polls Firebase Realtime DB for pending queries
    and processes them via the correct pipeline.

    Supported task values:
      "vqa"     → Gemma 4 VQA   (writes to /results/)
      "caption" → Gemma 4 caption (writes to /results/)
      "refer"   → Gemma 4 referring expression (writes to /results/)
      "agent"   → Full AI Agent pipeline (writes to /agent_results/)
      "cnn"     → CNN-only pipeline (writes to /cnn_results/)
    """

    # Maximum age (seconds) for a pending query before it is considered stale.
    _STALE_AFTER_S = 1800  # 30 minutes

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self._running = False
        self._thread = None
        self._poll_interval = 2        # seconds between polls
        self._auth_error_count = 0
        self._last_auth_error = ""
        self._in_flight = set()        # query IDs currently being processed
        # Allow parallel Gemma API calls (cloud, not CPU-bound)
        self._inference_sem = threading.Semaphore(3)

    def start(self):
        """Start the background listener thread."""
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="FirebaseListener"
        )
        self._thread.start()
        print("  [OK] Firebase listener started (polling every 2s)")

    def stop(self):
        """Stop the listener."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("  Firebase listener stopped")

    # ─────────────────────────────────────────────────────────────────────────
    # Poll loop
    # ─────────────────────────────────────────────────────────────────────────
    def _poll_loop(self):
        """Main polling loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        while self._running:
            try:
                self._process_pending()
                if self._auth_error_count > 0:
                    self._auth_error_count = 0
                    self._poll_interval = 2
                    print("  [OK] Firebase connection restored")
            except Exception as e:
                err_str = str(e)
                is_jwt = "JWT" in err_str or "invalid_grant" in err_str
                if is_jwt:
                    self._auth_error_count += 1
                    if err_str != self._last_auth_error:
                        print(
                            f"  [WARN] Firebase JWT auth error "
                            f"(count={self._auth_error_count}): "
                            "Invalid JWT Signature — run: w32tm /resync /force"
                        )
                        self._last_auth_error = err_str
                    self._poll_interval = min(60, 2 ** min(self._auth_error_count, 6))
                else:
                    print(f"  Firebase listener error: {e}")
            time.sleep(self._poll_interval)
        self._loop.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Pending query processing
    # ─────────────────────────────────────────────────────────────────────────
    def _process_pending(self):
        """Find pending queries, drop stale ones, dispatch fresh ones newest-first."""
        if not is_initialized():
            return

        all_queries = db_get(DB_QUERIES_PATH)
        if not all_queries:
            return

        now_ms = int(time.time() * 1000)
        stale_cutoff_ms = now_ms - self._STALE_AFTER_S * 1000
        orphan_cutoff_ms = now_ms - 240_000  # 4 minutes

        # ── Pass 1: recover orphaned 'processing' queries ─────────────────
        for query_id, query_data in all_queries.items():
            if not isinstance(query_data, dict):
                continue
            if query_data.get("status") != "processing":
                continue
            if query_id in self._in_flight:
                continue

            ts = query_data.get("timestamp", now_ms)
            last_hb_s = query_data.get("heartbeat")
            last_active_ms = ts + int(last_hb_s) * 1000 if last_hb_s else ts

            if last_active_ms < orphan_cutoff_ms:
                print(
                    f"  [RECOVER] Orphaned query {query_id} "
                    f"(last_active={(now_ms - last_active_ms)//1000}s ago) — marking error"
                )
                db_update(f"{DB_QUERIES_PATH}/{query_id}", {
                    "status": "error",
                    "error":  "Query was abandoned mid-processing. Please resubmit.",
                })

        # ── Pass 2: collect genuine pending entries ──────────────────────
        pending = []
        for query_id, query_data in all_queries.items():
            if not isinstance(query_data, dict):
                continue
            if query_data.get("status") != "pending":
                continue
            if query_id in self._in_flight:
                continue

            ts = query_data.get("timestamp", now_ms)
            if ts < stale_cutoff_ms:
                print(f"  [SKIP] Stale query {query_id} (age={(now_ms-ts)//1000}s) — marking error")
                db_update(f"{DB_QUERIES_PATH}/{query_id}", {
                    "status": "error",
                    "error":  "Query expired while backend was offline",
                })
                continue

            pending.append((ts, query_id, query_data))

        # Newest first
        pending.sort(key=lambda x: x[0], reverse=True)

        for ts, query_id, query_data in pending:
            self._in_flight.add(query_id)
            db_update(f"{DB_QUERIES_PATH}/{query_id}", {"status": "processing"})

            worker = threading.Thread(
                target=self._run_query_thread,
                args=(query_id, query_data),
                daemon=True,
                name=f"QW-{query_id[:8]}",
            )
            worker.start()

    def _run_query_thread(self, query_id: str, query_data: dict):
        """Run one query with semaphore, then release."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            print(f"  [WAIT] {query_id[:12]} waiting for inference slot...")
            with self._inference_sem:
                print(f"  [RUN]  {query_id[:12]} acquired slot")
                self._handle_query(query_id, query_data, loop)
        finally:
            self._in_flight.discard(query_id)
            loop.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Query dispatch
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_query(self, query_id: str, query_data: dict, loop: asyncio.AbstractEventLoop):
        """Dispatch to the correct pipeline based on task type."""
        task     = query_data.get("task", "vqa")
        question = str(query_data.get("question", ""))[:60]
        print(f"  [IN]  Query {query_id} | task={task} | '{question}'")
        t_dispatch = time.time()

        # Heartbeat thread so frontend knows backend is alive
        _stop_hb = threading.Event()

        def _heartbeat():
            while not _stop_hb.wait(20):
                elapsed = int(time.time() - t_dispatch)
                try:
                    db_update(f"{DB_QUERIES_PATH}/{query_id}", {
                        "heartbeat": elapsed,
                        "status":    "processing",
                    })
                    print(f"  [HB]  {query_id[:12]} still running ({elapsed}s elapsed)")
                except Exception:
                    pass

        _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
        _hb_thread.start()

        try:
            if task == "agent":
                # Agent pipeline needs a PIL image
                image = self._load_pil_image(query_data)
                self._handle_agent(query_id, query_data, image, task, loop)
            elif task == "cnn":
                image = self._load_pil_image(query_data)
                self._handle_cnn(query_id, query_data, image, task, loop)
            else:
                # VQA / caption / refer → Gemma 4 via OpenRouter
                # Pass raw base64 (no PIL conversion needed)
                self._handle_vlm(query_id, query_data, task)

            print(f"  [OK]  Query {query_id[:12]} done in {time.time()-t_dispatch:.1f}s")

        except Exception as e:
            print(f"  [ERR] Query {query_id[:12]} failed after {time.time()-t_dispatch:.1f}s: {e}")
            self._write_error(query_id, task, str(e))
        finally:
            _stop_hb.set()

    # ─────────────────────────────────────────────────────────────────────────
    # VLM pipeline (Gemma 4 via OpenRouter)
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_vlm(self, query_id: str, query_data: dict, task: str):
        """
        Handle VQA / caption / refer tasks via Gemma 4 (OpenRouter).
        Reads image_base64 directly from query_data and sends to Gemma.
        Writes result to /results/{queryId}.
        """
        question   = query_data.get("question", "Describe this image.")
        image_b64  = query_data.get("image_base64", "")
        expression = query_data.get("expression", question)  # for 'refer' task

        if not image_b64:
            self._write_error(query_id, task, "No image_base64 found in query.")
            return

        try:
            if task == "caption":
                result = self.model_manager.generate_caption(image_b64)
                answer = result.get("caption", result.get("answer", "No caption generated."))
                confidence = float(result.get("confidence", 0.75))
            elif task == "refer":
                result = self.model_manager.locate_object(image_b64, expression)
                answer = result.get("raw_output", result.get("answer", "No output."))
                confidence = float(result.get("confidence", 0.75))
            else:
                # vqa (default for any unrecognized task too)
                result = self.model_manager.answer_vqa(image_b64, question)
                answer = result.get("answer", "No answer returned.")
                confidence = float(result.get("confidence", 0.75))

        except Exception as e:
            print(f"  [VLM] Gemma inference error for {query_id[:12]}: {e}")
            self._write_error(query_id, task, f"Gemma VQA error: {e}")
            return

        model_used = result.get("model", "google/gemma-4-31b-it:free")

        payload = {
            "answer":       answer,
            "confidence":   round(confidence, 3),
            "task":         task,
            "model":        model_used,
            "processed_at": int(time.time() * 1000),
        }

        db_set(f"{DB_RESULTS_PATH}/{query_id}", payload)
        db_update(f"{DB_QUERIES_PATH}/{query_id}", {"status": "done"})
        print(
            f"  [OK] Gemma done {query_id[:12]} "
            f"| model={model_used} | conf={confidence:.2f} "
            f"| ans='{answer[:60]}'"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # AI Agent pipeline (YOLO + Gemma 4 combined)
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_agent(self, query_id, query_data, image, task, loop: asyncio.AbstractEventLoop):
        """Handle full AI Agent pipeline — writes to /agent_results/."""
        question = query_data.get("question", "")

        import sys
        if "AI_Agents.agent" in sys.modules:
            agent_mod = sys.modules["AI_Agents.agent"]
        else:
            import importlib.util as _ilu
            from pathlib import Path
            _ai_dir = Path(__file__).resolve().parent.parent / "AI Agents"
            if "AI_Agents" not in sys.modules:
                _pkg_spec = _ilu.spec_from_file_location(
                    "AI_Agents", _ai_dir / "__init__.py",
                    submodule_search_locations=[str(_ai_dir)],
                )
                _pkg_mod = _ilu.module_from_spec(_pkg_spec)
                sys.modules["AI_Agents"] = _pkg_mod
                for _sub in ["router", "planner", "tools", "executor", "verifier", "merger", "agent"]:
                    _ss = _ilu.spec_from_file_location(f"AI_Agents.{_sub}", _ai_dir / f"{_sub}.py")
                    if _ss:
                        _sm = _ilu.module_from_spec(_ss)
                        sys.modules[f"AI_Agents.{_sub}"] = _sm
                        _ss.loader.exec_module(_sm)
                _pkg_spec.loader.exec_module(_pkg_mod)
            agent_mod = sys.modules["AI_Agents.agent"]

        sat_agent = agent_mod.sat_agent
        print(f"  [Agent] Starting pipeline for {query_id[:12]}...")

        try:
            report = loop.run_until_complete(
                asyncio.wait_for(sat_agent.run(image, question), timeout=240)
            )
            rd = report.to_dict()
        except asyncio.TimeoutError:
            print(f"  [Agent] Pipeline TIMED OUT after 4 min for {query_id[:12]}")
            error_payload = {
                "summary":          "Analysis timed out after 4 minutes. Try a simpler query.",
                "vlm_answer":       None,
                "object_counts":    {},
                "total_detections": 0,
                "bboxes":           [],
                "land_cover":       {},
                "confidence":       0.0,
                "tools_used":       [],
                "routing_mode":     "timeout",
                "errors":           ["Pipeline timed out after 240 seconds."],
                "duration_ms":      240000.0,
                "question":         question,
                "processed_at":     int(time.time() * 1000),
            }
            db_set(f"{DB_AGENT_PATH}/{query_id}", error_payload)
            db_update(f"{DB_QUERIES_PATH}/{query_id}", {"status": "done"})
            return
        except Exception as agent_err:
            print(f"  [Agent] Pipeline error: {agent_err}")
            error_payload = {
                "summary":          f"Agent pipeline error: {agent_err}",
                "vlm_answer":       None,
                "object_counts":    {},
                "total_detections": 0,
                "bboxes":           [],
                "land_cover":       {},
                "confidence":       0.0,
                "tools_used":       [],
                "routing_mode":     "error",
                "errors":           [str(agent_err)],
                "duration_ms":      0.0,
                "question":         question,
                "processed_at":     int(time.time() * 1000),
            }
            db_set(f"{DB_AGENT_PATH}/{query_id}", error_payload)
            db_update(f"{DB_QUERIES_PATH}/{query_id}", {"status": "done", "error": str(agent_err)})
            return

        payload = {
            "summary":             rd.get("summary", ""),
            "vlm_answer":          rd.get("vlm_answer"),
            "caption":             rd.get("caption"),
            "object_counts":       rd.get("object_counts", {}),
            "total_detections":    rd.get("total_detections", 0),
            "bboxes":              rd.get("bboxes", []),
            "bbox_labels":         rd.get("bbox_labels", []),
            "bbox_confidences":    rd.get("bbox_confidences", []),
            "land_cover":          rd.get("land_cover", {}),
            "land_cover_display":  rd.get("land_cover_display", {}),
            "dominant_land_cover": rd.get("dominant_land_cover", ""),
            "confidence":          rd.get("confidence", 0.0),
            "tools_used":          rd.get("tools_used", []),
            "routing_mode":        rd.get("routing_mode", ""),
            "reasoning_trace":     rd.get("reasoning_trace", ""),
            "errors":              rd.get("errors", []),
            "duration_ms":         rd.get("duration_ms", 0.0),
            "question":            question,
            "processed_at":        int(time.time() * 1000),
        }

        db_set(f"{DB_AGENT_PATH}/{query_id}", payload)
        db_update(f"{DB_QUERIES_PATH}/{query_id}", {"status": "done"})
        print(
            f"  [OK] Agent done {query_id[:12]}: "
            f"conf={rd.get('confidence', 0):.2f} | "
            f"tools={rd.get('tools_used', [])} | "
            f"{rd.get('duration_ms', 0):.0f}ms"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # CNN pipeline
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_cnn(self, query_id, query_data, image, task, loop: asyncio.AbstractEventLoop):
        """Handle CNN-only pipeline (detect + segment) — writes to /cnn_results/."""
        tasks_param    = query_data.get("cnn_tasks", ["detect", "segment"])
        conf_threshold = float(query_data.get("conf_threshold", 0.25))

        from CNN.cnn_pipeline import cnn_pipeline

        result = loop.run_until_complete(
            cnn_pipeline.run(image, tasks=tasks_param, conf_threshold=conf_threshold)
        )
        rd = result.to_dict()

        payload = {
            "object_counts":          rd.get("object_counts", {}),
            "total_detections":       rd.get("total_detections", 0),
            "bboxes":                 rd.get("bboxes", []),
            "bbox_labels":            rd.get("bbox_labels", []),
            "bbox_confidences":       rd.get("bbox_confidences", []),
            "land_cover_percentages": rd.get("land_cover_percentages", {}),
            "land_cover_display":     rd.get("land_cover_display", {}),
            "dominant_land_cover":    rd.get("dominant_land_cover", ""),
            "tasks_run":              rd.get("tasks_run", []),
            "overall_confidence":     rd.get("overall_confidence", 0.0),
            "total_duration_ms":      rd.get("total_duration_ms", 0.0),
            "errors":                 rd.get("errors", []),
            "processed_at":           int(time.time() * 1000),
        }

        db_set(f"{DB_CNN_PATH}/{query_id}", payload)
        db_update(f"{DB_QUERIES_PATH}/{query_id}", {"status": "done"})
        print(
            f"  [OK] CNN done {query_id[:12]}: "
            f"detections={rd.get('total_detections', 0)} | "
            f"{rd.get('total_duration_ms', 0):.0f}ms"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Error writer
    # ─────────────────────────────────────────────────────────────────────────
    def _write_error(self, query_id: str, task: str, error_msg: str):
        """Write error to /queries/ and the appropriate results path."""
        print(f"  [ERROR] {query_id[:12]}: {error_msg}")

        result_path = (
            DB_AGENT_PATH if task == "agent"
            else DB_CNN_PATH if task == "cnn"
            else DB_RESULTS_PATH
        )

        db_update(f"{DB_QUERIES_PATH}/{query_id}", {
            "status": "error",
            "error":  error_msg,
        })
        db_set(f"{result_path}/{query_id}", {
            "answer":       f"Error: {error_msg}",
            "confidence":   0.0,
            "task":         task,
            "processed_at": int(time.time() * 1000),
            "error":        error_msg,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Image loading helpers
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _load_pil_image(query_data: dict) -> Image.Image:
        """Load PIL image from base64 or URL in the query payload (for CNN/Agent)."""
        image_b64 = query_data.get("image_base64", "")
        image_url = query_data.get("image_url", "")

        if image_b64:
            raw = base64.b64decode(image_b64)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        elif image_url:
            req = urllib.request.Request(image_url, headers={"User-Agent": "SatQueryAI/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return Image.open(io.BytesIO(resp.read())).convert("RGB")
        else:
            raise ValueError("Query has neither image_base64 nor image_url")

    # ─────────────────────────────────────────────────────────────────────────
    # Keyword helpers (used by Agent tools for routing)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _is_counting_query(question: str) -> bool:
        q = question.lower()
        return any(kw in q for kw in [
            "how many", "count", "number of", "total", "detect", "find all",
            "locate all", "identify all", "show all",
        ])

    @staticmethod
    def _detect_object_keyword(question: str):
        q = question.lower()
        OBJECT_MAP = [
            (["airplane", "aircraft", "plane", "planes", "airplanes", "aeroplane", "jet", "helicopter"], "airplane"),
            (["car", "cars", "vehicle", "vehicles", "automobile"], "car"),
            (["ship", "ships", "vessel", "vessels", "boat", "boats"], "ship"),
            (["building", "buildings", "structure", "structures", "house"], "building"),
            (["truck", "trucks"], "truck"),
            (["tank", "tanks", "storage tank"], "storage tank"),
            (["bridge", "bridges"], "bridge"),
            (["person", "people", "pedestrian", "human"], "person"),
        ]
        for keywords, label in OBJECT_MAP:
            if any(kw in q for kw in keywords):
                return label
        return None
