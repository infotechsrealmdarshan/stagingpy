import cv2
import os
import json
import uuid
import threading
import numpy as np
import re
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
SESSIONS_DB  = os.path.join(BASE_DIR, "sessions.json")
ALLOWED_EXT  = {'.jpg', '.jpeg', '.png', '.webp'}

# In-memory stitch-job status  {session_id: "pending"|"running"|"done"|"error"}
_stitch_status: dict = {}
_stitch_lock = threading.Lock()

os.makedirs(SESSIONS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────
def _load_sessions() -> dict:
    if os.path.exists(SESSIONS_DB):
        with open(SESSIONS_DB, 'r') as f:
            return json.load(f)
    return {}


def _save_sessions(data: dict) -> None:
    with open(SESSIONS_DB, 'w') as f:
        json.dump(data, f, indent=2)


def _session_dirs(session_id: str):
    base = os.path.join(SESSIONS_DIR, session_id)
    return base, os.path.join(base, "captures"), os.path.join(base, "results")


def _natural_key(s: str):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]


def _capture_count(captures_dir: str) -> int:
    if not os.path.exists(captures_dir):
        return 0
    return sum(1 for f in os.listdir(captures_dir)
               if os.path.splitext(f)[1].lower() in ALLOWED_EXT)


def _sorted_captures(captures_dir: str):
    if not os.path.exists(captures_dir):
        return []
    files = [f for f in os.listdir(captures_dir)
             if os.path.splitext(f)[1].lower() in ALLOWED_EXT]
    return sorted(files, key=_natural_key)


# ──────────────────────────────────────────────────────────────────────
# Background stitching worker
# ──────────────────────────────────────────────────────────────────────
def _run_stitch(session_id: str, image_paths: list, results_dir: str) -> None:
    with _stitch_lock:
        _stitch_status[session_id] = "running"
    try:
        from robust_stitcher import stitch_images_robustly
        print(f"[*] Background stitching started: {len(image_paths)} images …")
        result_paths = stitch_images_robustly(image_paths, results_dir)

        sessions = _load_sessions()
        if session_id in sessions:
            sessions[session_id]['capture_count'] = len(image_paths)
            sessions[session_id]['stitch_count']  = len(result_paths)
            _save_sessions(sessions)

        with _stitch_lock:
            _stitch_status[session_id] = "done" if result_paths else "error"
        print(f"[*] Stitching done: {len(result_paths)} panorama(s) saved.")
    except Exception as exc:
        import traceback
        print(f"[ERROR] Stitching failed: {exc}")
        print(traceback.format_exc())
        with _stitch_lock:
            _stitch_status[session_id] = "error"


# ──────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────
@app.route('/api/session/new', methods=['POST', 'OPTIONS'])
def api_new_session():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        data = request.get_json(silent=True) or {}
        name = (data.get('name') or 'New Session').strip()[:60]
        sid  = uuid.uuid4().hex[:12]

        _, captures_dir, results_dir = _session_dirs(sid)
        os.makedirs(captures_dir, exist_ok=True)
        os.makedirs(results_dir,  exist_ok=True)

        sessions = _load_sessions()
        sessions[sid] = {
            "name":       name,
            "created_at": datetime.utcnow().isoformat(),
        }
        _save_sessions(sessions)
        return jsonify({"success": True, "session_id": sid, "name": name})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/upload/<session_id>', methods=['POST', 'OPTIONS'])
def api_upload(session_id: str):
    if request.method == 'OPTIONS':
        return '', 200

    sessions = _load_sessions()
    if session_id not in sessions:
        sessions[session_id] = {
            "name":       f"Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            "created_at": datetime.utcnow().isoformat(),
        }
        _save_sessions(sessions)

    file = request.files.get('image')
    if not file:
        return jsonify({"success": False, "message": "No image provided"}), 400

    ext = os.path.splitext(file.filename or '')[1].lower() or '.jpg'
    if ext not in ALLOWED_EXT:
        ext = '.jpg'

    _, captures_dir, _ = _session_dirs(session_id)
    os.makedirs(captures_dir, exist_ok=True)

    count    = _capture_count(captures_dir)
    filename = f"capture_{count + 1:04d}{ext}"
    save_path = os.path.join(captures_dir, filename)
    file.save(save_path)

    # Basic validation – make sure OpenCV can open it
    img = cv2.imread(save_path)
    if img is None:
        os.remove(save_path)
        return jsonify({"success": False, "message": "Invalid image file"}), 400

    total = _capture_count(captures_dir)
    return jsonify({"success": True, "filename": filename, "total": total})


@app.route('/api/stitch/<session_id>', methods=['POST', 'OPTIONS'])
def api_stitch(session_id: str):
    if request.method == 'OPTIONS':
        return '', 200

    sessions = _load_sessions()
    if session_id not in sessions:
        return jsonify({"success": False, "message": "Session not found"}), 404

    _, captures_dir, results_dir = _session_dirs(session_id)

    sorted_files = _sorted_captures(captures_dir)
    if len(sorted_files) < 2:
        return jsonify({"success": False,
                        "message": f"Need at least 2 images (found {len(sorted_files)})"}), 400

    image_paths = [os.path.join(captures_dir, f) for f in sorted_files]

    # Check if already running
    with _stitch_lock:
        if _stitch_status.get(session_id) == "running":
            return jsonify({"success": True, "message": "Stitching already in progress",
                            "status": "running"})

    # Launch background thread
    t = threading.Thread(target=_run_stitch,
                         args=(session_id, image_paths, results_dir),
                         daemon=True)
    t.start()

    return jsonify({"success": True,
                    "message": f"Stitching {len(image_paths)} images in background …",
                    "status":  "running"})


@app.route('/api/stitch-status/<session_id>', methods=['GET', 'OPTIONS'])
def api_stitch_status(session_id: str):
    if request.method == 'OPTIONS':
        return '', 200

    sessions = _load_sessions()
    if session_id not in sessions:
        return jsonify({"success": False, "message": "Session not found"}), 404

    _, captures_dir, results_dir = _session_dirs(session_id)

    with _stitch_lock:
        status = _stitch_status.get(session_id)

    # Derive from disk if not in memory
    if status is None:
        pano_path = os.path.join(results_dir, "panorama.jpg")
        if os.path.exists(pano_path):
            status = "done"
        elif len(_sorted_captures(captures_dir)) >= 2:
            status = "pending"
        else:
            status = "pending"

    # Build panorama URLs if done
    panorama_urls = []
    if status == "done":
        for fname in sorted(os.listdir(results_dir)):
            if fname.endswith('.jpg'):
                panorama_urls.append(
                    f"/static/sessions/{session_id}/results/{fname}")

    return jsonify({
        "status":        status,
        "panorama_urls": panorama_urls,
        "panorama_url":  panorama_urls[0] if panorama_urls else None,
    })


@app.route('/api/sessions', methods=['GET'])
def api_sessions():
    sessions = _load_sessions()
    result = []
    for sid, meta in sessions.items():
        _, captures_dir, results_dir = _session_dirs(sid)
        pano_path = os.path.join(results_dir, "panorama.jpg")
        pano_url  = (f"/static/sessions/{sid}/results/panorama.jpg"
                     if os.path.exists(pano_path) else None)
        cap_count = (meta.get("capture_count")
                     if meta.get("capture_count") is not None
                     else _capture_count(captures_dir))
        result.append({
            "id":            sid,
            "name":          meta.get("name", "Untitled"),
            "created_at":    meta.get("created_at", ""),
            "capture_count": cap_count,
            "panorama_url":  pano_url,
        })
    result.sort(key=lambda s: s["created_at"], reverse=True)
    stitched_count  = sum(1 for s in result if s["panorama_url"])
    total_captures  = sum(s["capture_count"] for s in result)
    return jsonify({"sessions": result,
                    "stitched_count": stitched_count,
                    "total_captures": total_captures})


@app.route('/api/session/<session_id>', methods=['GET'])
def api_session_get(session_id: str):
    sessions = _load_sessions()
    if session_id not in sessions:
        return jsonify({"success": False, "message": "Session not found"}), 404

    meta = sessions[session_id]
    _, captures_dir, results_dir = _session_dirs(session_id)
    pano_path = os.path.join(results_dir, "panorama.jpg")
    pano_url  = (f"/static/sessions/{session_id}/results/panorama.jpg"
                 if os.path.exists(pano_path) else None)
    cap_count = (meta.get("capture_count")
                 if meta.get("capture_count") is not None
                 else _capture_count(captures_dir))

    return jsonify({
        "success":       True,
        "id":            session_id,
        "name":          meta.get("name", "Untitled"),
        "created_at":    meta.get("created_at", ""),
        "capture_count": cap_count,
        "panorama_url":  pano_url,
    })


@app.route('/api/captures-count/<session_id>', methods=['GET'])
def api_captures_count(session_id: str):
    sessions = _load_sessions()
    if session_id not in sessions:
        return jsonify({"success": False, "message": "Session not found"}), 404
    _, captures_dir, _ = _session_dirs(session_id)
    return jsonify({"success": True, "count": _capture_count(captures_dir)})


@app.route('/api/session/<session_id>/captures', methods=['GET'])
def api_list_captures(session_id: str):
    """Return a list of uploaded capture filenames for a session."""
    sessions = _load_sessions()
    if session_id not in sessions:
        return jsonify({"success": False, "message": "Session not found"}), 404
    _, captures_dir, _ = _session_dirs(session_id)
    files = _sorted_captures(captures_dir)
    urls  = [f"/static/sessions/{session_id}/captures/{f}" for f in files]
    return jsonify({"success": True, "captures": urls, "count": len(files)})


@app.route('/api/session/<session_id>', methods=['DELETE'])
def api_delete_session(session_id: str):
    """Delete a session and all its files."""
    import shutil
    sessions = _load_sessions()
    if session_id not in sessions:
        return jsonify({"success": False, "message": "Session not found"}), 404
    base, _, _ = _session_dirs(session_id)
    if os.path.exists(base):
        shutil.rmtree(base)
    del sessions[session_id]
    _save_sessions(sessions)
    with _stitch_lock:
        _stitch_status.pop(session_id, None)
    return jsonify({"success": True})


@app.route('/static/sessions/<path:filename>', methods=['GET', 'OPTIONS'])
def session_static(filename: str):
    if request.method == 'OPTIONS':
        resp = jsonify({})
        resp.headers['Access-Control-Allow-Origin']  = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = '*'
        return resp, 200
    return send_from_directory(SESSIONS_DIR, filename)


# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    print("Server starting on port 5000 …")
    app.run(host="0.0.0.0", port=5000, debug=False)