import cv2
import os
import uuid
import json
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__)

# Allow all origins in development; restrict in production via env var.
CORS(app, origins=os.environ.get("ALLOWED_ORIGINS", "*"))

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

# On Vercel the filesystem is read-only except /tmp.
_ON_VERCEL    = bool(os.environ.get("VERCEL"))
_WRITABLE_DIR = "/tmp" if _ON_VERCEL else BASE_DIR

SESSIONS_DIR = os.path.join(_WRITABLE_DIR, "sessions")   # sessions/<id>/captures/
SESSIONS_DB  = os.path.join(_WRITABLE_DIR, "sessions.json")

ALLOWED_EXT   = {'.jpg', '.jpeg', '.png', '.webp'}
MAX_UPLOAD_MB = 20


# ── Session helpers ───────────────────────────────────────────────────────────

def _load_sessions():
    if os.path.exists(SESSIONS_DB):
        with open(SESSIONS_DB, 'r') as f:
            return json.load(f)
    return {}


def _save_sessions(data):
    with open(SESSIONS_DB, 'w') as f:
        json.dump(data, f, indent=2)


def _session_dirs(session_id):
    base     = os.path.join(SESSIONS_DIR, session_id)
    captures = os.path.join(base, "captures")
    results  = os.path.join(base, "results")
    return base, captures, results


def _capture_count(captures_dir):
    if not os.path.exists(captures_dir):
        return 0
    return sum(
        1 for f in os.listdir(captures_dir)
        if os.path.splitext(f)[1].lower() in ALLOWED_EXT
    )


def _get_sessions_list():
    sessions = _load_sessions()
    result = []
    for sid, meta in sessions.items():
        _, captures_dir, results_dir = _session_dirs(sid)
        pano_path = os.path.join(results_dir, "panorama.jpg")
        pano_url  = f"/static/sessions/{sid}/results/panorama.jpg" if os.path.exists(pano_path) else None

        saved_count   = meta.get("capture_count")
        current_count = saved_count if saved_count is not None else _capture_count(captures_dir)

        result.append({
            "id":            sid,
            "name":          meta.get("name", "Untitled"),
            "created_at":    meta.get("created_at", ""),
            "capture_count": current_count,
            "panorama_url":  pano_url,
        })
    result.sort(key=lambda s: s["created_at"], reverse=True)
    return result


# ── Panorama stitching ───────────────────────────────────────────────────────

def stitch_panorama(captures_dir, results_dir):
    image_paths = []
    if os.path.exists(captures_dir):
        for fname in sorted(os.listdir(captures_dir)):
            if os.path.splitext(fname)[1].lower() in ALLOWED_EXT:
                image_paths.append(os.path.join(captures_dir, fname))

    if len(image_paths) < 2:
        return False, f"Need at least 2 images (found {len(image_paths)})"

    images = [cv2.imread(p) for p in image_paths]
    for img, path in zip(images, image_paths):
        if img is None:
            return False, f"Could not load image: {os.path.basename(path)}"

    stitcher = cv2.Stitcher_create()
    status, pano = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        os.makedirs(results_dir, exist_ok=True)
        pano_path = os.path.join(results_dir, "panorama.jpg")
        cv2.imwrite(pano_path, pano)

        import shutil
        if os.path.exists(captures_dir):
            shutil.rmtree(captures_dir, ignore_errors=True)

        return True, "Panorama stitched successfully"

    codes = {
        1: "Need more overlapping images — try capturing with more overlap",
        2: "Homography estimation failed — try more overlapping shots",
        3: "Camera parameters adjustment failed",
    }
    return False, codes.get(status, f"Unknown stitcher error (code {status})")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/favicon.ico')
def favicon():
    return '', 204


# ── Frontend Routes ──────────────────────────────────────────────────────────

@app.route('/')
def serve_index():
    frontend_dir = os.path.join(BASE_DIR, '../frontend')
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/<path:path>')
def serve_frontend(path):
    frontend_dir = os.path.join(BASE_DIR, '../frontend')
    if os.path.exists(os.path.join(frontend_dir, path)):
        return send_from_directory(frontend_dir, path)
    return send_from_directory(frontend_dir, 'index.html')


# ── Sessions API ─────────────────────────────────────────────────────────────

@app.route('/api/sessions', methods=['GET'])
def api_sessions():
    """Return all sessions with stats."""
    sessions = _get_sessions_list()
    stitched_count = sum(1 for s in sessions if s["panorama_url"])
    total_captures = sum(s["capture_count"] for s in sessions)
    return jsonify({
        "sessions":      sessions,
        "stitched_count": stitched_count,
        "total_captures": total_captures,
    })


@app.route('/api/session/new', methods=['POST'])
def api_new_session():
    """Create a new capture session."""
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or 'New Session').strip()[:60]

    sid = uuid.uuid4().hex[:12]
    _, captures_dir, results_dir = _session_dirs(sid)
    os.makedirs(captures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    sessions = _load_sessions()
    sessions[sid] = {
        "name":       name,
        "created_at": datetime.utcnow().isoformat(),
    }
    _save_sessions(sessions)

    return jsonify({"success": True, "session_id": sid, "name": name})


@app.route('/api/session/<session_id>', methods=['GET'])
def api_get_session(session_id):
    """Get metadata for a single session."""
    sessions = _load_sessions()
    if session_id not in sessions:
        return jsonify({"success": False, "message": "Session not found"}), 404

    meta = sessions[session_id]
    _, captures_dir, results_dir = _session_dirs(session_id)
    pano_path = os.path.join(results_dir, "panorama.jpg")
    pano_url  = f"/static/sessions/{session_id}/results/panorama.jpg" if os.path.exists(pano_path) else None

    saved_count   = meta.get("capture_count")
    current_count = saved_count if saved_count is not None else _capture_count(captures_dir)

    return jsonify({
        "success":       True,
        "id":            session_id,
        "name":          meta.get("name", "Untitled"),
        "created_at":    meta.get("created_at", ""),
        "capture_count": current_count,
        "panorama_url":  pano_url,
    })


@app.route('/api/session/<session_id>/delete', methods=['POST'])
def api_delete_session(session_id):
    sessions = _load_sessions()
    if session_id not in sessions:
        return jsonify({"success": False, "message": "Session not found"}), 404

    import shutil
    base, _, _ = _session_dirs(session_id)
    if os.path.exists(base):
        shutil.rmtree(base)

    del sessions[session_id]
    _save_sessions(sessions)
    return jsonify({"success": True})


@app.route('/api/session/<session_id>/rename', methods=['POST'])
def api_rename_session(session_id):
    sessions = _load_sessions()
    if session_id not in sessions:
        return jsonify({"success": False, "message": "Session not found"}), 404

    data = request.get_json(silent=True) or {}
    name = (data.get('name') or 'Untitled').strip()[:60]
    sessions[session_id]['name'] = name
    _save_sessions(sessions)
    return jsonify({"success": True, "name": name})


# ── Upload ───────────────────────────────────────────────────────────────────

@app.route('/api/upload/<session_id>', methods=['POST'])
def api_upload(session_id):
    """Accept a single image frame from the camera page."""
    sessions = _load_sessions()
    if session_id not in sessions:
        # Auto-create the session on the first upload
        sessions[session_id] = {
            "name": f"Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            "created_at": datetime.utcnow().isoformat(),
        }
        _save_sessions(sessions)

    content_length = request.content_length or 0
    if content_length > MAX_UPLOAD_MB * 1024 * 1024:
        return jsonify({"success": False, "message": f"File too large (max {MAX_UPLOAD_MB} MB)"}), 413

    file = request.files.get('image')
    if not file:
        return jsonify({"success": False, "message": "No image file received"}), 400

    ext = os.path.splitext(file.filename)[1].lower() if file.filename else '.jpg'
    if ext not in ALLOWED_EXT:
        ext = '.jpg'

    _, captures_dir, _ = _session_dirs(session_id)
    os.makedirs(captures_dir, exist_ok=True)

    count    = _capture_count(captures_dir)
    filename = f"capture_{count + 1:04d}{ext}"
    file.save(os.path.join(captures_dir, filename))

    new_count = _capture_count(captures_dir)
    return jsonify({"success": True, "filename": filename, "total": new_count})


# ── Stitch ───────────────────────────────────────────────────────────────────

@app.route('/api/stitch/<session_id>', methods=['POST'])
def api_stitch(session_id):
    sessions = _load_sessions()
    if session_id not in sessions:
        return jsonify({"success": False, "message": "Session not found"}), 404

    _, captures_dir, results_dir = _session_dirs(session_id)

    current_count = _capture_count(captures_dir)

    ok, msg = stitch_panorama(captures_dir, results_dir)

    if ok:
        sessions[session_id]['capture_count'] = current_count
        _save_sessions(sessions)

    pano_url = f"/static/sessions/{session_id}/results/panorama.jpg" if ok else None
    return jsonify({"success": ok, "message": msg, "panorama_url": pano_url})


# ── Captures count ───────────────────────────────────────────────────────────

@app.route('/api/captures-count/<session_id>')
def api_captures_count(session_id):
    _, captures_dir, _ = _session_dirs(session_id)
    return jsonify({"count": _capture_count(captures_dir)})


# ── Import panorama ──────────────────────────────────────────────────────────

@app.route('/api/import/<session_id>', methods=['POST'])
def api_import_photo(session_id):
    """Import an existing equirectangular panorama (2:1 ratio)."""
    sessions = _load_sessions()
    if session_id not in sessions:
        # Auto-create the session on the first upload
        sessions[session_id] = {
            "name": f"Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            "created_at": datetime.utcnow().isoformat(),
        }
        _save_sessions(sessions)

    file = request.files.get('image')
    if not file:
        return jsonify({"success": False, "message": "No file received"}), 400

    import numpy as np
    data = file.read()
    arr  = np.frombuffer(data, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"success": False, "message": "Could not decode image"}), 400

    h, w = img.shape[:2]
    ratio = w / h
    if abs(ratio - 2.0) > 0.15:
        return jsonify({
            "success": False,
            "message": f"Image must be 2:1 ratio (equirectangular). Got {w}×{h} = {ratio:.2f}:1"
        }), 400

    _, _, results_dir = _session_dirs(session_id)
    os.makedirs(results_dir, exist_ok=True)
    pano_path = os.path.join(results_dir, "panorama.jpg")
    cv2.imwrite(pano_path, img)

    pano_url = f"/static/sessions/{session_id}/results/panorama.jpg"
    return jsonify({"success": True, "panorama_url": pano_url, "dimensions": f"{w}×{h}"})


# ── Static session files ──────────────────────────────────────────────────────

@app.route('/static/sessions/<path:filename>')
def session_static(filename):
    return send_from_directory(SESSIONS_DIR, filename)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True, ssl_context="adhoc")
