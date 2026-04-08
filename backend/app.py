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


def _next_capture_index(captures_dir: str) -> int:
    """Return the next sequential index for naming captures."""
    if not os.path.exists(captures_dir):
        return 0
    existing = [f for f in os.listdir(captures_dir)
                if os.path.splitext(f)[1].lower() in ALLOWED_EXT]
    if not existing:
        return 0
    # Extract all numeric suffixes from camera_capture_N.jpg
    indices = []
    for f in existing:
        m = re.search(r'camera_capture_(\d+)', f)
        if m:
            indices.append(int(m.group(1)))
    return max(indices) + 1 if indices else len(existing)


# ──────────────────────────────────────────────────────────────────────
# Background stitching worker
# ──────────────────────────────────────────────────────────────────────
def _run_stitch(session_id: str, image_paths: list, results_dir: str, preserve_order: bool = False) -> None:
    with _stitch_lock:
        _stitch_status[session_id] = "running"
    try:
        n = len(image_paths)
        if n >= 20 and n % 10 == 0:
            from robust_stitcher import stitch_images_robustly_3layer
            print(f"[*] Background 3-layer stitching started: {n} images …")
            result_paths = stitch_images_robustly_3layer(image_paths, results_dir, preserve_order)
        else:
            from robust_stitcher import stitch_images_robustly
            print(f"[*] Background stitching started: {n} images …")
            result_paths = stitch_images_robustly(image_paths, results_dir, preserve_order)

        # Handle None result_paths to prevent crash
        if result_paths is None:
            print("[!] Stitching returned None, treating as error")
            result_paths = []

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
    """
    Upload a single image to a session.
    Original filename is preserved to maintain the user's expected order.
    """
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
        print(f"[UPLOAD] ❌  No image in request  (session={session_id})")
        return jsonify({"success": False, "message": "No image provided"}), 400

    idx = request.args.get('idx')
    ext = os.path.splitext(file.filename or '')[1].lower()
    if ext not in ALLOWED_EXT:
        ext = '.jpg'

    _, captures_dir, results_dir = _session_dirs(session_id)
    os.makedirs(captures_dir, exist_ok=True)
    os.makedirs(results_dir,  exist_ok=True)

    # ── USE INDEX FOR NAMING (Ensures 1, 2, 3... order) ──────────────
    if idx is not None and idx.isdigit():
        # Use 1-based indexing for filenames with leading zero (01.jpg, 02.jpg, ...)
        filename = f"{int(idx) + 1:02d}{ext}"
    else:
        # Fallback to original filename but sanitized and force leading zero if it's a simple number
        base_name = os.path.splitext(file.filename or 'unnamed')[0]
        if base_name.isdigit():
            filename = f"{int(base_name):02d}{ext}"
        else:
            filename = f"{base_name}{ext}"
    
    save_path = os.path.join(captures_dir, filename)
    
    # ALWAYS OVERWRITE to maintain sequence (fix duplicate naming issue)
    file.save(save_path)

    # Basic validation – make sure OpenCV can open it
    img = cv2.imread(save_path)
    if img is None:
        if os.path.exists(save_path):
            os.remove(save_path)
        print(f"[UPLOAD] ❌  Invalid image file rejected  (session={session_id}, file={filename})")
        return jsonify({"success": False, "message": "Invalid image file"}), 400

    size_kb = os.path.getsize(save_path) / 1024
    total   = _capture_count(captures_dir)
    h, w    = img.shape[:2]
    ts      = datetime.utcnow().strftime('%H:%M:%S')

    print(f"[UPLOAD] ✅  [{ts}]  session={session_id}")
    print(f"           📸  {filename}  |  {w}×{h}px  |  {size_kb:.1f} KB")
    print(f"           📦  Total captures in session: {total}")
    print(f"           {'─'*48}")

    return jsonify({"success": True, "filename": filename, "total": total})


# ── NEW: batch upload multiple images at once ──────────────────────────
@app.route('/api/upload-batch/<session_id>', methods=['POST', 'OPTIONS'])
def api_upload_batch(session_id: str):
    """
    Upload multiple images in one request.
    Original filenames are preserved to maintain the user's expected order.

    Form field name for each file: 'images' (repeated) or 'image0', 'image1', …
    """
    if request.method == 'OPTIONS':
        return '', 200

    sessions = _load_sessions()
    if session_id not in sessions:
        sessions[session_id] = {
            "name":       f"Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            "created_at": datetime.utcnow().isoformat(),
        }
        _save_sessions(sessions)

    _, captures_dir, results_dir = _session_dirs(session_id)
    os.makedirs(captures_dir, exist_ok=True)
    os.makedirs(results_dir,  exist_ok=True)

    # Collect files — accept field name 'images' (getlist) or 'image0'..'imageN'
    files = request.files.getlist('images')
    if not files:
        # Fallback: try image0, image1, …
        files = []
        i = 0
        while True:
            f = request.files.get(f'image{i}')
            if f is None:
                break
            files.append(f)
            i += 1

    if not files:
        return jsonify({"success": False, "message": "No images provided"}), 400

    saved = []

    for file in files:
        ext = os.path.splitext(file.filename or '')[1].lower()
        if ext not in ALLOWED_EXT:
            ext = '.jpg'

        # ── PRESERVE ORIGINAL FILENAME ──────────
        original_name = file.filename or 'unnamed.jpg'
        base_name = os.path.splitext(original_name)[0]
        filename = f"{base_name}{ext}"
        save_path = os.path.join(captures_dir, filename)
        
        # Handle duplicate filenames by appending number
        counter = 1
        while os.path.exists(save_path):
            filename = f"{base_name}_{counter:02d}{ext}"
            save_path = os.path.join(captures_dir, filename)
            counter += 1
        
        file.save(save_path)

        img = cv2.imread(save_path)
        if img is None:
            os.remove(save_path)
            print(f"[BATCH] ❌  Invalid image skipped: {file.filename}")
            continue

        h, w = img.shape[:2]
        size_kb = os.path.getsize(save_path) / 1024
        print(f"[BATCH] ✅  {filename}  |  {w}×{h}px  |  {size_kb:.1f} KB")
        saved.append(filename)

    total = _capture_count(captures_dir)
    print(f"[BATCH] 📦  {len(saved)} images saved  |  session total: {total}")

    return jsonify({
        "success":  True,
        "saved":    saved,
        "total":    total,
    })


@app.route('/api/stitch/<session_id>', methods=['POST', 'OPTIONS'])
def api_stitch(session_id: str):
    if request.method == 'OPTIONS':
        return '', 200

    sessions = _load_sessions()
    if session_id not in sessions:
        return jsonify({"success": False, "message": "Session not found"}), 404

    # Check for preserve_order flag in request
    preserve_order = False
    try:
        data = request.get_json(silent=True) or {}
        preserve_order = data.get('preserve_order', False)
    except:
        pass
    
    if preserve_order:
        print(f"[STITCH] ORDER-PRESERVING mode enabled for session {session_id}")

    _, captures_dir, results_dir = _session_dirs(session_id)
    os.makedirs(results_dir, exist_ok=True)

    sorted_files = _sorted_captures(captures_dir)
    if len(sorted_files) < 2:
        return jsonify({"success": False,
                        "message": f"Need at least 2 images (found {len(sorted_files)})"}), 400

    image_paths = [os.path.join(captures_dir, f) for f in sorted_files]

    # Print the ordered list exactly like the user requested
    print(f"\n[STITCH] Images to stitch ({len(image_paths)} total), sorted by filename:")
    for i, p in enumerate(image_paths, 1):
        print(f"  {i}. {os.path.basename(p)}")
    print()

    # Check if already running
    with _stitch_lock:
        if _stitch_status.get(session_id) == "running":
            return jsonify({"success": True, "message": "Stitching already in progress",
                            "status": "running"})

    # Launch background thread with preserve_order flag
    t = threading.Thread(target=_run_stitch,
                         args=(session_id, image_paths, results_dir, preserve_order),
                         daemon=True)
    t.start()

    return jsonify({
        "success": True,
        "message": f"Stitching {len(image_paths)} images in background …",
        "status":  "running",
        "preserve_order": preserve_order,
        "images":  image_paths,
    })


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

    if status is None:
        pano_path = os.path.join(results_dir, "panorama.jpg")
        if os.path.exists(pano_path):
            status = "done"
        else:
            status = "pending"

    panorama_urls = []
    if status == "done" and os.path.exists(results_dir):
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
    stitched_count = sum(1 for s in result if s["panorama_url"])
    total_captures = sum(s["capture_count"] for s in result)
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
    import socket as _socket
    os.makedirs(SESSIONS_DIR, exist_ok=True)

    frontend_dir = os.path.join(BASE_DIR, '..', 'frontend')
    cert_file = os.path.join(frontend_dir, 'cert.pem')
    key_file  = os.path.join(frontend_dir, 'key.pem')

    def get_local_ip():
        try:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    ip = get_local_ip()

    if os.path.exists(cert_file) and os.path.exists(key_file):
        ssl_ctx = (cert_file, key_file)
        print(f"\n{'='*52}")
        print(f"  🔒 Flask backend running over HTTPS")
        print(f"  Local:    https://localhost:5000")
        print(f"  Network:  https://{ip}:5000")
        print(f"{'='*52}\n")
        app.run(host="0.0.0.0", port=5000, debug=False, ssl_context=ssl_ctx)
    else:
        print(f"\n⚠️  No SSL certs found — running HTTP (mobile may fail)")
        print(f"  Local:    http://localhost:5000")
        print(f"  Network:  http://{ip}:5000\n")
        app.run(host="0.0.0.0", port=5000, debug=False)