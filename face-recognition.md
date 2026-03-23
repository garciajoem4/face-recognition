"""
GetMyPhotos Face Recognition Server
Uses InsightFace (buffalo_l) for face detection, encoding, and matching.
"""

import os
import io
import json
import logging
import tempfile
import time

import cv2
import numpy as np
from flask import Flask, request, jsonify
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from PIL import Image

# --- Configuration ---
PORT = int(os.environ.get("GMP_FACE_PORT", 5200))
BIND_HOST = os.environ.get("GMP_FACE_HOST", "127.0.0.1")
# Face match distance threshold (lower = stricter). 0.4 is good for high accuracy.
MATCH_THRESHOLD = float(os.environ.get("GMP_FACE_MATCH_THRESHOLD", "0.4"))
# DBSCAN clustering distance for no-bib grouping
CLUSTER_EPS = float(os.environ.get("GMP_FACE_CLUSTER_EPS", "0.5"))
# Minimum images in a cluster to form a group
CLUSTER_MIN_SAMPLES = int(os.environ.get("GMP_FACE_CLUSTER_MIN", "2"))
# Detection size (larger = more accurate but slower)
DET_SIZE = int(os.environ.get("GMP_FACE_DET_SIZE", "640"))

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("gmp-face")

# --- Initialize InsightFace ---
logger.info("Loading InsightFace buffalo_l model...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))
_current_det_size = DET_SIZE
logger.info("InsightFace model loaded successfully.")

app = Flask(__name__)


def load_image(file_path: str) -> np.ndarray | None:
    """Load an image from a file path and return as BGR numpy array."""
    try:
        if not os.path.isfile(file_path):
            logger.warning(f"File not found: {file_path}")
            return None
        img = cv2.imread(file_path)
        if img is None:
            # Fallback: try PIL for formats OpenCV can't handle
            pil_img = Image.open(file_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        logger.error(f"Failed to load image {file_path}: {e}")
        return None


def load_image_from_bytes(data: bytes) -> np.ndarray | None:
    """Load an image from raw bytes."""
    try:
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            pil_img = Image.open(io.BytesIO(data)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        logger.error(f"Failed to load image from bytes: {e}")
        return None


def prepare_det_size(det_size: int):
    """Re-prepare the model if the requested det_size differs from current."""
    global _current_det_size
    if det_size != _current_det_size:
        logger.info(f"Switching detection size from {_current_det_size} to {det_size}")
        face_app.prepare(ctx_id=0, det_size=(det_size, det_size))
        _current_det_size = det_size


def get_best_face(img: np.ndarray):
    """Detect faces and return the largest/best quality face."""
    faces = face_app.get(img)
    if not faces:
        return None
    # Pick the face with the largest bounding box area (most prominent)
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return best


def face_distance(enc1: np.ndarray, enc2: np.ndarray) -> float:
    """Cosine distance between two face encodings."""
    return float(1.0 - np.dot(enc1, enc2) / (np.linalg.norm(enc1) * np.linalg.norm(enc2) + 1e-8))


def encoding_to_str(enc: np.ndarray) -> str:
    """Serialize a face encoding to a compact string."""
    return ",".join(f"{v:.6f}" for v in enc.tolist())


# -------------------------------------------------------------------
# ENDPOINT: /health
# -------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "buffalo_l",
        "engine": "insightface",
        "version": "1.0.0",
    })


# -------------------------------------------------------------------
# ENDPOINT: /select-index
# Receives a group of photos, detects faces, picks the one with the
# largest and clearest face as the index photo for that group.
# -------------------------------------------------------------------
@app.route("/select-index", methods=["POST"])
def select_index():
    data = request.get_json(silent=True)
    if not data or "photos" not in data:
        return jsonify({"error": "Missing photos array"}), 400

    photos = data["photos"]
    group_id = data.get("group_id")
    det_size = int(data.get("det_size", DET_SIZE))

    logger.info(f"select-index: group_id={group_id}, photos={len(photos)}, det_size={det_size}")
    prepare_det_size(det_size)

    best_photo_id = None
    best_score = -1.0
    best_encoding = None

    for p in photos:
        file_path = p.get("file_path")
        photo_id = p.get("photo_id")
        if not file_path or not photo_id:
            continue

        img = load_image(file_path)
        if img is None:
            continue

        face = get_best_face(img)
        if face is None:
            continue

        # Score: combine bbox area (normalized) and detection score
        bbox = face.bbox
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        img_area = img.shape[0] * img.shape[1]
        relative_area = area / img_area if img_area > 0 else 0
        det_score = float(face.det_score) if hasattr(face, "det_score") else 0.5
        quality_score = relative_area * 0.6 + det_score * 0.4

        if quality_score > best_score:
            best_score = quality_score
            best_photo_id = photo_id
            best_encoding = face.normed_embedding

    if best_photo_id is None:
        logger.info(f"select-index: no faces detected in group {group_id}")
        return jsonify({"index_photo_id": None, "face_encoding": None, "quality_score": 0})

    logger.info(f"select-index: group {group_id} → photo {best_photo_id} (score={best_score:.3f})")
    return jsonify({
        "index_photo_id": best_photo_id,
        "face_encoding": encoding_to_str(best_encoding),
        "quality_score": round(best_score, 4),
    })


# -------------------------------------------------------------------
# ENDPOINT: /cluster-faces
# Receives a list of photos (no bib), detects faces, clusters them
# by face similarity using DBSCAN, returns groups.
# -------------------------------------------------------------------
@app.route("/cluster-faces", methods=["POST"])
def cluster_faces():
    data = request.get_json(silent=True)
    if not data or "photos" not in data:
        return jsonify({"error": "Missing photos array"}), 400

    photos = data["photos"]
    event_id = data.get("event_id")
    det_size = int(data.get("det_size", DET_SIZE))

    logger.info(f"cluster-faces: event_id={event_id}, photos={len(photos)}, det_size={det_size}")
    prepare_det_size(det_size)

    # Step 1: Detect faces and extract encodings
    face_data = []  # list of (photo_id, encoding, quality_score)
    for p in photos:
        file_path = p.get("file_path")
        photo_id = p.get("photo_id")
        if not file_path or not photo_id:
            continue

        img = load_image(file_path)
        if img is None:
            continue

        face = get_best_face(img)
        if face is None:
            continue

        bbox = face.bbox
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        img_area = img.shape[0] * img.shape[1]
        relative_area = area / img_area if img_area > 0 else 0
        det_score = float(face.det_score) if hasattr(face, "det_score") else 0.5
        quality_score = relative_area * 0.6 + det_score * 0.4

        face_data.append((photo_id, face.normed_embedding, quality_score))

    if len(face_data) < CLUSTER_MIN_SAMPLES:
        logger.info(f"cluster-faces: not enough faces found ({len(face_data)})")
        return jsonify({"groups": []})

    # Step 2: Build distance matrix and cluster with DBSCAN
    encodings = np.array([fd[1] for fd in face_data])
    # Cosine distance matrix
    norms = np.linalg.norm(encodings, axis=1, keepdims=True) + 1e-8
    normalized = encodings / norms
    cos_sim = np.dot(normalized, normalized.T)
    distance_matrix = 1.0 - cos_sim
    np.fill_diagonal(distance_matrix, 0)

    clustering = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES, metric="precomputed")
    labels = clustering.fit_predict(distance_matrix)

    # Step 3: Build groups from cluster labels
    cluster_map = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue  # Noise / unclustered
        if label not in cluster_map:
            cluster_map[label] = []
        cluster_map[label].append(i)

    groups = []
    for label, indices in cluster_map.items():
        # Pick the best quality photo as the index
        best_idx = max(indices, key=lambda i: face_data[i][2])
        photo_ids = [face_data[i][0] for i in indices]
        index_photo_id = face_data[best_idx][0]
        index_encoding = face_data[best_idx][1]
        index_quality = face_data[best_idx][2]

        groups.append({
            "photo_ids": photo_ids,
            "index_photo_id": index_photo_id,
            "face_encoding": encoding_to_str(index_encoding),
            "quality_score": round(index_quality, 4),
        })

    logger.info(f"cluster-faces: event {event_id} → {len(groups)} groups from {len(face_data)} faces")
    return jsonify({"groups": groups})


# -------------------------------------------------------------------
# ENDPOINT: /search-face
# Receives a selfie (multipart file) and a JSON array of indexed
# face groups. Compares the selfie face to each group's encoding
# and returns matching group IDs.
# -------------------------------------------------------------------
@app.route("/search-face", methods=["POST"])
def search_face():
    # Get the selfie file
    if "selfie" not in request.files:
        return jsonify({"error": "Missing selfie file"}), 400

    selfie_file = request.files["selfie"]
    selfie_bytes = selfie_file.read()
    if not selfie_bytes:
        return jsonify({"error": "Empty selfie file"}), 400

    # Get the groups JSON
    groups_json = request.form.get("groups")
    if not groups_json:
        return jsonify({"error": "Missing groups data"}), 400

    try:
        groups = json.loads(groups_json)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid groups JSON"}), 400

    logger.info(f"search-face: comparing selfie against {len(groups)} groups")

    # Detect face in selfie
    selfie_img = load_image_from_bytes(selfie_bytes)
    if selfie_img is None:
        return jsonify({"error": "Could not read selfie image"}), 400

    selfie_face = get_best_face(selfie_img)
    if selfie_face is None:
        return jsonify({
            "matched_group_ids": [],
            "message": "No face detected in the selfie. Please upload a clear front-facing photo.",
        })

    selfie_encoding = selfie_face.normed_embedding

    # Compare against each group's stored encoding
    matched_group_ids = []
    best_confidence = 0.0

    for g in groups:
        group_id = g.get("group_id")
        encoding_str = g.get("face_encoding")
        if not group_id or not encoding_str:
            continue

        try:
            group_enc = np.array([float(x) for x in encoding_str.split(",")])
        except (ValueError, TypeError):
            continue

        dist = face_distance(selfie_encoding, group_enc)
        if dist < MATCH_THRESHOLD:
            matched_group_ids.append(group_id)
            confidence = 1.0 - dist
            if confidence > best_confidence:
                best_confidence = confidence

    logger.info(f"search-face: matched {len(matched_group_ids)} groups (best confidence={best_confidence:.3f})")

    return jsonify({
        "matched_group_ids": matched_group_ids,
        "confidence": round(best_confidence, 3) if matched_group_ids else 0,
    })


# -------------------------------------------------------------------
# ENDPOINT: /crop-faces
# Receives a batch of photos, detects faces, crops the face region
# with padding, and saves cropped images to the specified output paths.
# -------------------------------------------------------------------
@app.route("/crop-faces", methods=["POST"])
def crop_faces():
    data = request.get_json(silent=True)
    if not data or "photos" not in data:
        return jsonify({"error": "Missing photos array"}), 400

    photos = data["photos"]
    det_size = int(data.get("det_size", DET_SIZE))
    padding = float(data.get("padding", 0.3))

    logger.info(f"crop-faces: {len(photos)} photos, det_size={det_size}, padding={padding}")
    prepare_det_size(det_size)

    results = []
    for p in photos:
        file_path = p.get("file_path")
        photo_id = p.get("photo_id")
        output_path = p.get("output_path")
        if not file_path or not photo_id or not output_path:
            results.append({"photo_id": photo_id, "success": False, "reason": "missing_params"})
            continue

        img = load_image(file_path)
        if img is None:
            results.append({"photo_id": photo_id, "success": False, "reason": "load_failed"})
            continue

        face = get_best_face(img)
        if face is None:
            results.append({"photo_id": photo_id, "success": False, "reason": "no_face"})
            continue

        # Crop with padding
        x1, y1, x2, y2 = face.bbox
        w, h = x2 - x1, y2 - y1
        pad_x = w * padding
        pad_y = h * padding
        crop_x1 = max(0, int(x1 - pad_x))
        crop_y1 = max(0, int(y1 - pad_y))
        crop_x2 = min(img.shape[1], int(x2 + pad_x))
        crop_y2 = min(img.shape[0], int(y2 + pad_y))

        cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save as JPEG
        try:
            cv2.imwrite(output_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except Exception as e:
            logger.error(f"crop-faces: failed to save {output_path}: {e}")
            results.append({"photo_id": photo_id, "success": False, "reason": "save_failed"})
            continue

        bbox_str = f"{int(x1)},{int(y1)},{int(x2)},{int(y2)}"
        det_score = float(face.det_score) if hasattr(face, "det_score") else 0.5
        results.append({
            "photo_id": photo_id,
            "success": True,
            "bbox": bbox_str,
            "quality_score": round(det_score, 4),
        })

    cropped_count = sum(1 for r in results if r.get("success"))
    logger.info(f"crop-faces: {cropped_count}/{len(photos)} faces cropped successfully")

    return jsonify({"results": results})


# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------
if __name__ == "__main__":
    logger.info(f"Starting GetMyPhotos Face Recognition Server on {BIND_HOST}:{PORT}")
    app.run(host=BIND_HOST, port=PORT, debug=False)