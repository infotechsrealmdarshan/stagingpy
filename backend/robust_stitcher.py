import cv2
import numpy as np
import os
import traceback
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1 ─ OpenCV built-in Stitcher (best quality, handles lens distortion)
# ─────────────────────────────────────────────────────────────────────────────
def _try_opencv_stitcher(images: List[np.ndarray]) -> Optional[np.ndarray]:
    """Use OpenCV's own panorama Stitcher pipeline."""
    try:
        print("[stitch] Trying OpenCV Stitcher …")
        stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        stitcher.setRegistrationResol(0.6)
        stitcher.setSeamEstimationResol(0.1)
        stitcher.setCompositingResol(cv2.Stitcher_ORIG_RESOL)
        stitcher.setPanoConfidenceThresh(0.5)

        status, result = stitcher.stitch(images)
        if status == cv2.Stitcher_OK and result is not None:
            print("[stitch] OpenCV Stitcher succeeded ✓")
            return result
        else:
            codes = {
                cv2.Stitcher_ERR_NEED_MORE_IMGS: "need more images",
                cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "homography failed",
                cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "camera params failed",
            }
            reason = codes.get(status, f"status={status}")
            print(f"[stitch] OpenCV Stitcher failed: {reason}")
            return None
    except Exception as e:
        print(f"[stitch] OpenCV Stitcher exception: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2 ─ Homography-based stitching (SIFT + RANSAC)
# ─────────────────────────────────────────────────────────────────────────────
def _find_homography(img_a: np.ndarray, img_b: np.ndarray) -> Optional[np.ndarray]:
    """Find homography from img_b to img_a coordinate space."""
    try:
        sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)
        
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        
        # CLAHE to improve contrast in dark areas
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_a = clahe.apply(gray_a)
        gray_b = clahe.apply(gray_b)
        
        kp_a, des_a = sift.detectAndCompute(gray_a, None)
        kp_b, des_b = sift.detectAndCompute(gray_b, None)
        
        if des_a is None or des_b is None or len(des_a) < 8 or len(des_b) < 8:
            return None
        
        bf = cv2.BFMatcher()
        raw = bf.knnMatch(des_a, des_b, k=2)
        
        good = [m for m, n in raw if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return None
        
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        inliers = int(mask.sum()) if mask is not None else 0
        print(f"    [H] {len(good)} matches → {inliers} inliers")
        if inliers < 8:
            return None
        return H
    except Exception as e:
        print(f"    [H] Exception: {e}")
        return None


def _warp_and_blend(base: np.ndarray, next_img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Warp next_img onto base using homography with multiband blending."""
    h_b, w_b = base.shape[:2]
    h_n, w_n = next_img.shape[:2]
    
    # Compute destination canvas size
    corners_n = np.float32([[0, 0], [w_n, 0], [w_n, h_n], [0, h_n]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_n, H)
    all_corners = np.concatenate([
        [[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]],
        warped_corners.reshape(-1, 2)
    ])
    
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
    
    x_min = min(x_min, 0)
    y_min = min(y_min, 0)
    
    out_w = x_max - x_min
    out_h = y_max - y_min
    
    if out_w > 40000 or out_h > 15000:
        # Sanity guard — homography went haywire
        return base
    
    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    H_shifted = T @ H
    
    warped = cv2.warpPerspective(next_img, H_shifted, (out_w, out_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
    
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[-y_min:h_b + (-y_min), -x_min:w_b + (-x_min)] = base
    
    # Mask-based blending in overlap zone
    mask_b = np.zeros((out_h, out_w), dtype=np.uint8)
    mask_b[-y_min:h_b + (-y_min), -x_min:w_b + (-x_min)] = 255
    
    mask_n = np.zeros((out_h, out_w), dtype=np.uint8)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    mask_n[warped_gray > 0] = 255
    
    overlap = cv2.bitwise_and(mask_b, mask_n)
    only_n  = cv2.bitwise_and(mask_n, cv2.bitwise_not(mask_b))
    
    # Fill non-overlap area with warped image
    canvas[only_n > 0] = warped[only_n > 0]
    
    # Blend overlap zone with linear gradient
    overlap_coords = np.where(overlap > 0)
    if len(overlap_coords[0]) > 0:
        xs = overlap_coords[1]
        x_ov_min, x_ov_max = xs.min(), xs.max()
        ov_w = max(x_ov_max - x_ov_min, 1)
        
        alpha_map = np.zeros((out_h, out_w, 1), dtype=np.float32)
        for r, c in zip(overlap_coords[0], overlap_coords[1]):
            alpha_map[r, c, 0] = (c - x_ov_min) / ov_w  # left=base, right=warped
        
        b_f = canvas.astype(np.float32)
        w_f = warped.astype(np.float32)
        ov_mask = (overlap[:, :, np.newaxis] / 255.0).astype(np.float32)
        
        blended = b_f * (1.0 - alpha_map) + w_f * alpha_map
        canvas = (b_f * (1.0 - ov_mask) + blended * ov_mask).clip(0, 255).astype(np.uint8)
    
    return canvas


def _try_homography_stitcher(images: List[np.ndarray]) -> Optional[np.ndarray]:
    """Sequential homography-based stitching."""
    print("[stitch] Trying Homography stitcher …")
    try:
        # Normalize brightness first
        images = _normalize_brightness(images)
        
        base = images[0].copy()
        failed = 0
        
        for i in range(1, len(images)):
            H = _find_homography(base, images[i])
            if H is None:
                print(f"    [H] Pair {i}: no homography, using translation fallback")
                dx, dy = _phase_corr_offset(base, images[i])
                # Construct a pure translation matrix
                H = np.array([[1, 0, float(dx)],
                               [0, 1, float(dy)],
                               [0, 0, 1.0]], dtype=np.float64)
                failed += 1
            
            print(f"    [H] Stitching pair {i}/{len(images)-1}")
            base = _warp_and_blend(base, images[i], H)
        
        if failed == len(images) - 1:
            print("[stitch] All pairs used translation only — low confidence")
            return None  # Let translation stitcher handle this
        
        print("[stitch] Homography stitcher succeeded ✓")
        return base
    except Exception as e:
        print(f"[stitch] Homography stitcher exception: {e}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 3 ─ Simple translation mosaic (always works, less accurate warp)
# ─────────────────────────────────────────────────────────────────────────────
def _phase_corr_offset(img_l: np.ndarray, img_r: np.ndarray) -> Tuple[int, int]:
    """Estimate horizontal translation via phase correlation on overlap strips."""
    try:
        h_l, w_l = img_l.shape[:2]
        h_r, w_r = img_r.shape[:2]
        
        scan = int(min(w_l, w_r) * 0.6)
        strip_l = cv2.cvtColor(img_l[:, -scan:], cv2.COLOR_BGR2GRAY).astype(np.float32)
        strip_r = cv2.cvtColor(img_r[:, :scan],  cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Match heights
        h_min = min(strip_l.shape[0], strip_r.shape[0])
        strip_l = strip_l[:h_min, :]
        strip_r = strip_r[:h_min, :]
        
        (dx_p, dy_p), resp = cv2.phaseCorrelate(strip_l, strip_r)
        
        # Convert strip-relative dx to canvas dx
        dx_canvas = w_l - scan + int(dx_p)
        dx_canvas = int(np.clip(dx_canvas, w_l - int(w_r * 0.85), w_l - int(w_r * 0.05)))
        return dx_canvas, int(dy_p)
    except:
        return int(img_l.shape[1] * 0.65), 0


def _translation_mosaic(images: List[np.ndarray]) -> Optional[np.ndarray]:
    """Simple left-to-right translation stitching — always produces output."""
    print("[stitch] Using translation mosaic (fallback) …")
    try:
        images = _normalize_brightness(images)
        
        pano = images[0].copy()
        
        for i in range(1, len(images)):
            img = images[i].copy()
            dx, dy = _phase_corr_offset(pano, img)
            
            h_p, w_p = pano.shape[:2]
            h_i, w_i = img.shape[:2]
            
            # Vertical padding for dy
            pad_top_p = max(0, -dy)
            pad_top_i = max(0,  dy)
            pad_bot_p = max(0,  dy)
            pad_bot_i = max(0, -dy)
            
            pano = cv2.copyMakeBorder(pano, pad_top_p, pad_bot_p, 0, 0, cv2.BORDER_CONSTANT)
            img  = cv2.copyMakeBorder(img,  pad_top_i, pad_bot_i, 0, 0, cv2.BORDER_CONSTANT)
            
            h_p, w_p = pano.shape[:2]
            h_i, w_i = img.shape[:2]
            h_c = min(h_p, h_i)
            pano = pano[:h_c, :]
            img  = img[:h_c, :]
            
            ov = w_p - dx
            ov = max(10, min(ov, w_i - 10))
            new_w = w_p + w_i - ov
            
            res = np.zeros((h_c, new_w, 3), dtype=np.uint8)
            res[:, :w_p] = pano
            
            # Sigmoid blend in overlap zone
            if ov > 10:
                t = np.linspace(0, 1, ov)
                ramp = 1.0 / (1.0 + np.exp(-12 * (t - 0.5)))
                alpha = np.tile(ramp, (h_c, 1))[:, :, np.newaxis].astype(np.float32)
                
                roi_l = pano[:, w_p-ov:w_p].astype(np.float32)
                roi_r = img[:, :ov].astype(np.float32)
                blended = (roi_l * (1 - alpha) + roi_r * alpha).clip(0, 255).astype(np.uint8)
                res[:, w_p-ov:w_p] = blended
                res[:, w_p:] = img[:, ov:]
            else:
                res[:, w_p:] = img
            
            pano = res
            print(f"    [trans] Stitched {i}/{len(images)-1}: dx={dx}, dy={dy}, ov={ov}px")
        
        print("[stitch] Translation mosaic succeeded ✓")
        return pano
    except Exception as e:
        print(f"[stitch] Translation mosaic exception: {e}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_brightness(images: List[np.ndarray]) -> List[np.ndarray]:
    """Equalize brightness and color temperature across all images."""
    if not images:
        return images
    
    labs = []
    for img in images:
        small = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        labs.append({'L': float(np.mean(L)), 'a': float(np.mean(a)), 'b': float(np.mean(b))})
    
    tL = float(np.median([s['L'] for s in labs]))
    ta = float(np.median([s['a'] for s in labs]))
    tb = float(np.median([s['b'] for s in labs]))
    
    result = []
    for img, s in zip(images, labs):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, a, b = cv2.split(lab)
        gain = np.clip(tL / (s['L'] + 1e-6), 0.65, 1.5)
        L = np.clip(L * gain, 0, 255)
        a = np.clip(a + (ta - s['a']), 0, 255)
        b = np.clip(b + (tb - s['b']), 0, 255)
        lab = cv2.merge([L, a, b]).clip(0, 255).astype(np.uint8)
        result.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
    return result


def _trim_borders(image: np.ndarray) -> np.ndarray:
    """Crop black/empty borders. Never returns None."""
    if image is None:
        return image
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray > 8).astype(np.uint8)
        coords = cv2.findNonZero(mask)
        if coords is None:
            return image
        x, y, w, h = cv2.boundingRect(coords)
        # Add a 2px safety margin
        x = max(0, x - 2); y = max(0, y - 2)
        w = min(image.shape[1] - x, w + 4)
        h = min(image.shape[0] - y, h + 4)
        return image[y:y+h, x:x+w]
    except:
        return image


def _post_process(image: np.ndarray) -> np.ndarray:
    """Gentle HDR-like post processing."""
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12, 12))
        L = clahe.apply(L)
        L_f = L.astype(np.float32) / 255.0
        L_f = np.power(L_f, 0.88)
        L = (L_f * 255).clip(0, 255).astype(np.uint8)
        image = cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)
        return image
    except:
        return image


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def stitch_images_robustly(image_paths: List[str], results_dir: str) -> List[str]:
    print(f"[stitch] Loading {len(image_paths)} images …")
    
    images = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  [!] Could not read: {p}")
            continue
        # Resize so largest dimension ≤ 1400px  (speed + memory)
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1400:
            scale = 1400 / max_dim
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        images.append(img)
    
    if len(images) < 2:
        print("[stitch] Not enough valid images — need at least 2")
        return []
    
    print(f"[stitch] Processing {len(images)} images …")
    
    result = None
    
    # ── Strategy 1: OpenCV built-in
    result = _try_opencv_stitcher(images)
    
    # ── Strategy 2: Homography-based
    if result is None:
        result = _try_homography_stitcher(images)
    
    # ── Strategy 3: Translation mosaic (guaranteed output)
    if result is None:
        result = _translation_mosaic(images)
    
    if result is None:
        print("[stitch] All strategies failed.")
        return []
    
    # Post-process and trim
    result = _post_process(result)
    result = _trim_borders(result)
    
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "panorama.jpg")
    cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[stitch] Saved panorama → {out_path}  ({result.shape[1]}×{result.shape[0]}px)")
    return [out_path]
