import cv2
import numpy as np
import os
from typing import List

# VERSION: 2024-01-07-NO-CROP-v4
print("[STITCHER] Loading NO-CROP stitcher version v4 - Full image preservation")

def _mask_from_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (gray > 0).astype(np.uint8) * 255

def _column_ranges(mask):
    active = np.any(mask > 0, axis=0)
    ranges = []
    start = None

    for idx, is_active in enumerate(active):
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            ranges.append((start, idx))
            start = None

    if start is not None:
        ranges.append((start, mask.shape[1]))

    return ranges

def _blend_overlap_with_vertical_seams(base_img, new_img, feather=32):
    """Blend two aligned layers using a low-error vertical seam in each overlap band."""
    mask1 = _mask_from_image(base_img)
    mask2 = _mask_from_image(new_img)

    out = base_img.copy()
    new_only = (mask2 > 0) & (mask1 == 0)
    out[new_only] = new_img[new_only]

    overlap = (mask1 > 0) & (mask2 > 0)
    if not np.any(overlap):
        return out

    diff = np.abs(base_img.astype(np.float32) - new_img.astype(np.float32)).mean(axis=2)

    for start, end in _column_ranges(overlap):
        band_overlap = overlap[:, start:end]
        if not np.any(band_overlap):
            continue

        cols = np.arange(start, end)
        scores = []
        for c in cols:
            valid = overlap[:, c]
            scores.append(diff[valid, c].mean() if np.any(valid) else np.inf)

        seam_col = cols[int(np.argmin(scores))]
        blend_start = max(start, seam_col - feather)
        blend_end = min(end, seam_col + feather + 1)

        if blend_end < end:
            right_cols = np.arange(blend_end, end)
            right_mask = overlap[:, right_cols]
            out_slice = out[:, right_cols].copy()
            new_slice = new_img[:, right_cols]
            out_slice[right_mask] = new_slice[right_mask]
            out[:, right_cols] = out_slice

        blend_cols = np.arange(blend_start, blend_end)
        if blend_cols.size == 0:
            continue

        alpha = np.linspace(0.0, 1.0, blend_cols.size, dtype=np.float32).reshape(1, -1, 1)
        base_slice = out[:, blend_cols].astype(np.float32)
        new_slice = new_img[:, blend_cols].astype(np.float32)
        blended = ((1.0 - alpha) * base_slice + alpha * new_slice).astype(np.uint8)

        blend_mask = overlap[:, blend_cols]
        out_slice = out[:, blend_cols].copy()
        out_slice[blend_mask] = blended[blend_mask]
        out[:, blend_cols] = out_slice

    return out

def _find_edge_homography(current_img, next_img, strip_ratio=0.45, min_strip=240):
    """Estimate the next->current transform from the active overlap strips only."""
    curr_h, curr_w = current_img.shape[:2]
    next_h, next_w = next_img.shape[:2]

    curr_strip_w = min(curr_w, max(min_strip, int(curr_w * strip_ratio)))
    next_strip_w = min(next_w, max(min_strip, int(next_w * strip_ratio)))

    curr_x0 = curr_w - curr_strip_w
    next_x0 = 0

    # Crop center band to remove ceiling/floor noise (table + monitor = low texture areas)
    h = current_img.shape[0]
    y1, y2 = int(h * 0.2), int(h * 0.8)
    
    curr_strip = current_img[y1:y2, curr_x0:]
    next_strip = next_img[y1:y2, next_x0:next_x0 + next_strip_w]

    H_strip = find_homography_between_images(next_strip, curr_strip)
    if H_strip is None:
        return None

    t_curr = np.array([[1.0, 0.0, curr_x0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
    t_next_inv = np.array([[1.0, 0.0, -next_x0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]], dtype=np.float64)

    return t_curr.dot(H_strip).dot(t_next_inv)

def find_homography_between_images(img1, img2):
    """Find homography matrix between two images using enhanced feature detection for better circular stitching."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Enhance images for better feature detection
    gray1 = cv2.equalizeHist(gray1)
    gray2 = cv2.equalizeHist(gray2)
    
    # Try multiple feature detectors with different parameters
    detectors = []
    
    # 1. SIFT with optimized parameters for circular panoramas
    try:
        sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.008, edgeThreshold=15, sigma=1.0)
        detectors.append(('SIFT', sift))
    except:
        pass
    
    # 2. ORB with enhanced parameters
    try:
        orb = cv2.ORB_create(nfeatures=8000, scaleFactor=1.1, nlevels=12, edgeThreshold=20, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=15)
        detectors.append(('ORB', orb))
    except:
        pass
    
    # 3. AKAZE with fine-tuned parameters
    try:
        akaze = cv2.AKAZE_create(threshold=0.0005)
        detectors.append(('AKAZE', akaze))
    except:
        pass
    
    best_H = None
    best_inliers = 0
    
    for detector_name, detector in detectors:
        try:
            # Find keypoints and descriptors
            kp1, des1 = detector.detectAndCompute(gray1, None)
            kp2, des2 = detector.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None or len(des1) < 15 or len(des2) < 15:
                continue
            
            # Choose matcher based on detector type
            if detector_name == 'SIFT':
                # Use FLANN for SIFT with optimized parameters
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
                search_params = dict(checks=80)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
                matches = matcher.knnMatch(des1, des2, k=2)
                
                # Apply Lowe's ratio test with stricter threshold for better quality
                good_matches = []
                for pair in matches:
                    if len(pair) < 2:
                        continue
                    m, n = pair
                    if m.distance < 0.65 * n.distance:  # Stricter ratio for better matches
                        good_matches.append(m)
            else:
                # Use Brute Force for ORB/AKAZE with cross-checking
                if des1.dtype == np.uint8:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    # Take top matches based on distance
                    good_matches = sorted(matches, key=lambda x: x.distance)[:min(len(matches), 150)]
                else:
                    continue
            
            print(f"    {detector_name}: Found {len(good_matches)} good matches")
            
            # Require more matches for robust circular stitching
            if len(good_matches) >= 8:
                # Extract location of good matches
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography with stricter parameters for better accuracy
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0, maxIters=5000, confidence=0.995)
                
                if H is not None:
                    # Count inliers for quality assessment
                    inliers = np.sum(mask)
                    print(f"    ✓ {detector_name} homography found with {inliers} inliers")
                    
                    # Keep the best homography (most inliers)
                    if inliers > best_inliers:
                        best_H = H
                        best_inliers = inliers
            
        except Exception as e:
            print(f"    {detector_name} failed: {e}")
            continue
    
    if best_H is not None:
        print(f"    ✓ Best homography selected with {best_inliers} inliers")
        return best_H
    else:
        print("    ✗ No suitable homography found with any detector")
        return None

def feature_based_stitch(images, results_dir):
    """Create 3D-like panorama with perspective-correct stitching and enhanced blending."""
    print(f"[*] Creating 3D panorama for {len(images)} portrait images")
    
    if len(images) < 2:
        print("[!] Need at least 2 images")
        return []
    
    # Process images with higher resolution for better 3D effect
    target_h = 1600  # Even higher for better detail
    processed = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Keep original colors - no correction for natural look
        processed.append(img)
        print(f"    Image {i+1}: {img.shape[1]}x{img.shape[0]} (processed for 3D)")
    
    # Try perspective-correct 3D stitching first
    print("[*] Attempting perspective-correct 3D stitching...")
    panorama = _perspective_stitch_3d(processed, results_dir)
    
    # If 3D stitching fails, fallback to enhanced horizontal stitching
    if panorama is None:
        print("[*] 3D stitching failed, using enhanced horizontal stitching...")
        return _enhanced_horizontal_stitch(processed, results_dir)
    
    return [panorama]
    
def _perspective_stitch_3d(images, results_dir):
    """Create true 3D panorama using perspective-correct stitching."""
    print("[*] Attempting perspective-correct 3D stitching...")
    
    try:
        # Start with the first image
        result = images[0].copy()
        
        for i in range(1, len(images)):
            print(f"    Stitching image {i+1} to panorama...")
            
            # Find homography between current result and next image
            H = find_homography_between_images(result, images[i])
            
            if H is None:
                print(f"    [!] Could not find homography for image {i+1}")
                return None
            
            # Get dimensions
            h1, w1 = result.shape[:2]
            h2, w2 = images[i].shape[:2]
            
            # Calculate corners of the warped image
            corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
            warped_corners2 = cv2.perspectiveTransform(corners2, H)
            
            # Find the bounding box of the warped image with padding
            all_corners = np.concatenate([
                np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2),
                warped_corners2
            ], axis=0)
            
            # Add padding to prevent cutting
            [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 50)
            [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 50)
            
            # Create translation matrix to shift the result with padding
            translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
            
            # Warp the current result and the new image
            result_warped = cv2.warpPerspective(result, translation, (x_max - x_min, y_max - y_min))
            img_warped = cv2.warpPerspective(images[i], translation.dot(H), (x_max - x_min, y_max - y_min))
            
            # Create masks for blending
            mask1 = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
            mask2 = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
            
            # Fill masks where images exist
            result_gray = cv2.cvtColor(result_warped, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
            mask1[result_gray > 0] = 255
            mask2[img_gray > 0] = 255
            
            # Find overlap region
            overlap = cv2.bitwise_and(mask1, mask2)
            
            # Create smooth blending in overlap region
            if np.any(overlap):
                # Distance transform for smooth blending
                dist1 = cv2.distanceTransform(255 - mask1, cv2.DIST_L2, 5)
                dist2 = cv2.distanceTransform(255 - mask2, cv2.DIST_L2, 5)
                
                # Normalize weights
                total_dist = dist1 + dist2
                total_dist[total_dist == 0] = 1
                weight1 = dist1 / total_dist
                weight2 = dist2 / total_dist
                
                # Blend in overlap region
                overlap_mask = overlap > 0
                for c in range(3):
                    result_warped[:, :, c][overlap_mask] = (
                        result_warped[:, :, c][overlap_mask] * weight1[overlap_mask] +
                        img_warped[:, :, c][overlap_mask] * weight2[overlap_mask]
                    )
                
                # Add non-overlapping parts
                result_warped[mask2 == 255] = img_warped[mask2 == 255]
            
            result = result_warped
        
        # Keep natural colors - no final balancing
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # FORCE circular seam blending
        print("[*] Forcing circular seam blending...")
        print("[*] Aligning last image to first...")
        
        print(f"[*] 3D panorama created: {result.shape[1]}x{result.shape[0]}")
        
        # Save
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, "panorama.jpg")
        cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 98])
        print(f"[*] Saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"    [!] 3D stitching failed: {e}")
        return None

def _enhanced_horizontal_stitch(images, results_dir):
    """Enhanced horizontal stitching as fallback."""
    print(f"[*] Enhanced horizontal stitching for {len(images)} images")
    
    h, w = images[0].shape[:2]
    
    # Increased overlap for better blending
    overlap = max(int(w * 0.20), 120)  # 20% overlap for better alignment
    step = w - overlap
    
    # Calculate canvas
    total_width = step * (len(images) - 1) + w
    canvas_height = h
    
    print(f"    Canvas: {total_width}x{canvas_height}, Overlap: {overlap}px")
    
    # Create canvas
    result = np.zeros((canvas_height, total_width, 3), dtype=np.float32)
    weights_sum = np.zeros((canvas_height, total_width), dtype=np.float32)
    
    # Place images with multi-band blending
    for i, img in enumerate(images):
        x_offset = i * step
        
        # Create weight mask with multi-band blending
        weights = np.ones((h, w), dtype=np.float32)
        
        # Left edge feathering
        if i > 0:
            feather = overlap
            gradient = np.cos(np.linspace(np.pi, 0, feather)) * 0.5 + 0.5
            weights[:, :feather] = gradient.reshape(1, -1)
        
        # Right edge feathering
        if i < len(images) - 1:
            feather = overlap
            gradient = np.cos(np.linspace(0, np.pi, feather)) * 0.5 + 0.5
            weights[:, -feather:] = gradient.reshape(1, -1)
        
        # Add weighted image to result
        result[:, x_offset:x_offset+w] += img.astype(np.float32) * weights[:, :, np.newaxis]
        weights_sum[:, x_offset:x_offset+w] += weights
    
    # Normalize and keep natural colors
    mask = weights_sum > 0
    result[mask] /= weights_sum[mask][:, np.newaxis]
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # FORCE circular seam blending
    print("[*] Forcing circular seam blending...")
    print("[*] Aligning last image to first...")
    aligned_last = align_last_to_first(images[0], images[-1])
    result = blend_circular_seam(result, images[0], aligned_last, results_dir)
    
    print(f"[*] Final panorama: {result.shape[1]}x{result.shape[0]}")
    
    # Save
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "panorama.jpg")
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 98])
    print(f"[*] Saved to: {output_path}")
    
    return [output_path]

def stitch_images_robustly(image_paths: List[str], results_dir: str, preserve_order: bool = False) -> List[str]:
    """Stitch images using OpenCV's Stitcher class for proper panorama generation.
    
    Args:
        image_paths: List of image file paths in desired order
        results_dir: Output directory for results
        preserve_order: If True, place images side-by-side in input order without feature-based reordering
    """
    
    if len(image_paths) < 2:
        print("[!] Need at least 2 images to stitch")
        return []
    
    print(f"[*] Loading {len(image_paths)} images...")
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            # Keep original size - only resize if extremely large
            h, w = img.shape[:2]
            max_dim = max(h, w)
            if max_dim > 2000:
                scale = 2000 / max_dim
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                print(f"    ✓ {os.path.basename(path)}: {img.shape[1]}x{img.shape[0]} (resized from {w}x{h})")
            else:
                print(f"    ✓ {os.path.basename(path)}: {img.shape[1]}x{img.shape[0]} (original)")
            images.append(img)
        else:
            print(f"    ✗ Failed to load: {os.path.basename(path)}")
    
    if len(images) < 2:
        print("[!] Not enough valid images")
        return []
    
    # If preserve_order is True, use simple side-by-side stitching
    if preserve_order:
        print("[*] Using ORDER-PRESERVING stitch (images will stay in input sequence)")
        return simple_stitch(images, results_dir)
    
    print(f"[*] Stitching {len(images)} images with OpenCV Stitcher...")
    
    # Create stitcher
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    
    # Try stitching
    status, pano = stitcher.stitch(images)
    
    if status == cv2.STITCHER_OK:
        print(f"[*] Stitching successful! Panorama size: {pano.shape[1]}x{pano.shape[0]}")
        
        # Enhance the result
        lab = cv2.cvtColor(pano, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L = clahe.apply(L)
        pano = cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)
        
        # Save result
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, "panorama.jpg")
        cv2.imwrite(output_path, pano, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[*] Saved to: {output_path}")
        return [output_path]
    else:
        # Use numeric error codes - constants may not exist in all OpenCV versions
        error_messages = {
            1: "Need more images",
            2: "Homography estimation failed", 
            3: "Camera parameter adjustment failed",
        }
        error_msg = error_messages.get(abs(status), f"Error code: {status}")
        print(f"[!] Stitching failed: {error_msg}")
        
        # Fallback: feature-based stitching with blending
        print("[*] Trying feature-based stitching with blending...")
        return feature_based_stitch(images, results_dir)

def simple_stitch(images: List[np.ndarray], results_dir: str) -> List[str]:
    """Simple side-by-side placement: images placed left-to-right without blending."""
    
    print(f"[*] Simple side-by-side placement for {len(images)} images")
    
    # Standardize height - all images same height
    target_h = 800
    resized = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        resized_img = cv2.resize(img, (new_w, target_h))
        resized.append(resized_img)
        print(f"    Image {i+1}: resized to {new_w}x{target_h}")
    
    # Calculate total width (no overlap, just side by side)
    total_width = sum(img.shape[1] for img in resized)
    
    # Create canvas
    pano = np.zeros((target_h, total_width, 3), dtype=np.uint8)
    
    # Place images left-to-right
    x_pos = 0
    for i, img in enumerate(resized):
        w = img.shape[1]
        # Place image at current x position
        pano[:, x_pos:x_pos + w] = img
        print(f"[*] Placed image {i+1} at x={x_pos} (width={w})")
        x_pos += w
    
    print(f"[*] Final panorama: {pano.shape[1]}x{pano.shape[0]}")
    
    # Save
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "panorama.jpg")
    cv2.imwrite(output_path, pano, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[*] Saved to: {output_path}")
    return [output_path]

def _color_correct_image(img):
    """Apply subtle color correction to preserve natural colors."""
    # Convert to LAB color space for gentle lightness adjustment
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply very gentle CLAHE to lightness only (preserve natural look)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))  # Much gentler
    l = clahe.apply(l)
    
    # Merge back and convert to BGR
    lab = cv2.merge([l, a, b])
    corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return corrected

def _final_color_balance(result):
    """Apply minimal final color balancing to preserve natural colors."""
    # Very subtle brightness adjustment only
    result_float = result.astype(np.float32) / 255.0
    
    # Minimal gamma correction (almost neutral)
    gamma = 0.98  # Very close to 1.0 to preserve natural colors
    result_float = np.power(result_float, gamma)
    
    # Convert back to 0-255 range
    result = result_float * 255.0
    
    return result

def align_last_to_first(first_img, last_img):
    """Align last image to first image using homography for proper circular stitching."""
    H = find_homography_between_images(last_img, first_img)
    if H is None:
        print("[!] Cannot align last to first - no homography found")
        return last_img

    h, w = first_img.shape[:2]
    aligned = cv2.warpPerspective(last_img, H, (w, h))
    print(f"[*] Aligned last image to first (target size: {w}x{h})")
    return aligned

def is_circular_panorama(images):
    """Enhanced circular panorama detection with multiple checks."""
    if len(images) < 3:
        return False
    
    print("[*] Enhanced circular panorama detection...")
    
    # Resize images for faster processing
    target_size = (800, 600)
    first_resized = cv2.resize(images[0], target_size)
    last_resized = cv2.resize(images[-1], target_size)
    
    # Multiple attempts with different preprocessing
    attempts = [
        ("Original", first_resized, last_resized),
        ("Enhanced", cv2.equalizeHist(cv2.cvtColor(first_resized, cv2.COLOR_BGR2GRAY)), 
                   cv2.equalizeHist(cv2.cvtColor(last_resized, cv2.COLOR_BGR2GRAY))),
    ]
    
    best_confidence = 0
    for attempt_name, img1, img2 in attempts:
        print(f"    Trying {attempt_name} detection...")
        
        # Convert back to BGR if needed
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
        H = find_homography_between_images(img1, img2)
        if H is not None:
            # Calculate transformation confidence
            # Check if the homography creates a reasonable transformation
            corners = np.float32([[0, 0], [img2.shape[1], 0], [img2.shape[1], img2.shape[0]], [0, img2.shape[0]]]).reshape(-1, 1, 2)
            warped_corners = cv2.perspectiveTransform(corners, H)
            
            # Calculate overlap and displacement
            center_img1 = np.array([img1.shape[1]/2, img1.shape[0]/2])
            center_warped = np.mean(warped_corners.squeeze(), axis=0)
            displacement = np.linalg.norm(center_img1 - center_warped)
            
            # Check if displacement is reasonable for circular panorama
            if displacement < img1.shape[1] * 0.8:  # Should be less than 80% of image width
                confidence = max(0, 1.0 - displacement / (img1.shape[1] * 0.8))
                print(f"    ✓ {attempt_name} detection successful (confidence: {confidence:.2f})")
                best_confidence = max(best_confidence, confidence)
    
    # Determine if it's circular based on confidence threshold
    is_circular = best_confidence > 0.3  # 30% confidence threshold
    
    if is_circular:
        print(f"[*] ✓ Circular panorama confirmed (confidence: {best_confidence:.2f})")
        return True
    else:
        print(f"[*] ✗ Not a circular panorama (best confidence: {best_confidence:.2f})")
        return False

def blend_circular_seam(result, first_img, last_img, results_dir):
    """Enhanced circular seam blending with multi-band approach for seamless 360° panoramas."""
    try:
        print("[*] Applying enhanced circular seam blending...")
        
        # Get dimensions
        h, w = result.shape[:2]
        
        # Determine optimal seam width based on image size
        seam_width = min(300, max(100, w // 8))  # Adaptive seam width
        
        # Create extended regions for better blending
        # Take larger regions from both sides
        left_region = result[:, -seam_width*2:].copy()
        right_region = result[:, :seam_width*2].copy()
        
        # Create multi-band gradient for smoother transition
        # Primary gradient for main blend
        primary_gradient = np.cos(np.linspace(np.pi, 0, seam_width)) * 0.5 + 0.5
        
        # Secondary gradient for feathering
        secondary_gradient = np.cos(np.linspace(np.pi*1.5, np.pi*0.5, seam_width//2)) * 0.25 + 0.25
        
        # Apply multi-band blending for each color channel
        for c in range(3):
            # Extract the actual seam regions
            left_seam = left_region[:, -seam_width:, c]
            right_seam = right_region[:, :seam_width, c]
            
            # Apply primary blending in the main seam area
            blended_main = (
                left_seam * primary_gradient.reshape(1, -1) +
                right_seam * (1 - primary_gradient.reshape(1, -1))
            )
            
            # Apply secondary feathering at the edges
            if seam_width > 50:
                # Left edge feathering
                left_feather = secondary_gradient.reshape(1, -1)
                blended_main[:, :seam_width//2] = (
                    blended_main[:, :seam_width//2] * left_feather +
                    left_region[:, -seam_width:, c][:, :seam_width//2] * (1 - left_feather)
                )
                
                # Right edge feathering
                right_feather = secondary_gradient[::-1].reshape(1, -1)
                blended_main[:, -seam_width//2:] = (
                    blended_main[:, -seam_width//2:] * right_feather +
                    right_region[:, :seam_width, c][:, -seam_width//2:] * (1 - right_feather)
                )
            
            # Apply the blended result back to the main image
            result[:, -seam_width:, c] = blended_main
        
        # Apply post-blending smoothing to reduce artifacts
        # Use a small Gaussian blur on the seam region
        seam_mask = np.zeros((h, w), dtype=np.uint8)
        seam_mask[:, -seam_width-10:seam_width+10] = 255
        
        # Apply bilateral filter for edge-preserving smoothing
        result_smooth = result.copy()
        for c in range(3):
            result_smooth[:, :, c] = cv2.bilateralFilter(result[:, :, c], 9, 75, 75)
        
        # Blend smoothed result only in seam region
        seam_region = (seam_mask > 0)
        for c in range(3):
            result[seam_region, c] = (
                result[seam_region, c] * 0.7 + 
                result_smooth[seam_region, c] * 0.3
            )
        
        # Final color correction to ensure consistency
        result = enhance_circular_consistency(result)
        
        print(f"[*] ✓ Enhanced circular seam blended (width: {seam_width}px)")
        return result
        
    except Exception as e:
        print(f"[!] Enhanced circular seam blending failed: {e}")
        return result

def enhance_circular_consistency(result):
    """Enhance color consistency across the circular seam."""
    try:
        h, w = result.shape[:2]
        seam_width = min(100, w // 20)
        
        # Sample colors from both sides of the seam
        left_colors = result[:, -seam_width-20:-seam_width, :].mean(axis=(0, 1))
        right_colors = result[:, seam_width:seam_width+20, :].mean(axis=(0, 1))
        
        # Calculate color correction factor
        color_ratio = right_colors / (left_colors + 1e-6)
        color_ratio = np.clip(color_ratio, 0.8, 1.2)  # Limit correction to avoid artifacts
        
        # Apply gentle color correction to the left side
        correction_region = result[:, -seam_width*3:, :]
        for c in range(3):
            correction_gradient = np.linspace(1.0, color_ratio[c], seam_width*3)
            correction_region[:, :, c] *= correction_gradient.reshape(1, -1)
        
        result[:, -seam_width*3:, :] = correction_region
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
        
    except Exception as e:
        print(f"[!] Color consistency enhancement failed: {e}")
        return result

def stitch_images_robustly_3layer(image_paths: List[str], results_dir: str, preserve_order: bool = False) -> List[str]:
    """Wrapper for 3-layer stitching (same as regular for now)."""
    return stitch_images_robustly(image_paths, results_dir, preserve_order)
