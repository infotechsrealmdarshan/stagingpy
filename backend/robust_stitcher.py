import cv2
import numpy as np
import os
import re
from typing import List, Tuple, Optional


class RobustStitcher:
    def __init__(self, min_matches: int = 8, ratio: float = 0.72, reproj: float = 3.5):
        self.min_matches = min_matches
        self.ratio_test = ratio
        self.reproj_thresh = reproj
        # SIFT for robust but potentially repetitive points
        self.sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.015)
        self.matcher = cv2.BFMatcher()

    def get_features(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use stronger CLAHE for better dark-area details
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        kp, des = self.sift.detectAndCompute(gray, None)
        return kp, des

    def match_features(self, des1, des2):
        if des1 is None or des2 is None or len(des1) < self.min_matches or len(des2) < self.min_matches:
            return []
        raw_matches = self.matcher.knnMatch(des1, des2, k=2)
        matches = [m for m, n in raw_matches if m.distance < self.ratio_test * n.distance]
        return matches

    def verify_translation(self, img_l: np.ndarray, img_r: np.ndarray, dx: int, dy: int) -> float:
        """Verify a translation hypothesis using template matching on a strip."""
        try:
            h_l, w_l = img_l.shape[:2]
            h_r, w_r = img_r.shape[:2]
            
            # The overlap region in img_l is [dx, w_l]
            # The overlap region in img_r is [0, w_l - dx]
            ov_w = w_l - dx
            if ov_w < 20 or ov_w > w_r: return 0.0
            
            # Pad images for dy
            h_common = min(h_l, h_r)
            strip_l = img_l[h_common//4:h_common*3//4, dx:dx+ov_w]
            strip_r = img_r[h_common//4:h_common*3//4, :ov_w]
            
            # Shift strip_r by dy
            if abs(dy) > h_common//8: return 0.0
            
            # Match gray versions
            g_l = cv2.cvtColor(strip_l, cv2.COLOR_BGR2GRAY)
            g_r = cv2.cvtColor(strip_r, cv2.COLOR_BGR2GRAY)
            
            if dy != 0:
                # Vertical shift for verification
                M = np.float32([[1, 0, 0], [0, 1, -dy]])
                g_r = cv2.warpAffine(g_r, M, (g_r.shape[1], g_r.shape[0]))
            
            res = cv2.matchTemplate(g_l, g_r, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            return max_val
        except:
            return 0.0

    def find_translation(self, img_l: np.ndarray, img_r: np.ndarray, expected_ov_pct: float = 0.35) -> Tuple[Optional[int], Optional[int], float]:
        """Robuster translation finding with geometric constraints and template validation."""
        h_l, w_l = img_l.shape[:2]
        h_r, w_r = img_r.shape[:2]
        
        # Scan a slightly larger area than expected
        scan_w = int(w_r * 0.90)
        strip_l = img_l[:, -scan_w:]
        strip_r = img_r[:, :scan_w]
        
        kp_l, des_l = self.get_features(strip_l)
        kp_r, des_r = self.get_features(strip_r)
        
        matches = self.match_features(des_l, des_r)
        if len(matches) < 4:
            return self.fallback_phase_corr(img_l, img_r, expected_ov_pct)
            
        dxs = [ (kp_l[m.queryIdx].pt[0] + (w_l - scan_w)) - kp_r[m.trainIdx].pt[0] for m in matches ]
        dys = [ kp_l[m.queryIdx].pt[1] - kp_r[m.trainIdx].pt[1] for m in matches ]
        
        # Tighter constraints for sequential photos
        # Overlap is typically 15% to 55%
        min_dx = w_l - int(w_r * 0.70)
        max_dx = w_l - int(w_r * 0.08)
        
        candidates = []
        for d_x, d_y in zip(dxs, dys):
            if min_dx <= d_x <= max_dx:
                candidates.append((d_x, d_y))
        
        if not candidates:
            return self.fallback_phase_corr(img_l, img_r, expected_ov_pct)
            
        # Group candidates into DX clusters
        best_dx, best_dy, best_score = None, None, -1.0
        
        # Simple clustering around common DX values
        for d_x, d_y in candidates:
            # Check this hypothesis with template alignment
            score = self.verify_translation(img_l, img_r, int(d_x), int(d_y))
            if score > best_score:
                best_score = score
                best_dx = int(d_x)
                best_dy = int(d_y)
                
        if best_score > 0.45:
            return best_dx, best_dy, best_score
            
        return self.fallback_phase_corr(img_l, img_r, expected_ov_pct)

    def fallback_phase_corr(self, img_l, img_r, expected_ov_pct) -> Tuple[Optional[int], Optional[int], float]:
        try:
            h_l, w_l = img_l.shape[:2]
            h_r, w_r = img_r.shape[:2]
            ov_search = int(w_r * 0.6)
            sl_gray = cv2.cvtColor(img_l[:, -ov_search:], cv2.COLOR_BGR2GRAY).astype(np.float32)
            sr_gray = cv2.cvtColor(img_r[:, :ov_search], cv2.COLOR_BGR2GRAY).astype(np.float32)
            (dx_p, dy_p), resp = cv2.phaseCorrelate(sl_gray, sr_gray)
            if resp > 0.05:
                calc_dx = w_l - ov_search + dx_p
                # Validate the fallback too
                score = self.verify_translation(img_l, img_r, int(calc_dx), int(dy_p))
                if score > 0.35:
                    return int(calc_dx), int(dy_p), score
        except: pass
        return None, None, 0.0

    @staticmethod
    def global_brightness_normalise(images: List[np.ndarray]) -> List[np.ndarray]:
        """Normalize brightness and balance color temperature (tint) in LAB space."""
        if not images: return images
        
        stats = []
        for img in images:
            # Use a slightly downsampled version for speed in stats calculation
            small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
            L, a, b = cv2.split(lab)
            stats.append({
                'L_avg': np.mean(L),
                'a_avg': np.mean(a),
                'b_avg': np.mean(b)
            })
            
        # Target based on medians to avoid outliers (e.g. bright lamps or dark corners)
        target_L = float(np.median([s['L_avg'] for s in stats]))
        target_a = float(np.median([s['a_avg'] for s in stats]))
        target_b = float(np.median([s['b_avg'] for s in stats]))
        
        res = []
        for img, s in zip(images, stats):
            # Process in float32 for precision
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            L, a, b = cv2.split(lab)
            
            # Exposure correction (L channel)
            gain = target_L / (s['L_avg'] + 1e-6)
            gain = np.clip(gain, 0.65, 1.5)
            L *= gain
            
            # Color balancing (neutralizing a/b shift towards global median)
            # This matches color temperature and tint
            a += (target_a - s['a_avg'])
            b += (target_b - s['b_avg'])
            
            # Clip and convert back
            lab = cv2.merge([L, a, b])
            lab = np.clip(lab, 0, 255).astype(np.uint8)
            res.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
        return res

    def manual_mosaic_fallback(self, images: List[np.ndarray], overlap_pct: float = 0.35, close_loop: bool = True) -> np.ndarray:
        if not images: return None
        # Pre-process global color/brightness
        images = self.global_brightness_normalise(images)
        
        pano = images[0].copy()
        first_ref = images[0].copy()
        
        for i in range(1, len(images)):
            img = images[i].copy()
            print(f"[seam {i}/{len(images)-1}]")
            
            # Find geometric alignment
            dx, dy, score = self.find_translation(pano, img, expected_ov_pct=overlap_pct)
            
            if dx is not None and score > 0.30:
                ov = pano.shape[1] - dx
                print(f"    [align] Match score={score:.2f}, overlap={ov}px, dy={dy}")
            else:
                ov = int(img.shape[1] * overlap_pct)
                dy = 0
                print(f"    [align] Low confidence, using fixed overlap {ov}px")
            
            # 1. Match local exposure and color of incoming 'img' to 'pano' based on overlap
            # This ensures that even if global normalization wasn't perfect, the transition is seamless.
            if ov > 20:
                h_p, w_p = pano.shape[:2]
                roi_p = pano[:, w_p-ov:w_p]
                roi_i = img[:, :ov]
                
                # Match means across channels for seamless transition
                for c in range(3):
                    m_p = np.mean(roi_p[:, :, c])
                    m_i = np.mean(roi_i[:, :, c])
                    if m_i > 5:
                        l_gain = m_p / (m_i + 1e-6)
                        # Slightly more restrictive gain to prevent runaway brightness
                        l_gain = np.clip(l_gain, 0.8, 1.25)
                        img[:, :, c] = cv2.convertScaleAbs(img[:, :, c], alpha=l_gain, beta=0)

            # 2. Handle canvas expansion and vertical shift
            pad_tp = max(0, -dy); pad_bp = max(0, dy)
            pad_ti = max(0, dy); pad_bi = max(0, -dy)
            pano = cv2.copyMakeBorder(pano, pad_tp, pad_bp, 0, 0, cv2.BORDER_CONSTANT, value=0)
            img = cv2.copyMakeBorder(img, pad_ti, pad_bi, 0, 0, cv2.BORDER_CONSTANT, value=0)
            
            h_p, w_p = pano.shape[:2]
            h_i, w_i = img.shape[:2]
            h_c = min(h_p, h_i)
            pano = pano[:h_c, :]
            img = img[:h_c, :]
            
            # 3. Create new canvas and blend
            new_w = w_p + img.shape[1] - ov
            res = np.zeros((h_c, new_w, 3), dtype=np.uint8)
            res[:, :w_p] = pano
            
            if ov > 10:
                roi_l = pano[:, w_p-ov:]
                roi_r = img[:, :ov]
                
                # Smoother Sigmoid for transition
                t = np.linspace(0, 1, ov)
                ramp = 1.0 / (1.0 + np.exp(-14 * (t - 0.5)))
                alpha = np.tile(ramp, (h_c, 1))[:, :, np.newaxis].astype(np.float32)
                
                blended_ov = self.multiband_blend_robust(roi_l, roi_r, alpha, levels=6)
                res[:, w_p-ov:w_p] = blended_ov
                res[:, w_p:] = img[:, ov:]
            else:
                res[:, w_p:] = img
            
            pano = res

        if close_loop and len(images) >= 4:
            pano = self.close_loop_seam_robust(pano, first_ref)
            
        # Post-process for soft, diffused, HDR-balanced lighting
        pano = self.post_process_hdr_lighting(pano)
        return pano

    def post_process_hdr_lighting(self, image: np.ndarray) -> np.ndarray:
        """Apply HDR-like balancing and soft diffusion."""
        print("[*] Balancing HDR lighting and softening shadows...")
        # 1. Shadow/Highlight recovery using LAB CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, a, b = cv2.split(lab)
        
        # Soft CLAHE to lift shadows and tame sunlight streaks
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(12, 12))
        L = clahe.apply(L)
        
        # 2. Global tone mapping simulation for photorealistic look
        # Linear to log-like mapping
        L_f = L.astype(np.float32) / 255.0
        L_f = cv2.pow(L_f, 0.85) # Gamma adjust to lift midtones
        L = (L_f * 255.0).astype(np.uint8)
        
        image = cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)
        
        # 3. Soft diffusion (Bilateral Filter) to match "soft indoor lighting"
        # D=5 is small enough to keep textures, but smooth noise/banding
        image = cv2.bilateralFilter(image, 5, 20, 20)
        
        return image

    def multiband_blend_robust(self, img_l: np.ndarray, img_r: np.ndarray, alpha: np.ndarray, levels: int = 6) -> np.ndarray:
        l_f = img_l.astype(np.float32)
        r_f = img_r.astype(np.float32)
        a_f = alpha.astype(np.float32)
        
        def build_gauss(img, n):
            gp = [img]
            for _ in range(n):
                if gp[-1].shape[0] < 2 or gp[-1].shape[1] < 2: break
                gp.append(cv2.pyrDown(gp[-1]))
            return gp
            
        def build_lap(gp):
            lp = []
            for i in range(len(gp)-1):
                up = cv2.pyrUp(gp[i+1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
                lp.append(gp[i] - up)
            lp.append(gp[-1])
            return lp
            
        gL = build_gauss(l_f, levels); gR = build_gauss(r_f, levels); gA = build_gauss(a_f, levels)
        lL = build_lap(gL); lR = build_lap(gR)
        
        blended = []
        for l, r, a in zip(lL, lR, gA):
            if a.ndim == 2: a = a[:, :, np.newaxis]
            if a.shape[:2] != l.shape[:2]:
                a = cv2.resize(a, (l.shape[1], l.shape[0]))
            blended.append(l * (1.0 - a) + r * a)
            
        res = blended[-1]
        for layer in reversed(blended[:-1]):
            res = cv2.pyrUp(res, dstsize=(layer.shape[1], layer.shape[0])) + layer
            
        return np.clip(res, 0, 255).astype(np.uint8)

    def close_loop_seam_robust(self, pano: np.ndarray, first_img: np.ndarray) -> np.ndarray:
        print("[*] Final 360-loop closure...")
        h, w = pano.shape[:2]
        # Look for first image in the last 45% of panorama
        search_w = int(w * 0.45)
        tail = pano[:, -search_w:]
        
        dx, dy, score = self.find_translation(tail, first_img)
        
        if dx is not None and score > 0.40 and dx < search_w * 0.96:
            match_x = (w - search_w) + dx
            print(f"  [loop] Closure found at x={match_x} (score={score:.2f})")
            pano = pano[:, :match_x]
            
            # Smoothly distribute dy drift across the whole pano
            if abs(dy) > 1:
                rows, cols = pano.shape[:2]
                src = np.float32([[0,0], [cols-1,0], [0,rows-1]])
                dst = np.float32([[0,0], [cols-1, -dy], [0,rows-1]])
                M = cv2.getAffineTransform(src, dst)
                pano = cv2.warpAffine(pano, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            print("  [loop] No strong 360-match found at the end.")
        return pano

    def trim_black_borders(self, image: np.ndarray) -> np.ndarray:
        if image is None: return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray > 12).astype(np.uint8)
        coords = cv2.findNonZero(mask)
        if coords is None: return image
        x, y, w, h = cv2.boundingRect(coords)
        
        res = image[y:y+h, x:x+w]
        h_r, w_r = res.shape[:2]
        gray_r = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        top, bot = 0, h_r
        # Require 85% content for a row to be valid
        row_content = np.count_nonzero(gray_r > 12, axis=1) / w_r
        for r in range(h_r // 3):
            if row_content[r] > 0.85:
                top = r; break
        for r in range(h_r - 1, h_r // 3 * 2, -1):
            if row_content[r] > 0.85:
                bot = r + 1; break
        return res[top:bot]


def stitch_images_robustly(image_paths: List[str], results_dir: str) -> List[str]:
    print(f"[*] Loading {len(image_paths)} images...")
    images = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is not None:
            # High 1400px height for capturing mall details but saving RAM
            if img.shape[0] > 1400:
                scale = 1400 / img.shape[0]
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
            images.append(img)
            
    if len(images) < 2: return []
    print(f"[*] Processing {len(images)} images")
    
    stitcher = RobustStitcher()
    print(f"[*] Sequential stitching starting from {len(images)} images")
    result = stitcher.manual_mosaic_fallback(images, close_loop=True)
    result = stitcher.trim_black_borders(result)
    
    if result is not None:
        out_path = os.path.join(results_dir, "panorama.jpg")
        cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 93])
        print(f"[*] Stitching done: 1 panorama(s) saved.")
        return [out_path]
    return []
