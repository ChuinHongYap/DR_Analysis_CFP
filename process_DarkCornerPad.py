import csv
import os
import shutil
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import cv2
import numpy as np


def _fit_circle_lstsq(pts: np.ndarray) -> Tuple[float, float, float]:
    """
    Fits a circle (xc, yc, r) to an (N, 2) array of (x, y) points
    using algebraic least squares: A*x + B*y + C = x^2 + y^2.
    """
    x = pts[:, 0]
    y = pts[:, 1]
    M = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    params, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
    xc = params[0] / 2.0
    yc = params[1] / 2.0
    r = np.sqrt(max(params[2] + xc**2 + yc**2, 1.0))
    return float(xc), float(yc), float(r)

def _compute_adaptive_threshold(max_channel: np.ndarray, min_thresh: int = 12) -> float:
    """
    Estimates the background intensity from the 4 corners of the image
    (which are always outside the circular fundus field of view) and
    combines it with Otsu thresholding to handle dark-grey backgrounds.
    """
    h, w = max_channel.shape
    patch = max(12, int(0.06 * min(h, w)))
    pad = 3  # Avoid 1-2px screenshot border artefacts

    corners = [
        max_channel[pad : pad + patch, pad : pad + patch],
        max_channel[pad : pad + patch, w - pad - patch : w - pad],
        max_channel[h - pad - patch : h - pad, pad : pad + patch],
        max_channel[h - pad - patch : h - pad, w - pad - patch : w - pad],
    ]
    # Use the 75th percentile across the 4 corners' 98th percentiles
    # so even if one corner has a label, the other corners set the background floor
    corner_p98 = [float(np.percentile(c, 98)) for c in corners if c.size > 0]
    bg_level = float(np.percentile(corner_p98, 75)) if corner_p98 else 0.0

    otsu_val, _ = cv2.threshold(
        max_channel.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Place threshold safely above the corner background floor
    adaptive_t = bg_level + max(10.0, 0.18 * max(float(otsu_val) - bg_level, 0.0))
    return max(float(min_thresh), adaptive_t)

def analyse_and_clean_cfp(
    image_bgr: np.ndarray,
    min_intensity_thresh: int = 12,
    ransac_iters: int = 400,
    radius_shrink_ratio: float = 0.003,
    min_radius_shrink_px: float = 2.5,
    min_outer_pixels: int = 15,
    rng_seed: int = 42,
) -> Tuple[bool, np.ndarray, Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Detects the optical circle of a CFP scan across arbitrary resolutions and
    background grey levels, masking the outer region to (0, 0, 0).

    Returns
    -------
    needs_fill : bool
        True if non-black background pixels or protruding notches exist outside the circle.
    cleaned_bgr : np.ndarray
        Image with everything outside the detected circle set to (0, 0, 0).
    metrics : dict
        Fitted circle centre, radius, threshold used, and outer artefact pixel count.
    inlier_pts : np.ndarray
        Contour points classified as part of the true circle (for plotting).
    outlier_pts : np.ndarray
        Contour points rejected as notches or clipped borders (for plotting).
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] < 3:
        raise ValueError("Expected a 3-channel colour image (H, W, 3).")

    img = image_bgr[:, :, :3].copy()
    h, w = img.shape[:2]
    min_dim = float(min(h, w))

    # Resolution-scaled tolerances
    border_margin = max(8, int(0.01 * min_dim))
    inlier_tol = max(3.0, 0.004 * min_dim)
    radius_shrink = max(min_radius_shrink_px, radius_shrink_ratio * min_dim)

    # 1. Adaptive foreground segmentation using max across RGB channels
    max_channel = np.max(img, axis=2)
    thresh = _compute_adaptive_threshold(max_channel, min_intensity_thresh)
    fg_mask = (max_channel > thresh).astype(np.uint8) * 255

    # Clear thin outer frame so borders cannot bridge disconnected regions
    fg_mask[:2, :] = 0
    fg_mask[-2:, :] = 0
    fg_mask[:, :2] = 0
    fg_mask[:, -2:] = 0

    # Morphological closing & opening to smooth blurry rims
    k_size = max(5, int(0.005 * min_dim) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # 2. Extract largest external contour
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return False, img, {}, np.empty((0, 2)), np.empty((0, 2))

    largest_contour = max(contours, key=cv2.contourArea)
    pts = largest_contour.reshape(-1, 2).astype(np.float64)

    # 3. Filter out clipped border points (near image edges or flat bounding extremes)
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    cx_init = 0.5 * (x_min + x_max)
    cy_init = 0.5 * (y_min + y_max)

    non_border_mask = (
        (pts[:, 0] > border_margin)
        & (pts[:, 0] < (w - 1 - border_margin))
        & (pts[:, 1] > border_margin)
        & (pts[:, 1] < (h - 1 - border_margin))
    )

    # Also detect if the scan is clipped along an internal crop line (straight vertical/horizontal segments)
    if np.sum(pts[:, 0] <= x_min + 2) > 0.08 * min_dim:
        non_border_mask &= pts[:, 0] > (x_min + border_margin)
    if np.sum(pts[:, 0] >= x_max - 2) > 0.08 * min_dim:
        non_border_mask &= pts[:, 0] < (x_max - border_margin)
    if np.sum(pts[:, 1] <= y_min + 2) > 0.08 * min_dim:
        non_border_mask &= pts[:, 1] > (y_min + border_margin)
    if np.sum(pts[:, 1] >= y_max - 2) > 0.08 * min_dim:
        non_border_mask &= pts[:, 1] < (y_max - border_margin)

    arc_pts = pts[non_border_mask]
    if len(arc_pts) < 30:
        arc_pts = pts

    # 4. 360-degree Polar Angular Binning (1 representative point per degree)
    angles = np.degrees(np.arctan2(arc_pts[:, 1] - cy_init, arc_pts[:, 0] - cx_init))
    bins = np.floor(angles + 180.0).astype(int)
    bins = np.clip(bins, 0, 359)

    binned_pts_list = []
    binned_angles_list = []
    for b in range(360):
        mask_b = bins == b
        if np.any(mask_b):
            pts_b = arc_pts[mask_b]
            # Median point in this 1-degree wedge
            binned_pts_list.append(np.median(pts_b, axis=0))
            binned_angles_list.append(float(b))

    binned_pts = np.array(binned_pts_list, dtype=np.float64)
    binned_angles = np.array(binned_angles_list, dtype=np.float64)

    if len(binned_pts) < 12:
        binned_pts = arc_pts
        binned_angles = np.degrees(
            np.arctan2(binned_pts[:, 1] - cy_init, binned_pts[:, 0] - cx_init)
        )

    # 5. Wide-Arc RANSAC on angularly binned points
    rng = np.random.default_rng(rng_seed)
    best_inliers = None
    best_score = -1.0
    n_bins = len(binned_pts)

    for _ in range(ransac_iters):
        idx = rng.choice(n_bins, size=3, replace=False)
        a0, a1, a2 = binned_angles[idx]

        # Angular separation check (>= 40 degrees apart around the circle)
        d01 = abs((a0 - a1 + 180.0) % 360.0 - 180.0)
        d12 = abs((a1 - a2 + 180.0) % 360.0 - 180.0)
        d02 = abs((a0 - a2 + 180.0) % 360.0 - 180.0)
        if min(d01, d12, d02) < 40.0:
            continue

        xc_s, yc_s, r_s = _fit_circle_lstsq(binned_pts[idx])

        # Sanity bounds: centre near image middle, radius reasonable
        if not (0.25 * w <= xc_s <= 0.75 * w and 0.25 * h <= yc_s <= 0.75 * h):
            continue
        if not (0.32 * min_dim <= r_s <= 0.70 * max(h, w)):
            continue

        dists = np.hypot(binned_pts[:, 0] - xc_s, binned_pts[:, 1] - yc_s)
        residuals = np.abs(dists - r_s)
        inliers = residuals <= inlier_tol

        # Huber-like score rewards tight inliers
        score = float(np.sum(np.maximum(0.0, 1.0 - residuals[inliers] / inlier_tol)))
        if score > best_score:
            best_score = score
            best_inliers = inliers

    # 6. Two-pass Least-Squares Refinement (rejects filleted notch bases)
    if best_inliers is not None and np.sum(best_inliers) >= 10:
        xc, yc, r = _fit_circle_lstsq(binned_pts[best_inliers])

        # Pass 2: tighten tolerance to 60% to exclude smooth notch transitions
        dists = np.hypot(binned_pts[:, 0] - xc, binned_pts[:, 1] - yc)
        tight_inliers = np.abs(dists - r) <= (0.65 * inlier_tol)
        if np.sum(tight_inliers) >= 10:
            xc, yc, r = _fit_circle_lstsq(binned_pts[tight_inliers])
    else:
        xc, yc, r = _fit_circle_lstsq(binned_pts)

    # Classify full contour points for diagnostic plotting
    all_dists = np.hypot(pts[:, 0] - xc, pts[:, 1] - yc)
    full_inlier_mask = non_border_mask & (np.abs(all_dists - r) <= inlier_tol)
    inlier_pts = pts[full_inlier_mask]
    outlier_pts = pts[~full_inlier_mask]

    # 7. Check whether the image needs outer circle filling
    yy, xx = np.ogrid[:h, :w]
    dist_sq = (xx - xc) ** 2 + (yy - yc) ** 2

    # Any non-black pixels outside (r + inlier_tol) indicate a notch or grey background
    outside_tol_zone = dist_sq > (r + inlier_tol) ** 2
    non_black = np.any(img > 0, axis=2)
    outer_artifact_pixels = int(np.sum(outside_tol_zone & non_black))
    needs_fill = outer_artifact_pixels >= min_outer_pixels

    effective_r = max(1.0, r - radius_shrink)
    inside_circle = dist_sq <= effective_r**2

    cleaned_bgr = img.copy()
    cleaned_bgr[~inside_circle] = (0, 0, 0)

    metrics = {
        "center_x": round(xc, 2),
        "center_y": round(yc, 2),
        "fitted_radius": round(r, 2),
        "masked_radius": round(effective_r, 2),
        "adaptive_thresh": round(thresh, 1),
        "outer_artifact_pixels": outer_artifact_pixels,
    }

    return needs_fill, cleaned_bgr, metrics, inlier_pts, outlier_pts

def process_cfp_folder(
    input_dir: str,
    output_dir: str,
    csv_name: str = "images_circleFilled.csv",
    valid_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    **kwargs,
) -> Path:
    """
    Processes all CFP images in `input_dir` and writes results to `output_dir`:
    - Images requiring outer-circle filling are masked to (0, 0, 0), saved to
      `output_dir`, and recorded in `images_circleFilled.csv`.
    - Clean images without outer artefacts are copied directly without modification.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    if not in_path.exists() or not in_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {in_path}")

    out_path.mkdir(parents=True, exist_ok=True)
    if not csv_name.lower().endswith(".csv"):
        csv_name += ".csv"
    csv_path = csv_name

    filled_records = []
    copied_count = 0
    filled_count = 0

    image_files = sorted(
        [f for f in in_path.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
    )

    for src_file in image_files:
        dst_file = out_path / src_file.name
        bgr = cv2.imread(str(src_file))
        if bgr is None:
            print(f"Skipping unreadable file: {src_file.name}")
            continue

        needs_fill, cleaned_bgr, metrics, _, _ = analyse_and_clean_cfp(bgr, **kwargs)

        if needs_fill:
            cv2.imwrite(str(dst_file), cleaned_bgr)
            filled_records.append({"filename": src_file.name, **metrics})
            filled_count += 1
        else:
            if src_file.resolve() != dst_file.resolve():
                shutil.copy2(src_file, dst_file)
            copied_count += 1

    fieldnames = [
        "filename",
        "center_x",
        "center_y",
        "fitted_radius",
        "masked_radius",
        "adaptive_thresh",
        "outer_artifact_pixels",
    ]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filled_records)

    print(
        f"Done: {filled_count} image(s) filled and logged, "
        f"{copied_count} image(s) copied directly.\n"
        f"CSV saved to: {csv_path}"
    )
    return csv_path

if __name__ == "__main__":
    # Example usage:
    INPUT_PATH = "./MMRDR/MMRDR-CFP/img"
    OUTPUT_PATH = "./MMRDR/MMRDR-CFP/img_dcp"

    process_cfp_folder(
        input_dir=INPUT_PATH,
        output_dir=OUTPUT_PATH,
        csv_name="images_record_DCP.csv",
    )