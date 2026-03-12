#!/usr/bin/env python3
"""
STL Alignment Tool — Align 2+ STL pieces by minimizing border gaps.

Extracts print-plate outlines (z-cross-section), finds shared borders,
and uses robust optimization to find the best translation (+ optional rotation).

Outputs:
- Aligned overlay plot showing all pieces
- Per-pair border gap QC with histogram

Designed to work standalone (CLI) and as an importable module.

Usage:
    # Basic
    conda run -n demgis python3 align_stls.py piece1.stl piece2.stl

    # With initial pose from slicer screenshot
    conda run -n demgis python3 align_stls.py piece1.stl piece2.stl --init-image screenshot.png

    # Multiple pieces
    conda run -n demgis python3 align_stls.py piece1.stl piece2.stl piece3.stl -o my_align_
"""

import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import Normalize
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from shapely.affinity import rotate, translate
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

# ============================================================
# OUTLINE EXTRACTION
# ============================================================


def extract_outline(stl_path, z_height=0.5, min_island_frac=0.01):
    """Extract 2D outline polygon from STL at given z height.

    Args:
        stl_path: Path to STL file.
        z_height: Cross-section height in mm.
        min_island_frac: Drop polygons smaller than this fraction of the
            largest component's area.  Set to 0 to keep everything.

    Returns (outline_polygon, mesh_bounds) or (None, bounds) on failure.
    """
    mesh = trimesh.load(stl_path)
    sec = mesh.section(plane_origin=[0, 0, z_height], plane_normal=[0, 0, 1])
    if sec is None:
        print(f"  Warning: No cross-section at z={z_height} for {stl_path}")
        return None, mesh.bounds

    path2d = sec.to_2D()
    polys = path2d[0].polygons_full if isinstance(path2d, tuple) else path2d.polygons_full
    tf = path2d[1] if isinstance(path2d, tuple) else np.eye(4)

    all_polys = []
    for p in polys:
        shifted = translate(p, xoff=tf[0, 3], yoff=tf[1, 3])
        if shifted.is_valid and shifted.area > 1.0:
            all_polys.append(shifted)

    if not all_polys:
        return None, mesh.bounds

    # Deduplicate near-identical polygons — trimesh polygons_full can
    # produce duplicates at base z-heights, and unary_union of duplicates
    # creates exterior rings that snake through sub-mm slivers.
    if len(all_polys) > 1:
        unique = [all_polys[0]]
        for p in all_polys[1:]:
            is_dup = False
            for u in unique:
                if abs(p.area - u.area) < 1.0:
                    sym_diff = p.symmetric_difference(u).area
                    if sym_diff < 1.0:
                        is_dup = True
                        break
            if not is_dup:
                unique.append(p)
        if len(unique) < len(all_polys):
            print(f"    Deduplicated: {len(all_polys)} → {len(unique)} polygons")
            all_polys = unique

    # Filter small islands
    if min_island_frac > 0 and len(all_polys) > 1:
        max_area = max(p.area for p in all_polys)
        threshold = max_area * min_island_frac
        before = len(all_polys)
        all_polys = [p for p in all_polys if p.area >= threshold]
        dropped = before - len(all_polys)
        if dropped:
            print(f"    Dropped {dropped} small island(s) "
                  f"(< {min_island_frac*100:.0f}% of largest = {threshold:.1f} mm²)")

    combined = unary_union(all_polys)

    # Close sub-mm sliver passages that create straight-line artifacts —
    # these come from near-vertical walls in the mesh at the base z-height.
    combined = combined.buffer(1.5).buffer(-1.5)

    return combined, mesh.bounds


def sample_boundary(outline, spacing=0.5):
    """Sample points along polygon boundary at regular spacing (mm)."""
    if outline is None:
        return np.empty((0, 2))

    points = []

    def _sample_ring(ring):
        length = ring.length
        n = max(10, int(length / spacing))
        for i in range(n):
            pt = ring.interpolate(i / n, normalized=True)
            points.append([pt.x, pt.y])

    if outline.geom_type == "MultiPolygon":
        for poly in outline.geoms:
            _sample_ring(poly.exterior)
    elif outline.geom_type == "Polygon":
        _sample_ring(outline.exterior)

    return np.array(points)


# ============================================================
# IMAGE-BASED INITIAL POSE
# ============================================================


def estimate_offsets_from_image(image_path, outlines):
    """Estimate initial offsets from a slicer screenshot.

    Detects yellow regions in the image, matches them to outlines by area,
    and computes relative centroid offsets in mm.

    Returns list of (dx, dy) for each outline, with first = (0, 0).
    """
    try:
        from PIL import Image
    except ImportError:
        print("  WARNING: PIL not available, skipping image-based initialization")
        return [(0.0, 0.0)] * len(outlines)

    try:
        img = np.array(Image.open(image_path).convert("RGB"))
    except Exception as e:
        print(f"  WARNING: Could not load image {image_path}: {e}")
        return [(0.0, 0.0)] * len(outlines)

    print(f"  Image: {img.shape[1]}x{img.shape[0]} pixels")

    # Convert to HSV using numpy (no cv2 dependency)
    img_float = img.astype(np.float32) / 255.0
    r, g, b = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue (0-360)
    hue = np.zeros_like(cmax)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)
    hue[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    hue[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    hue[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    # Saturation (0-1)
    sat = np.where(cmax > 0, delta / cmax, 0)

    # Value (0-1)
    val = cmax

    # Threshold for yellow: H in [30, 70], S > 0.3, V > 0.5
    yellow_mask = (hue >= 30) & (hue <= 70) & (sat > 0.3) & (val > 0.5)

    # Morphological close to fill terrain texture holes (manual dilation + erosion)
    kernel_size = 15
    from scipy.ndimage import binary_closing, label

    yellow_mask = binary_closing(yellow_mask, structure=np.ones((kernel_size, kernel_size)))

    # Find connected components
    labeled, n_components = label(yellow_mask)
    if n_components < len(outlines):
        print(f"  WARNING: Found {n_components} blobs, expected {len(outlines)}")
        return [(0.0, 0.0)] * len(outlines)

    # Get component areas and centroids
    blob_data = []
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        area_px = comp_mask.sum()
        if area_px < 100:
            continue
        ys, xs = np.where(comp_mask)
        cx = xs.mean()
        cy = ys.mean()
        blob_data.append({"area": area_px, "cx": cx, "cy": cy})

    blob_data.sort(key=lambda b: b["area"], reverse=True)
    blob_data = blob_data[: len(outlines)]

    if len(blob_data) < len(outlines):
        print(f"  WARNING: Only {len(blob_data)} significant blobs found")
        return [(0.0, 0.0)] * len(outlines)

    print(f"  Found {len(blob_data)} blobs in image")

    # Match blobs to outlines by area ranking
    outline_areas = []
    for o in outlines:
        if o is not None:
            outline_areas.append(o.area)
        else:
            outline_areas.append(0)

    # Sort both by area descending, match by rank
    outline_order = np.argsort(outline_areas)[::-1]
    # blob_data is already sorted by area descending

    # Compute scale: pixels per mm
    total_blob_area = sum(b["area"] for b in blob_data)
    total_outline_area = sum(outline_areas)
    if total_outline_area > 0:
        scale_px_per_mm = np.sqrt(total_blob_area / total_outline_area)
    else:
        scale_px_per_mm = 1.0

    print(f"  Scale: {scale_px_per_mm:.1f} pixels/mm")

    # Build offset mapping: blob_rank -> outline_index
    rank_to_idx = {}
    for rank, idx in enumerate(outline_order):
        rank_to_idx[idx] = rank

    # Reference blob (matches first outline = outline index 0)
    ref_rank = rank_to_idx[0]
    ref_blob = blob_data[ref_rank]

    offsets = []
    for i in range(len(outlines)):
        if i == 0:
            offsets.append((0.0, 0.0))
            continue

        blob_rank = rank_to_idx[i]
        blob = blob_data[blob_rank]

        # Pixel offset from reference
        dpx = blob["cx"] - ref_blob["cx"]
        dpy = blob["cy"] - ref_blob["cy"]

        # Convert to mm (image Y is inverted vs STL Y)
        dx_mm = dpx / scale_px_per_mm
        dy_mm = -dpy / scale_px_per_mm

        # Adjust for outline centroid difference (the offsets should move
        # outline[i] relative to outline[0], accounting for their current centroids)
        if outlines[0] is not None and outlines[i] is not None:
            c0 = np.array(outlines[0].centroid.coords[0])
            ci = np.array(outlines[i].centroid.coords[0])
            # dx_mm is where outline[i] centroid should be relative to outline[0] centroid
            # So the offset to apply to outline[i] is:
            dx_mm = dx_mm - (ci[0] - c0[0])
            dy_mm = dy_mm - (ci[1] - c0[1])

        offsets.append((dx_mm, dy_mm))
        print(f"  Piece {i}: image offset → dx={dx_mm:.1f}, dy={dy_mm:.1f} mm")

    return offsets


# ============================================================
# ALIGNMENT
# ============================================================


def find_shared_border(outline1, outline2, threshold_mm=5.0):
    """Find shared border points between two outlines.

    Returns (pts1, pts2, distances) where pts1[i] is near pts2[i].
    Only includes pairs within threshold_mm of each other.
    """
    pts1 = sample_boundary(outline1, spacing=0.5)
    pts2 = sample_boundary(outline2, spacing=0.5)

    if len(pts1) == 0 or len(pts2) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), np.array([])

    tree2 = cKDTree(pts2)
    dists1, idx1 = tree2.query(pts1)

    # Keep only border-adjacent points
    mask = dists1 < threshold_mm
    border_pts1 = pts1[mask]
    border_pts2 = pts2[idx1[mask]]
    border_dists = dists1[mask]

    return border_pts1, border_pts2, border_dists


def _apply_transform(outline, dx, dy, theta=0):
    """Apply translation + optional rotation to an outline."""
    moved = translate(outline, xoff=dx, yoff=dy)
    if theta != 0:
        cx, cy = moved.centroid.x, moved.centroid.y
        moved = rotate(moved, theta, origin=(cx, cy), use_radians=True)
    return moved


def _bidirectional_border_cost(outline1, outline2, threshold_mm,
                               border_region=None, region_only=False):
    """Compute bidirectional border gap cost (both directions).

    Returns (mean_gap_squared, n_border_pts, overlap_area).
    Bidirectional: checks outline1→outline2 AND outline2→outline1,
    so both borders are pulled toward each other.

    If border_region is (x1, y1, x2, y2):
      - region_only=False (default): points inside weighted 5x, outside 1x
      - region_only=True: ONLY consider points inside the region
    """
    pts1 = sample_boundary(outline1, spacing=0.5)
    pts2 = sample_boundary(outline2, spacing=0.5)

    if len(pts1) == 0 or len(pts2) == 0:
        return 1e6, 0, 0

    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)

    # Forward: outline1 points near outline2
    dists_fwd, _ = tree2.query(pts1)
    mask_fwd = dists_fwd < threshold_mm

    # Reverse: outline2 points near outline1
    dists_rev, _ = tree1.query(pts2)
    mask_rev = dists_rev < threshold_mm

    border_dists_fwd = dists_fwd[mask_fwd]
    border_dists_rev = dists_rev[mask_rev]

    # Apply region filtering/weighting if specified
    if border_region is not None:
        x1, y1, x2, y2 = border_region

        pts1_border = pts1[mask_fwd]
        in_fwd = ((pts1_border[:, 0] >= x1) & (pts1_border[:, 0] <= x2) &
                  (pts1_border[:, 1] >= y1) & (pts1_border[:, 1] <= y2))

        pts2_border = pts2[mask_rev]
        in_rev = ((pts2_border[:, 0] >= x1) & (pts2_border[:, 0] <= x2) &
                  (pts2_border[:, 1] >= y1) & (pts2_border[:, 1] <= y2))

        if region_only:
            # Strict: only use points inside the region
            border_dists_fwd = border_dists_fwd[in_fwd]
            border_dists_rev = border_dists_rev[in_rev]
            all_border_dists = np.concatenate([border_dists_fwd, border_dists_rev])
            n_pts = len(all_border_dists)
            if n_pts < 3:
                return 1e6, 0, 0
            mean_gap_sq = float(np.mean(all_border_dists**2))
        else:
            # Weighted: 5x inside, 1x outside
            w_fwd = np.where(in_fwd, 5.0, 1.0)
            w_rev = np.where(in_rev, 5.0, 1.0)
            all_border_dists = np.concatenate([border_dists_fwd, border_dists_rev])
            all_weights = np.concatenate([w_fwd, w_rev])
            n_pts = len(all_border_dists)
            if n_pts < 5:
                return 1e6, 0, 0
            weighted_sq = all_weights * all_border_dists**2
            mean_gap_sq = float(np.sum(weighted_sq) / np.sum(all_weights))
    else:
        all_border_dists = np.concatenate([border_dists_fwd, border_dists_rev])
        n_pts = len(all_border_dists)
        if n_pts < 5:
            return 1e6, 0, 0
        mean_gap_sq = float(np.mean(all_border_dists**2))

    # Overlap area (expensive — only compute if we have border points)
    overlap = outline1.intersection(outline2)
    overlap_area = overlap.area if not overlap.is_empty else 0

    return mean_gap_sq, n_pts, overlap_area


def align_pair(outline1, outline2, init_dx=0, init_dy=0, allow_rotation=False,
               border_threshold=5.0, border_region=None, region_only=False):
    """Align outline2 to outline1 using ICP warm-up + scipy optimization.

    Uses bidirectional border matching and a balanced cost function that
    allows small overlaps to find the tightest fit.

    If border_region is (x1, y1, x2, y2):
      - region_only=False: points inside weighted 5x, outside 1x
      - region_only=True: ONLY use border points inside the region

    Returns (dx, dy, theta, stats_dict, convergence_history).
    convergence_history is a list of dicts with keys:
        phase, iteration, dx, dy, cost, median_gap, n_border, overlap_area
    """
    if outline1 is None or outline2 is None:
        return init_dx, init_dy, 0, {"median": float("inf"), "n_border": 0}, []

    history = []

    # Pre-sample outline1 boundary (fixed, reusable)
    pts1_fixed = sample_boundary(outline1, spacing=0.5)
    tree1_fixed = cKDTree(pts1_fixed)

    def _record(phase, iteration, ddx, ddy, theta=0):
        """Snapshot current alignment quality into history."""
        moved = _apply_transform(outline2, ddx, ddy, theta)
        gap_sq, n_pts, ov_area = _bidirectional_border_cost(
            outline1, moved, border_threshold, border_region, region_only
        )
        # Compute median gap for the snapshot
        bpts1, bpts2, bdists = find_shared_border(outline1, moved, border_threshold)
        med_gap = float(np.median(bdists)) if len(bdists) > 0 else float("inf")
        history.append({
            "phase": phase,
            "iteration": iteration,
            "dx": ddx,
            "dy": ddy,
            "cost": gap_sq if gap_sq < 1e5 else float("nan"),
            "median_gap": med_gap,
            "n_border": n_pts,
            "overlap_area": ov_area,
        })

    # Phase 1: ICP warm-up (translation only, bidirectional border matching)
    print("  Phase 1: ICP warm-up...")
    dx, dy = float(init_dx), float(init_dy)
    _record("ICP", 0, dx, dy)

    best_icp_cost = float("inf")
    best_icp_dx, best_icp_dy = dx, dy
    stall_count = 0

    for iteration in range(50):
        moved = _apply_transform(outline2, dx, dy)
        pts2 = sample_boundary(moved, spacing=0.5)

        if len(pts2) == 0:
            break

        tree2 = cKDTree(pts2)

        # Forward: pts1 near moved outline2
        dists_fwd, idx_fwd = tree2.query(pts1_fixed)
        mask_fwd = dists_fwd < border_threshold

        # Reverse: pts2 near fixed outline1
        dists_rev, idx_rev = tree1_fixed.query(pts2)
        mask_rev = dists_rev < border_threshold

        if mask_fwd.sum() < 10 and mask_rev.sum() < 10:
            # Not enough border points — move centroids closer
            c1 = np.array(outline1.centroid.coords[0])
            moved_c = np.array(moved.centroid.coords[0])
            delta = (c1 - moved_c) * 0.3
            dx += delta[0]
            dy += delta[1]
            _record("ICP", iteration + 1, dx, dy)
            continue

        if mask_rev.sum() > 0:
            delta = np.mean(pts1_fixed[idx_rev[mask_rev]] - pts2[mask_rev], axis=0)
        elif mask_fwd.sum() > 0:
            delta = -np.mean(pts2[idx_fwd[mask_fwd]] - pts1_fixed[mask_fwd], axis=0)
        else:
            break

        dx += delta[0] * 0.5  # Dampen to avoid oscillation
        dy += delta[1] * 0.5

        _record("ICP", iteration + 1, dx, dy)

        # Track best ICP position; stop if drifting
        current_cost = history[-1]["cost"]
        if not np.isnan(current_cost) and current_cost < best_icp_cost:
            best_icp_cost = current_cost
            best_icp_dx, best_icp_dy = dx, dy
            stall_count = 0
        else:
            stall_count += 1
            if stall_count >= 5:
                # ICP is drifting — revert to best and hand off to scipy
                dx, dy = best_icp_dx, best_icp_dy
                break

        if np.linalg.norm(delta) < 0.005:
            break

    print(f"    ICP result: dx={dx:.2f}, dy={dy:.2f} ({iteration+1} iterations)")

    # Phase 2: Scipy refinement with tight tolerances
    print("  Phase 2: Scipy optimization...")

    eval_count = [0]
    # Record scipy history at regular intervals (every 10 evals)
    scipy_iter_offset = len(history)

    def cost_fn(params):
        eval_count[0] += 1
        ddx, ddy = params[0], params[1]
        theta = params[2] if allow_rotation else 0

        moved = _apply_transform(outline2, ddx, ddy, theta)

        gap_sq, n_pts, overlap_area = _bidirectional_border_cost(
            outline1, moved, border_threshold, border_region, region_only
        )

        if n_pts < 5:
            return 1e6

        # Gap-dominated cost: overlap penalty is very gentle so optimizer
        # finds the tightest fit.  Only penalize gross overlap (>10mm²).
        excess_overlap = max(0, overlap_area - 10.0)
        cost = gap_sq + 0.01 * excess_overlap

        # Record every 10th evaluation to track convergence without overhead
        if eval_count[0] % 10 == 1:
            bpts1, bpts2, bdists = find_shared_border(outline1, moved, border_threshold)
            med_gap = float(np.median(bdists)) if len(bdists) > 0 else float("inf")
            history.append({
                "phase": "Scipy",
                "iteration": eval_count[0],
                "dx": ddx,
                "dy": ddy,
                "cost": cost,
                "median_gap": med_gap,
                "n_border": n_pts,
                "overlap_area": overlap_area,
            })

        return cost

    x0 = [dx, dy, 0.0] if allow_rotation else [dx, dy]

    # Use tight tolerances and more iterations
    result = minimize(
        cost_fn, x0, method="Nelder-Mead",
        options={
            "maxiter": 2000,
            "maxfev": 3000,
            "xatol": 0.001,   # 1 micron position tolerance
            "fatol": 0.0001,  # tight cost tolerance
            "adaptive": True,
        },
    )

    dx_opt, dy_opt = result.x[0], result.x[1]
    theta_opt = result.x[2] if allow_rotation else 0

    print(f"    {eval_count[0]} evaluations, converged={result.success}")

    # Record final state
    _record("Final", eval_count[0], dx_opt, dy_opt, theta_opt)

    # Compute final stats with bidirectional border
    moved = _apply_transform(outline2, dx_opt, dy_opt, theta_opt)
    bpts1, bpts2, bdists = find_shared_border(outline1, moved, border_threshold)

    # Also get reverse direction
    bpts2r, bpts1r, bdists_rev = find_shared_border(moved, outline1, border_threshold)
    all_dists = np.concatenate([bdists, bdists_rev]) if len(bdists_rev) > 0 else bdists

    overlap = outline1.intersection(moved)
    overlap_area = overlap.area if not overlap.is_empty else 0

    stats = {
        "n_border": len(all_dists),
        "overlap_area": overlap_area,
    }
    if len(all_dists) > 0:
        stats.update({
            "min": float(np.min(all_dists)),
            "max": float(np.max(all_dists)),
            "mean": float(np.mean(all_dists)),
            "median": float(np.median(all_dists)),
            "p95": float(np.percentile(all_dists, 95)),
            "std": float(np.std(all_dists)),
        })
    else:
        stats.update({"min": 0, "max": 0, "mean": 0, "median": 0, "p95": 0, "std": 0})

    print(f"    Optimized: dx={dx_opt:.2f}, dy={dy_opt:.2f}" +
          (f", θ={np.degrees(theta_opt):.2f}°" if allow_rotation else ""))
    print(f"    Border: {stats['n_border']} pts, median gap={stats['median']:.3f}mm, "
          f"overlap={overlap_area:.1f}mm²")

    return dx_opt, dy_opt, theta_opt, stats, history


def compute_border_gaps(outline1, outline2, dx, dy, theta=0, threshold_mm=5.0):
    """Compute detailed border gap analysis after alignment.

    Returns dict with border_pts, gap_distances, overlap_area, stats.
    """
    moved = _apply_transform(outline2, dx, dy, theta)
    bpts1, bpts2, bdists = find_shared_border(outline1, moved, threshold_mm)

    overlap = outline1.intersection(moved)
    overlap_area = overlap.area if not overlap.is_empty else 0

    stats = {"n_border": len(bdists), "overlap_area": overlap_area}
    if len(bdists) > 0:
        stats.update({
            "min": float(np.min(bdists)),
            "max": float(np.max(bdists)),
            "mean": float(np.mean(bdists)),
            "median": float(np.median(bdists)),
            "p95": float(np.percentile(bdists, 95)),
            "std": float(np.std(bdists)),
        })
    else:
        stats.update({"min": 0, "max": 0, "mean": 0, "median": 0, "p95": 0, "std": 0})

    return {
        "border_pts1": bpts1,
        "border_pts2": bpts2,
        "gap_distances": bdists,
        "overlap_area": overlap_area,
        "overlap_geom": overlap if not overlap.is_empty else None,
        "stats": stats,
    }


# ============================================================
# PLOTTING
# ============================================================

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]


def _plot_outline(ax, outline, color, label, alpha=0.15, linewidth=1.5):
    """Plot a polygon outline on an axis.

    Uses a buffered version for the fill to close narrow strait cuts that
    create straight-line artifacts, while keeping the real outline for stroke.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    if outline is None:
        return

    def _poly_to_path(poly):
        """Convert a Shapely polygon (with holes) to a matplotlib Path."""
        rings = [np.array(poly.exterior.coords)]
        rings.extend(np.array(r.coords) for r in poly.interiors)
        codes_list = []
        verts_list = []
        for ring in rings:
            n = len(ring)
            codes = [Path.MOVETO] + [Path.LINETO] * (n - 2) + [Path.CLOSEPOLY]
            codes_list.extend(codes)
            verts_list.append(ring)
        return Path(np.concatenate(verts_list), codes_list)

    # Buffer slightly to close narrow strait cuts for fill rendering
    fill_geom = outline.buffer(1.5).buffer(-1.0)
    fill_polys = list(fill_geom.geoms) if fill_geom.geom_type == "MultiPolygon" else [fill_geom]
    for fp in fill_polys:
        if fp.is_empty:
            continue
        patch = PathPatch(_poly_to_path(fp), facecolor=color, edgecolor="none",
                          alpha=alpha, zorder=1)
        ax.add_patch(patch)

    # Draw the real outline stroke
    geoms = list(outline.geoms) if outline.geom_type == "MultiPolygon" else [outline]
    for i, poly in enumerate(geoms):
        x, y = poly.exterior.xy
        ax.plot(x, y, color=color, linewidth=linewidth,
                label=label if i == 0 else None)


def plot_alignment(outlines, offsets, names, output_path):
    """Plot all aligned pieces together (2-panel: overview + border zoom)."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Apply transforms
    moved_outlines = []
    for i, (outline, (dx, dy)) in enumerate(zip(outlines, offsets)):
        if outline is not None and (dx != 0 or dy != 0):
            moved_outlines.append(translate(outline, xoff=dx, yoff=dy))
        else:
            moved_outlines.append(outline)

    # Left panel: Full overview
    ax = axes[0]
    ax.set_title("Aligned Pieces — Overview", fontsize=14)
    for i, (outline, name) in enumerate(zip(moved_outlines, names)):
        _plot_outline(ax, outline, COLORS[i % len(COLORS)], name, alpha=0.2)

    # Show overlap between consecutive pairs
    for i in range(1, len(moved_outlines)):
        if moved_outlines[0] is not None and moved_outlines[i] is not None:
            overlap = moved_outlines[0].intersection(moved_outlines[i])
            if not overlap.is_empty and overlap.area > 0.1:
                _plot_outline(ax, overlap, "purple",
                              f"Overlap ({overlap.area:.1f}mm²)", alpha=0.4)

    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="best")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(True, alpha=0.3)

    # Right panel: Zoom on shared border region
    ax = axes[1]
    ax.set_title("Border Region Zoom", fontsize=14)

    # Find the shared border zone to zoom into
    all_border_pts = []
    for i in range(1, len(moved_outlines)):
        if moved_outlines[0] is not None and moved_outlines[i] is not None:
            bpts1, bpts2, bdists = find_shared_border(
                moved_outlines[0], moved_outlines[i], threshold_mm=8.0
            )
            if len(bpts1) > 0:
                all_border_pts.append(bpts1)
                all_border_pts.append(bpts2)

    for i, (outline, name) in enumerate(zip(moved_outlines, names)):
        _plot_outline(ax, outline, COLORS[i % len(COLORS)], name, alpha=0.2)

    if all_border_pts:
        all_bp = np.vstack(all_border_pts)
        margin = 15
        ax.set_xlim(all_bp[:, 0].min() - margin, all_bp[:, 0].max() + margin)
        ax.set_ylim(all_bp[:, 1].min() - margin, all_bp[:, 1].max() + margin)

        # Color border points by gap distance
        for i in range(1, len(moved_outlines)):
            if moved_outlines[0] is None or moved_outlines[i] is None:
                continue
            bpts1, _, bdists = find_shared_border(
                moved_outlines[0], moved_outlines[i], threshold_mm=8.0
            )
            if len(bpts1) > 0:
                sc = ax.scatter(bpts1[:, 0], bpts1[:, 1], c=bdists, cmap="RdYlGn_r",
                                s=3, vmin=0, vmax=3, zorder=5)
                plt.colorbar(sc, ax=ax, label="Gap (mm)", shrink=0.7)

    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="best")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_border_detail(border_data, name1, name2, output_path):
    """Plot detailed border gap analysis for one pair (2-panel)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    bpts1 = border_data["border_pts1"]
    bdists = border_data["gap_distances"]
    stats = border_data["stats"]

    # Left: Spatial view of border gaps
    ax = axes[0]
    ax.set_title(f"Border Gaps: {name1} ↔ {name2}", fontsize=13)

    if len(bpts1) > 0:
        sc = ax.scatter(bpts1[:, 0], bpts1[:, 1], c=bdists, cmap="RdYlGn_r",
                        s=8, vmin=0, vmax=max(3, np.percentile(bdists, 95)))
        plt.colorbar(sc, ax=ax, label="Gap distance (mm)", shrink=0.8)

        # Add stats annotation
        stats_text = (
            f"N = {stats['n_border']} pts\n"
            f"Min: {stats['min']:.3f} mm\n"
            f"Mean: {stats['mean']:.3f} mm\n"
            f"Median: {stats['median']:.3f} mm\n"
            f"P95: {stats['p95']:.3f} mm\n"
            f"Max: {stats['max']:.3f} mm\n"
            f"Overlap: {stats['overlap_area']:.1f} mm²"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))
    else:
        ax.text(0.5, 0.5, "No shared border found", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="red")

    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(True, alpha=0.3)

    # Right: Histogram of gap distances
    ax = axes[1]
    ax.set_title("Gap Distance Distribution", fontsize=13)

    if len(bdists) > 0:
        ax.hist(bdists, bins=50, color="#3498db", alpha=0.7, edgecolor="white")
        ax.axvline(stats["median"], color="red", linewidth=2, linestyle="--",
                   label=f"Median: {stats['median']:.3f}mm")
        ax.axvline(stats["mean"], color="orange", linewidth=2, linestyle=":",
                   label=f"Mean: {stats['mean']:.3f}mm")
        ax.axvline(stats["p95"], color="purple", linewidth=1.5, linestyle="-.",
                   label=f"P95: {stats['p95']:.3f}mm")
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, "No border data", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="red")

    ax.set_xlabel("Gap distance (mm)")
    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_convergence(history, name1, name2, output_path):
    """Plot optimization convergence: cost, median gap, dx/dy, and overlap over iterations."""
    if not history:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Convergence: {name1} ↔ {name2}", fontsize=14, y=0.98)

    iters = list(range(len(history)))
    phases = [h["phase"] for h in history]
    costs = [h["cost"] for h in history]
    med_gaps = [h["median_gap"] for h in history]
    dxs = [h["dx"] for h in history]
    dys = [h["dy"] for h in history]
    n_borders = [h["n_border"] for h in history]
    overlaps = [h["overlap_area"] for h in history]

    # Find phase boundaries for shading
    icp_end = max((i for i, p in enumerate(phases) if p == "ICP"), default=0)

    phase_colors = {"ICP": "#3498db", "Scipy": "#e67e22", "Final": "#2ecc71"}

    def _shade_phases(ax):
        ax.axvspan(0, icp_end + 0.5, alpha=0.08, color="#3498db", label="ICP phase")
        ax.axvspan(icp_end + 0.5, len(iters) - 0.5, alpha=0.08, color="#e67e22", label="Scipy phase")

    # Top-left: Cost function
    ax = axes[0, 0]
    ax.set_title("Cost Function (mean gap² + overlap penalty)")
    _shade_phases(ax)
    valid_costs = [(i, c) for i, c in zip(iters, costs) if not np.isnan(c) and c < 1e5]
    if valid_costs:
        ci, cv = zip(*valid_costs)
        ax.plot(ci, cv, ".-", color="#2c3e50", markersize=4, linewidth=1)
    ax.set_ylabel("Cost")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # Top-right: Median gap
    ax = axes[0, 1]
    ax.set_title("Median Border Gap (mm)")
    _shade_phases(ax)
    valid_gaps = [(i, g) for i, g in zip(iters, med_gaps) if np.isfinite(g)]
    if valid_gaps:
        gi, gv = zip(*valid_gaps)
        ax.plot(gi, gv, ".-", color="#e74c3c", markersize=4, linewidth=1)
    ax.set_ylabel("Median gap (mm)")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.3)

    # Bottom-left: dx, dy trajectory
    ax = axes[1, 0]
    ax.set_title("Translation (dx, dy)")
    _shade_phases(ax)
    ax.plot(iters, dxs, ".-", color="#3498db", markersize=3, linewidth=1, label="dx")
    ax.plot(iters, dys, ".-", color="#e74c3c", markersize=3, linewidth=1, label="dy")
    ax.set_ylabel("Offset (mm)")
    ax.set_xlabel("Step")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Border points and overlap
    ax = axes[1, 1]
    ax.set_title("Border Points & Overlap")
    _shade_phases(ax)
    ax.plot(iters, n_borders, ".-", color="#2ecc71", markersize=3, linewidth=1, label="Border pts")
    ax.set_ylabel("# Border points", color="#2ecc71")
    ax.tick_params(axis="y", labelcolor="#2ecc71")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(iters, overlaps, ".-", color="#9b59b6", markersize=3, linewidth=1, label="Overlap")
    ax2.set_ylabel("Overlap area (mm²)", color="#9b59b6")
    ax2.tick_params(axis="y", labelcolor="#9b59b6")

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

    ax.set_xlabel("Step")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_animation(outline1, outline2, history, name1, name2, output_path,
                   border_threshold=5.0, fps=6):
    """Generate an animated GIF showing the alignment process step-by-step.

    Each frame shows the reference (fixed) and moving piece at that iteration's
    position, with border gap points colored by distance.
    """
    if not history:
        print("  No history to animate")
        return

    from matplotlib.animation import FuncAnimation, FFMpegWriter

    # Subsample if many frames — aim for ~40-80 frames max
    max_frames = 60
    if len(history) > max_frames:
        # Always include first, last, and evenly spaced in between
        indices = [0]
        step = (len(history) - 1) / (max_frames - 2)
        indices += [int(round(step * i)) for i in range(1, max_frames - 1)]
        indices.append(len(history) - 1)
        indices = sorted(set(indices))
        frames = [history[i] for i in indices]
    else:
        frames = list(history)

    # Hold final frame for 2 seconds
    n_hold = max(1, int(fps * 2))
    frames = frames + [frames[-1]] * n_hold

    # Compute fixed view bounds from full alignment range
    all_xs, all_ys = [], []
    if outline1 is not None:
        b = outline1.bounds
        all_xs.extend([b[0], b[2]])
        all_ys.extend([b[1], b[3]])
    for h in history:
        moved = _apply_transform(outline2, h["dx"], h["dy"])
        b = moved.bounds
        all_xs.extend([b[0], b[2]])
        all_ys.extend([b[1], b[3]])
    margin = 8
    xlim = (min(all_xs) - margin, max(all_xs) + margin)
    ylim = (min(all_ys) - margin, max(all_ys) + margin)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    def _draw_frame(frame_idx):
        ax.clear()
        h = frames[frame_idx]
        dx, dy = h["dx"], h["dy"]
        phase = h["phase"]
        iteration = h["iteration"]
        med_gap = h.get("median_gap", float("nan"))
        n_border = h.get("n_border", 0)
        overlap_area = h.get("overlap_area", 0)

        moved = _apply_transform(outline2, dx, dy)

        # Draw pieces
        _plot_outline(ax, outline1, COLORS[0], name1, alpha=0.2, linewidth=1.5)
        _plot_outline(ax, moved, COLORS[1], name2, alpha=0.2, linewidth=1.5)

        # Draw overlap
        try:
            overlap = outline1.intersection(moved)
            if not overlap.is_empty and overlap.area > 0.1:
                _plot_outline(ax, overlap, "purple", None, alpha=0.35, linewidth=0)
        except Exception:
            pass

        # Draw border gap points
        try:
            bpts1, bpts2, bdists = find_shared_border(
                outline1, moved, border_threshold
            )
            if len(bpts1) > 0:
                ax.scatter(bpts1[:, 0], bpts1[:, 1], c=bdists, cmap="RdYlGn_r",
                           s=6, vmin=0, vmax=3, zorder=5)
        except Exception:
            pass

        # Status text
        gap_str = f"{med_gap:.3f}" if np.isfinite(med_gap) else "---"
        ax.set_title(
            f"{name1} ↔ {name2}\n"
            f"{phase} step {iteration}  |  "
            f"dx={dx:.1f}  dy={dy:.1f}  |  "
            f"median gap={gap_str}mm  |  "
            f"border pts={n_border}  |  "
            f"overlap={overlap_area:.1f}mm²",
            fontsize=11, fontfamily="monospace",
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")

    anim = FuncAnimation(fig, _draw_frame, frames=len(frames),
                         interval=1000 // fps, repeat=False)

    writer = FFMpegWriter(fps=fps, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p"])
    anim.save(output_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Align 2+ STL pieces by minimizing border gaps"
    )
    parser.add_argument("stls", nargs="+", help="STL files (first is reference)")
    parser.add_argument("--init-image", help="Slicer screenshot for initial pose estimate")
    parser.add_argument("--output-prefix", "-o", default=None,
                        help="Output file prefix (default: align_YYYYMMDD_HHMMSS_)")
    parser.add_argument("--z-height", type=float, default=0.5,
                        help="Cross-section z height in mm (default: 0.5)")
    parser.add_argument("--allow-rotation", action="store_true", default=True,
                        help="Optimize rotation angle (default: on)")
    parser.add_argument("--no-rotation", action="store_true",
                        help="Disable rotation optimization")
    parser.add_argument("--border-threshold", type=float, default=5.0,
                        help="Max distance (mm) for shared border detection (default: 5.0)")
    parser.add_argument("--keep-islands", action="store_true",
                        help="Keep all small islands (by default, drops < 1%% of largest)")
    parser.add_argument("--min-island-frac", type=float, default=0.01,
                        help="Min island area as fraction of largest (default: 0.01 = 1%%)")
    parser.add_argument("--init-poses", help="JSON file with initial poses (from GUI or previous run)")
    args = parser.parse_args()

    if len(args.stls) < 2:
        print("ERROR: Need at least 2 STL files")
        sys.exit(1)

    # Generate unique output prefix with timestamp
    if args.output_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_prefix = f"align_{timestamp}_"

    # Extract names from filenames
    names = [os.path.splitext(os.path.basename(p))[0] for p in args.stls]

    # Step 1: Extract outlines
    print("=== Extracting outlines ===")
    outlines = []
    for path in args.stls:
        print(f"  Loading {os.path.basename(path)}...")
        island_frac = 0 if args.keep_islands else args.min_island_frac
        outline, bounds = extract_outline(path, args.z_height, min_island_frac=island_frac)
        if outline is not None:
            print(f"    Area: {outline.area:.0f} mm², bounds: "
                  f"({outline.bounds[0]:.0f}, {outline.bounds[1]:.0f}) to "
                  f"({outline.bounds[2]:.0f}, {outline.bounds[3]:.0f})")
        else:
            print("    WARNING: Could not extract outline")
        outlines.append(outline)

    # Step 2: Get initial offsets (dx, dy, theta)
    initial_offsets = [(0.0, 0.0, 0.0)] * len(outlines)
    if args.init_poses:
        print(f"\n=== Loading initial poses from {args.init_poses} ===")
        with open(args.init_poses) as f:
            pose_data = json.load(f)
        for p in pose_data.get("pieces", []):
            pname = p.get("name", "")
            for i, n in enumerate(names):
                if n == pname and not p.get("is_reference", False):
                    init_dx = p.get("dx", 0)
                    init_dy = p.get("dy", 0)
                    init_theta = np.radians(p.get("theta", 0))
                    initial_offsets[i] = (init_dx, init_dy, init_theta)
                    print(f"  {n}: dx={init_dx:.1f}, dy={init_dy:.1f}, "
                          f"\u03b8={p.get('theta', 0):.1f}\u00b0")
                    break
    elif args.init_image:
        print(f"\n=== Estimating initial pose from {args.init_image} ===")
        img_offsets = estimate_offsets_from_image(args.init_image, outlines)
        initial_offsets = [(dx, dy, 0.0) for dx, dy in img_offsets]

    # Step 3: Align each piece to reference
    print("\n=== Aligning pieces ===")
    final_offsets = [(0.0, 0.0)]  # First piece is fixed
    all_histories = []

    do_rotation = args.allow_rotation and not args.no_rotation
    for i in range(1, len(outlines)):
        print(f"\nAligning {names[i]} to {names[0]}...")
        dx, dy, theta, stats, history = align_pair(
            outlines[0], outlines[i],
            init_dx=initial_offsets[i][0],
            init_dy=initial_offsets[i][1],
            allow_rotation=do_rotation,
            border_threshold=args.border_threshold,
        )
        final_offsets.append((dx, dy))
        all_histories.append(history)

    # Step 4: Generate outputs
    print("\n=== Generating plots ===")

    # Main alignment plot
    aligned_path = f"{args.output_prefix}aligned.png"
    plot_alignment(outlines, final_offsets, names, aligned_path)

    # Per-pair border detail + convergence
    for i in range(1, len(outlines)):
        dx, dy = final_offsets[i]
        border_data = compute_border_gaps(
            outlines[0], outlines[i], dx, dy,
            threshold_mm=args.border_threshold,
        )
        detail_path = f"{args.output_prefix}border_{names[i]}.png"
        plot_border_detail(border_data, names[0], names[i], detail_path)

        conv_path = f"{args.output_prefix}convergence_{names[i]}.png"
        plot_convergence(all_histories[i - 1], names[0], names[i], conv_path)

        anim_path = f"{args.output_prefix}animation_{names[i]}.mp4"
        plot_animation(outlines[0], outlines[i], all_histories[i - 1],
                       names[0], names[i], anim_path,
                       border_threshold=args.border_threshold)

    # Summary table
    print("\n=== Alignment Summary ===")
    print(f"{'Piece':<30} {'dx':>8} {'dy':>8} {'Border Pts':>12} "
          f"{'Median Gap':>12} {'P95 Gap':>10} {'Overlap':>10}")
    print("-" * 100)
    print(f"{names[0]:<30} {'0.00':>8} {'0.00':>8} {'(reference)':>12} "
          f"{'---':>12} {'---':>10} {'---':>10}")

    for i in range(1, len(outlines)):
        dx, dy = final_offsets[i]
        border_data = compute_border_gaps(
            outlines[0], outlines[i], dx, dy,
            threshold_mm=args.border_threshold,
        )
        s = border_data["stats"]
        print(f"{names[i]:<30} {dx:>8.2f} {dy:>8.2f} {s['n_border']:>12} "
              f"{s['median']:>12.3f} {s['p95']:>10.3f} {s['overlap_area']:>10.1f}")


if __name__ == "__main__":
    main()
