#!/usr/bin/env python3
"""Proof-of-concept: split a single STL into two pieces along an axis-aligned
midline, with one dovetail joint baked in so the pieces interlock.

Investigation deliverable for Bead 11. Not production code.

Approach (boolean cut + add):
  1. Load STL, compute bbox.
  2. Pick cut axis (default: longer of X/Y).
  3. Build a "splitter solid" = half-space prism along the cut axis whose
     boundary on the cut plane carries a trapezoidal (dovetail-shaped) bump.
  4. Piece A = mesh ∩ splitter (gets the tab)
     Piece B = mesh − splitter (gets the matching slot)
  5. Write both pieces; verify (a) each piece is a watertight volume and
     (b) their union closes back to the original (interlock check).

Dovetail geometry: a single trapezoidal tab centred on the cut line. Wider at
its tip than at its base, so it cannot pull apart in-plane (the classic
sliding-dovetail constraint — assembly is by sliding along the joint normal,
i.e. along Z, which is fine for print-and-snap-together if the joint is
sized for a friction fit).

Usage:
    conda run -n demgis python3 bin/poc_dovetail_split.py \\
        --in /workspace/STLs/Korea_Japan/20260512T210211Z/Japan_solid.stl \\
        --out-prefix docs/dovetail_split_poc

    # Or synthetic slab:
    conda run -n demgis python3 bin/poc_dovetail_split.py --synthetic \\
        --out-prefix docs/dovetail_split_poc
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon


# ---- Dovetail joint geometry (defaults; mm) -----------------------------
# These are the load-bearing parameters. Sized for the 200-mm-class slabs
# the print bed actually splits at; the implementation bead will need to
# generalise this (see report §4).
#
# Flare ratio (tip / base) controls how aggressively the tab widens as it
# extends from the cut line. 1.5+ is a classic furniture dovetail; for FDM
# prints with millimetre-scale layer adhesion, 1.15–1.25 is plenty for the
# pull-out constraint and looks much less obtrusive at the cut line.
DEFAULT_TAB_BASE_MM = 30.0   # width of tab along the cut line, at the base
DEFAULT_TAB_TIP_MM = 36.0    # width of tab at the tip (1.2× base = mild flare)
DEFAULT_TAB_DEPTH_MM = 15.0  # how far the tab protrudes from the cut line
DEFAULT_CLEARANCE_MM = 0.20  # gap between tab and slot (FDM friction fit)


def manifold_clean(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Round-trip through manifold3d so STL float32 export doesn't expose
    duplicate faces. Mirrors make_all_sa_with_vector_clip.manifold_clean."""
    from manifold3d import Manifold, Mesh as M3Mesh
    m3 = M3Mesh(
        vert_properties=mesh.vertices.astype(np.float32),
        tri_verts=mesh.faces.astype(np.uint32),
    )
    out = Manifold(m3).to_mesh()
    cleaned = trimesh.Trimesh(
        vertices=np.array(out.vert_properties)[:, :3],
        faces=np.array(out.tri_verts),
    )
    return cleaned


def build_splitter_solid(
    bounds: np.ndarray,
    cut_axis: str,
    cut_coord: float,
    tab_base_mm: float,
    tab_tip_mm: float,
    tab_depth_mm: float,
    clearance_mm: float = 0.0,
    perp_mid_override: float | None = None,
    base_only_z_max: float | None = None,
) -> trimesh.Trimesh:
    """Build a prism that occupies the half-space {cut_axis < cut_coord},
    with a dovetail tab protruding into the OTHER half along +cut_axis.

    The "with clearance" variant shrinks the tab by `clearance_mm/2` on each
    edge so that when used to cut the slot, the slot is slightly oversized.
    The opposite (with negative clearance) would grow the tab. For this PoC
    we pass clearance=0 for the tab piece and clearance=clearance_mm for
    the slot piece, so the slot is slightly larger than the tab.

    Args:
        bounds: mesh bbox, (2,3) array [[xmin,ymin,zmin],[xmax,ymax,zmax]]
        cut_axis: 'x' or 'y'
        cut_coord: position along cut axis where the cut plane sits
        tab_*_mm: trapezoid dimensions (base = at cut line, tip = away from it)
        clearance_mm: enlarge tab outline by this much per side (used for the
                      slot piece). 0 for the tab piece.
    """
    (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
    # Pad in z so the cutter clears mesh top/bottom even after numerical jitter.
    z_pad = max(1.0, (zmax - zmin) * 0.05)
    z_lo = zmin - z_pad
    z_hi = zmax + z_pad

    if cut_axis == "x":
        perp_lo, perp_hi = ymin, ymax
        perp_mid = (perp_mid_override
                    if perp_mid_override is not None
                    else 0.5 * (perp_lo + perp_hi))
    else:
        perp_lo, perp_hi = xmin, xmax
        perp_mid = (perp_mid_override
                    if perp_mid_override is not None
                    else 0.5 * (perp_lo + perp_hi))

    # Build the 2D outline in the (cut_axis, perp_axis) plane.
    # Half-space is everything with cut_axis < cut_coord.
    # Dovetail tab juts out in +cut_axis direction, centred at perp_mid.
    # Trapezoid: base at cut_coord (width = tab_base), tip at cut_coord+depth
    # (width = tab_tip > base ⇒ dovetail).
    base_half = tab_base_mm * 0.5 + clearance_mm
    tip_half = tab_tip_mm * 0.5 + clearance_mm
    depth = tab_depth_mm + clearance_mm  # also lengthen the slot

    # March anticlockwise around the half-space + tab outline.
    # Pad the half-space rectangle so it clears the mesh bbox by ~2 mm.
    pad = 2.0
    if cut_axis == "x":
        outline_xy = [
            (xmin - pad,        perp_lo - pad),                        # SW corner
            (cut_coord,         perp_lo - pad),                        # SE corner (base)
            (cut_coord,         perp_mid - base_half),                 # tab base S
            (cut_coord + depth, perp_mid - tip_half),                  # tab tip S
            (cut_coord + depth, perp_mid + tip_half),                  # tab tip N
            (cut_coord,         perp_mid + base_half),                 # tab base N
            (cut_coord,         perp_hi + pad),                        # NE corner
            (xmin - pad,        perp_hi + pad),                        # NW corner
        ]
    else:
        # cut_axis == 'y'; outline is in (x,y), tab juts in +y
        outline_xy = [
            (perp_lo - pad,        ymin - pad),
            (perp_hi + pad,        ymin - pad),
            (perp_hi + pad,        cut_coord),
            (perp_mid + base_half, cut_coord),
            (perp_mid + tip_half,  cut_coord + depth),
            (perp_mid - tip_half,  cut_coord + depth),
            (perp_mid - base_half, cut_coord),
            (perp_lo - pad,        cut_coord),
        ]

    poly = Polygon(outline_xy)
    assert poly.is_valid, "splitter outline is not a valid polygon"

    if base_only_z_max is None:
        # Original behaviour: dovetail through full mesh height.
        prism = trimesh.creation.extrude_polygon(poly, height=z_hi - z_lo)
        prism.apply_translation([0.0, 0.0, z_lo])
        return prism

    # Base-only dovetail: the dovetail-shaped prism only exists up to
    # base_only_z_max (typically the base slab top, z=2 mm). Above that,
    # the splitter is just a plain half-space at the cut plane — the
    # terrain above the base meets at a clean vertical cut, no joint
    # geometry. This forces the pieces to assemble by sliding together
    # along the cut plane (rather than Z-dropping), and the terrain above
    # the base then locks them against Z-lift (each piece's terrain
    # half rests on the other's base at the seam).
    dovetail_prism = trimesh.creation.extrude_polygon(
        poly, height=base_only_z_max - z_lo,
    )
    dovetail_prism.apply_translation([0.0, 0.0, z_lo])

    # Above z=base_only_z_max: plain half-space rectangle (no tab).
    if cut_axis == "x":
        plain_outline = [
            (xmin - pad, perp_lo - pad),
            (cut_coord,  perp_lo - pad),
            (cut_coord,  perp_hi + pad),
            (xmin - pad, perp_hi + pad),
        ]
    else:
        plain_outline = [
            (perp_lo - pad, ymin - pad),
            (perp_hi + pad, ymin - pad),
            (perp_hi + pad, cut_coord),
            (perp_lo - pad, cut_coord),
        ]
    plain_prism = trimesh.creation.extrude_polygon(
        Polygon(plain_outline), height=z_hi - base_only_z_max,
    )
    plain_prism.apply_translation([0.0, 0.0, base_only_z_max])

    combined = dovetail_prism.union(plain_prism, engine="manifold")
    return combined


def _cross_section_perp_range(
    mesh: trimesh.Trimesh,
    cut_axis: str,
    cut_coord: float,
    slab_mm: float = 0.5,
) -> tuple[float, float] | None:
    """Find the perpendicular-axis range where the mesh has material at
    the cut plane. Returns (perp_min, perp_max) or None if the cut plane
    doesn't intersect any material.

    Without this, the dovetail tab is centred on the full mesh bbox
    midpoint — but for irregular shapes (Cuba's elongated banana, Chile's
    long thin profile, etc.) the country's actual cross-section at the
    cut line is offset from the bbox midpoint. Placing the tab on the
    bbox midpoint leaves part of the tab in mid-air, where the boolean
    intersection produces no material, and the joint becomes a half-tab
    nubbin rather than a proper dovetail.
    """
    axis_idx = 0 if cut_axis == "x" else 1
    perp_idx = 1 - axis_idx
    near = mesh.vertices[
        np.abs(mesh.vertices[:, axis_idx] - cut_coord) < slab_mm
    ]
    if len(near) == 0:
        return None
    return float(near[:, perp_idx].min()), float(near[:, perp_idx].max())


def _keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return only the largest connected component of `mesh` by face count.

    Used post-cut so the output STL is a single connected piece — the
    user's concern with Japan's archipelago was that the cut produced
    >2 disconnected pieces (one per intersected island). For single-
    landmass countries (Cuba) this drops only the tiny offshore islets
    (Isle of Youth + sliver), leaving the main island intact.
    """
    comps = mesh.split(only_watertight=False)
    if len(comps) <= 1:
        return mesh
    largest = max(comps, key=lambda m: len(m.faces))
    n_dropped = len(comps) - 1
    dropped_faces = sum(len(c.faces) for c in comps if c is not largest)
    print(f"    kept largest component ({len(largest.faces)} faces); "
          f"dropped {n_dropped} smaller components ({dropped_faces} faces total)")
    return largest


def split_with_dovetail(
    mesh: trimesh.Trimesh,
    cut_axis: str = "auto",
    cut_coord: float | None = None,
    tab_base_mm: float = DEFAULT_TAB_BASE_MM,
    tab_tip_mm: float = DEFAULT_TAB_TIP_MM,
    tab_depth_mm: float = DEFAULT_TAB_DEPTH_MM,
    clearance_mm: float = DEFAULT_CLEARANCE_MM,
    keep_largest: bool = False,
    base_only_z_max: float | None = None,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Return (tab_piece, slot_piece). Tab piece is the half whose tab juts
    into the slot piece's territory.

    If keep_largest is True, drops disconnected components (e.g. offshore
    islets that fall entirely on one side of the cut) so each output STL
    is a single connected piece. Without this flag, multi-island inputs
    can produce STLs containing several disconnected components that the
    slicer treats as separate parts.
    """
    if cut_axis == "auto":
        ex = mesh.extents
        cut_axis = "x" if ex[0] >= ex[1] else "y"
    (xmin, ymin, _), (xmax, ymax, _) = mesh.bounds
    if cut_coord is None:
        cut_coord = 0.5 * (xmin + xmax) if cut_axis == "x" else 0.5 * (ymin + ymax)

    # Find where the country actually has material at the cut line, so
    # the tab is centred on the cross-section midpoint rather than the
    # bbox midpoint (which would leave part of the tab in mid-air for
    # irregular country shapes).
    section = _cross_section_perp_range(mesh, cut_axis, cut_coord)
    if section is None:
        print(f"  WARNING: no mesh material at cut={cut_coord:.2f}; "
              f"falling back to bbox-midpoint placement")
        perp_mid_override = None
        section_width = None
    else:
        perp_lo, perp_hi = section
        perp_mid_override = 0.5 * (perp_lo + perp_hi)
        section_width = perp_hi - perp_lo
        print(f"  cross-section at cut: perp range [{perp_lo:.2f}, "
              f"{perp_hi:.2f}] mm (width {section_width:.2f} mm, "
              f"midpoint {perp_mid_override:.2f})")
        # The 70% rule: tab tip width should not exceed 70% of the
        # cross-section width. Otherwise the slot piece's material
        # above and below the slot becomes paper-thin, the dovetail
        # tip approaches the country's coastline (so the trapezoid
        # north/south sides get clipped at the coast), and the joint
        # visually degenerates into a Z-shaped cut rather than a
        # clean trapezoidal tab. Also factor in clearance for the
        # slot side.
        max_tip_70 = 0.70 * section_width
        # Slot piece's effective half-width occupied = (tab_tip/2) + clearance
        # for the north-tip and south-tip walls. Total effective = tab_tip + 2*clearance.
        effective_tip = tab_tip_mm + 2.0 * clearance_mm
        if effective_tip > max_tip_70:
            print(f"  WARNING: tab tip + 2× clearance "
                  f"({effective_tip:.2f} mm = {effective_tip/section_width*100:.0f}% "
                  f"of cross-section) exceeds the 70% rule "
                  f"(max {max_tip_70:.2f} mm). The slot piece's "
                  f"ceiling/floor material will be < "
                  f"{(section_width - effective_tip) / 2:.2f} mm thick "
                  f"and the joint may degenerate into a Z-shaped cut. "
                  f"Shrink tab_tip_mm, reduce clearance, or pick a wider "
                  f"cut location.")

    print(f"  splitting along {cut_axis} = {cut_coord:.2f} mm")
    print(f"  tab: base={tab_base_mm}  tip={tab_tip_mm}  depth={tab_depth_mm}  "
          f"clearance={clearance_mm}  keep_largest={keep_largest}")

    # Tab piece = mesh ∩ splitter_tight  (the "<cut" half plus the tab)
    splitter_tight = build_splitter_solid(
        mesh.bounds, cut_axis, cut_coord,
        tab_base_mm, tab_tip_mm, tab_depth_mm, clearance_mm=0.0,
        perp_mid_override=perp_mid_override,
        base_only_z_max=base_only_z_max,
    )
    # Slot piece = mesh − splitter_loose  (the ">cut" half, with a slightly
    # oversized slot carved out where the tab sits)
    splitter_loose = build_splitter_solid(
        mesh.bounds, cut_axis, cut_coord,
        tab_base_mm, tab_tip_mm, tab_depth_mm, clearance_mm=clearance_mm,
        perp_mid_override=perp_mid_override,
        base_only_z_max=base_only_z_max,
    )

    t0 = time.perf_counter()
    tab_piece = mesh.intersection(splitter_tight, engine="manifold")
    t_int = time.perf_counter() - t0
    print(f"  intersection (tab piece): {t_int:.2f} s, {len(tab_piece.faces)} faces")

    t0 = time.perf_counter()
    slot_piece = mesh.difference(splitter_loose, engine="manifold")
    t_diff = time.perf_counter() - t0
    print(f"  difference (slot piece): {t_diff:.2f} s, {len(slot_piece.faces)} faces")

    tab_piece = manifold_clean(tab_piece)
    slot_piece = manifold_clean(slot_piece)

    if keep_largest:
        print("  keep_largest: tab piece")
        tab_piece = _keep_largest_component(tab_piece)
        print("  keep_largest: slot piece")
        slot_piece = _keep_largest_component(slot_piece)

    return tab_piece, slot_piece


def make_synthetic_slab(
    width_mm: float = 400.0,
    depth_mm: float = 200.0,
    height_mm: float = 5.0,
) -> trimesh.Trimesh:
    """A 400×200×5 mm slab in the +octant, origin at (0,0,0)."""
    box = trimesh.creation.box(extents=[width_mm, depth_mm, height_mm])
    box.apply_translation([width_mm * 0.5, depth_mm * 0.5, height_mm * 0.5])
    return box


def verify_interlock(
    tab_piece: trimesh.Trimesh,
    slot_piece: trimesh.Trimesh,
    original: trimesh.Trimesh,
) -> dict:
    """Reassemble and compare against the original mesh as a sanity check."""
    out = {}
    out["tab_watertight"] = bool(tab_piece.is_watertight)
    out["slot_watertight"] = bool(slot_piece.is_watertight)
    out["tab_volume"] = float(tab_piece.volume) if tab_piece.is_volume else None
    out["slot_volume"] = float(slot_piece.volume) if slot_piece.is_volume else None
    out["original_volume"] = float(original.volume) if original.is_volume else None

    try:
        t0 = time.perf_counter()
        reassembled = tab_piece.union(slot_piece, engine="manifold")
        out["union_time_s"] = time.perf_counter() - t0
        out["union_watertight"] = bool(reassembled.is_watertight)
        out["union_volume"] = float(reassembled.volume) if reassembled.is_volume else None
        if out["original_volume"] is not None and out["union_volume"] is not None:
            # With a clearance gap the slot is slightly larger than the tab,
            # so the union of the two pieces (tab+slot) ought to be slightly
            # SMALLER than the original (the gap is missing material).
            out["volume_loss_ratio"] = 1.0 - out["union_volume"] / out["original_volume"]
    except Exception as e:
        out["union_error"] = str(e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path,
                    help="Input STL (omit with --synthetic)")
    ap.add_argument("--synthetic", action="store_true",
                    help="Use a 400×200×5 mm synthetic slab instead of --in")
    ap.add_argument("--out-prefix", type=Path, required=True,
                    help="Output path prefix; writes <prefix>_left.stl and <prefix>_right.stl")
    ap.add_argument("--cut-axis", default="auto", choices=["x", "y", "auto"])
    ap.add_argument("--cut-coord", type=float, default=None,
                    help="Position along cut axis (mm). Default: midline.")
    ap.add_argument("--tab-base-mm", type=float, default=DEFAULT_TAB_BASE_MM)
    ap.add_argument("--tab-tip-mm", type=float, default=DEFAULT_TAB_TIP_MM)
    ap.add_argument("--tab-depth-mm", type=float, default=DEFAULT_TAB_DEPTH_MM)
    ap.add_argument("--clearance-mm", type=float, default=DEFAULT_CLEARANCE_MM)
    ap.add_argument("--keep-largest", action="store_true",
                    help="After the cut, drop disconnected components on "
                         "each side (small offshore islands etc.) so each "
                         "output STL is a single connected piece.")
    ap.add_argument("--xy-scale", type=float, default=1.0,
                    help="Scale the input mesh XY (not Z) by this factor "
                         "before splitting. Useful for PoC print tests "
                         "where the production-scale STL is too small to "
                         "carry printable dovetail features. Defaults to "
                         "1.0 (no scaling).")
    ap.add_argument("--base-only-z-max", type=float, default=None,
                    help="Confine the dovetail shape to z < this value "
                         "(typically the base slab top, 2 mm). Above this "
                         "z, the cut is a plain vertical plane. With this, "
                         "the pieces assemble by sliding along the cut "
                         "plane on the build plate; terrain above the "
                         "base then resists Z-lift. Omit for a full-height "
                         "dovetail prism (assemble by Z-lowering).")
    args = ap.parse_args()

    if args.synthetic:
        mesh = make_synthetic_slab()
        print(f"Synthetic slab: extents={mesh.extents.tolist()}, faces={len(mesh.faces)}")
    else:
        if args.in_path is None:
            ap.error("either --in or --synthetic is required")
        mesh = trimesh.load(args.in_path)
        if not isinstance(mesh, trimesh.Trimesh):
            ap.error(f"loaded scene, not a single mesh: {type(mesh)}")
        print(f"Loaded {args.in_path}: extents={mesh.extents.tolist()}, faces={len(mesh.faces)}")

    if args.xy_scale != 1.0:
        # Scale XY only (preserve Z) so terrain doesn't get exaggerated.
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] = args.xy_scale
        scale_matrix[1, 1] = args.xy_scale
        mesh.apply_transform(scale_matrix)
        # Re-translate so the mesh starts at the origin in XY
        mesh.apply_translation([-mesh.bounds[0, 0], -mesh.bounds[0, 1], 0])
        print(f"  applied --xy-scale={args.xy_scale}: extents now "
              f"{mesh.extents.tolist()}")
    print(f"  watertight={mesh.is_watertight}  volume={mesh.is_volume}")

    t0 = time.perf_counter()
    tab_piece, slot_piece = split_with_dovetail(
        mesh,
        cut_axis=args.cut_axis,
        cut_coord=args.cut_coord,
        tab_base_mm=args.tab_base_mm,
        tab_tip_mm=args.tab_tip_mm,
        tab_depth_mm=args.tab_depth_mm,
        clearance_mm=args.clearance_mm,
        keep_largest=args.keep_largest,
        base_only_z_max=args.base_only_z_max,
    )
    t_split = time.perf_counter() - t0
    print(f"split total: {t_split:.2f} s")

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    left_path = args.out_prefix.with_name(args.out_prefix.name + "_left.stl")
    right_path = args.out_prefix.with_name(args.out_prefix.name + "_right.stl")
    tab_piece.export(left_path)
    slot_piece.export(right_path)
    print(f"wrote {left_path} ({len(tab_piece.faces)} faces)")
    print(f"wrote {right_path} ({len(slot_piece.faces)} faces)")

    report = verify_interlock(tab_piece, slot_piece, mesh)
    print("interlock verification:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
