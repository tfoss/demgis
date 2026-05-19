"""paint_elevation_3mf.py — wrap a country / ocean STL in a Bambu-native
3MF whose triangles carry per-triangle filament assignments encoded as
``paint_color`` attributes (the format Bambu Studio actually reads).

Format reverse-engineered from a saved Bambu 3MF (sample at
``tmp/holder_filament_candycane.3mf``): in
``3D/Objects/object_N.model`` each ``<triangle>`` may carry
``paint_color="K"`` where ``K`` is a single hex char (0–F).
For whole-triangle painting (no sub-triangle subdivision), ``K`` is
the 1-based filament/extruder slot. Triangles with no
``paint_color`` attribute default to filament 1.

Per-triangle classification: each triangle's centroid Z is compared to
the band thresholds and assigned a slot.

Bands (mesh-mm z; z_mm = 2.0 + 0.0020·elevation_m). Authoritative
band tables live in ``paint_bands.toml``; the summary below tracks
that file. Three tile types:

    country tile (--type country; pure-land, no bead-04 extension):
        7 elevation bands from slot 1 (lowland green) up to slot 8
        (purple peaks); slot 4 reserved for ocean blue.

    ocean tile (--type ocean; standalone bead-04 tile):
        z ≤ 1.98 → slot 4 (ocean blue); z > 1.98 → slot 1.

    country with ocean extension (--type country_ocean; e.g. Cuba_Caribbean,
    Korea_Japan, Sri_Lanka, UK_Ireland):
        z ≤ 1.98 → slot 4 (ocean blue — extension region was bridge-
        lowered to 1.5 mm); then the 7 country bands above 1.98.

The 3MF layout mirrors the Bambu sample:
  3D/3dmodel.model            — stub referencing /3D/Objects/object_1.model
  3D/Objects/object_1.model   — the mesh with painted triangles
  3D/_rels/3dmodel.model.rels
  Metadata/model_settings.config — Bambu's per-object record
  Metadata/slice_info.config  — required version stub
  [Content_Types].xml, _rels/.rels

Usage:
    python paint_elevation_3mf.py \\
        --stl STLs/France/.../France_solid.stl --type country
"""

from __future__ import annotations

import argparse
import datetime
import os
import tomllib
import uuid
import zipfile
from typing import Optional
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import trimesh

# ---------------------------------------------------------------------------
# Band table — loaded from ``paint_bands.toml`` so the assignments can be
# edited without touching code. Each band is a 4-tuple matching the legacy
# in-code format: (z_upper_bound_mm, filament_slot_1based, hex_color, label).
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paint_bands.toml")


def load_bands_config(path: Optional[str] = None) -> dict:
    """Load the elevation→slot mapping from a TOML file. Returns a dict
    with keys ``country``, ``ocean``, and ``country_ocean`` (each a list
    of band tuples) and ``min_band_faces`` (int). ``inf`` in the file
    becomes ``float('inf')``.
    """
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "rb") as f:
        data = tomllib.load(f)

    def _to_bands(rows: list) -> list[tuple[float, int, str, str]]:
        out = []
        for row in rows:
            z_max = row["z_max"]
            z_max = float("inf") if z_max == float("inf") else float(z_max)
            out.append((z_max, int(row["slot"]), str(row["color"]), str(row["label"])))
        return out

    return {
        "country": _to_bands(data["country"]),
        "ocean": _to_bands(data["ocean"]),
        "country_ocean": _to_bands(data["country_ocean"]),
        "min_band_faces": int(data.get("min_band_faces", 0)),
    }


_CFG = load_bands_config()
COUNTRY_BANDS = _CFG["country"]
OCEAN_BANDS = _CFG["ocean"]
COUNTRY_OCEAN_BANDS = _CFG["country_ocean"]
MIN_BAND_FACES = _CFG["min_band_faces"]


def classify_faces(centroid_z: np.ndarray, bands: list) -> np.ndarray:
    """Return (n_faces,) int array of 1-based filament slot per triangle."""
    out = np.full(centroid_z.shape, bands[-1][1], dtype=np.int32)
    assigned = np.zeros(centroid_z.shape, dtype=bool)
    for z_upper, slot, _, _ in bands:
        m = (~assigned) & (centroid_z <= z_upper)
        out[m] = slot
        assigned |= m
    return out


def subdivide_at_band_boundaries(mesh: trimesh.Trimesh, bands: list) -> trimesh.Trimesh:
    """Slice the mesh by horizontal planes at every band boundary so no
    output triangle straddles a band — Bambu's native paint tool produces
    clean horizontal cuts because it subdivides crossing triangles; we
    reproduce that by mesh pre-subdivision.

    Each band's upper Z is a cut plane. We slice the mesh by each plane in
    turn, keeping BOTH halves and concatenating — the slice operation
    introduces new vertices/triangles along the cut line so every output
    triangle is wholly on one side.

    Cap=False so we don't add interior fill quads; the original mesh's
    side walls + base + relief stay intact, only edge-crossing triangles
    are split.
    """
    upper_zs = [z for (z, _, _, _) in bands if z != float("inf")]
    current = mesh
    for z in upper_zs:
        if z <= current.bounds[0, 2] + 1e-9 or z >= current.bounds[1, 2] - 1e-9:
            continue
        try:
            above = current.slice_plane(
                plane_origin=[0.0, 0.0, z],
                plane_normal=[0.0, 0.0, 1.0],
                cap=False,
            )
            below = current.slice_plane(
                plane_origin=[0.0, 0.0, z],
                plane_normal=[0.0, 0.0, -1.0],
                cap=False,
            )
        except Exception as e:
            print(f"  warning: slice at z={z:.3f} failed ({e}), skipping")
            continue
        parts = [m for m in (above, below) if m is not None and len(m.faces) > 0]
        if not parts:
            continue
        current = trimesh.util.concatenate(parts)
    # Merge duplicate vertices created on the cut line so the seam is shared.
    current.merge_vertices()
    return current


# ---------------------------------------------------------------------------
# Bambu 3MF writers — multi-file layout matching the sample. Per-triangle
# filament assignment is done via ``paint_color="K"`` (single hex char).
# ---------------------------------------------------------------------------

NS_CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
NS_PROD = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
NS_BBL  = "http://schemas.bambulab.com/package/2021"


def _make_uuid() -> str:
    return str(uuid.uuid4())


def _slot_to_paint_code(slot: int) -> str:
    """Encode an extruder slot as a Bambu/Orca/Prusa ``paint_color`` value.

    Format from ``TriangleSelector::serialize`` in OrcaSlicer's
    ``src/libslic3r/TriangleSelector.cpp`` (around line 1643):

    Each leaf triangle is encoded by 4 or 8 bits (xxyy or zzzzxxyy):
      bits 0-1 (yy): number_of_split_sides (0 for leaf)
      bits 2-3 (xx): state if state ∈ [0,1,2]; else 0b11 (extended marker)
      bits 4-7 (zzzz): state - 3, only when the extended marker is set

    Bitstream is packed LSB-first into bytes; the byte's hex
    representation is written high-nibble-first.

    Resulting mapping (extruder N = state N for MMU painting):
      state 0  → omitted        (default extruder = filament 1)
      state 1  → "4"            (extruder 1, explicit)
      state 2  → "8"            (extruder 2)
      state N (3..18) → f"{N-3:X}C"  (extruder N)

    For filament-slot semantics, slot K (1-based, matching Bambu's UI)
    means paint with extruder K. We omit paint_color for slot 1 so that
    Bambu uses the default extruder.
    """
    if slot == 1:
        return ""             # default extruder; no attribute
    if slot == 2:
        return "8"            # state 2 — bits xx=10, yy=00 → 0b1000 = 8
    if 3 <= slot <= 18:
        # state slot → extended marker + (slot - 3) as a 4-bit nibble.
        # Byte = 0b1100 (low nibble C) | ((slot-3) << 4) (high nibble).
        # As a hex string (high nibble first): f"{slot-3:X}C".
        return f"{slot - 3:X}C"
    raise ValueError(f"slot {slot} outside supported range 1-18")


def build_object_model(
    mesh: trimesh.Trimesh,
    face_slot: np.ndarray,
    object_uuid: str,
) -> bytes:
    """Build a ``3D/Objects/object_N.model`` with painted triangles."""
    parts: list[str] = []
    parts.append(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<model unit="millimeter" xml:lang="en-US"\n'
        f'       xmlns="{NS_CORE}"\n'
        f'       xmlns:BambuStudio="{NS_BBL}"\n'
        f'       xmlns:p="{NS_PROD}"\n'
        '       requiredextensions="p">\n'
        ' <metadata name="BambuStudio:3mfVersion">1</metadata>\n'
        ' <resources>\n'
        f'  <object id="1" p:UUID="{object_uuid}" type="model">\n'
        '   <mesh>\n    <vertices>\n'
    )
    for x, y, z in mesh.vertices:
        parts.append(f'     <vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>\n')
    parts.append('    </vertices>\n    <triangles>\n')
    for (v0, v1, v2), slot in zip(mesh.faces, face_slot):
        slot = int(slot)
        code = _slot_to_paint_code(slot)
        if not code:
            # Default filament (slot 1) — no paint_color attribute.
            parts.append(
                f'     <triangle v1="{int(v0)}" v2="{int(v1)}" v3="{int(v2)}"/>\n'
            )
        else:
            parts.append(
                f'     <triangle v1="{int(v0)}" v2="{int(v1)}" v3="{int(v2)}" '
                f'paint_color="{code}"/>\n'
            )
    parts.append(
        '    </triangles>\n   </mesh>\n  </object>\n'
        ' </resources>\n'
        ' <build/>\n'
        '</model>\n'
    )
    return "".join(parts).encode("utf-8")


def build_top_model(
    base_name: str,
    assembly_id: int,
    assembly_uuid: str,
    component_uuid: str,
    build_item_uuid: str,
    build_uuid: str,
) -> bytes:
    """Build ``3D/3dmodel.model`` — production-extension stub referencing
    the single mesh file at ``/3D/Objects/object_1.model``."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<model unit="millimeter" xml:lang="en-US"\n'
        f'       xmlns="{NS_CORE}"\n'
        f'       xmlns:BambuStudio="{NS_BBL}"\n'
        f'       xmlns:p="{NS_PROD}"\n'
        '       requiredextensions="p">\n'
        ' <metadata name="Application">DEMGIS pipeline / paint_elevation_3mf</metadata>\n'
        ' <metadata name="BambuStudio:3mfVersion">1</metadata>\n'
        f' <metadata name="Title">{xml_escape(base_name)}</metadata>\n'
        ' <resources>\n'
        f'  <object id="{assembly_id}" p:UUID="{assembly_uuid}" type="model">\n'
        '   <components>\n'
        f'    <component p:path="/3D/Objects/object_1.model" objectid="1" '
        f'p:UUID="{component_uuid}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>\n'
        '   </components>\n'
        '  </object>\n'
        ' </resources>\n'
        f' <build p:UUID="{build_uuid}">\n'
        f'  <item objectid="{assembly_id}" p:UUID="{build_item_uuid}" '
        'transform="1 0 0 0 1 0 0 0 1 128 128 0" printable="1"/>\n'
        ' </build>\n'
        '</model>\n'
    ).encode("utf-8")


def build_model_settings(
    base_name: str,
    assembly_id: int,
    n_faces: int,
    source_stl: str,
) -> bytes:
    """Build ``Metadata/model_settings.config`` — Bambu's per-object record.
    Single-part: the mesh is one volume with per-triangle paint_color."""
    src = os.path.basename(source_stl)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<config>\n'
        f'  <object id="{assembly_id}">\n'
        f'    <metadata key="name" value="{xml_escape(base_name)}"/>\n'
        f'    <metadata face_count="{n_faces}"/>\n'
        '    <part id="1" subtype="normal_part">\n'
        f'      <metadata key="name" value="{xml_escape(base_name)}"/>\n'
        f'      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
        f'      <metadata key="source_file" value="{xml_escape(src)}"/>\n'
        f'      <metadata key="source_object_id" value="0"/>\n'
        f'      <metadata key="source_volume_id" value="0"/>\n'
        f'      <metadata key="source_offset_x" value="0"/>\n'
        f'      <metadata key="source_offset_y" value="0"/>\n'
        f'      <metadata key="source_offset_z" value="0"/>\n'
        f'      <mesh_stat face_count="{n_faces}" edges_fixed="0" '
        'degenerate_facets="0" facets_removed="0" facets_reversed="0" backwards_edges="0"/>\n'
        '    </part>\n'
        '  </object>\n'
        '  <plate>\n'
        '    <metadata key="plater_id" value="1"/>\n'
        '    <metadata key="plater_name" value=""/>\n'
        '    <metadata key="locked" value="false"/>\n'
        '    <metadata key="filament_map_mode" value="Auto For Flush"/>\n'
        '    <model_instance>\n'
        f'      <metadata key="object_id" value="{assembly_id}"/>\n'
        '      <metadata key="instance_id" value="0"/>\n'
        '      <metadata key="identify_id" value="1001"/>\n'
        '    </model_instance>\n'
        '  </plate>\n'
        '  <assemble>\n'
        f'   <assemble_item object_id="{assembly_id}" instance_id="0" '
        'transform="1 0 0 0 1 0 0 0 1 128 128 0" offset="0 0 0" />\n'
        '  </assemble>\n'
        '</config>\n'
    ).encode("utf-8")


def build_3dmodel_rels() -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        ' <Relationship Target="/3D/Objects/object_1.model" Id="rel-1" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
        '</Relationships>\n'
    ).encode("utf-8")


CONTENT_TYPES_BAMBU = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
    ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
    ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
    '</Types>\n'
).encode("utf-8")

PKG_RELS_BAMBU = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
    ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
    '</Relationships>\n'
).encode("utf-8")

SLICE_INFO_STUB = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<config>\n'
    '  <header>\n'
    '    <header_item key="X-BBL-Client-Type" value="slicer"/>\n'
    '    <header_item key="X-BBL-Client-Version" value="02.04.00.70"/>\n'
    '  </header>\n'
    '</config>\n'
).encode("utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def paint_and_save(
    stl_path: str,
    out_path: str,
    tile_type: str = "country",
    config_path: Optional[str] = None,
) -> dict:
    cfg = load_bands_config(config_path) if config_path else {
        "country": COUNTRY_BANDS,
        "ocean": OCEAN_BANDS,
        "country_ocean": COUNTRY_OCEAN_BANDS,
    }
    # process=True dedupes vertices and merges coincident faces.
    mesh = trimesh.load(stl_path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"loaded geometry is not a single mesh: {type(mesh)}")
    if tile_type not in cfg:
        raise ValueError(
            f"unknown tile_type {tile_type!r}; expected one of "
            f"{sorted(k for k in cfg if k != 'min_band_faces')}"
        )
    bands = cfg[tile_type]

    # Pre-subdivide triangles that straddle a band boundary so the painted
    # bands have clean horizontal cuts (matches Bambu's native paint).
    mesh = subdivide_at_band_boundaries(mesh, bands)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    centroid_z = vertices[faces].mean(axis=1)[:, 2]
    face_slot = classify_faces(centroid_z, bands)
    counts = {label: int((face_slot == slot).sum()) for _, slot, _, label in bands}

    base_name = os.path.splitext(os.path.basename(stl_path))[0]
    assembly_id = 2
    assembly_uuid = _make_uuid()
    component_uuid = _make_uuid()
    build_item_uuid = _make_uuid()
    build_uuid = _make_uuid()
    object_uuid = _make_uuid()

    top_xml = build_top_model(
        base_name=base_name,
        assembly_id=assembly_id,
        assembly_uuid=assembly_uuid,
        component_uuid=component_uuid,
        build_item_uuid=build_item_uuid,
        build_uuid=build_uuid,
    )
    object_xml = build_object_model(mesh, face_slot, object_uuid)
    settings_xml = build_model_settings(
        base_name=base_name,
        assembly_id=assembly_id,
        n_faces=int(len(faces)),
        source_stl=stl_path,
    )
    rels_xml = build_3dmodel_rels()

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES_BAMBU)
        z.writestr("_rels/.rels", PKG_RELS_BAMBU)
        z.writestr("3D/3dmodel.model", top_xml)
        z.writestr("3D/_rels/3dmodel.model.rels", rels_xml)
        z.writestr("3D/Objects/object_1.model", object_xml)
        z.writestr("Metadata/model_settings.config", settings_xml)
        z.writestr("Metadata/slice_info.config", SLICE_INFO_STUB)

    return {
        "out": out_path,
        "tile_type": tile_type,
        "n_vertices": int(len(vertices)),
        "n_faces": int(len(faces)),
        "face_counts_by_band": counts,
    }


def _default_out_path(stl_path: str) -> str:
    """Each regen lands in its own ``banded_<UTC-ts>`` subdir so iterations
    don't overwrite (per CLAUDE.md)."""
    stl_dir = os.path.dirname(os.path.abspath(stl_path))
    base = os.path.splitext(os.path.basename(stl_path))[0]
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(stl_dir, f"banded_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base}.3mf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stl", required=True)
    ap.add_argument(
        "--type",
        choices=("country", "ocean", "country_ocean"),
        default="country",
        help=(
            "country: pure-land tile (Tajikistan, Germany, France-mainland). "
            "ocean: standalone ocean tile. "
            "country_ocean: country tile with bead-04 ocean extension "
            "(Cuba_Caribbean, Korea_Japan, Sri_Lanka, UK_Ireland)."
        ),
    )
    ap.add_argument("--out", help="explicit output path (default: new timestamped subdir)")
    ap.add_argument(
        "--config",
        help=f"path to a TOML band-config (default: {DEFAULT_CONFIG_PATH})",
    )
    args = ap.parse_args()
    out = args.out or _default_out_path(args.stl)
    info = paint_and_save(args.stl, out, args.type, config_path=args.config)
    print(f"wrote {out}")
    print(f"  vertices: {info['n_vertices']}, faces: {info['n_faces']}")
    print(f"  faces per band:")
    for k, v in info["face_counts_by_band"].items():
        print(f"    {k:22s} {v:>8d}")


if __name__ == "__main__":
    main()
