#!/usr/bin/env python3
"""
Interactive STL Alignment & QC GUI

QPainter-based tool for visually aligning 2+ STL pieces:
- Drag to translate, right-drag to rotate
- Real-time border gap statistics
- Run optimizer from current manual pose
- Set border regions to focus alignment on specific edges
- Save/load poses as JSON
- Export QC images

Usage:
    conda run -n demgis python3 stl_align_gui.py piece1.stl piece2.stl [piece3.stl ...]
    conda run -n demgis python3 stl_align_gui.py piece1.stl piece2.stl --poses saved.json
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPointF, QRectF, QEvent
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QPainterPath, QPolygonF, QFont,
    QWheelEvent, QMouseEvent, QPaintEvent, QResizeEvent,
)
from PyQt6.QtWidgets import QGestureEvent

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox, QGroupBox,
    QFileDialog, QSplitter, QMessageBox, QProgressBar,
    QScrollArea, QCheckBox,
)
from shapely.affinity import translate, rotate
from shapely.geometry import Point, box

from align_stls import (
    extract_outline, find_shared_border, align_pair, compute_border_gaps,
    COLORS, plot_alignment, plot_border_detail, plot_animation,
)

# Colors as QColor
QCOLORS = [QColor(c) for c in COLORS]


def _simplify_for_display(geom, tolerance=0.3):
    """Convert shapely geometry to list of numpy coordinate arrays, simplified."""
    if geom is None:
        return []
    simplified = geom.simplify(tolerance, preserve_topology=True)
    if simplified.geom_type == "MultiPolygon":
        polys = list(simplified.geoms)
    elif simplified.geom_type == "Polygon":
        polys = [simplified]
    else:
        return []
    result = []
    for p in polys:
        coords = np.array(p.exterior.coords)
        result.append(coords)
        for ring in p.interiors:
            result.append(np.array(ring.coords))
    return result


def _apply_offset(coords_list, dx, dy, theta_deg, centroid_xy):
    """Apply translation + rotation to coordinate arrays."""
    if dx == 0 and dy == 0 and theta_deg == 0:
        return [c.copy() for c in coords_list]
    result = []
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = centroid_xy
    # Rotate around original centroid, then translate
    for coords in coords_list:
        c = coords.copy()
        if theta_deg != 0:
            rx = c[:, 0] - cx
            ry = c[:, 1] - cy
            c[:, 0] = rx * cos_t - ry * sin_t + cx
            c[:, 1] = rx * sin_t + ry * cos_t + cy
        c[:, 0] += dx
        c[:, 1] += dy
        result.append(c)
    return result


# ============================================================
# FAST CANVAS (QPainter-based)
# ============================================================

class MapCanvas(QWidget):
    """High-performance polygon rendering widget using QPainter."""

    piece_clicked = pyqtSignal(int)  # emits piece index on click
    drag_finished = pyqtSignal()     # emits when drag/rotate ends
    border_region_set = pyqtSignal(int, float, float, float, float)  # piece, x1, y1, x2, y2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Data: list of dicts per piece
        # {base_coords: [...], land_coords: [...], color: QColor, name: str,
        #  centroid: (cx, cy), offset: [dx, dy, theta]}
        self.pieces = []
        self.selected = 1
        self.border_points = []  # list of (x, y, dist) for gap coloring
        self.border_regions = []  # list of (x1, y1, x2, y2) or None per piece
        self.show_dots = True  # toggled by checkbox in main window

        # View transform: screen = (world - pan) * zoom
        self._zoom = 1.0
        self._pan = np.array([0.0, 0.0])  # world coords of center
        self._auto_fitted = False

        # Interaction state
        self._dragging = False
        self._rotating = False
        self._panning = False
        self._border_mode = False  # drawing border region rectangle
        self._drag_start_screen = None
        self._drag_start_world = None
        self._drag_offset_start = None
        self._rotate_start_x = None
        self._rotate_start_theta = None
        self._pan_start_screen = None
        self._pan_start_pan = None
        self._border_rect_start = None
        self._border_rect_end = None

        # Pinch gesture for two-finger rotate + zoom
        self.grabGesture(Qt.GestureType.PinchGesture)
        self._pinch_start_theta = None
        self._pinch_start_zoom = None

        # Legend font
        self._font = QFont("monospace", 10)

    def set_border_mode(self, enabled):
        self._border_mode = enabled
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def auto_fit(self):
        """Fit all pieces in view."""
        if not self.pieces:
            return
        all_x, all_y = [], []
        for p in self.pieces:
            moved = _apply_offset(p["base_coords"], *p["offset"], p["centroid"])
            for coords in moved:
                all_x.extend(coords[:, 0])
                all_y.extend(coords[:, 1])
        if not all_x:
            return
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
        w = self.width()
        h = self.height()
        margin = 40
        data_w = xmax - xmin or 1
        data_h = ymax - ymin or 1
        self._zoom = min((w - 2 * margin) / data_w, (h - 2 * margin) / data_h)
        self._pan = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
        self._auto_fitted = True

    def world_to_screen(self, wx, wy):
        """Convert world coords to screen pixel coords."""
        cx, cy = self.width() / 2, self.height() / 2
        sx = cx + (wx - self._pan[0]) * self._zoom
        sy = cy - (wy - self._pan[1]) * self._zoom  # flip Y
        return sx, sy

    def screen_to_world(self, sx, sy):
        """Convert screen pixel coords to world coords."""
        cx, cy = self.width() / 2, self.height() / 2
        wx = (sx - cx) / self._zoom + self._pan[0]
        wy = -(sy - cy) / self._zoom + self._pan[1]
        return wx, wy

    def _coords_to_qpolygon(self, coords):
        """Convert numpy coords array to QPolygonF in screen space."""
        poly = QPolygonF()
        for x, y in coords:
            sx, sy = self.world_to_screen(x, y)
            poly.append(QPointF(sx, sy))
        return poly

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(255, 255, 255))

        # Draw grid
        self._draw_grid(painter)

        # Draw pieces (base then land)
        for i, p in enumerate(self.pieces):
            self._draw_piece(painter, i, layer="base")
        for i, p in enumerate(self.pieces):
            self._draw_piece(painter, i, layer="land")

        # Draw border gap points
        if self.border_points and self.show_dots:
            self._draw_border_points(painter)

        # Draw border regions
        for i, region in enumerate(self.border_regions):
            if region is not None:
                self._draw_border_region(painter, i, region)

        # Draw border region being drawn
        if self._border_rect_start and self._border_rect_end:
            self._draw_temp_rect(painter)

        # Legend
        self._draw_legend(painter)

        painter.end()

    def _draw_grid(self, painter):
        """Draw coordinate grid."""
        pen = QPen(QColor(220, 220, 220), 1)
        painter.setPen(pen)

        # Determine grid spacing based on zoom
        pixels_per_unit = self._zoom
        target_pixels = 80  # desired grid spacing in pixels
        raw_spacing = target_pixels / pixels_per_unit
        # Round to nice numbers
        mag = 10 ** math.floor(math.log10(max(raw_spacing, 0.001)))
        nice = [1, 2, 5, 10]
        spacing = mag
        for n in nice:
            if n * mag >= raw_spacing:
                spacing = n * mag
                break

        # Get visible world bounds
        wx0, wy0 = self.screen_to_world(0, self.height())
        wx1, wy1 = self.screen_to_world(self.width(), 0)

        # Vertical lines
        x = math.floor(wx0 / spacing) * spacing
        text_pen = QPen(QColor(180, 180, 180), 1)
        painter.setFont(QFont("sans-serif", 8))
        while x <= wx1:
            sx, _ = self.world_to_screen(x, 0)
            painter.setPen(pen)
            painter.drawLine(int(sx), 0, int(sx), self.height())
            painter.setPen(text_pen)
            painter.drawText(int(sx) + 2, self.height() - 4, f"{x:.0f}")
            x += spacing

        # Horizontal lines
        y = math.floor(wy0 / spacing) * spacing
        while y <= wy1:
            _, sy = self.world_to_screen(0, y)
            painter.setPen(pen)
            painter.drawLine(0, int(sy), self.width(), int(sy))
            painter.setPen(text_pen)
            painter.drawText(4, int(sy) - 2, f"{y:.0f}")
            y += spacing

    def _draw_piece(self, painter, idx, layer="base"):
        p = self.pieces[idx]
        coords_key = "base_coords" if layer == "base" else "land_coords"
        coords_list = p.get(coords_key, [])
        if not coords_list:
            return

        moved = _apply_offset(coords_list, *p["offset"], p["centroid"])
        color = QColor(p["color"])

        if layer == "base":
            fill_alpha = 20 if idx != self.selected else 30
            stroke_alpha = 100
            lw = 1.0 if idx != self.selected else 2.0
        else:
            fill_alpha = 60 if idx != self.selected else 80
            stroke_alpha = 200
            lw = 1.5 if idx != self.selected else 2.5

        fill_color = QColor(color)
        fill_color.setAlpha(fill_alpha)
        stroke_color = QColor(color)
        stroke_color.setAlpha(stroke_alpha)

        painter.setPen(QPen(stroke_color, lw))
        painter.setBrush(QBrush(fill_color))

        for coords in moved:
            qpoly = self._coords_to_qpolygon(coords)
            painter.drawPolygon(qpoly)

    def _draw_border_points(self, painter):
        """Draw colored border gap points."""
        for x, y, dist in self.border_points:
            sx, sy = self.world_to_screen(x, y)
            # Color: green (0mm) -> yellow (1.5mm) -> red (3mm+)
            t = min(dist / 3.0, 1.0)
            if t < 0.5:
                r = int(255 * (t * 2))
                g = 255
            else:
                r = 255
                g = int(255 * (1 - (t - 0.5) * 2))
            color = QColor(r, g, 0, 200)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            sz = max(3, min(6, 3 * self._zoom / 2))
            painter.drawEllipse(QPointF(sx, sy), sz, sz)

    def _draw_border_region(self, painter, idx, region):
        """Draw a border region rectangle for a piece."""
        x1, y1, x2, y2 = region
        p = self.pieces[idx]
        # Apply piece offset to region corners
        dx, dy, theta = p["offset"]
        cx, cy = p["centroid"]

        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        if theta != 0:
            th = math.radians(theta)
            cos_t, sin_t = math.cos(th), math.sin(th)
            rx = corners[:, 0] - cx
            ry = corners[:, 1] - cy
            corners[:, 0] = rx * cos_t - ry * sin_t + cx
            corners[:, 1] = rx * sin_t + ry * cos_t + cy
        corners[:, 0] += dx
        corners[:, 1] += dy

        color = QColor(p["color"])
        color.setAlpha(80)
        painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
        painter.setBrush(QBrush(QColor(255, 255, 0, 30)))
        qpoly = QPolygonF()
        for x, y in corners:
            sx, sy = self.world_to_screen(x, y)
            qpoly.append(QPointF(sx, sy))
        painter.drawPolygon(qpoly)

    def _draw_temp_rect(self, painter):
        """Draw the rectangle being drawn in border region mode."""
        sx1, sy1 = self._border_rect_start
        sx2, sy2 = self._border_rect_end
        painter.setPen(QPen(QColor(255, 200, 0), 2, Qt.PenStyle.DashLine))
        painter.setBrush(QBrush(QColor(255, 255, 0, 40)))
        painter.drawRect(QRectF(QPointF(sx1, sy1), QPointF(sx2, sy2)))

    def _draw_legend(self, painter):
        """Draw piece names legend."""
        painter.setFont(self._font)
        x, y = 10, 20
        for i, p in enumerate(self.pieces):
            color = QColor(p["color"])
            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(color))
            painter.drawRect(x, y - 8, 12, 12)
            painter.setPen(QPen(QColor(0, 0, 0)))
            suffix = " [ref]" if i == 0 else (" *" if i == self.selected else "")
            painter.drawText(x + 18, y + 3, f"{p['name']}{suffix}")
            y += 20

    # --- Gesture events (two-finger pinch: rotate selected + zoom) ---

    def event(self, event):
        if event.type() == QEvent.Type.Gesture:
            return self._gesture_event(event)
        return super().event(event)

    def _gesture_event(self, event: QGestureEvent):
        from PyQt6.QtWidgets import QPinchGesture
        pinch = event.gesture(Qt.GestureType.PinchGesture)
        if pinch is None:
            return False

        state = pinch.state()
        if state == Qt.GestureState.GestureStarted:
            self._pinch_start_zoom = self._zoom
            if self.selected >= 1:
                self._pinch_start_theta = self.pieces[self.selected]["offset"][2]
            else:
                self._pinch_start_theta = None

        elif state == Qt.GestureState.GestureUpdated:
            # Zoom from scale factor
            if self._pinch_start_zoom is not None:
                self._zoom = self._pinch_start_zoom * pinch.totalScaleFactor()

            # Rotate selected piece from rotation angle
            if self._pinch_start_theta is not None and self.selected >= 1:
                rot_delta = pinch.totalRotationAngle()
                self.pieces[self.selected]["offset"][2] = self._pinch_start_theta + rot_delta

            self.update()

        elif state in (Qt.GestureState.GestureFinished,
                       Qt.GestureState.GestureCanceled):
            if self._pinch_start_theta is not None:
                self.drag_finished.emit()
            self._pinch_start_zoom = None
            self._pinch_start_theta = None

        event.accept()
        return True

    # --- Mouse events ---

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position()
        sx, sy = pos.x(), pos.y()
        wx, wy = self.screen_to_world(sx, sy)

        # Border region mode
        if self._border_mode and event.button() == Qt.MouseButton.LeftButton:
            self._border_rect_start = (sx, sy)
            self._border_rect_end = (sx, sy)
            return

        # Middle button = pan
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start_screen = (sx, sy)
            self._pan_start_pan = self._pan.copy()
            return

        # Right button = rotate selected
        if event.button() == Qt.MouseButton.RightButton:
            if self.selected >= 1:
                self._rotating = True
                self._rotate_start_x = sx
                self._rotate_start_theta = self.pieces[self.selected]["offset"][2]
            return

        # Left button = select + drag
        if event.button() == Qt.MouseButton.LeftButton:
            hit = self._hit_test(wx, wy)
            if hit >= 0:
                if hit >= 1:
                    self.selected = hit
                    self.piece_clicked.emit(hit)
                    self._dragging = True
                    self._drag_start_world = (wx, wy)
                    self._drag_offset_start = self.pieces[hit]["offset"][:2].copy()
                elif hit == 0:
                    self.piece_clicked.emit(0)
            else:
                # Click on empty space = pan
                self._panning = True
                self._pan_start_screen = (sx, sy)
                self._pan_start_pan = self._pan.copy()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        sx, sy = pos.x(), pos.y()

        if self._border_mode and self._border_rect_start:
            self._border_rect_end = (sx, sy)
            self.update()
            return

        if self._panning:
            dx_px = sx - self._pan_start_screen[0]
            dy_px = sy - self._pan_start_screen[1]
            self._pan[0] = self._pan_start_pan[0] - dx_px / self._zoom
            self._pan[1] = self._pan_start_pan[1] + dy_px / self._zoom
            self.update()
            return

        if self._dragging and self.selected >= 1:
            wx, wy = self.screen_to_world(sx, sy)
            dx_delta = wx - self._drag_start_world[0]
            dy_delta = wy - self._drag_start_world[1]
            self.pieces[self.selected]["offset"][0] = self._drag_offset_start[0] + dx_delta
            self.pieces[self.selected]["offset"][1] = self._drag_offset_start[1] + dy_delta
            self.update()
            return

        if self._rotating and self.selected >= 1:
            dx_pixels = sx - self._rotate_start_x
            angle_delta = dx_pixels * 0.2
            self.pieces[self.selected]["offset"][2] = self._rotate_start_theta + angle_delta
            self.update()
            return

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._border_mode and self._border_rect_start:
            pos = event.position()
            sx2, sy2 = pos.x(), pos.y()
            sx1, sy1 = self._border_rect_start
            # Convert screen rect to world coords (un-transformed piece space)
            wx1, wy1 = self.screen_to_world(min(sx1, sx2), max(sy1, sy2))
            wx2, wy2 = self.screen_to_world(max(sx1, sx2), min(sy1, sy2))
            # Un-apply selected piece's offset to get region in piece-local coords
            if self.selected >= 1:
                p = self.pieces[self.selected]
                dx, dy, theta = p["offset"]
                cx, cy = p["centroid"]
                # Reverse: subtract translation, then reverse rotation
                lwx1, lwy1 = wx1 - dx, wy1 - dy
                lwx2, lwy2 = wx2 - dx, wy2 - dy
                if theta != 0:
                    th = math.radians(-theta)
                    cos_t, sin_t = math.cos(th), math.sin(th)
                    for wx, wy in [(lwx1, lwy1), (lwx2, lwy2)]:
                        pass
                    rx1, ry1 = lwx1 - cx, lwy1 - cy
                    lwx1 = rx1 * cos_t - ry1 * sin_t + cx
                    lwy1 = rx1 * sin_t + ry1 * cos_t + cy
                    rx2, ry2 = lwx2 - cx, lwy2 - cy
                    lwx2 = rx2 * cos_t - ry2 * sin_t + cx
                    lwy2 = rx2 * sin_t + ry2 * cos_t + cy
                self.border_region_set.emit(
                    self.selected,
                    min(lwx1, lwx2), min(lwy1, lwy2),
                    max(lwx1, lwx2), max(lwy1, lwy2),
                )
            self._border_rect_start = None
            self._border_rect_end = None
            self.update()
            return

        was_interacting = self._dragging or self._rotating
        self._dragging = False
        self._rotating = False
        self._panning = False
        self._drag_start_world = None

        if was_interacting:
            self.drag_finished.emit()

    def wheelEvent(self, event: QWheelEvent):
        pos = event.position()
        sx, sy = pos.x(), pos.y()
        wx, wy = self.screen_to_world(sx, sy)

        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15

        self._zoom *= factor
        # Keep world point under cursor fixed
        self._pan[0] = wx - (sx - self.width() / 2) / self._zoom
        self._pan[1] = wy + (sy - self.height() / 2) / self._zoom
        self.update()

    def _hit_test(self, wx, wy):
        """Find which piece contains the world point."""
        pt = Point(wx, wy)
        # Check land outlines first (more precise), then base
        for i in range(len(self.pieces) - 1, -1, -1):
            p = self.pieces[i]
            land = p.get("land_geom")
            if land is not None:
                moved = translate(land, xoff=p["offset"][0], yoff=p["offset"][1])
                if p["offset"][2] != 0:
                    base = translate(p["base_geom"], xoff=p["offset"][0], yoff=p["offset"][1])
                    cx, cy = base.centroid.x, base.centroid.y
                    moved = rotate(moved, p["offset"][2], origin=(cx, cy))
                if moved.contains(pt):
                    return i
        for i in range(len(self.pieces) - 1, -1, -1):
            p = self.pieces[i]
            base = p.get("base_geom")
            if base is not None:
                moved = translate(base, xoff=p["offset"][0], yoff=p["offset"][1])
                if p["offset"][2] != 0:
                    cx, cy = moved.centroid.x, moved.centroid.y
                    moved = rotate(moved, p["offset"][2], origin=(cx, cy))
                if moved.contains(pt):
                    return i
        return -1

    def resizeEvent(self, event: QResizeEvent):
        if not self._auto_fitted:
            self.auto_fit()
        super().resizeEvent(event)


# ============================================================
# WORKER THREAD
# ============================================================

class AlignWorker(QThread):
    """Run align_pair in background, then generate QC outputs."""
    finished = pyqtSignal(int, float, float, float, dict, list)  # idx, dx, dy, theta, stats, output_files

    def __init__(self, outline_ref, outline_moving, piece_idx,
                 init_dx, init_dy, init_theta, border_region=None,
                 region_only=False, allow_rotation=True,
                 name_ref="ref", name_moving="moving", output_prefix=""):
        super().__init__()
        self.outline_ref = outline_ref
        self.outline_moving = outline_moving
        self.piece_idx = piece_idx
        self.init_dx = init_dx
        self.init_dy = init_dy
        self.init_theta = init_theta
        self.border_region = border_region
        self.region_only = region_only
        self.allow_rotation = allow_rotation
        self.name_ref = name_ref
        self.name_moving = name_moving
        self.output_prefix = output_prefix

    def run(self):
        kwargs = dict(
            init_dx=self.init_dx, init_dy=self.init_dy,
            allow_rotation=self.allow_rotation, border_threshold=5.0,
        )
        if self.border_region is not None:
            kwargs["border_region"] = self.border_region
            kwargs["region_only"] = self.region_only
        dx, dy, theta, stats, history = align_pair(
            self.outline_ref, self.outline_moving, **kwargs,
        )

        # Generate QC outputs in this thread (CPU work, no GUI)
        output_files = []
        if self.output_prefix:
            try:
                border_data = compute_border_gaps(
                    self.outline_ref, self.outline_moving,
                    dx, dy, theta=theta, threshold_mm=5.0,
                )
                detail_path = f"{self.output_prefix}border_{self.name_moving}.png"
                plot_border_detail(border_data, self.name_ref, self.name_moving, detail_path)
                output_files.append(detail_path)
            except Exception as e:
                print(f"  Warning: border detail failed: {e}")

            try:
                anim_path = f"{self.output_prefix}animation_{self.name_moving}.mp4"
                plot_animation(self.outline_ref, self.outline_moving, history,
                               self.name_ref, self.name_moving, anim_path)
                output_files.append(anim_path)
            except Exception as e:
                print(f"  Warning: animation failed: {e}")

        self.finished.emit(self.piece_idx, dx, dy, theta, stats, output_files)


class StatsWorker(QThread):
    """Compute border stats in background."""
    finished = pyqtSignal(str, list)  # text, border_points

    def __init__(self, pieces, names, selected, use_land=True):
        super().__init__()
        geom_key = "land_geom" if use_land else "base_geom"
        self.pieces = [(p.get(geom_key) or p["base_geom"], p["offset"][:]) for p in pieces]
        self.names = names[:]
        self.selected = selected

    def run(self):
        lines = []
        border_pts = []
        ref_geom, ref_off = self.pieces[0]
        if ref_geom is None:
            self.finished.emit("No reference outline", [])
            return

        ref_moved = translate(ref_geom, xoff=ref_off[0], yoff=ref_off[1])
        if ref_off[2] != 0:
            ref_moved = rotate(ref_moved, ref_off[2], origin="centroid")

        for i in range(1, len(self.pieces)):
            geom, off = self.pieces[i]
            if geom is None:
                continue
            moved = translate(geom, xoff=off[0], yoff=off[1])
            if off[2] != 0:
                moved = rotate(moved, off[2], origin="centroid")
            try:
                bpts1, _bpts2, bdists = find_shared_border(ref_moved, moved, threshold_mm=8.0)
                if len(bdists) > 0:
                    overlap = ref_moved.intersection(moved)
                    ov_area = overlap.area if not overlap.is_empty else 0
                    sel = " *" if i == self.selected else ""
                    lines.append(f"{self.names[i]}{sel}")
                    lines.append(f"  Border: {len(bdists)} pts")
                    lines.append(f"  Median: {np.median(bdists):.3f} mm")
                    lines.append(f"  P95:    {np.percentile(bdists, 95):.3f} mm")
                    lines.append(f"  Overlap:{ov_area:.1f} mm2")
                    lines.append("")
                    for j in range(len(bpts1)):
                        border_pts.append((bpts1[j, 0], bpts1[j, 1], bdists[j]))
                else:
                    lines.append(f"{self.names[i]}: no border")
                    lines.append("")
            except Exception as e:
                lines.append(f"{self.names[i]}: error ({e})")
                lines.append("")

        self.finished.emit("\n".join(lines) if lines else "No borders detected", border_pts)


# ============================================================
# MAIN WINDOW
# ============================================================

class AlignGUI(QMainWindow):
    def __init__(self, stl_paths, poses_path=None):
        super().__init__()
        self.stl_paths = stl_paths
        self.names = [os.path.splitext(os.path.basename(p))[0] for p in stl_paths]

        self.selected = max(0, min(1, len(stl_paths) - 1))
        self._worker = None
        self._stats_worker = None
        self._spinboxes = []

        self.setWindowTitle("STL Alignment Tool")
        self.setMinimumSize(1200, 800)

        self._build_ui()
        self._load_outlines()

        if poses_path:
            self._load_poses_file(poses_path)

        self.canvas.auto_fit()
        self.canvas.update()
        if len(self.canvas.pieces) >= 2:
            self._schedule_stats()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left: fast canvas
        self.canvas = MapCanvas()
        self.canvas.piece_clicked.connect(self._on_piece_clicked)
        self.canvas.drag_finished.connect(self._on_drag_finished)
        self.canvas.border_region_set.connect(self._on_border_region_set)
        splitter.addWidget(self.canvas)

        # Right: controls (scrollable)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(panel)
        splitter.addWidget(scroll)

        splitter.setSizes([900, 300])

        # Pieces
        self._pieces_container = QVBoxLayout()
        panel_layout.addLayout(self._pieces_container)

        # Stats
        stats_group = QGroupBox("Border Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_label = QLabel("Loading...")
        self.stats_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        panel_layout.addWidget(stats_group)

        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        self.align_btn = QPushButton("Run Alignment (selected)")
        self.align_btn.clicked.connect(self._run_alignment)
        actions_layout.addWidget(self.align_btn)

        self.align_all_btn = QPushButton("Align All Pieces")
        self.align_all_btn.clicked.connect(self._run_alignment_all)
        actions_layout.addWidget(self.align_all_btn)

        self.rotation_cb = QCheckBox("Allow rotation")
        self.rotation_cb.setChecked(True)
        actions_layout.addWidget(self.rotation_cb)

        self.use_land_cb = QCheckBox("Use land outlines")
        self.use_land_cb.setChecked(False)
        self.use_land_cb.setToolTip(
            "Unchecked: align using base outlines (z=0.5mm) — full STL footprint (default)\n"
            "Checked: align using land outlines (z=1.5mm) — island shapes only"
        )
        self.use_land_cb.toggled.connect(lambda: self._schedule_stats())
        actions_layout.addWidget(self.use_land_cb)

        self.show_dots_cb = QCheckBox("Show gap dots")
        self.show_dots_cb.setChecked(True)
        self.show_dots_cb.setToolTip("Show/hide colored border gap dots on the map")
        self.show_dots_cb.toggled.connect(self._toggle_dots)
        actions_layout.addWidget(self.show_dots_cb)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        actions_layout.addWidget(self.progress)

        # Border region controls
        border_group = QGroupBox("Border Region")
        border_layout = QVBoxLayout(border_group)

        self.border_btn = QPushButton("Draw Border Region")
        self.border_btn.setCheckable(True)
        self.border_btn.toggled.connect(self._toggle_border_mode)
        border_layout.addWidget(self.border_btn)

        self.clear_border_btn = QPushButton("Clear Border Region")
        self.clear_border_btn.clicked.connect(self._clear_border_region)
        border_layout.addWidget(self.clear_border_btn)

        self.region_only_cb = QCheckBox("Region only (strict)")
        self.region_only_cb.setToolTip(
            "Checked: only consider border points inside the region\n"
            "Unchecked: weight region points 5x, still use all borders"
        )
        border_layout.addWidget(self.region_only_cb)

        self.border_info = QLabel("Draw a rectangle on the selected\npiece to focus alignment on\nthat border region.")
        self.border_info.setStyleSheet("font-size: 10px; color: gray;")
        border_layout.addWidget(self.border_info)

        actions_layout.addWidget(border_group)

        # File ops
        save_btn = QPushButton("Save Poses")
        save_btn.clicked.connect(self._save_poses)
        actions_layout.addWidget(save_btn)

        load_btn = QPushButton("Load Poses")
        load_btn.clicked.connect(self._load_poses)
        actions_layout.addWidget(load_btn)

        export_btn = QPushButton("Export QC Snapshot")
        export_btn.clicked.connect(self._export_qc)
        actions_layout.addWidget(export_btn)

        reset_btn = QPushButton("Reset Selected")
        reset_btn.clicked.connect(self._reset_selected)
        actions_layout.addWidget(reset_btn)

        fit_btn = QPushButton("Fit View")
        fit_btn.clicked.connect(self._fit_view)
        actions_layout.addWidget(fit_btn)

        add_stl_btn = QPushButton("Add STL...")
        add_stl_btn.clicked.connect(self._add_stl)
        actions_layout.addWidget(add_stl_btn)

        panel_layout.addWidget(actions_group)
        panel_layout.addStretch()

        self.statusBar().showMessage("Ready")

    def _build_pieces_panel(self):
        while self._pieces_container.count():
            item = self._pieces_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._spinboxes = []

        for i, name in enumerate(self.names):
            group = QGroupBox(f"{'[REF] ' if i == 0 else ''}{name}")
            if i == self.selected:
                group.setStyleSheet("QGroupBox { font-weight: bold; }")
            glayout = QVBoxLayout(group)

            if i == 0:
                lbl = QLabel("Reference piece (fixed)")
                lbl.setStyleSheet("color: gray; font-style: italic;")
                glayout.addWidget(lbl)
                self._spinboxes.append(None)
            else:
                row1 = QHBoxLayout()
                row1.addWidget(QLabel("dx:"))
                dx_spin = QDoubleSpinBox()
                dx_spin.setRange(-2000, 2000)
                dx_spin.setSingleStep(0.5)
                dx_spin.setDecimals(1)
                dx_spin.setValue(self.canvas.pieces[i]["offset"][0])
                dx_spin.valueChanged.connect(lambda v, idx=i: self._spin_changed(idx, "dx", v))
                row1.addWidget(dx_spin)

                row1.addWidget(QLabel("dy:"))
                dy_spin = QDoubleSpinBox()
                dy_spin.setRange(-2000, 2000)
                dy_spin.setSingleStep(0.5)
                dy_spin.setDecimals(1)
                dy_spin.setValue(self.canvas.pieces[i]["offset"][1])
                dy_spin.valueChanged.connect(lambda v, idx=i: self._spin_changed(idx, "dy", v))
                row1.addWidget(dy_spin)
                glayout.addLayout(row1)

                row2 = QHBoxLayout()
                row2.addWidget(QLabel("\u03b8:"))
                theta_spin = QDoubleSpinBox()
                theta_spin.setRange(-180, 180)
                theta_spin.setSingleStep(0.5)
                theta_spin.setDecimals(1)
                theta_spin.setValue(self.canvas.pieces[i]["offset"][2])
                theta_spin.setSuffix("\u00b0")
                theta_spin.valueChanged.connect(lambda v, idx=i: self._spin_changed(idx, "theta", v))
                row2.addWidget(theta_spin)

                select_btn = QPushButton("Select")
                select_btn.clicked.connect(lambda checked, idx=i: self._select_piece(idx))
                row2.addWidget(select_btn)
                glayout.addLayout(row2)

                self._spinboxes.append((dx_spin, dy_spin, theta_spin))

            self._pieces_container.addWidget(group)

    # --------------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------------

    def _load_outlines(self):
        self.statusBar().showMessage("Loading STLs...")
        QApplication.processEvents()

        for idx, path in enumerate(self.stl_paths):
            self.statusBar().showMessage(f"Loading {os.path.basename(path)}...")
            QApplication.processEvents()

            base_geom, _bounds = extract_outline(path, z_height=0.5)
            land_geom, _ = extract_outline(path, z_height=1.5, min_island_frac=0)

            # Pre-compute simplified coordinate arrays for fast drawing
            base_coords = _simplify_for_display(base_geom, tolerance=0.5)
            land_coords = _simplify_for_display(land_geom, tolerance=0.3)

            # Centroid for rotation
            centroid = (0.0, 0.0)
            if base_geom is not None:
                centroid = (base_geom.centroid.x, base_geom.centroid.y)

            self.canvas.pieces.append({
                "name": self.names[idx],
                "color": COLORS[idx % len(COLORS)],
                "base_coords": base_coords,
                "land_coords": land_coords,
                "base_geom": base_geom,
                "land_geom": land_geom,
                "centroid": centroid,
                "offset": [0.0, 0.0, 0.0],
            })
            self.canvas.border_regions.append(None)

        self._build_pieces_panel()
        self.statusBar().showMessage(f"Loaded {len(self.canvas.pieces)} pieces")

    def _add_stl(self):
        """Add one or more STL files via file dialog."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add STL files", "",
            "STL files (*.stl);;All files (*)"
        )
        if not paths:
            return

        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            # Skip if already loaded
            if name in self.names:
                self.statusBar().showMessage(f"Skipped {name} (already loaded)")
                continue

            self.statusBar().showMessage(f"Loading {name}...")
            QApplication.processEvents()

            try:
                base_geom, _bounds = extract_outline(path, z_height=0.5)
                land_geom, _ = extract_outline(path, z_height=1.5, min_island_frac=0)
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Failed to load {name}:\n{e}")
                continue

            base_coords = _simplify_for_display(base_geom, tolerance=0.5)
            land_coords = _simplify_for_display(land_geom, tolerance=0.3)

            centroid = (0.0, 0.0)
            if base_geom is not None:
                centroid = (base_geom.centroid.x, base_geom.centroid.y)

            self.stl_paths.append(path)
            self.names.append(name)
            self.canvas.pieces.append({
                "name": name,
                "color": COLORS[len(self.canvas.pieces) % len(COLORS)],
                "base_coords": base_coords,
                "land_coords": land_coords,
                "base_geom": base_geom,
                "land_geom": land_geom,
                "centroid": centroid,
                "offset": [0.0, 0.0, 0.0],
            })
            self.canvas.border_regions.append(None)

        # If we went from 1 to 2+ pieces, auto-select piece 1
        if self.selected == 0 and len(self.canvas.pieces) >= 2:
            self.selected = 1
            self.canvas.selected = 1

        self._build_pieces_panel()
        self.canvas.auto_fit()
        self.canvas.update()
        self._schedule_stats()
        self.statusBar().showMessage(f"Loaded {len(self.canvas.pieces)} pieces total")

    def _load_poses_file(self, path):
        try:
            with open(path) as f:
                data = json.load(f)
            for p in data.get("pieces", []):
                name = p.get("name", "")
                for i, n in enumerate(self.names):
                    if n == name:
                        self.canvas.pieces[i]["offset"] = [
                            p.get("dx", 0), p.get("dy", 0), p.get("theta", 0)
                        ]
                        break
            self._sync_spinboxes()
            self.statusBar().showMessage(f"Loaded poses from {os.path.basename(path)}")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading poses: {e}")

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    def _on_piece_clicked(self, idx):
        if idx == 0:
            self.statusBar().showMessage("Reference piece (fixed)")
        elif idx >= 1:
            self._select_piece(idx)

    def _on_drag_finished(self):
        self._sync_spinboxes()
        self._schedule_stats()

    def _on_border_region_set(self, piece_idx, x1, y1, x2, y2):
        self.canvas.border_regions[piece_idx] = (x1, y1, x2, y2)
        self.border_btn.setChecked(False)
        self.canvas.set_border_mode(False)
        self.statusBar().showMessage(
            f"Border region set for {self.names[piece_idx]}: "
            f"({x1:.0f},{y1:.0f}) to ({x2:.0f},{y2:.0f})"
        )
        self.canvas.update()

    def _toggle_border_mode(self, checked):
        self.canvas.set_border_mode(checked)
        if checked:
            self.statusBar().showMessage("Draw a rectangle on the selected piece to set border region")
        else:
            self.statusBar().showMessage("Ready")

    def _toggle_dots(self, checked):
        self.canvas.show_dots = checked
        self.canvas.update()

    def _clear_border_region(self):
        if self.selected >= 1:
            self.canvas.border_regions[self.selected] = None
            self.canvas.update()
            self.statusBar().showMessage(f"Cleared border region for {self.names[self.selected]}")

    def _schedule_stats(self):
        """Compute stats in background thread."""
        if len(self.canvas.pieces) < 2:
            return  # need at least 2 pieces
        if self._stats_worker is not None and self._stats_worker.isRunning():
            return  # skip if already computing

        self._stats_worker = StatsWorker(
            self.canvas.pieces, self.names, self.selected,
            use_land=self.use_land_cb.isChecked(),
        )
        self._stats_worker.finished.connect(self._stats_done)
        self._stats_worker.start()

    def _stats_done(self, text, border_pts):
        self.stats_label.setText(text)
        self.canvas.border_points = border_pts
        self.canvas.update()

    # --------------------------------------------------------
    # SPINBOX SYNC
    # --------------------------------------------------------

    def _sync_spinboxes(self):
        for i, spins in enumerate(self._spinboxes):
            if spins is None:
                continue
            dx_s, dy_s, th_s = spins
            off = self.canvas.pieces[i]["offset"]
            dx_s.blockSignals(True)
            dy_s.blockSignals(True)
            th_s.blockSignals(True)
            dx_s.setValue(off[0])
            dy_s.setValue(off[1])
            th_s.setValue(off[2])
            dx_s.blockSignals(False)
            dy_s.blockSignals(False)
            th_s.blockSignals(False)

    def _spin_changed(self, idx, field, value):
        off = self.canvas.pieces[idx]["offset"]
        if field == "dx":
            off[0] = value
        elif field == "dy":
            off[1] = value
        elif field == "theta":
            off[2] = value
        self.canvas.update()
        self._schedule_stats()

    def _select_piece(self, idx):
        if idx < 1:
            return
        self.selected = idx
        self.canvas.selected = idx
        self._build_pieces_panel()
        self.canvas.update()
        self.statusBar().showMessage(f"Selected: {self.names[idx]}")

    def _reset_selected(self):
        if self.selected >= 1:
            self.canvas.pieces[self.selected]["offset"] = [0.0, 0.0, 0.0]
            self._sync_spinboxes()
            self.canvas.update()
            self._schedule_stats()

    def _fit_view(self):
        self.canvas.auto_fit()
        self.canvas.update()

    # --------------------------------------------------------
    # ALIGNMENT
    # --------------------------------------------------------

    def _run_alignment(self):
        if self.selected < 1:
            return
        self._start_alignment([self.selected])

    def _run_alignment_all(self):
        indices = list(range(1, len(self.canvas.pieces)))
        self._start_alignment(indices)

    def _start_alignment(self, indices):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._align_prefix = f"align_{ts}_"
        self._align_queue = list(indices)
        self._align_output_files = []
        self._align_next()

    def _align_next(self):
        if not self._align_queue:
            self.align_btn.setEnabled(True)
            self.align_all_btn.setEnabled(True)
            self.progress.setVisible(False)
            self.canvas.update()
            self._schedule_stats()

            # Generate overview plot now that all pieces are aligned
            try:
                outlines = [p["base_geom"] for p in self.canvas.pieces]
                offsets = [(p["offset"][0], p["offset"][1]) for p in self.canvas.pieces]
                aligned_path = f"{self._align_prefix}aligned.png"
                plot_alignment(outlines, offsets, self.names, aligned_path)
                self._align_output_files.append(aligned_path)
            except Exception as e:
                print(f"  Warning: overview plot failed: {e}")

            if self._align_output_files:
                self.statusBar().showMessage(
                    f"Alignment complete — {len(self._align_output_files)} QC files saved"
                )
            return

        idx = self._align_queue.pop(0)
        self.align_btn.setEnabled(False)
        self.align_all_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.statusBar().showMessage(f"Aligning {self.names[idx]}...")

        off = self.canvas.pieces[idx]["offset"]
        border_region = self.canvas.border_regions[idx]

        self._worker = AlignWorker(
            self.canvas.pieces[0]["base_geom"],
            self.canvas.pieces[idx]["base_geom"],
            idx, off[0], off[1], math.radians(off[2]),
            border_region=border_region,
            region_only=self.region_only_cb.isChecked(),
            allow_rotation=self.rotation_cb.isChecked(),
            name_ref=self.names[0],
            name_moving=self.names[idx],
            output_prefix=self._align_prefix,
        )
        self._worker.finished.connect(self._alignment_done)
        self._worker.start()

    def _alignment_done(self, idx, dx, dy, theta, stats, output_files):
        self.canvas.pieces[idx]["offset"] = [dx, dy, math.degrees(theta)]
        self._sync_spinboxes()
        self._align_output_files.extend(output_files)
        med = stats.get("median", float("inf"))
        n = stats.get("n_border", 0)
        self.statusBar().showMessage(
            f"Aligned {self.names[idx]}: dx={dx:.1f} dy={dy:.1f} "
            f"\u03b8={math.degrees(theta):.1f}\u00b0 | median gap={med:.3f}mm ({n} pts)"
        )
        self._align_next()

    # --------------------------------------------------------
    # SAVE / LOAD / EXPORT
    # --------------------------------------------------------

    def _save_poses(self):
        default_name = "poses_" + "_".join(self.names) + ".json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Poses", default_name, "JSON Files (*.json)"
        )
        if not path:
            return
        data = {"version": 1, "pieces": []}
        for i, name in enumerate(self.names):
            off = self.canvas.pieces[i]["offset"]
            piece_data = {
                "name": name,
                "stl": self.stl_paths[i],
                "dx": off[0], "dy": off[1], "theta": off[2],
                "is_reference": i == 0,
            }
            if self.canvas.border_regions[i] is not None:
                piece_data["border_region"] = list(self.canvas.border_regions[i])
            data["pieces"].append(piece_data)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.statusBar().showMessage(f"Saved poses to {path}")

    def _load_poses(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Poses", "", "JSON Files (*.json)"
        )
        if not path:
            return
        self._load_poses_file(path)
        self.canvas.auto_fit()
        self.canvas.update()
        self._schedule_stats()

    def _export_qc(self):
        """Snapshot current pose as QC PNGs (no re-alignment)."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"qc_snapshot_{ts}_"

        outlines = [p["base_geom"] for p in self.canvas.pieces]
        offsets = [(p["offset"][0], p["offset"][1]) for p in self.canvas.pieces]

        self.statusBar().showMessage("Exporting QC snapshot...")
        QApplication.processEvents()

        files = []
        aligned_path = f"{prefix}aligned.png"
        plot_alignment(outlines, offsets, self.names, aligned_path)
        files.append(aligned_path)

        for i in range(1, len(self.canvas.pieces)):
            off = self.canvas.pieces[i]["offset"]
            border_data = compute_border_gaps(
                outlines[0], outlines[i],
                off[0], off[1], theta=math.radians(off[2]),
                threshold_mm=5.0,
            )
            detail_path = f"{prefix}border_{self.names[i]}.png"
            plot_border_detail(border_data, self.names[0], self.names[i], detail_path)
            files.append(detail_path)

        self.statusBar().showMessage(f"Exported {len(files)} QC snapshots")
        QMessageBox.information(self, "Export Complete",
                                f"QC snapshots saved:\n" + "\n".join(files))


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Interactive STL Alignment GUI")
    parser.add_argument("stls", nargs="*", help="STL files (first is reference)")
    parser.add_argument("--poses", help="JSON file with initial poses")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    if not args.stls:
        # No CLI args — open file picker
        paths, _ = QFileDialog.getOpenFileNames(
            None, "Select STL files (first = reference)",
            "", "STL files (*.stl);;All files (*)"
        )
        if not paths:
            sys.exit(0)
        args.stls = paths

    window = AlignGUI(args.stls, poses_path=args.poses)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
