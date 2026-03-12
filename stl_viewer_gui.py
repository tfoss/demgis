#!/usr/bin/env python3
"""
STL Viewer & Comparison GUI

Lightweight tool for viewing and comparing multiple STL iterations:
- All pieces movable (no fixed reference)
- Supports same-filename STLs from different directories
- Drag to translate, right-drag to rotate, scroll to zoom

Usage:
    conda run -n demgis python3 stl_viewer_gui.py piece1.stl piece2.stl [...]
    conda run -n demgis python3 stl_viewer_gui.py --poses saved.json
    conda run -n demgis python3 stl_viewer_gui.py  # opens file picker
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QEvent
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QPolygonF, QFont,
    QWheelEvent, QMouseEvent, QPaintEvent, QResizeEvent,
)
from PyQt6.QtWidgets import QGestureEvent

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox, QGroupBox,
    QFileDialog, QSplitter, QMessageBox, QScrollArea,
)

from align_stls import extract_outline, COLORS

DEFAULT_BASE_Z = 0.5
DEFAULT_LAND_Z = 1.5

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
# VIEWER CANVAS
# ============================================================

class ViewerCanvas(QWidget):
    """Polygon rendering widget — all pieces movable."""

    piece_clicked = pyqtSignal(int)
    drag_finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 400)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.pieces = []
        self.selected = 0

        # View transform
        self._zoom = 1.0
        self._pan = np.array([0.0, 0.0])
        self._auto_fitted = False

        # Interaction state
        self._dragging = False
        self._rotating = False
        self._panning = False
        self._drag_start_screen = None
        self._drag_start_world = None
        self._drag_offset_start = None
        self._rotate_start_x = None
        self._rotate_start_theta = None
        self._pan_start_screen = None
        self._pan_start_pan = None

        # Pinch gesture
        self.grabGesture(Qt.GestureType.PinchGesture)
        self._pinch_start_theta = None
        self._pinch_start_zoom = None

        self._font = QFont("monospace", 10)

    def auto_fit(self):
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
        cx, cy = self.width() / 2, self.height() / 2
        sx = cx + (wx - self._pan[0]) * self._zoom
        sy = cy - (wy - self._pan[1]) * self._zoom
        return sx, sy

    def screen_to_world(self, sx, sy):
        cx, cy = self.width() / 2, self.height() / 2
        wx = (sx - cx) / self._zoom + self._pan[0]
        wy = -(sy - cy) / self._zoom + self._pan[1]
        return wx, wy

    def _coords_to_qpolygon(self, coords):
        poly = QPolygonF()
        for x, y in coords:
            sx, sy = self.world_to_screen(x, y)
            poly.append(QPointF(sx, sy))
        return poly

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        self._draw_grid(painter)
        for i in range(len(self.pieces)):
            self._draw_piece(painter, i, layer="base")
        for i in range(len(self.pieces)):
            self._draw_piece(painter, i, layer="land")
        self._draw_legend(painter)
        painter.end()

    def _draw_grid(self, painter):
        pen = QPen(QColor(220, 220, 220), 1)
        painter.setPen(pen)
        pixels_per_unit = self._zoom
        target_pixels = 80
        raw_spacing = target_pixels / pixels_per_unit
        mag = 10 ** math.floor(math.log10(max(raw_spacing, 0.001)))
        nice = [1, 2, 5, 10]
        spacing = mag
        for n in nice:
            if n * mag >= raw_spacing:
                spacing = n * mag
                break
        wx0, wy0 = self.screen_to_world(0, self.height())
        wx1, wy1 = self.screen_to_world(self.width(), 0)
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

    def _draw_legend(self, painter):
        painter.setFont(self._font)
        x, y = 10, 20
        for i, p in enumerate(self.pieces):
            color = QColor(p["color"])
            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(color))
            painter.drawRect(x, y - 8, 12, 12)
            painter.setPen(QPen(QColor(0, 0, 0)))
            suffix = " *" if i == self.selected else ""
            painter.drawText(x + 18, y + 3, f"{p['name']}{suffix}")
            y += 20

    # --- Gesture events ---

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
            self._pinch_start_theta = self.pieces[self.selected]["offset"][2]
        elif state == Qt.GestureState.GestureUpdated:
            if self._pinch_start_zoom is not None:
                self._zoom = self._pinch_start_zoom * pinch.totalScaleFactor()
            if self._pinch_start_theta is not None:
                rot_delta = pinch.totalRotationAngle()
                self.pieces[self.selected]["offset"][2] = self._pinch_start_theta + rot_delta
            self.update()
        elif state in (Qt.GestureState.GestureFinished, Qt.GestureState.GestureCanceled):
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

        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start_screen = (sx, sy)
            self._pan_start_pan = self._pan.copy()
            return

        if event.button() == Qt.MouseButton.RightButton:
            self._rotating = True
            self._rotate_start_x = sx
            self._rotate_start_theta = self.pieces[self.selected]["offset"][2]
            return

        if event.button() == Qt.MouseButton.LeftButton:
            hit = self._hit_test(wx, wy)
            if hit >= 0:
                self.selected = hit
                self.piece_clicked.emit(hit)
                self._dragging = True
                self._drag_start_world = (wx, wy)
                self._drag_offset_start = self.pieces[hit]["offset"][:2].copy()
            else:
                self._panning = True
                self._pan_start_screen = (sx, sy)
                self._pan_start_pan = self._pan.copy()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        sx, sy = pos.x(), pos.y()

        if self._panning:
            dx_px = sx - self._pan_start_screen[0]
            dy_px = sy - self._pan_start_screen[1]
            self._pan[0] = self._pan_start_pan[0] - dx_px / self._zoom
            self._pan[1] = self._pan_start_pan[1] + dy_px / self._zoom
            self.update()
            return

        if self._dragging:
            wx, wy = self.screen_to_world(sx, sy)
            dx_delta = wx - self._drag_start_world[0]
            dy_delta = wy - self._drag_start_world[1]
            self.pieces[self.selected]["offset"][0] = self._drag_offset_start[0] + dx_delta
            self.pieces[self.selected]["offset"][1] = self._drag_offset_start[1] + dy_delta
            self.update()
            return

        if self._rotating:
            dx_pixels = sx - self._rotate_start_x
            angle_delta = dx_pixels * 0.2
            self.pieces[self.selected]["offset"][2] = self._rotate_start_theta + angle_delta
            self.update()
            return

    def mouseReleaseEvent(self, event: QMouseEvent):
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
        self._pan[0] = wx - (sx - self.width() / 2) / self._zoom
        self._pan[1] = wy + (sy - self.height() / 2) / self._zoom
        self.update()

    def _hit_test(self, wx, wy):
        from shapely.geometry import Point
        from shapely.affinity import translate, rotate
        pt = Point(wx, wy)
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
# MAIN WINDOW
# ============================================================

class ViewerGUI(QMainWindow):
    def __init__(self, stl_paths, poses_path=None):
        super().__init__()
        self.stl_paths = list(stl_paths)
        self.names = []
        self._name_counts = {}  # stem -> count for disambiguation

        for p in self.stl_paths:
            self.names.append(self._make_display_name(p))

        self.selected = 0
        self._spinboxes = []

        self.setWindowTitle("STL Viewer")
        self.setMinimumSize(1200, 800)

        self._build_ui()
        self._load_outlines()

        if poses_path:
            self._load_poses_file(poses_path)

        self.canvas.auto_fit()
        self.canvas.update()

    def _make_display_name(self, path):
        """Create display name, adding parent dir prefix on collision."""
        stem = os.path.splitext(os.path.basename(path))[0]
        parent = os.path.basename(os.path.dirname(os.path.abspath(path)))

        count = self._name_counts.get(stem, 0)
        self._name_counts[stem] = count + 1

        if count == 0:
            # First occurrence — check if we need to retroactively fix existing
            return stem
        elif count == 1:
            # Second occurrence — retroactively add prefix to the first one
            for i, n in enumerate(self.names):
                if n == stem:
                    old_path = self.stl_paths[i]
                    old_parent = os.path.basename(os.path.dirname(os.path.abspath(old_path)))
                    self.names[i] = f"{old_parent}/{stem}"
                    if hasattr(self, 'canvas') and i < len(self.canvas.pieces):
                        self.canvas.pieces[i]["name"] = self.names[i]
                    break
            return f"{parent}/{stem}"
        else:
            return f"{parent}/{stem}"

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left: canvas
        self.canvas = ViewerCanvas()
        self.canvas.piece_clicked.connect(self._on_piece_clicked)
        self.canvas.drag_finished.connect(self._on_drag_finished)
        splitter.addWidget(self.canvas)

        # Right: controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(panel)
        splitter.addWidget(scroll)

        splitter.setSizes([900, 300])

        # Pieces container
        self._pieces_container = QVBoxLayout()
        panel_layout.addLayout(self._pieces_container)

        # Actions
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        add_btn = QPushButton("Add STL...")
        add_btn.clicked.connect(self._add_stl)
        actions_layout.addWidget(add_btn)

        save_btn = QPushButton("Save Poses")
        save_btn.clicked.connect(self._save_poses)
        actions_layout.addWidget(save_btn)

        load_btn = QPushButton("Load Poses")
        load_btn.clicked.connect(self._load_poses)
        actions_layout.addWidget(load_btn)

        export_btn = QPushButton("Export PNG")
        export_btn.clicked.connect(self._export_png)
        actions_layout.addWidget(export_btn)

        reset_btn = QPushButton("Reset Selected")
        reset_btn.clicked.connect(self._reset_selected)
        actions_layout.addWidget(reset_btn)

        fit_btn = QPushButton("Fit View")
        fit_btn.clicked.connect(self._fit_view)
        actions_layout.addWidget(fit_btn)

        panel_layout.addWidget(actions_group)

        # Z-slice control
        z_group = QGroupBox("Z Slice")
        z_layout = QVBoxLayout(z_group)

        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("Base z:"))
        self._base_z_spin = QDoubleSpinBox()
        self._base_z_spin.setRange(0.01, 50.0)
        self._base_z_spin.setSingleStep(0.1)
        self._base_z_spin.setDecimals(2)
        self._base_z_spin.setValue(DEFAULT_BASE_Z)
        self._base_z_spin.setSuffix(" mm")
        z_row.addWidget(self._base_z_spin)
        z_layout.addLayout(z_row)

        z_row2 = QHBoxLayout()
        z_row2.addWidget(QLabel("Land z:"))
        self._land_z_spin = QDoubleSpinBox()
        self._land_z_spin.setRange(0.01, 50.0)
        self._land_z_spin.setSingleStep(0.1)
        self._land_z_spin.setDecimals(2)
        self._land_z_spin.setValue(DEFAULT_LAND_Z)
        self._land_z_spin.setSuffix(" mm")
        z_row2.addWidget(self._land_z_spin)
        z_layout.addLayout(z_row2)

        z_apply_btn = QPushButton("Re-slice")
        z_apply_btn.clicked.connect(self._reslice)
        z_layout.addWidget(z_apply_btn)

        panel_layout.addWidget(z_group)
        panel_layout.addStretch()

        self.statusBar().showMessage("Ready")

    def _build_pieces_panel(self):
        while self._pieces_container.count():
            item = self._pieces_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._spinboxes = []

        for i, name in enumerate(self.names):
            group = QGroupBox(name)
            if i == self.selected:
                group.setStyleSheet("QGroupBox { font-weight: bold; }")
            glayout = QVBoxLayout(group)

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

            base_coords = _simplify_for_display(base_geom, tolerance=0.5)
            land_coords = _simplify_for_display(land_geom, tolerance=0.3)

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

        self._build_pieces_panel()
        self.statusBar().showMessage(f"Loaded {len(self.canvas.pieces)} pieces")

    def _add_stl(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add STL files", "",
            "STL files (*.stl);;All files (*)"
        )
        if not paths:
            return

        for path in paths:
            name = self._make_display_name(path)

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

        self._build_pieces_panel()
        self.canvas.auto_fit()
        self.canvas.update()
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
        self._select_piece(idx)

    def _on_drag_finished(self):
        self._sync_spinboxes()

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

    def _select_piece(self, idx):
        self.selected = idx
        self.canvas.selected = idx
        self._build_pieces_panel()
        self.canvas.update()
        self.statusBar().showMessage(f"Selected: {self.names[idx]}")

    def _reslice(self):
        """Re-extract outlines at the current z-slice heights."""
        base_z = self._base_z_spin.value()
        land_z = self._land_z_spin.value()
        self.statusBar().showMessage(f"Re-slicing at base z={base_z:.2f}, land z={land_z:.2f}...")
        QApplication.processEvents()

        for idx, path in enumerate(self.stl_paths):
            self.statusBar().showMessage(f"Re-slicing {self.names[idx]}...")
            QApplication.processEvents()

            base_geom, _ = extract_outline(path, z_height=base_z)
            land_geom, _ = extract_outline(path, z_height=land_z, min_island_frac=0)

            p = self.canvas.pieces[idx]
            p["base_coords"] = _simplify_for_display(base_geom, tolerance=0.5)
            p["land_coords"] = _simplify_for_display(land_geom, tolerance=0.3)
            p["base_geom"] = base_geom
            p["land_geom"] = land_geom

            if base_geom is not None:
                p["centroid"] = (base_geom.centroid.x, base_geom.centroid.y)

        self.canvas.update()
        self.statusBar().showMessage(
            f"Re-sliced {len(self.stl_paths)} pieces at base z={base_z:.2f}, land z={land_z:.2f}"
        )

    def _reset_selected(self):
        self.canvas.pieces[self.selected]["offset"] = [0.0, 0.0, 0.0]
        self._sync_spinboxes()
        self.canvas.update()

    def _fit_view(self):
        self.canvas.auto_fit()
        self.canvas.update()

    # --------------------------------------------------------
    # SAVE / LOAD / EXPORT
    # --------------------------------------------------------

    def _save_poses(self):
        default_name = "viewer_poses_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Poses", default_name, "JSON Files (*.json)"
        )
        if not path:
            return
        data = {"version": 1, "pieces": []}
        for i, name in enumerate(self.names):
            off = self.canvas.pieces[i]["offset"]
            data["pieces"].append({
                "name": name,
                "stl": self.stl_paths[i],
                "dx": off[0], "dy": off[1], "theta": off[2],
            })
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

    def _export_png(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"stl_viewer_{ts}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PNG", default_name, "PNG Files (*.png)"
        )
        if not path:
            return
        pixmap = self.canvas.grab()
        pixmap.save(path)
        self.statusBar().showMessage(f"Exported {path}")


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="STL Viewer & Comparison GUI")
    parser.add_argument("stls", nargs="*", help="STL files to view")
    parser.add_argument("--poses", help="JSON file with initial poses")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    if not args.stls:
        paths, _ = QFileDialog.getOpenFileNames(
            None, "Select STL files",
            "", "STL files (*.stl);;All files (*)"
        )
        if not paths:
            sys.exit(0)
        args.stls = paths

    window = ViewerGUI(args.stls, poses_path=args.poses)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
