#!/usr/bin/env python3
# gui_results.py
"""
GUI tool for visualizing beam optimization results
Run with: python gui_results.py [optional_results_file.h5]
"""

import sys
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Ellipse
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import time
import traceback

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QSplitter, QGroupBox, QLabel, QPushButton, QSlider, QComboBox,
                           QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget, QSizePolicy,
                           QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QScrollArea,
                           QStatusBar, QMessageBox, QProgressBar, QFrame, QShortcut, QCheckBox, QGridLayout)
from PyQt5.QtCore import (Qt, QTimer, QThread, pyqtSignal, QSize, QRectF, QPointF, QPropertyAnimation,
                         QEasingCurve, QSettings)
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QPen, QBrush, QColor, QLinearGradient, 
                        QPalette, QFont, QKeySequence, QCursor, QKeyEvent)

class HDF5LoaderThread(QThread):
    """Thread for loading HDF5 files in background to keep GUI responsive"""
    finished = pyqtSignal(dict, str)
    error = pyqtSignal(str)
    
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self._is_running = True
    
    def run(self):
        try:
            if not self._is_running:
                return
                
            data = self.load_hdf5_data(self.filename)
            if self._is_running:  # Only emit if thread is still active
                self.finished.emit(data, self.filename)
        except Exception as e:
            if self._is_running:  # Only emit if thread is still active
                error_msg = f"Failed to load file: {str(e)}\n{traceback.format_exc()}"
                self.error.emit(error_msg)
    
    def stop(self):
        """Safely stop the thread"""
        self._is_running = False
    
    def load_hdf5_data(self, filename):
        """Load data from HDF5 file into a dictionary structure"""
        with h5py.File(filename, 'r') as f:
            # Build history dictionary
            history = {
                'filename': filename,
                'algorithm': f['metadata'].attrs.get('algorithm', 'Unknown'),
                'budget': f['metadata'].attrs.get('budget', 0),
                'early_stop': f['metadata'].attrs.get('early_stop', False),
                'stop_iteration': f['metadata'].attrs.get('stop_iteration', 0),
                'best_iteration_index': f['metadata'].attrs.get('best_iteration_index', 0),
                'device_pvs': [],
                'device_names': [],
                'has_images': True,  # New format always contains images
                'iteration_images': [],
                'iteration_metrics': [],
                'parameters': [],
                'values': [],
                'iterations': [],
                'physical_sizes': [],
                'size_x': [],
                'size_y': [],
                'roundness': [],
                'centroid_x': [],
                'centroid_y': [],
                'is_best': []
            }
            
            # Get device info from metadata
            if 'metadata' in f:
                metadata = f['metadata']
                if 'device_pvs' in metadata:
                    history['device_pvs'] = [pv.decode('utf-8') for pv in metadata['device_pvs'][:]]
                if 'device_names' in metadata:
                    history['device_names'] = [name.decode('utf-8') for name in metadata['device_names'][:]]
            
            # Get basic data from convergence group
            if 'convergence' in f:
                convergence = f['convergence']
                if 'iterations' in convergence:
                    history['iterations'] = list(convergence['iterations'][:])
                    history['is_best'] = [False] * len(history['iterations'])
                if 'scores' in convergence:
                    history['values'] = list(convergence['scores'][:])
                if 'physical_sizes' in convergence:
                    history['physical_sizes'] = list(convergence['physical_sizes'][:])
            
            # Load detailed data from iterations group
            if 'iterations' in f:
                iterations_group = f['iterations']
                iteration_nums = sorted([int(name.split('_')[1]) for name in iterations_group.keys()])
                
                for iter_num in iteration_nums:
                    iter_group = iterations_group[f'iter_{iter_num}']
                    
                    # Load image
                    if 'image' in iter_group:
                        history['iteration_images'].append(iter_group['image'][:])
                    
                    # Load parameters
                    if 'parameters' in iter_group:
                        history['parameters'].append(list(iter_group['parameters'][:]))
                    
                    # Load metrics
                    metrics = {}
                    for attr_name, attr_value in iter_group.attrs.items():
                        metrics[attr_name] = attr_value
                        if attr_name == 'physical_size' and len(history['physical_sizes']) < iter_num:
                            history['physical_sizes'].append(attr_value)
                        elif attr_name == 'size_x' and len(history['size_x']) < iter_num:
                            history['size_x'].append(attr_value)
                        elif attr_name == 'size_y' and len(history['size_y']) < iter_num:
                            history['size_y'].append(attr_value)
                        elif attr_name == 'roundness' and len(history['roundness']) < iter_num:
                            history['roundness'].append(attr_value)
                        elif attr_name == 'centroid_x' and len(history['centroid_x']) < iter_num:
                            history['centroid_x'].append(attr_value)
                        elif attr_name == 'centroid_y' and len(history['centroid_y']) < iter_num:
                            history['centroid_y'].append(attr_value)
                    
                    history['iteration_metrics'].append(metrics)
            
            # Get results from summary group
            if 'summary' in f:
                summary = f['summary'].attrs
                history['initial_physical_size'] = summary.get('initial_physical_size', 0)
                history['best_physical_size'] = summary.get('best_physical_size', 0)
                history['initial_roundness'] = summary.get('initial_roundness', 0)
                history['best_roundness'] = summary.get('best_roundness', 0)
                history['initial_score'] = summary.get('initial_score', 0)
                history['best_score'] = summary.get('best_score', 0)
                history['improvement_percent'] = summary.get('improvement_percent', 0)
            
            # Ensure all lists have consistent length
            max_len = max(len(history['iterations']), len(history['values']), len(history['parameters']))
            if len(history['iterations']) < max_len:
                history['iterations'] = list(range(1, max_len + 1))
            
            # Set best iteration marking
            if history['best_iteration_index'] < len(history['is_best']):
                history['is_best'][history['best_iteration_index']] = True
            
            # Set best and initial parameters
            best_idx = history['best_iteration_index']
            if best_idx < len(history['parameters']):
                history['best_params'] = history['parameters'][best_idx]
            else:
                history['best_params'] = history['parameters'][-1] if history['parameters'] else []
            
            if history['parameters']:
                history['initial_values'] = history['parameters'][0]
                if history['values']:
                    history['initial_score'] = history['values'][0]
            
            # Set initial metrics
            if history['iteration_metrics'] and len(history['iteration_metrics']) > 0:
                initial_metrics = history['iteration_metrics'][0]
                history['initial_size_x'] = initial_metrics.get('size_x', 0)
                history['initial_size_y'] = initial_metrics.get('size_y', 0)
                history['initial_centroid_x'] = initial_metrics.get('centroid_x', 0)
                history['initial_centroid_y'] = initial_metrics.get('centroid_y', 0)
            
            # Set best metrics
            if best_idx < len(history['iteration_metrics']):
                best_metrics = history['iteration_metrics'][best_idx]
                history['best_size_x'] = best_metrics.get('size_x', 0)
                history['best_size_y'] = best_metrics.get('size_y', 0)
                history['best_centroid_x'] = best_metrics.get('centroid_x', 0)
                history['best_centroid_y'] = best_metrics.get('centroid_y', 0)
            
            return history


class BeamImageView(QGraphicsView):
    """Custom graphics view for displaying beam images with annotations"""
    mouseMoved = pyqtSignal(int, int, float)  # x, y, intensity

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        # Create scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        # Image item
        self.image_item = None
        self.overlay_items = []
        # Image data
        self.image_data = None
        
        # Enable mouse tracking
        self.setMouseTracking(True)
    
    def set_image(self, image_data, centroid_x=None, centroid_y=None, size_x=None, size_y=None, roundness=None):
        """Set image data and display with optional beam annotations - FIXED VERSION"""
        if image_data is None:
            return
        
        # Make a copy to avoid modifying original data
        img = image_data.copy()
        
        # Handle 3D arrays (multi-channel)
        if img.ndim == 3:
            img = img[0]  # Take first channel
        
        # Transpose the image to match expected orientation (CRITICAL FIX)
        img = img.T
        
        self.image_data = img
        
        # Create a figure with matplotlib and render it to an image
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        im = ax.imshow(img, cmap='viridis')
        ax.set_xlabel('X pixels')
        ax.set_ylabel('Y pixels')
        fig.colorbar(im, ax=ax, label='Intensity')
        
        # Add beam annotations if metrics provided
        if centroid_x is not None and centroid_y is not None and size_x is not None and size_y is not None:
            # Swap coordinates for transposed image
            transposed_centroid_x = centroid_y
            transposed_centroid_y = centroid_x
            transposed_size_x = size_y
            transposed_size_y = size_x
            
            # Calculate bounding box
            x_min = max(0, transposed_centroid_x - transposed_size_x/2)
            x_max = min(img.shape[1]-1, transposed_centroid_x + transposed_size_x/2)
            y_min = max(0, transposed_centroid_y - transposed_size_y/2)
            y_max = min(img.shape[0]-1, transposed_centroid_y + transposed_size_y/2)
            
            # Draw rectangle
            rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                            linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            
            # Draw centroid cross
            ax.plot(transposed_centroid_x, transposed_centroid_y, 'w+', 
                   markersize=15, linewidth=2, markeredgewidth=2)
            
            # Draw ellipse if needed
            aspect_ratio = max(transposed_size_x, transposed_size_y) / min(transposed_size_x, transposed_size_y) if min(transposed_size_x, transposed_size_y) > 0 else 1.0
            if aspect_ratio > 1.2 and roundness is not None and roundness < 0.8:
                rotation_angle = 30 if aspect_ratio > 1.5 else 0
                ellipse = Ellipse((transposed_centroid_x, transposed_centroid_y),
                                 transposed_size_x, transposed_size_y,
                                 angle=rotation_angle,
                                 edgecolor='yellow', facecolor='none', linewidth=2)
                ax.add_patch(ellipse)
            
            # Add text annotations
            ax.text(0.05, 0.95, f"X size: {size_x:.1f}px", transform=ax.transAxes,
                   color='white', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", fc="blue", ec="none", alpha=0.7))
            ax.text(0.05, 0.90, f"Y size: {size_y:.1f}px", transform=ax.transAxes,
                   color='white', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", fc="blue", ec="none", alpha=0.7))
            ax.text(0.05, 0.85, f"Center: ({centroid_x:.1f}, {centroid_y:.1f})", transform=ax.transAxes,
                   color='white', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", fc="green", ec="none", alpha=0.7))
            ax.text(0.05, 0.80, f"Roundness: {roundness:.3f}", transform=ax.transAxes,
                   color='white', fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3", fc="red", ec="none", alpha=0.7))
        
        fig.tight_layout()
        
        # Convert matplotlib figure to QImage - FIXED THIS PART
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        buffer = fig.canvas.buffer_rgba()
        
        # FIX: Use buffer correctly for memoryview
        # For newer matplotlib versions that return memoryview
        if isinstance(buffer, memoryview):
            qimage = QImage(buffer, width, height, QImage.Format_RGBA8888)
        else:
            # Fallback for older versions
            qimage = QImage(buffer.data, width, height, QImage.Format_RGBA8888)
        
        plt.close(fig)  # Close the figure to free memory
        
        # Create or update image item
        if self.image_item is None:
            self.image_item = QGraphicsPixmapItem()
            self.scene.addItem(self.image_item)
        self.image_item.setPixmap(QPixmap.fromImage(qimage))
        
        # Set scene rectangle
        self.scene.setSceneRect(0, 0, width, height)
        
        # Reset view
        self.resetTransform()
        self.centerOn(width/2, height/2) 
    def add_beam_annotation(self, centroid_x, centroid_y, size_x, size_y, roundness=None):
        """Add beam annotations to the image - FIXED VERSION"""
        if self.image_data is None:
            return
            
        # Calculate bounding box (with bounds checking)
        height, width = self.image_data.shape
        x_min = max(0, centroid_x - size_x/2)
        x_max = min(width-1, centroid_x + size_x/2)
        y_min = max(0, centroid_y - size_y/2)
        y_max = min(height-1, centroid_y + size_y/2)
        
        # Create bounding box (blue dashed line)
        rect = self.scene.addRect(x_min, y_min, x_max-x_min, y_max-y_min, 
                                QPen(QColor(0, 255, 255), 2, Qt.DashLine))
        self.overlay_items.append(rect)
        
        # Create centroid (white cross)
        cross_size = 10
        h_line = self.scene.addLine(centroid_x - cross_size, centroid_y, 
                                centroid_x + cross_size, centroid_y,
                                QPen(QColor(255, 255, 255), 2))
        v_line = self.scene.addLine(centroid_x, centroid_y - cross_size,
                                centroid_x, centroid_y + cross_size,
                                QPen(QColor(255, 255, 255), 2))
        self.overlay_items.extend([h_line, v_line])
        
        # Create ellipse fit (if roundness indicates elliptical shape)
        if roundness is not None and roundness < 0.8:
            # Estimate ellipse angle based on aspect ratio
            aspect_ratio = max(size_x, size_y) / min(size_x, size_y) if min(size_x, size_y) > 0 else 1.0
            rotation_angle = 30 if aspect_ratio > 1.5 else 0
            
            # Create ellipse
            ellipse = self.scene.addEllipse(
                centroid_x - size_x/2, centroid_y - size_y/2,
                size_x, size_y,
                QPen(QColor(255, 255, 0), 2)
            )
            self.overlay_items.append(ellipse)
    
    def reset_transform(self):
        """Reset zoom and pan"""
        self.resetTransform()  # Use resetTransform() instead of resetMatrix()
        self.centerOn(self.scene.width()/2, self.scene.height()/2)
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(zoom_factor, zoom_factor)
    
    def mouseMoveEvent(self, event):
        """Handle mouse movement for pixel value display"""
        super().mouseMoveEvent(event)
        if self.image_data is not None:
            scene_pos = self.mapToScene(event.pos())
            x = int(scene_pos.x())
            y = int(scene_pos.y())
            
            # Check if coordinates are within image bounds
            if 0 <= x < self.image_data.shape[1] and 0 <= y < self.image_data.shape[0]:
                intensity = self.image_data[y, x]
                self.mouseMoved.emit(x, y, intensity)


class MetricsPlot(FigureCanvas):
    """FigureCanvas for displaying metric plots"""
    
    def __init__(self, parent=None, width=4, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)
        
        self.axes = fig.add_subplot(111)
        self.axes.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#f8f9fa')
    
    def plot_metric(self, data, title, ylabel, current_iteration, best_iteration, 
                   current_value=None, best_value=None, show_legend=True):
        """Plot metric evolution over iterations"""
        self.axes.clear()
        self.axes.set_facecolor('#f8f9fa')
        
        iterations = list(range(1, len(data) + 1))
        
        # Plot main curve
        self.axes.plot(iterations, data, 'b-', linewidth=2, alpha=0.7, label=ylabel)
        
        # Mark current point
        if current_iteration is not None and 0 <= current_iteration < len(data):
            self.axes.scatter([iterations[current_iteration]], [data[current_iteration]], 
                            c='red', s=80, zorder=5, label='Current')
        
        # Mark best point
        if best_iteration is not None and 0 <= best_iteration < len(data):
            self.axes.scatter([iterations[best_iteration]], [data[best_iteration]], 
                            c='gold', s=100, marker='*', edgecolors='black', zorder=5, 
                            label='Best')
        
        # Add grid
        self.axes.grid(True, alpha=0.3)
        
        # Set title and labels with larger font sizes
        self.axes.set_title(title, fontsize=10, fontweight='bold')
        self.axes.set_xlabel('Iteration', fontsize=9)
        self.axes.set_ylabel(ylabel, fontsize=9)
        
        # Set ticks
        self.axes.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # Add legend
        if show_legend:
            self.axes.legend(loc='best', fontsize=8)
        
        # Adjust layout
        self.figure.tight_layout()
        self.draw()


class BeamOptimizationGUI(QMainWindow):
    """Main window for beam optimization results visualization"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beam Optimization Results Visualization v1.0")
        self.setGeometry(100, 100, 1200, 800)  # New window size
        
        # Track active threads
        self.active_threads = []
        
        # Set application style
        self.set_style()
        
        # Initialize data
        self.optimization_data = None
        self.current_iteration = 0
        self.best_iteration = 0
        self.loader_thread = None  # Keep reference to loader thread
        
        # Create UI
        self.create_ui()
        
        # Load settings
        self.settings = QSettings("BeamOptimization", "VisualizationTool")
        self.load_settings()
        
        # Load file if specified as command line argument
        if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
            self.load_file(sys.argv[1])
        else:
            # Try to load latest file
            self.load_latest_file()
    
    def set_style(self):
        """Set application style and palette with larger fonts"""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(245, 245, 245))
        palette.setColor(QPalette.WindowText, QColor(40, 40, 40))
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(240, 248, 255))
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
        palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
        palette.setColor(QPalette.Text, QColor(40, 40, 40))
        palette.setColor(QPalette.Button, QColor(230, 230, 230))
        palette.setColor(QPalette.ButtonText, QColor(40, 40, 40))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Highlight, QColor(70, 130, 180))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        self.setPalette(palette)
        
        # Set larger font for entire application
        font = QFont("Arial", 14)  # Increased to 12
        QApplication.setFont(font)
    
    def create_ui(self):
        """Create the main user interface with redesigned layout"""
        # Create central widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Create main layout
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 1. Top control bar
        top_bar = self.create_top_bar()
        main_layout.addWidget(top_bar)
        
        # 2. Main content area (two-column layout)
        content_splitter = QSplitter(Qt.Horizontal)
        
        # 2.1 Center image display area
        center_panel = self.create_center_panel()
        content_splitter.addWidget(center_panel)
        
        # 2.2 Right data analysis panel
        right_panel = self.create_right_panel()
        content_splitter.addWidget(right_panel)
        
        # Set initial sizes
        content_splitter.setSizes([800, 400])  # More space for images
        main_layout.addWidget(content_splitter, 1)
        
        # 3. Bottom control panel (was left panel)
        bottom_panel = self.create_bottom_panel()
        main_layout.addWidget(bottom_panel)
        
        # 4. Status bar
        self.create_status_bar()
    
    def create_top_bar(self):
        """Create the top control bar with larger text"""
        top_bar = QWidget()
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(10, 10, 10, 10)
        top_layout.setSpacing(12)
        
        # Title with larger font
        title_label = QLabel("Beam Optimization Results Visualization v1.0")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        top_layout.addWidget(title_label, 1)
        
        # Button group
        button_group = QGroupBox()
        button_group.setStyleSheet("border: none;")
        button_layout = QHBoxLayout(button_group)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(10)
        
        # Open file button
        open_btn = QPushButton("📂 Open File")
        open_btn.setStyleSheet("font-size: 12px; padding: 5px 10px;")
        open_btn.clicked.connect(self.open_file_dialog)
        button_layout.addWidget(open_btn)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Latest")
        refresh_btn.setStyleSheet("font-size: 12px; padding: 5px 10px;")
        refresh_btn.clicked.connect(self.load_latest_file)
        button_layout.addWidget(refresh_btn)
        
        # Save button
        save_btn = QPushButton("💾 Save As")
        save_btn.setStyleSheet("font-size: 12px; padding: 5px 10px;")
        save_btn.clicked.connect(self.save_current_view)
        button_layout.addWidget(save_btn)
        
        top_layout.addWidget(button_group)
        
        # Status indicators
        status_group = QGroupBox()
        status_group.setStyleSheet("border: none;")
        status_layout = QHBoxLayout(status_group)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(12)
        
        # File path
        self.file_path_label = QLabel("No file loaded")
        self.file_path_label.setStyleSheet("color: #666666; font-size: 12px;")
        self.file_path_label.setToolTip("No file loaded")
        status_layout.addWidget(self.file_path_label)
        
        # Iteration counter with larger font
        self.iteration_counter = QLabel("Iteration 0/0")
        self.iteration_counter.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50;")
        status_layout.addWidget(self.iteration_counter)
        
        # Status indicator
        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(20, 20)  # Slightly larger
        self.set_status_indicator("normal")
        status_layout.addWidget(self.status_indicator)
        
        top_layout.addWidget(status_group)
        
        return top_bar
    
    def create_center_panel(self):
        """Create the center image display panel with larger controls"""
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(10, 10, 10, 10)
        center_layout.setSpacing(12)
        
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("font-size: 12px;")
        
        # 1. Comparison view
        comparison_widget = QWidget()
        comparison_layout = QVBoxLayout(comparison_widget)
        
        # Current iteration image
        current_group = QGroupBox("Current Iteration Beam Spot")
        current_group.setStyleSheet("font-size: 12px; font-weight: bold;")
        current_layout = QVBoxLayout(current_group)
        
        self.current_image_view = BeamImageView()
        self.current_image_view.setMinimumHeight(350)  # Increased height
        self.current_image_view.mouseMoved.connect(self.update_image_status)
        current_layout.addWidget(self.current_image_view)
        
        comparison_layout.addWidget(current_group)
        
        # Initial iteration image
        initial_group = QGroupBox("Initial Beam Spot")
        initial_group.setStyleSheet("font-size: 12px; font-weight: bold;")
        initial_layout = QVBoxLayout(initial_group)
        
        self.initial_image_view = BeamImageView()
        self.initial_image_view.setMinimumHeight(350)  # Increased height
        self.initial_image_view.mouseMoved.connect(self.update_image_status)
        initial_layout.addWidget(self.initial_image_view)
        
        comparison_layout.addWidget(initial_group)
        
        tab_widget.addTab(comparison_widget, "Comparison View")
        
        # 2. Single view
        single_widget = QWidget()
        single_layout = QVBoxLayout(single_widget)
        
        self.single_image_view = BeamImageView()
        self.single_image_view.setMinimumHeight(700)  # Increased height
        self.single_image_view.mouseMoved.connect(self.update_image_status)
        single_layout.addWidget(self.single_image_view)
        
        tab_widget.addTab(single_widget, "Single View")
        
        center_layout.addWidget(tab_widget, 1)
        
        return center_panel
    
    def create_right_panel(self):
        """Create the right data analysis panel with larger fonts"""
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(12)
        
        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("font-size: 12px;")
        
        # 1. Current iteration metrics
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        
        # Metrics cards
        self.metrics_cards = QGroupBox("📊 Current Iteration Metrics")
        self.metrics_cards.setStyleSheet("font-size: 12px; font-weight: bold;")
        metrics_cards_layout = QGridLayout(self.metrics_cards)
        metrics_cards_layout.setContentsMargins(12, 12, 12, 12)
        metrics_cards_layout.setSpacing(12)
        
        # Beam physical size
        self.size_card = self.create_metric_card("Beam Physical Size", "0.0", "px", "#2980b9")
        metrics_cards_layout.addWidget(self.size_card, 0, 0)
        
        # X direction size
        self.size_x_card = self.create_metric_card("X Direction Size", "0.0", "px", "#27ae60")
        metrics_cards_layout.addWidget(self.size_x_card, 0, 1)
        
        # Y direction size
        self.size_y_card = self.create_metric_card("Y Direction Size", "0.0", "px", "#c0392b")
        metrics_cards_layout.addWidget(self.size_y_card, 1, 0)
        
        # Roundness coefficient
        self.roundness_card = self.create_metric_card("Roundness Coefficient", "0.0", "", "#8e44ad")
        metrics_cards_layout.addWidget(self.roundness_card, 1, 1)
        
        metrics_layout.addWidget(self.metrics_cards)
        
        # Real-time charts
        charts_group = QGroupBox("📈 Metric Evolution")
        charts_group.setStyleSheet("font-size: 12px; font-weight: bold;")
        charts_layout = QVBoxLayout(charts_group)
        
        # Beam size chart
        self.size_plot = MetricsPlot(width=5, height=2.5)  # Larger charts
        charts_layout.addWidget(self.size_plot)
        
        # Roundness chart
        self.roundness_plot = MetricsPlot(width=5, height=2.5)
        charts_layout.addWidget(self.roundness_plot)
        
        metrics_layout.addWidget(charts_group)
        
        tab_widget.addTab(metrics_tab, "Current Metrics")
        
        # 2. Historical comparison
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        
        # Comparison table with larger font
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(4)
        self.comparison_table.setHorizontalHeaderLabels(["Metric", "Initial", "Current", "Best"])
        self.comparison_table.horizontalHeader().setStretchLastSection(True)
        self.comparison_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.comparison_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.comparison_table.setSelectionMode(QTableWidget.SingleSelection)
        self.comparison_table.setStyleSheet("font-size: 12px;")
        
        history_layout.addWidget(self.comparison_table)
        
        tab_widget.addTab(history_tab, "Historical Comparison")
        
        # 3. Parameter status
        params_tab = QWidget()
        params_layout = QVBoxLayout(params_tab)
        
        # Parameters table with larger font
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(3)
        self.params_table.setHorizontalHeaderLabels(["Device", "Current Value", "Best Value"])
        self.params_table.horizontalHeader().setStretchLastSection(True)
        self.params_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.params_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.params_table.setSelectionMode(QTableWidget.SingleSelection)
        self.params_table.setStyleSheet("font-size: 12px;")
        
        params_layout.addWidget(self.params_table)
        
        # Parameters evolution chart
        self.params_plot = MetricsPlot(width=5, height=2.5)
        params_layout.addWidget(self.params_plot)
        
        tab_widget.addTab(params_tab, "Parameter Status")
        
        right_layout.addWidget(tab_widget, 1)
        
        # Collapse button
        collapse_btn = QPushButton("Collapse Panel ▶")
        collapse_btn.setStyleSheet("font-size: 11px; padding: 4px;")
        collapse_btn.clicked.connect(lambda: self.toggle_panel(right_panel, collapse_btn))
        right_layout.addWidget(collapse_btn)
        
        return right_panel
    
    def create_bottom_panel(self):
        """Create the bottom control panel (was left panel)"""
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(10, 10, 10, 10)
        bottom_layout.setSpacing(10)
        
        # File selection area
        file_group = QGroupBox("📁 File History")
        file_group.setStyleSheet("font-size: 12px; font-weight: bold;")
        file_layout = QVBoxLayout(file_group)
        
        # File history combo box
        self.file_history_combo = QComboBox()
        self.file_history_combo.setStyleSheet("font-size: 12px; padding: 4px;")
        self.file_history_combo.currentIndexChanged.connect(self.file_history_changed)
        file_layout.addWidget(self.file_history_combo)
        
        bottom_layout.addWidget(file_group)
        
        # Current iteration display area
        current_group = QGroupBox("🎯 Current Iteration")
        current_group.setStyleSheet("font-size: 12px; font-weight: bold;")
        current_layout = QVBoxLayout(current_group)
        
        # Current iteration display
        self.current_iter_display = QLabel("Current: Iteration 0")
        self.current_iter_display.setStyleSheet("font-size: 14px; font-weight: bold; color: #2980b9;")
        self.current_iter_display.setAlignment(Qt.AlignCenter)
        current_layout.addWidget(self.current_iter_display)
        
        # Metrics display for current iteration
        metrics_layout = QGridLayout()
        
        self.size_label = QLabel("Size: 0.0 px")
        self.size_label.setStyleSheet("font-size: 12px; color: #27ae60;")
        metrics_layout.addWidget(self.size_label, 0, 0)
        
        self.roundness_label = QLabel("Roundness: 0.0")
        self.roundness_label.setStyleSheet("font-size: 12px; color: #2980b9;")
        metrics_layout.addWidget(self.roundness_label, 0, 1)
        
        self.score_label = QLabel("Score: 0.0")
        self.score_label.setStyleSheet("font-size: 12px; color: #e74c3c;")
        metrics_layout.addWidget(self.score_label, 1, 0, 1, 2)
        
        current_layout.addLayout(metrics_layout)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        self.first_btn = QPushButton("⏮ Initial")
        self.first_btn.setStyleSheet("font-size: 11px; padding: 4px;")
        self.first_btn.setToolTip("Go to initial state")
        self.first_btn.clicked.connect(self.go_to_first_iteration)
        nav_layout.addWidget(self.first_btn)
        
        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.setStyleSheet("font-size: 11px; padding: 4px;")
        self.prev_btn.setToolTip("Previous iteration")
        self.prev_btn.clicked.connect(self.previous_iteration)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.setStyleSheet("font-size: 11px; padding: 4px;")
        self.next_btn.setToolTip("Next iteration")
        self.next_btn.clicked.connect(self.next_iteration)
        nav_layout.addWidget(self.next_btn)
        
        self.best_btn = QPushButton("⭐ Best")
        self.best_btn.setStyleSheet("font-size: 11px; padding: 4px;")
        self.best_btn.setToolTip("Go to best iteration")
        self.best_btn.clicked.connect(self.go_to_best_iteration)
        nav_layout.addWidget(self.best_btn)
        
        current_layout.addLayout(nav_layout)
        
        bottom_layout.addWidget(current_group, 1)
        
        # Iteration control area
        iteration_group = QGroupBox("📊 Iteration Control")
        iteration_group.setStyleSheet("font-size: 12px; font-weight: bold;")
        iteration_layout = QVBoxLayout(iteration_group)
        
        # Current iteration value display
        self.slider_value_label = QLabel("Iteration: 0")
        self.slider_value_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50;")
        self.slider_value_label.setAlignment(Qt.AlignCenter)
        iteration_layout.addWidget(self.slider_value_label)
        
        # Main slider with larger size and tick marks
        self.iteration_slider = QSlider(Qt.Horizontal)
        self.iteration_slider.setMinimum(0)
        self.iteration_slider.setMaximum(0)
        self.iteration_slider.setTickPosition(QSlider.TicksBelow)
        self.iteration_slider.setTickInterval(5)  # Tick marks every 5 iterations
        self.iteration_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: #f0f0f0;
                height: 12px;
                border-radius: 6px;
            }
            QSlider::handle:horizontal {
                background: #2980b9;
                border: 1px solid #2980b9;
                width: 20px;
                margin: -5px 0;
                border-radius: 10px;
            }
        """)
        self.iteration_slider.valueChanged.connect(self.slider_value_changed)
        iteration_layout.addWidget(self.iteration_slider)
        
        # Labels for min and max iterations
        range_layout = QHBoxLayout()
        self.min_iter_label = QLabel("0")
        self.min_iter_label.setStyleSheet("font-size: 11px; color: #7f8c8d;")
        range_layout.addWidget(self.min_iter_label)
        
        range_layout.addStretch()
        
        self.max_iter_label = QLabel("0")
        self.max_iter_label.setStyleSheet("font-size: 11px; color: #7f8c8d;")
        range_layout.addWidget(self.max_iter_label)
        
        iteration_layout.addLayout(range_layout)
        
        bottom_layout.addWidget(iteration_group)
        
        return bottom_panel
    
    def create_status_bar(self):
        """Create the bottom status bar with larger font"""
        status_bar = QStatusBar()
        status_bar.setStyleSheet("font-size: 12px;")
        self.setStatusBar(status_bar)
        
        # Optimization metadata
        self.meta_label = QLabel("No data loaded")
        self.meta_label.setStyleSheet("font-weight: bold;")
        status_bar.addWidget(self.meta_label, 1)
        
        # Technical info
        self.tech_label = QLabel("Memory: 0MB | Version: 1.0")
        status_bar.addPermanentWidget(self.tech_label)
    
    def create_metric_card(self, title, value, unit, color):
        """Create a metric card widget with larger text"""
        card = QGroupBox()
        card.setStyleSheet(f"""
            QGroupBox {{
                border: 2px solid {color};
                border-radius: 6px;
                margin-top: 1ex;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                font-size: 12px;
                font-weight: bold;
                color: {color};
            }}
        """)
        
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 12px;")
        layout.addWidget(title_label)
        
        value_layout = QHBoxLayout()
        value_label = QLabel(value)
        value_label.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {color};")  # Larger value font
        value_layout.addWidget(value_label)
        
        if unit:
            unit_label = QLabel(unit)
            unit_label.setStyleSheet(f"font-size: 14px; color: {color}; margin-left: 5px;")
            value_layout.addWidget(unit_label)
        
        value_layout.addStretch()
        layout.addLayout(value_layout)
        
        # Save reference
        setattr(self, f"{title.replace(' ', '_')}_value_label", value_label)
        
        return card
    
    def toggle_panel(self, panel, button):
        """Toggle panel visibility"""
        if panel.isVisible():
            panel.hide()
            button.setText("▶ Expand Panel" if "right" in str(panel) else "Expand Panel ▶")
        else:
            panel.show()
            button.setText("◀ Collapse Panel" if "right" in str(panel) else "Collapse Panel ◀")
    
    def find_latest_results_file(self):
        """Find the latest results file in the results directory"""
        supported_extensions = ['.h5']
        results_files = []
        
        # Search in results directory first
        results_dir = "results"
        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            for ext in supported_extensions:
                results_files.extend(glob.glob(f'{results_dir}/*{ext}'))
        
        # Also search in current directory
        for ext in supported_extensions:
            results_files.extend(glob.glob(f'*{ext}'))
        
        if not results_files:
            return None
        
        # Sort by modification time
        latest_file = max(results_files, key=os.path.getmtime)
        return latest_file
    
    def load_latest_file(self):
        """Load the latest results file"""
        self.set_status_indicator("loading")
        self.statusBar().showMessage("Searching for latest file...", 2000)
        
        try:
            latest_file = self.find_latest_results_file()
            if latest_file:
                self.update_file_history(latest_file)
                self.load_file(latest_file)
            else:
                self.statusBar().showMessage("No results files found", 3000)
                self.set_status_indicator("error")
        except Exception as e:
            error_msg = f"Error loading latest file: {str(e)}"
            self.statusBar().showMessage(error_msg, 5000)
            self.set_status_indicator("error")
            QMessageBox.critical(self, "Error", error_msg)
    
    def open_file_dialog(self):
        """Open file dialog to select results file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Results File",
            "results",
            "HDF5 Files (*.h5);;All Files (*)"
        )
        
        if file_path:
            self.update_file_history(file_path)
            self.load_file(file_path)
    
    def update_file_history(self, file_path):
        """Update file history combo box"""
        current_files = [self.file_history_combo.itemText(i) for i in range(self.file_history_combo.count())]
        
        # Add new file to history if not already present
        if file_path not in current_files:
            self.file_history_combo.insertItem(0, file_path)
            
            # Limit history size
            if self.file_history_combo.count() > 10:
                self.file_history_combo.removeItem(self.file_history_combo.count() - 1)
        
        # Select the new file
        self.file_history_combo.setCurrentIndex(0)
    
    def file_history_changed(self, index):
        """Handle file history selection change"""
        if index >= 0:
            file_path = self.file_history_combo.itemText(index)
            if os.path.exists(file_path):
                self.load_file(file_path)
    
    def load_file(self, filename):
        """Load results file"""
        self.file_path_label.setText(os.path.basename(filename))
        self.file_path_label.setToolTip(filename)
        self.set_status_indicator("loading")
        self.statusBar().showMessage(f"Loading file: {filename}", 2000)
        
        # Stop any existing loader thread
        if self.loader_thread and self.loader_thread.isRunning():
            self.loader_thread.stop()
            self.loader_thread.wait(1000)  # Wait up to 1 second
        
        # Create and start new loader thread
        self.loader_thread = HDF5LoaderThread(filename)
        self.active_threads.append(self.loader_thread)
        self.loader_thread.finished.connect(lambda data, fname: self.on_file_loaded(data, fname))
        self.loader_thread.error.connect(lambda error: self.on_file_load_error(error))
        self.loader_thread.finished.connect(self.cleanup_thread)
        self.loader_thread.error.connect(self.cleanup_thread)
        self.loader_thread.start()
    
    def cleanup_thread(self):
        """Clean up finished threads"""
        sender = self.sender()
        if sender in self.active_threads:
            self.active_threads.remove(sender)
    
    def on_file_loaded(self, data, filename):
        """Handle successful file load"""
        self.optimization_data = data
        self.file_path_label.setText(os.path.basename(filename))
        self.file_path_label.setToolTip(filename)
        
        # Update UI
        self.update_ui_with_data()
        
        # Default to best iteration
        self.best_iteration = data['best_iteration_index']
        self.current_iteration = self.best_iteration
        self.update_iteration_display()
        
        self.set_status_indicator("normal")
        self.statusBar().showMessage(f"File loaded successfully: {filename}", 3000)
        
        # Update status bar metadata
        self.update_status_bar_metadata()
    
    def on_file_load_error(self, error_msg):
        """Handle file load error"""
        self.set_status_indicator("error")
        self.statusBar().showMessage(f"Load failed: {error_msg}", 5000)
        QMessageBox.critical(self, "Load Error", error_msg)
    
    def update_ui_with_data(self):
        """Update UI with loaded data"""
        if not self.optimization_data:
            return
        
        data = self.optimization_data
        
        # Update iteration controls
        max_iteration = len(data['iterations']) - 1 if data['iterations'] else 0
        self.iteration_slider.setMaximum(max_iteration)
        self.min_iter_label.setText("1")
        self.max_iter_label.setText(str(max_iteration + 1))
        self.iteration_counter.setText(f"Iteration {self.current_iteration+1}/{max_iteration+1}")
        self.slider_value_label.setText(f"Iteration: {self.current_iteration+1}")
        
        # Update charts
        self.update_plots()
        
        # Update tables
        self.update_comparison_table()
        self.update_params_table()
        
        # Update images
        self.update_image_views()
    
    def update_plots(self):
        """Update all charts"""
        if not self.optimization_data:
            return
        
        data = self.optimization_data
        
        # Beam size evolution chart
        if 'values' in data and data['values']:
            self.size_plot.plot_metric(
                data['values'],
                "Comprehensive Score Evolution",
                "Score",
                self.current_iteration,
                self.best_iteration
            )
        
        # Roundness evolution chart
        if 'iteration_metrics' in data and data['iteration_metrics']:
            roundness_values = [metrics.get('roundness', 0) for metrics in data['iteration_metrics']]
            if roundness_values:
                self.roundness_plot.plot_metric(
                    roundness_values,
                    "Roundness Evolution",
                    "Roundness",
                    self.current_iteration,
                    self.best_iteration
                )
        
        # Parameters evolution chart
        if 'parameters' in data and data['parameters']:
            # Plot first parameter's evolution
            param_index = 0
            param_values = [params[param_index] for params in data['parameters'] if param_index < len(params)]
            if param_values:
                param_name = data['device_pvs'][param_index].split(':')[-1] if data['device_pvs'] else f"Param {param_index+1}"
                self.params_plot.plot_metric(
                    param_values,
                    f"{param_name} Evolution",
                    param_name,
                    self.current_iteration,
                    self.best_iteration,
                    show_legend=False
                )
    
    def update_comparison_table(self):
        """Update comparison table with metrics"""
        if not self.optimization_data:
            return
        
        data = self.optimization_data
        self.comparison_table.setRowCount(0)
        
        # Add rows for metrics
        metrics = [
            ("Beam Physical Size", "physical_size", "px"),
            ("X Direction Size", "size_x", "px"),
            ("Y Direction Size", "size_y", "px"),
            ("Roundness Coefficient", "roundness", ""),
            ("Comprehensive Score", "score", "")
        ]
        
        for row, (name, key, unit) in enumerate(metrics):
            self.comparison_table.insertRow(row)
            self.comparison_table.setItem(row, 0, QTableWidgetItem(name))
            
            # Initial value
            if key == 'physical_size':
                initial_value = data.get('initial_physical_size', 0)
            elif key == 'roundness':
                initial_value = data.get('initial_roundness', 0)
            elif key == 'score':
                initial_value = data.get('initial_score', 0)
            else:
                initial_value = data['iteration_metrics'][0].get(key, 0) if 'iteration_metrics' in data and data['iteration_metrics'] else 0
            
            initial_item = QTableWidgetItem(f"{initial_value:.2f} {unit}")
            self.comparison_table.setItem(row, 1, initial_item)
            
            # Current value
            if self.current_iteration < len(data.get('iteration_metrics', [])):
                current_metrics = data['iteration_metrics'][self.current_iteration]
                current_value = current_metrics.get(key, 0)
            else:
                current_value = 0
            
            current_item = QTableWidgetItem(f"{current_value:.2f} {unit}")
            # Color code: green if better than initial (smaller for size/score, larger for roundness)
            if (key in ['physical_size', 'size_x', 'size_y', 'score'] and current_value < initial_value) or \
               (key == 'roundness' and current_value > initial_value):
                current_item.setBackground(QColor(144, 238, 144))  # Light green
            else:
                current_item.setBackground(QColor(255, 182, 193))  # Light red
            
            self.comparison_table.setItem(row, 2, current_item)
            
            # Best value
            if self.best_iteration < len(data.get('iteration_metrics', [])):
                best_metrics = data['iteration_metrics'][self.best_iteration]
                best_value = best_metrics.get(key, 0)
            else:
                best_value = 0
            
            best_item = QTableWidgetItem(f"{best_value:.2f} {unit}")
            best_item.setBackground(QColor(255, 215, 0))  # Gold
            self.comparison_table.setItem(row, 3, best_item)
        
        self.comparison_table.resizeColumnsToContents()
    
    def update_params_table(self):
        """Update parameters table"""
        if not self.optimization_data:
            return
        
        data = self.optimization_data
        self.params_table.setRowCount(0)
        
        if 'device_pvs' not in data or 'parameters' not in data:
            return
        
        device_pvs = data['device_pvs']
        current_params = data['parameters'][self.current_iteration] if self.current_iteration < len(data['parameters']) else []
        best_params = data['best_params']
        
        for row, pv in enumerate(device_pvs):
            self.params_table.insertRow(row)
            
            # Device name
            device_name = pv.split(':')[-1] if ':' in pv else pv
            self.params_table.setItem(row, 0, QTableWidgetItem(device_name))
            
            # Current value
            current_value = current_params[row] if row < len(current_params) else 0
            current_item = QTableWidgetItem(f"{current_value:.4f}")
            self.params_table.setItem(row, 1, current_item)
            
            # Best value
            best_value = best_params[row] if row < len(best_params) else 0
            best_item = QTableWidgetItem(f"{best_value:.4f}")
            best_item.setBackground(QColor(220, 220, 255))
            self.params_table.setItem(row, 2, best_item)
        
        self.params_table.resizeColumnsToContents()
    
    def update_image_views(self):
        """Update image views with current iteration data"""
        if not self.optimization_data or not self.optimization_data.get('has_images', False):
            print("Warning: No image data available or has_images flag is False")
            return
        
        data = self.optimization_data
        
        # Get current iteration image and metrics
        if self.current_iteration < len(data.get('iteration_images', [])):
            current_image = data['iteration_images'][self.current_iteration]
            current_metrics = data['iteration_metrics'][self.current_iteration] if self.current_iteration < len(data.get('iteration_metrics', [])) else {}
        else:
            current_image = None
            current_metrics = {}
        
        # Get initial image and metrics
        initial_image = data.get('iteration_images', [None])[0] if data.get('iteration_images') else None
        initial_metrics = data['iteration_metrics'][0] if 'iteration_metrics' in data and data['iteration_metrics'] else {}
        
        # Update current image
        if current_image is not None and current_metrics:
            centroid_x = current_metrics.get('centroid_x', 0)
            centroid_y = current_metrics.get('centroid_y', 0)
            size_x = current_metrics.get('size_x', 0)
            size_y = current_metrics.get('size_y', 0)
            roundness = current_metrics.get('roundness', 0)
            physical_size = current_metrics.get('physical_size', 0)
            score = current_metrics.get('score', 0)
            
            self.current_image_view.set_image(current_image, centroid_x, centroid_y, size_x, size_y, roundness)
            
            # Update current iteration display
            self.current_iter_display.setText(f"Current: Iteration {self.current_iteration+1}")
            self.size_label.setText(f"Size: {physical_size:.1f} px")
            self.roundness_label.setText(f"Roundness: {roundness:.3f}")
            self.score_label.setText(f"Score: {score:.2f}")
        else:
            print(f"Warning: No image data for current iteration {self.current_iteration}")
        
        # Update initial image
        if initial_image is not None and initial_metrics:
            centroid_x = initial_metrics.get('centroid_x', 0)
            centroid_y = initial_metrics.get('centroid_y', 0)
            size_x = initial_metrics.get('size_x', 0)
            size_y = initial_metrics.get('size_y', 0)
            roundness = initial_metrics.get('roundness', 0)
            
            self.initial_image_view.set_image(initial_image, centroid_x, centroid_y, size_x, size_y, roundness)
            self.single_image_view.set_image(initial_image, centroid_x, centroid_y, size_x, size_y, roundness)
        else:
            print("Warning: No initial image data available")
        
        # Update metric cards
        self.update_metric_cards(current_metrics)
    
    def update_metric_cards(self, metrics):
        """Update metric cards with current values"""
        if not metrics:
            return
        
        # Beam physical size
        physical_size = metrics.get('physical_size', 0)
        getattr(self, "Beam_Physical_Size_value_label").setText(f"{physical_size:.2f}")
        
        # X direction size
        size_x = metrics.get('size_x', 0)
        getattr(self, "X_Direction_Size_value_label").setText(f"{size_x:.2f}")
        
        # Y direction size
        size_y = metrics.get('size_y', 0)
        getattr(self, "Y_Direction_Size_value_label").setText(f"{size_y:.2f}")
        
        # Roundness coefficient
        roundness = metrics.get('roundness', 0)
        getattr(self, "Roundness_Coefficient_value_label").setText(f"{roundness:.3f}")
    
    def update_iteration_display(self):
        """Update display for current iteration"""
        if not self.optimization_data:
            return
        
        max_iteration = len(self.optimization_data['iterations']) - 1 if self.optimization_data['iterations'] else 0
        self.iteration_slider.setValue(self.current_iteration)
        self.iteration_counter.setText(f"Iteration {self.current_iteration+1}/{max_iteration+1}")
        self.slider_value_label.setText(f"Iteration: {self.current_iteration+1}")
        
        # Update images and charts
        self.update_image_views()
        self.update_plots()
        self.update_comparison_table()
        self.update_params_table()
    
    def slider_value_changed(self, value):
        """Handle slider value change"""
        self.current_iteration = value
        self.update_iteration_display()
    
    def previous_iteration(self):
        """Go to previous iteration"""
        if self.optimization_data and self.current_iteration > 0:
            self.current_iteration -= 1
            self.update_iteration_display()
    
    def next_iteration(self):
        """Go to next iteration"""
        if self.optimization_data:
            max_iteration = len(self.optimization_data['iterations']) - 1 if self.optimization_data['iterations'] else 0
            if self.current_iteration < max_iteration:
                self.current_iteration += 1
                self.update_iteration_display()
    
    def go_to_first_iteration(self):
        """Go to first iteration"""
        if self.optimization_data:
            self.current_iteration = 0
            self.update_iteration_display()
    
    def go_to_best_iteration(self):
        """Go to best iteration"""
        if self.optimization_data:
            self.current_iteration = self.best_iteration
            self.update_iteration_display()
    
    def reset_image_views(self):
        """Reset image views to default zoom/pan"""
        self.current_image_view.reset_transform()
        self.initial_image_view.reset_transform()
        self.single_image_view.reset_transform()
    
    def update_image_status(self, x, y, intensity):
        """Update status bar with image pixel info"""
        self.statusBar().showMessage(f"Coordinates: ({x}, {y}) | Intensity: {intensity:.1f}", 1000)
    
    def save_current_image(self):
        """Save current image to file"""
        if not self.optimization_data:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            f"beam_iteration_{self.current_iteration+1}.png",
            "PNG Files (*.png);;All Files (*)"
        )
        
        if file_path:
            try:
                # Get current displayed image
                if self.current_image_view.image_data is not None:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(self.current_image_view.image_data, cmap='viridis')
                    plt.colorbar(label='Intensity')
                    plt.title(f'Beam Spot Image - Iteration {self.current_iteration+1}')
                    plt.xlabel('X pixels')
                    plt.ylabel('Y pixels')
                    plt.tight_layout()
                    plt.savefig(file_path, dpi=300)
                    plt.close()
                    
                    self.statusBar().showMessage(f"Image saved to: {file_path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save image: {str(e)}")
    
    def save_current_view(self):
        """Save current view to file"""
        if not self.optimization_data:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Current View",
            f"optimization_view_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG Files (*.png);;All Files (*)"
        )
        
        if file_path:
            try:
                # Create screenshot of current view
                screen = QApplication.primaryScreen()
                screenshot = screen.grabWindow(self.winId())
                screenshot.save(file_path)
                
                self.statusBar().showMessage(f"Current view saved to: {file_path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save view: {str(e)}")
    
    def update_status_bar_metadata(self):
        """Update status bar with optimization metadata"""
        if not self.optimization_data:
            return
        
        data = self.optimization_data
        
        # Optimization metadata
        algorithm = data.get('algorithm', 'Unknown')
        budget = data.get('budget', 0)
        iterations = len(data.get('iterations', []))
        early_stop = data.get('early_stop', False)
        timestamp = data.get('timestamp', 'Unknown')
        
        meta_text = f"Algorithm: {algorithm} | Iterations: {iterations}/{budget} | Early Stop: {'Yes' if early_stop else 'No'} | Time: {timestamp}"
        self.meta_label.setText(meta_text)
        
        # Technical info (estimated)
        file_size = data.get('file_size', 0)
        tech_text = f"File: {file_size:.1f}MB | Version: 1.0"
        self.tech_label.setText(tech_text)
    
    def set_status_indicator(self, status):
        """Set status indicator color"""
        if status == "normal":
            self.status_indicator.setStyleSheet("background-color: #27ae60; border-radius: 10px;")
        elif status == "loading":
            self.status_indicator.setStyleSheet("background-color: #f39c12; border-radius: 10px;")
        elif status == "error":
            self.status_indicator.setStyleSheet("background-color: #e74c3c; border-radius: 10px;")
    
    def load_settings(self):
        """Load application settings"""
        # Restore window geometry
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Restore file history
        file_history = self.settings.value("file_history", [])
        for file_path in file_history:
            if os.path.exists(file_path):
                self.file_history_combo.addItem(file_path)
    
    def closeEvent(self, event):
        """Handle window close event"""
        print("Window closing - cleaning up resources...")
        
        # Stop all active threads
        for thread in self.active_threads[:]:  # Iterate over a copy of the list
            if thread.isRunning():
                print(f"Stopping thread: {thread.__class__.__name__}")
                thread.stop()
                thread.wait(1000)  # Wait up to 1 second for thread to finish
        
        # Clear active threads list
        self.active_threads.clear()
        print("All threads stopped")
        
        # Stop and cleanup loader thread if it exists
        if self.loader_thread:
            if self.loader_thread.isRunning():
                print("Stopping loader thread")
                self.loader_thread.stop()
                self.loader_thread.wait(1000)
            self.loader_thread = None
        
        # Save settings
        self.settings.setValue("geometry", self.saveGeometry())
        
        # Save file history
        file_history = [self.file_history_combo.itemText(i) for i in range(self.file_history_combo.count())]
        self.settings.setValue("file_history", file_history)
        print("Settings saved")
        
        print("Window closed successfully")
        event.accept()


def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    
    # Set application attributes
    app.setApplicationName("BeamOptimizationVisualizer")
    app.setOrganizationName("AcceleratorLab")
    
    # Create and show main window
    window = BeamOptimizationGUI()
    window.show()
    
    # Set application exit code
    exit_code = app.exec_()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()