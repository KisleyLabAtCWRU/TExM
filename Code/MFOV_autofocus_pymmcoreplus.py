import pymmcore_plus
import math
from typing import Any
import numpy as np
import cv2
import time
import os
import sys
import traceback
from datetime import datetime

# GUI Imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QInputDialog, QFileDialog, 
                             QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, 
                             QMessageBox, QComboBox, QLineEdit, QSizePolicy)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QSettings, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush

""" global variables """

# Base configuration: 20X = 0.51 um/px
# Constant = Mag * PixelSize = 20 * 0.51 = 10.2
BASE_MAG_CONSTANT = 10.2 
IMAGE_SIZE = 1412

""" Helper Functions """

def tenengrad(image: Any) -> float:
    """
    Tenegrad algorithm - edge detection based on gradient 
    """
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    tenengrad = np.sqrt(sobel_x**2 + sobel_y**2)
    focus_score = np.mean(tenengrad)
    return focus_score

""" Classes for GUI """

class LiveHistogram(QWidget):
    """Widget to display real-time histogram of 16-bit image intensities."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(120)
        self.num_bins = 512 # Number of display bins for the 16-bit range
        self.hist_data = np.zeros(self.num_bins)
        self.max_intensity = 0
        self.setStyleSheet("background-color: #222; border: 1px solid #444;")

    def update_data(self, img_array):
        # Image is expected to be uint16 (0-65535)
        # We compute histogram over full 16-bit range
        self.max_intensity = np.max(img_array)
        
        # Calculate histogram: Bin the 65536 values into self.num_bins for display
        hist, _ = np.histogram(img_array.ravel(), bins=self.num_bins, range=(0, 65536))
        self.hist_data = hist
        self.update() # Trigger repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        if np.max(self.hist_data) == 0:
            return

        # Log scale helps visualize lower counts in presence of background
        # Add 1 to avoid log(0)
        log_hist = np.log1p(self.hist_data)
        max_val = np.max(log_hist)
        
        w = self.width()
        h = self.height()
        
        # Draw bars
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 200, 255, 180)) # Cyan color

        bin_width = w / self.num_bins
        
        for i in range(self.num_bins):
            val = log_hist[i]
            if max_val > 0:
                bar_height = (val / max_val) * h
            else:
                bar_height = 0
            
            x = i * bin_width
            y = h - bar_height
            painter.drawRect(int(x), int(y), int(math.ceil(bin_width)), int(bar_height))
            
        # --- Saturation Warning ---
        # If max intensity is near 16-bit limit (65535), draw red line
        if self.max_intensity >= 65530:
             painter.setPen(QPen(Qt.red, 3))
             painter.drawLine(w-3, 0, w-3, h)
             
             # Draw "SATURATED" text
             painter.setPen(Qt.red)
             painter.drawText(w - 70, 15, "SATURATION")

class MFOVConfigDialog(QDialog):
    """Popup dialog to get Grid Size, Overlap, Magnification, Laser, Prefix, etc."""
    def __init__(self, current_exp_ms, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MFOV Capture Settings")
        self.resize(450, 400)
        self.current_exp_ms = current_exp_ms
        self.settings = QSettings("MyLab", "MFOV_Microscope") # Persistent settings
        
        layout = QFormLayout()
        
        # 1. Prefix
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setText(self.settings.value("prefix", "Experiment"))
        layout.addRow("File Prefix:", self.prefix_edit)

        # 2. Grid Size
        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(1, 20)
        self.grid_spin.setValue(int(self.settings.value("grid_n", 5)))
        layout.addRow("Grid Size (NxN):", self.grid_spin)
        
        # 3. Overlap
        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setRange(0, 99)
        self.overlap_spin.setValue(float(self.settings.value("overlap", 80.0)))
        self.overlap_spin.setSuffix("%")
        layout.addRow("Overlap (%):", self.overlap_spin)

        # 4. Magnification
        self.mag_combo = QComboBox()
        self.mag_combo.addItems(["10X", "20X", "40X","50X", "60X", "100X"])
        saved_mag = self.settings.value("objective", "20X")
        self.mag_combo.setCurrentText(str(saved_mag))
        self.mag_combo.currentTextChanged.connect(self.update_pixel_size_label)
        
        self.pixel_label = QLabel("0.510 um/px")
        layout.addRow("Objective:", self.mag_combo)
        layout.addRow("Pixel Size:", self.pixel_label)
        
        # 5. Laser
        self.laser_combo = QComboBox()
        self.laser_combo.addItems(["488", "561", "633"])
        saved_laser = self.settings.value("laser", "488")
        self.laser_combo.setCurrentText(str(saved_laser))
        layout.addRow("Laser (nm):", self.laser_combo)
        
        # 6. Directory
        self.dir_btn = QPushButton("Browse...")
        last_dir = self.settings.value("save_dir", os.getcwd())
        self.dir_label = QLabel(last_dir)
        self.dir_btn.clicked.connect(self.select_directory)
        layout.addRow("Save Directory:", self.dir_label)
        layout.addRow("", self.dir_btn)
        
        # Buttons
        self.ok_btn = QPushButton("Start Capture")
        self.ok_btn.clicked.connect(self.save_and_accept)
        layout.addRow(self.ok_btn)
        
        self.setLayout(layout)
        self.selected_path = last_dir
        self.current_pixel_size = 0.51
        
        # Init label
        self.update_pixel_size_label(self.mag_combo.currentText())

    def update_pixel_size_label(self, text):
        try:
            mag = int(text.replace("X", ""))
            self.current_pixel_size = BASE_MAG_CONSTANT / mag
            self.pixel_label.setText(f"{self.current_pixel_size:.3f} um/px")
        except ValueError:
            pass

    def select_directory(self):
        d = QFileDialog.getExistingDirectory(self, "Select Save Directory", self.selected_path)
        if d:
            self.selected_path = d
            self.dir_label.setText(d)

    def save_and_accept(self):
        # Save settings for next time
        self.settings.setValue("prefix", self.prefix_edit.text())
        self.settings.setValue("grid_n", self.grid_spin.value())
        self.settings.setValue("overlap", self.overlap_spin.value())
        self.settings.setValue("objective", self.mag_combo.currentText())
        self.settings.setValue("laser", self.laser_combo.currentText())
        self.settings.setValue("save_dir", self.selected_path)
        self.accept()

    def get_data(self):
        return {
            'prefix': self.prefix_edit.text(),
            'grid_n': self.grid_spin.value(),
            'overlap': self.overlap_spin.value(),
            'objective': self.mag_combo.currentText(),
            'laser': self.laser_combo.currentText(),
            'path': self.selected_path,
            'pixel_size': self.current_pixel_size,
            'exp_ms': self.current_exp_ms
        }

class MicroscopeWindow(QMainWindow):
    """Main GUI Window showing Live View and Controls."""
    def __init__(self, mmc: pymmcore_plus.CMMCorePlus, exposure_ms: float):
        super().__init__()
        self.mmc = mmc
        self.exposure_ms = exposure_ms
        self.setWindowTitle("Microscope Live View")
        self.setGeometry(100, 100, 900, 950)
        
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Live View Label
        self.video_label = QLabel("Initializing Camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(600, 600)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #444;")
        layout.addWidget(self.video_label)
        
        # Histogram
        layout.addWidget(QLabel("Live Histogram (16-bit):"))
        self.histogram_widget = LiveHistogram()
        layout.addWidget(self.histogram_widget)

        # Controls Layout
        controls_layout = QHBoxLayout()
        
        # Capture Button
        self.btn_capture = QPushButton("Capture MFOV")
        self.btn_capture.setFixedHeight(50)
        self.btn_capture.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #4CAF50; color: white;")
        self.btn_capture.clicked.connect(self.handle_mfov_click)
        controls_layout.addWidget(self.btn_capture)

        # Quit Button
        self.btn_quit = QPushButton("Quit / Safe Reset")
        self.btn_quit.setFixedHeight(50)
        self.btn_quit.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #f44336; color: white;")
        self.btn_quit.clicked.connect(self.safe_quit)
        controls_layout.addWidget(self.btn_quit)
        
        layout.addLayout(controls_layout)
        
        # Timer for Live View
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Start Live
        self.start_live()

    def start_live(self):
        try:
            if not self.mmc.isSequenceRunning():
                self.mmc.startContinuousSequenceAcquisition(0)
            self.timer.start(30) # ~30fps update rate for GUI
        except Exception as e:
            print(f"Error starting live view: {e}")

    def stop_live(self):
        self.timer.stop()
        if self.mmc.isSequenceRunning():
            self.mmc.stopSequenceAcquisition()

    def safe_quit(self):
        """Stops hardware safely and closes window."""
        print("Stopping Live View...")
        self.stop_live()
        print("Resetting Core...")
        self.mmc.reset()
        print("Closing Application...")
        self.close()
        QApplication.quit()

    def update_frame(self):
        """Fetch image from circular buffer, auto-contrast, display, and update histogram."""
        try:
            if self.mmc.getRemainingImageCount() > 0:
                img = self.mmc.getLastImage()
                
                # --- Ensure 16-bit type ---
                img_16bit = img.astype(np.uint16)

                # --- Update Histogram with 16-bit data ---
                self.histogram_widget.update_data(img_16bit)

                # --- Display Logic (Downscale to 8-bit for Qt Display) ---
                # Auto Contrast for display only
                img_float = img_16bit.astype(float)
                min_val = np.min(img_float)
                max_val = np.max(img_float)
                
                if max_val > min_val:
                    img_norm = ((img_float - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    img_norm = np.zeros_like(img, dtype=np.uint8)

                height, width = img_norm.shape
                bytes_per_line = width
                q_img = QImage(img_norm.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                
                pixmap = QPixmap.fromImage(q_img)
                self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio))
        except Exception as e:
            pass 

    def handle_mfov_click(self):
        self.stop_live()
        
        # Pass exposure to dialog
        dlg = MFOVConfigDialog(self.exposure_ms, self)
        if dlg.exec_() == QDialog.Accepted:
            data = dlg.get_data()
            self.run_mfov_routine(data)
        
        print("MFOV finished. Resuming Live View...")
        self.start_live()

    def perform_autofocus(self):
        """2-stage autofocus: Rough +/-10um, Fine +/-2.5um."""
        current_z = self.mmc.getPosition()
        
        # Rough Scan
        rough_step = 1.0
        z_start_rough = current_z - 10.0
        z_end_rough = current_z + 10.0
        rough_range = np.arange(z_start_rough, z_end_rough + rough_step, rough_step)
        
        best_score = -1.0
        best_z = current_z

        for z in rough_range:
            self.mmc.setPosition(z)
            self.mmc.waitForSystem()
            self.mmc.snapImage()
            img = self.mmc.getImage()
            score = tenengrad(img)
            if score > best_score:
                best_score = score
                best_z = z
        
        # Fine Scan
        fine_step = 0.25
        z_start_fine = best_z - 2.5
        z_end_fine = best_z + 2.5
        fine_range = np.arange(z_start_fine, z_end_fine + fine_step, fine_step)
        
        best_score_fine = -1.0
        best_z_fine = best_z 

        for z in fine_range:
            self.mmc.setPosition(z)
            self.mmc.waitForSystem()
            self.mmc.snapImage()
            img = self.mmc.getImage()
            score = tenengrad(img)
            if score > best_score_fine:
                best_score_fine = score
                best_z_fine = z
        
        self.mmc.setPosition(best_z_fine)
        self.mmc.waitForSystem()
        return best_z_fine

    def run_mfov_routine(self, data):
        """Logic for Grid, Naming, Metadata, and Capture."""
        try:
            N = data['grid_n']
            overlap = data['overlap']
            pixel_size = data['pixel_size']
            save_dir = data['path']
            prefix = data['prefix']
            mag_str = data['objective']
            laser = data['laser']
            exp = data['exp_ms']

            # Construct Folder Name
            # Format: Prefix_MFOV_gridNxN_ObjectiveX_Lasernm_Exposurems
            folder_name = f"{prefix}_MFOV_grid{N}x{N}_{mag_str}_{laser}nm_{exp}ms"
            full_save_path = os.path.join(save_dir, folder_name)

            # Sequential check
            counter = 1
            original_path = full_save_path
            while os.path.exists(full_save_path):
                full_save_path = f"{original_path}_{counter}"
                counter += 1
            
            os.makedirs(full_save_path)
            print(f"\n--- Starting MFOV. Saving to: {full_save_path} ---")

            # Setup Metadata File
            metadata_path = os.path.join(full_save_path, "metadata.txt")
            
            # Grid Calculations
            x_center = self.mmc.getXPosition()
            y_center = self.mmc.getYPosition()
            
            fov_width_um = IMAGE_SIZE * pixel_size
            step_um = fov_width_um * (1 - (overlap / 100.0))
            
            offset_indices = np.arange(N) - (N - 1) / 2.0
            total_images = N * N
            count = 0
            
            # Open metadata file
            with open(metadata_path, "w") as meta_file:
                # --- Write Header with All Parameters ---
                meta_file.write("=== Acquisition Parameters ===\n")
                meta_file.write(f"Prefix: {prefix}\n")
                meta_file.write(f"Grid Size: {N}x{N}\n")
                meta_file.write(f"Overlap: {overlap}%\n")
                meta_file.write(f"Objective: {mag_str}\n")
                meta_file.write(f"Pixel Size: {pixel_size:.4f} um\n")
                meta_file.write(f"Laser: {laser} nm\n")
                meta_file.write(f"Exposure Time: {exp} ms\n")
                meta_file.write(f"Center Position: X={x_center:.2f}, Y={y_center:.2f}\n")
                meta_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
                meta_file.write("==============================\n\n")

                # Write Column Headers
                meta_file.write("Filename,X_Pos_um,Y_Pos_um,Z_Pos_um\n")
                
                for i, y_idx in enumerate(offset_indices):
                    for j, x_idx in enumerate(offset_indices):
                        count += 1
                        
                        target_x = x_center + (x_idx * step_um)
                        target_y = y_center + (y_idx * step_um)
                        
                        print(f"Grid ({j},{i}) -> Moving... ", end="")
                        self.mmc.setXYPosition(target_x, target_y)
                        self.mmc.waitForSystem()
                        
                        # Autofocus
                        print("Autofocusing... ", end="")
                        final_z = self.perform_autofocus()
                        
                        # Snap
                        self.mmc.snapImage()
                        img = self.mmc.getImage()
                        img_16bit = img.astype(np.uint16) # Ensure 16-bit save
                        
                        # Save Image
                        filename = f"Grid_Y{i}_X{j}.tif"
                        filepath = os.path.join(full_save_path, filename)
                        cv2.imwrite(filepath, img_16bit)
                        
                        # Save Metadata
                        meta_file.write(f"{filename},{target_x:.3f},{target_y:.3f},{final_z:.3f}\n")
                        meta_file.flush() # Ensure write to disk
                        
                        print(f"Saved {filename} [{count}/{total_images}]")
                        QApplication.processEvents()

            # Return to center
            self.mmc.setXYPosition(x_center, y_center)
            self.mmc.waitForSystem()
            
            QMessageBox.information(self, "Complete", f"Captured {total_images} images.\nSaved to: {full_save_path}")

        except Exception as e:
            print(f"Error during MFOV: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(e))


""" main """
def main(MMC:pymmcore_plus.CMMCorePlus):
    # 1. Ask for exposure time
    try:
        exp_input = input("Enter exposure time (ms): ")
        exp_time = float(exp_input)
    except ValueError:
        print("Invalid input, defaulting to 20ms")
        exp_time = 20.0

    set_exposure(MMC, exp_time)

    # 2. Launch Qt Application
    app = QApplication(sys.argv)
    
    # 3. Create and Show GUI (Pass exposure time)
    window = MicroscopeWindow(MMC, exp_time)
    window.show()
    
    # 4. Run Event Loop
    app.exec_()
    
    print("GUI Closed.")


""" functions """

def set_exposure(MMC:pymmcore_plus.CMMCorePlus, exp):
    new_exposure = MMC.setExposure(exp) 
    print(f"Exposure set to: {exp} ms")


""" execution """

if __name__ == "__main__":
    
    try:
        MMC = pymmcore_plus.CMMCorePlus.instance()  # Instance micromanager core
        print(MMC.getVersionInfo()) 
        print(MMC.getAPIVersionInfo()) 
        print(pymmcore_plus.find_micromanager())
        
        # NOTE: Verify this path exists on your machine
        config_path = "C:/Program Files/Micro-Manager-2.0gamma/MMConfig_Olympus112022.cfg"
        if not os.path.exists(config_path):
            print(f"WARNING: Config file not found at {config_path}")
        else:
            MMC.loadSystemConfiguration(config_path)
            
        print(MMC)
        
        # Initial Hardware Setup
        try:
            MMC.setProperty("Objective","State",2)
            MMC.setProperty("TransmittedIllumination 2","Brightness",0*212)
        except Exception as e:
            print(f"Hardware property error (ignoring for generic safety): {e}")

        time.sleep(3) 
        
        print("starting main")
        main(MMC)
        print("ending main")

        MMC.reset()
        print("ran this shit")

    except KeyboardInterrupt:
        MMC.reset()
        print("\nProgram interrupted by user (Ctrl+C). Exiting gracefully.")
    
    except Exception as e:
        MMC.reset()
        print("Error bitch                " + str(e))
        traceback.print_exc()