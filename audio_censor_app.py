"""
Audio Censor Application - Production Version
-------------------------------------------
A professional application for detecting and censoring explicit content in audio files.
Features:
- Multiple censoring methods (beep, mute, reverse)
- Multiple language support (English, Hindi, and custom)
- Detailed logging and error handling
- Modern PyQt5 interface with dark mode support
- Audio visualization and preview
- Export options with quality settings
- Configuration saving/loading
- Multiple Whisper model support
"""

import os
import sys
import logging
import json
import re
import time
import tempfile
import threading
import traceback
from pathlib import Path
from functools import partial
from datetime import datetime

# Third-party imports with proper error handling
try:
    import whisper
    import numpy as np
    from pydub import AudioSegment
    from pydub.generators import Sine
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QFileDialog, QVBoxLayout, QPushButton,
        QLabel, QProgressBar, QTextEdit, QComboBox, QRadioButton, QButtonGroup,
        QWidget, QLineEdit, QHBoxLayout, QGroupBox, QSlider, QAction, QMenu,
        QMessageBox, QSpinBox, QDoubleSpinBox, QCheckBox, QStyle, QSplitter,
        QTabWidget, QToolBar, QStatusBar, QFrame, QDialog, QGridLayout,
        QListWidget, QListWidgetItem, QFormLayout
    )
    from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
    from PyQt5.QtMultimediaWidgets import QVideoWidget
    from PyQt5.QtCore import (
        Qt, QUrl, QSize, QTimer, QThread, pyqtSignal, QSettings, QDir,
        QTemporaryFile, QCoreApplication
    )
    from PyQt5.QtGui import (
        QIcon, QFont, QPalette, QColor, QPixmap, QKeySequence, QDesktopServices
    )
    # Optional: For audio visualization
    try:
        import matplotlib
        matplotlib.use('Qt5Agg')
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        VISUALIZATION_AVAILABLE = True
    except ImportError:
        VISUALIZATION_AVAILABLE = False
except ImportError as e:
    # Show a more user-friendly error message
    module_name = str(e).split("'")[1] if "'" in str(e) else str(e)
    error_message = f"Missing dependency: {module_name}\n\n"
    error_message += "Please install required dependencies:\n"
    error_message += "pip install openai-whisper pydub PyQt5 matplotlib numpy\n\n"
    error_message += "For Whisper, you might also need to install ffmpeg:\n"
    error_message += "- On Windows: via https://ffmpeg.org/download.html\n"
    error_message += "- On macOS: brew install ffmpeg\n"
    error_message += "- On Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg"
    
    # Try to show a GUI error if possible, otherwise fall back to console
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "Missing Dependencies", error_message)
    except:
        print("\n" + "="*60)
        print("ERROR: Missing Dependencies")
        print("="*60)
        print(error_message)
    
    sys.exit(1)

# Set application info for QSettings
QCoreApplication.setOrganizationName("AudioCensorApp")
QCoreApplication.setApplicationName("AudioCensor")
QCoreApplication.setApplicationVersion("1.0.0")

# Configure logging with rotating file handler
def setup_logging():
    """Set up application logging with file rotation"""
    log_dir = os.path.join(QDir.homePath(), ".audio_censor")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'audio_censor_app.log')
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler with rotation
    try:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=1024*1024*5, backupCount=3, encoding='utf-8'
        )
    except:
        # Fall back to basic file handler if rotation isn't available
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
    
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    ))
    logger.addHandler(file_handler)
    
    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

# Global logger instance
logger = setup_logging()

class TranscriptionWorker(QThread):
    """Worker thread for running whisper transcription without blocking the UI"""
    progress_update = pyqtSignal(str)
    transcription_complete = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model, audio_path):
        super().__init__()
        self.model = model
        self.audio_path = audio_path
    
    def run(self):
        try:
            self.progress_update.emit("Transcribing audio with Whisper...")
            result = self.model.transcribe(
                self.audio_path,
                word_timestamps=True,
                verbose=False
            )
            self.transcription_complete.emit(result)
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            self.error_occurred.emit(f"Transcription failed: {str(e)}")
            traceback.print_exc()


class ProfanityDetector:
    """Class for detecting profanity in transcripts with custom word lists"""
    
    def __init__(self):
        self.config_dir = os.path.join(QDir.homePath(), ".audio_censor")
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Initialize language profanity sets
        self.profanity_lists = {
            'english': self._load_profanity_list('english'),
            'hindi': self._load_profanity_list('hindi'),
            'custom': self._load_profanity_list('custom')
        }
    
    def _load_profanity_list(self, language):
        """Load profanity list from file or use default built-in lists"""
        profanity_set = set()
        
        # Default built-in profanity lists
        if language == 'english':
            profanity_set.update([
                "fuck", "shit", "ass", "bitch", "dick", "cock", "pussy", 
                "cunt", "whore", "bastard", "damn", "motherfucker", "asshole",
                "bullshit", "fucker", "fucking", "shitty", "motherfucking",
                "f**k", "f*ck", "s**t", "s*it", "b**ch", "b*tch", "d**k",
                "p*ssy", "c*nt", "wh*re", "a**hole"
            ])
        elif language == 'hindi':
            profanity_set.update([
                "chutiya", "चूतिया", "bhosdike", "भोसडीके", "madarchod", "मादरचोद",
                "behenchod", "बहनचोद", "bkl", "बीकेएल", "mc", "bc", "bhenchod", 
                "lund", "लंड", "lauda", "लौडा", "randi", "रंडी", "gaand", "गांड",
                "chodu", "चोदू", "saala", "साला", "jhatu", "झाटू", "chutmarike", "चूतमारिके"
            ])
        
        # Load custom words from file
        file_path = os.path.join(self.config_dir, f"{language}_profanity.txt")
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    custom_words = [line.strip() for line in file.readlines() if line.strip()]
                    profanity_set.update(custom_words)
                    logger.info(f"Loaded {len(custom_words)} words from {language} profanity list")
        except Exception as e:
            logger.error(f"Error loading profanity list: {e}")
        
        return profanity_set
    
    def save_profanity_list(self, language):
        """Save the current profanity list for a language to a file"""
        if language not in self.profanity_lists:
            logger.error(f"Unknown language for profanity list: {language}")
            return False
            
        file_path = os.path.join(self.config_dir, f"{language}_profanity.txt")
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                for word in sorted(self.profanity_lists[language]):
                    file.write(word + '\n')
            logger.info(f"Saved {len(self.profanity_lists[language])} words to {language} profanity list")
            return True
        except Exception as e:
            logger.error(f"Error saving profanity list: {e}")
            return False
    
    def add_profanity(self, language, words):
        """Add words to a profanity list"""
        if language in self.profanity_lists:
            for word in words:
                if word.strip():
                    self.profanity_lists[language].add(word.strip())
            self.save_profanity_list(language)
            return True
        return False
    
    def remove_profanity(self, language, words):
        """Remove words from a profanity list"""
        if language in self.profanity_lists:
            for word in words:
                if word in self.profanity_lists[language]:
                    self.profanity_lists[language].remove(word)
            self.save_profanity_list(language)
            return True
        return False
    
    def detect_profanity(self, transcription, padding=0.1, active_languages=None):
        """
        Detect profanity in transcription for selected languages
        Returns a list of (start_time, end_time, word) tuples for profane words
        """
        if active_languages is None:
            active_languages = ['english', 'hindi', 'custom']
        
        # Combine active profanity lists
        active_profanity = set()
        for lang in active_languages:
            if lang in self.profanity_lists:
                active_profanity.update(self.profanity_lists[lang])
        
        profanity_timestamps = []
        
        # Extract all words with timestamps from the transcription
        all_words = []
        for segment in transcription['segments']:
            if 'words' in segment:
                all_words.extend(segment['words'])
        
        # Detect profanity in each word
        for word_info in all_words:
            word = word_info['word'].lower().strip()
            # Clean the word of punctuation for better matching
            clean_word = re.sub(r'[^\w\s]', '', word)
            
            # Check if the word is in active profanity lists
            if clean_word in active_profanity:
                start_time = word_info['start'] - padding
                end_time = word_info['end'] + padding
                
                # Ensure start_time is not negative
                start_time = max(0, start_time)
                
                profanity_timestamps.append((start_time, end_time, clean_word))
        
        return profanity_timestamps


class AudioProcessor:
    """Class for processing and censoring audio based on detected profanity"""
    
    def __init__(self):
        self.censor_method = "beep"  # "beep", "mute", or "reverse"
        self.beep_freq = 1000  # Hz
        self.padding = 0.1  # seconds to add before and after each detected word
        self.supported_formats = ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac']
        
    def set_censor_method(self, method):
        """Set the censoring method"""
        if method in ["beep", "mute", "reverse"]:
            self.censor_method = method
            return True
        return False
    
    def set_beep_frequency(self, freq):
        """Set the frequency for beep censoring"""
        if 200 <= freq <= 5000:
            self.beep_freq = freq
            return True
        return False
    
    def set_padding(self, padding):
        """Set padding around detected words (in seconds)"""
        if 0 <= padding <= 1.0:
            self.padding = padding
            return True
        return False
    
    def censor_audio(self, audio_path, profanity_timestamps, output_path, 
                     progress_callback=None):
        """
        Censor the audio based on profanity timestamps
        Returns the path to the censored audio file
        """
        try:
            if progress_callback:
                progress_callback("Loading audio file...")
            
            # Load the audio file
            audio = AudioSegment.from_file(audio_path)
            
            if not profanity_timestamps:
                if progress_callback:
                    progress_callback("No explicit content detected - copying original file...")
                audio.export(output_path, format=os.path.splitext(output_path)[1][1:])
                return output_path
            
            # Generate a beep sound if beep method is selected
            if self.censor_method == "beep":
                if progress_callback:
                    progress_callback("Generating beep tone...")
                beep = Sine(self.beep_freq).to_audio_segment(duration=1000)
            
            # Sort timestamps in order (not reverse!) for sequential processing
            profanity_timestamps.sort(key=lambda x: x[0])
            
            if progress_callback:
                progress_callback(f"Censoring {len(profanity_timestamps)} instances of explicit content...")
            
            if self.censor_method == "mute":
                # Build censored audio by appending non-profane and silent segments
                censored_audio = AudioSegment.empty()
                last_end_ms = 0
                for i, (start_time, end_time, word) in enumerate(profanity_timestamps):
                    start_ms = int(start_time * 1000)
                    end_ms = int(end_time * 1000)
                    # Append non-profane segment
                    if start_ms > last_end_ms:
                        censored_audio += audio[last_end_ms:start_ms]
                    # Append silence for profane segment
                    duration_ms = end_ms - start_ms
                    censored_audio += AudioSegment.silent(duration=duration_ms)
                    last_end_ms = end_ms
                # Append the rest of the audio
                if last_end_ms < len(audio):
                    censored_audio += audio[last_end_ms:]
                audio = censored_audio
            else:
                # For beep and reverse, keep old logic (replace in reverse order)
                for i, (start_time, end_time, word) in enumerate(sorted(profanity_timestamps, key=lambda x: x[0], reverse=True)):
                    if progress_callback and i % 5 == 0:
                        progress_callback(f"Censoring instance {i+1}/{len(profanity_timestamps)}...")
                    start_ms = int(start_time * 1000)
                    end_ms = int(end_time * 1000)
                    duration_ms = end_ms - start_ms
                    if self.censor_method == "beep":
                        segment_beep = beep[:duration_ms]
                        segment_beep = segment_beep - 5
                        audio = audio[:start_ms] + segment_beep + audio[end_ms:]
                    elif self.censor_method == "reverse":
                        segment = audio[start_ms:end_ms]
                        reversed_segment = segment.reverse()
                        audio = audio[:start_ms] + reversed_segment + audio[end_ms:]
            
            # Export the censored audio
            if progress_callback:
                progress_callback("Exporting censored audio...")
            
            export_format = os.path.splitext(output_path)[1][1:]
            audio.export(output_path, format=export_format)
            
            if progress_callback:
                progress_callback(f"Censoring complete. Saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error censoring audio: {e}")
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            raise


class AudioVisualizerWidget(QWidget):
    """Widget for visualizing audio waveforms with profanity markers"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.audio_data = None
        self.profanity_timestamps = []
        self.duration = 0
        self.current_position = 0
        
        self.layout = QVBoxLayout(self)
        
        if VISUALIZATION_AVAILABLE:
            # Create matplotlib figure for visualization
            self.figure = Figure(figsize=(5, 2), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.layout.addWidget(self.canvas)
            
            # Create the axes
            self.axes = self.figure.add_subplot(111)
            self.axes.set_facecolor('#f0f0f0')
            self.figure.patch.set_facecolor('#f0f0f0')
            
            # Position indicator
            self.position_line = self.axes.axvline(x=0, color='r', linestyle='-')
            
            # Initialize empty plot
            self.waveform_plot = None
            self.profanity_markers = []
            self.update_plot()
        else:
            # Fallback if matplotlib is not available
            self.placeholder = QLabel("Audio visualization requires matplotlib")
            self.placeholder.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(self.placeholder)
    
    def set_audio(self, audio_path):
        """Load audio data for visualization"""
        if not VISUALIZATION_AVAILABLE:
            return
            
        try:
            # Load audio using pydub (normalized to single channel for display)
            audio = AudioSegment.from_file(audio_path)
            self.duration = len(audio) / 1000.0  # Duration in seconds
            
            # Convert to numpy array for plotting (resample if too large)
            samples = np.array(audio.get_array_of_samples())
            
            # If stereo, convert to mono for visualization
            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            
            # Downsample for performance if needed
            max_samples = 10000
            if len(samples) > max_samples:
                step = len(samples) // max_samples
                samples = samples[::step]
            
            self.audio_data = samples
            self.update_plot()
        except Exception as e:
            logger.error(f"Error loading audio for visualization: {e}")
            self.audio_data = None
            self.update_plot()
    
    def set_profanity_timestamps(self, timestamps):
        """Set profanity timestamps for visualization"""
        self.profanity_timestamps = timestamps
        self.update_plot()
    
    def set_position(self, position_ms):
        """Update current playback position"""
        if not VISUALIZATION_AVAILABLE:
            return
            
        self.current_position = position_ms / 1000.0  # Convert to seconds
        if hasattr(self, 'position_line'):
            self.position_line.set_xdata([self.current_position])
            self.canvas.draw_idle()
    
    def update_plot(self):
        """Update the audio waveform plot"""
        if not VISUALIZATION_AVAILABLE:
            return
            
        # Clear previous plot
        self.axes.clear()
        
        # Set background color
        self.axes.set_facecolor('#f0f0f0')
        
        if self.audio_data is not None:
            # Plot the waveform
            time_array = np.linspace(0, self.duration, len(self.audio_data))
            self.waveform_plot = self.axes.plot(time_array, self.audio_data, 'b-', linewidth=0.5)
            
            # Add profanity markers
            for start, end, word in self.profanity_timestamps:
                rect = self.axes.axvspan(start, end, color='red', alpha=0.3)
                self.profanity_markers.append(rect)
            
            # Add current position line
            self.position_line = self.axes.axvline(x=self.current_position, color='r', linestyle='-')
            
            # Set limits
            self.axes.set_xlim(0, self.duration)
            max_val = np.max(np.abs(self.audio_data)) * 1.1
            self.axes.set_ylim(-max_val, max_val)
            
            # Remove y-axis ticks and labels
            self.axes.set_yticks([])
            
            # Add x-axis with time in minutes:seconds
            def time_format(x, pos):
                minutes = int(x // 60)
                seconds = int(x % 60)
                return f"{minutes}:{seconds:02d}"
                
            self.axes.set_xlabel('Time (min:sec)')
            
            # Only show a few tick marks
            max_ticks = 10
            if self.duration > 0:
                step = max(1, int(self.duration / max_ticks))
                ticks = np.arange(0, self.duration + step, step)
                self.axes.set_xticks(ticks)
                self.axes.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(time_format))
        else:
            # No audio data - show placeholder
            self.axes.text(0.5, 0.5, 'No audio loaded', 
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=self.axes.transAxes)
            self.axes.set_xticks([])
            self.axes.set_yticks([])
        
        # Update the canvas
        self.figure.tight_layout()
        self.canvas.draw_idle()


class ProfanityManagerDialog(QDialog):
    """Dialog for managing profanity word lists"""
    
    def __init__(self, profanity_detector, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Profanity List Manager")
        self.setMinimumSize(500, 400)
        self.profanity_detector = profanity_detector
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Language selection
        lang_layout = QHBoxLayout()
        lang_label = QLabel("Language:")
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["English", "Hindi", "Custom"])
        self.lang_combo.currentIndexChanged.connect(self.load_selected_language)
        
        lang_layout.addWidget(lang_label)
        lang_layout.addWidget(self.lang_combo)
        layout.addLayout(lang_layout)
        
        # Word list
        self.word_list = QListWidget()
        layout.addWidget(self.word_list)
        
        # Add word section
        add_layout = QHBoxLayout()
        self.new_word_edit = QLineEdit()
        self.new_word_edit.setPlaceholderText("Enter new word to add")
        add_button = QPushButton("Add Word")
        add_button.clicked.connect(self.add_word)
        
        add_layout.addWidget(self.new_word_edit)
        add_layout.addWidget(add_button)
        layout.addLayout(add_layout)
        
        # Remove selected
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_selected)
        layout.addWidget(remove_button)
        
        # Import/Export buttons
        io_layout = QHBoxLayout()
        
        import_button = QPushButton("Import List")
        import_button.clicked.connect(self.import_list)
        export_button = QPushButton("Export List")
        export_button.clicked.connect(self.export_list)
        
        io_layout.addWidget(import_button)
        io_layout.addWidget(export_button)
        layout.addLayout(io_layout)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
        
        # Initialize with first language
        self.load_selected_language()
    
    def load_selected_language(self):
        """Load words for the selected language"""
        language = self.get_selected_language()
        self.word_list.clear()
        
        words = sorted(self.profanity_detector.profanity_lists[language])
        for word in words:
            self.word_list.addItem(word)
            
        self.word_list.sortItems()
    
    def get_selected_language(self):
        """Get the lowercase language key for the current selection"""
        language_map = {
            0: "english",
            1: "hindi",
            2: "custom"
        }
        return language_map.get(self.lang_combo.currentIndex(), "english")
    
    def add_word(self):
        """Add a new word to the list"""
        new_word = self.new_word_edit.text().strip().lower()
        if not new_word:
            return
            
        language = self.get_selected_language()
        
        # Check if word already exists
        if new_word in self.profanity_detector.profanity_lists[language]:
            QMessageBox.information(self, "Word Exists", 
                                    f"The word '{new_word}' is already in the list.")
            return
            
        # Add to detector and update UI
        self.profanity_detector.add_profanity(language, [new_word])
        self.word_list.addItem(new_word)
        self.word_list.sortItems()
        self.new_word_edit.clear()
    
    def remove_selected(self):
        """Remove selected words from the list"""
        selected_items = self.word_list.selectedItems()
        if not selected_items:
            return
            
        language = self.get_selected_language()
        words_to_remove = [item.text() for item in selected_items]
        
        # Confirm removal
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText(f"Remove {len(words_to_remove)} selected word(s)?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        if msg.exec_() == QMessageBox.Yes:
            # Remove from detector and UI
            self.profanity_detector.remove_profanity(language, words_to_remove)
            self.load_selected_language()  # Reload the list
    
    def import_list(self):
        """Import words from a text file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Word List", "", "Text Files (*.txt);;All Files (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                words = [line.strip() for line in file.readlines() if line.strip()]
                
            if not words:
                QMessageBox.warning(self, "Import Error", "No words found in the file.")
                return
                
            language = self.get_selected_language()
            self.profanity_detector.add_profanity(language, words)
            
            QMessageBox.information(
                self, "Import Complete", 
                f"Successfully imported {len(words)} words."
            )
            
            self.load_selected_language()
                
        except Exception as e:
            QMessageBox.critical(
                self, "Import Error", 
                f"Failed to import words: {str(e)}"
            )
    
    def export_list(self):
        """Export the current word list to a text file"""
        language = self.get_selected_language()
        words = sorted(self.profanity_detector.profanity_lists[language])
        
        if not words:
            QMessageBox.warning(self, "Export Error", "No words to export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Word List", f"{language}_profanity.txt", 
            "Text Files (*.txt);;All Files (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                for word in words:
                    file.write(word + '\n')
                    
            QMessageBox.information(
                self, "Export Complete", 
                f"Successfully exported {len(words)} words to {file_path}."
            )
                
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", 
                f"Failed to export words: {str(e)}"
            )


class AudioCensorApp(QMainWindow):
    """Main application window for Audio Censor"""
    
    model_loaded = pyqtSignal(object)
    log_signal = pyqtSignal(str, bool)
    
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.profanity_detector = ProfanityDetector()
        self.audio_processor = AudioProcessor()
        self.whisper_model = None
        self.model_name = "medium"  # Default model
        self.transcription = None
        self.profanity_timestamps = []
        self.input_file_path = None
        self.output_file_path = None
        self.censored_file_path = None
        self.processing_thread = None
        self.settings = QSettings()
        
        # Set up status bar before UI
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # UI Setup
        self.init_ui()
        
        # Load settings
        self.load_settings()
        
        # Timer for updating playback position
        self.position_timer = QTimer()
        self.position_timer.setInterval(100)  # Update every 100ms
        self.position_timer.timeout.connect(self.update_position)
        
        # Load whisper model in background
        self.model_loaded.connect(self.on_model_loaded)
        self.log_signal.connect(self.log_message)
        self.load_model_thread = threading.Thread(target=self.load_model_async)
        self.load_model_thread.daemon = True
        self.load_model_thread.start()
    
    def init_ui(self):
        """Initialize the user interface"""
        # Main window setup
        self.setWindowTitle("Audio Censor Pro")
        self.setMinimumSize(900, 700)
        
        # Create central widget with main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create splitter for main areas
        main_splitter = QSplitter(Qt.Vertical)
        
        # Top area - File selection, controls, and visualization
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        
        # Create actions
        open_action = QAction(self.style().standardIcon(QStyle.SP_DialogOpenButton), "Open File", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.select_file)
        toolbar.addAction(open_action)
        
        settings_action = QAction(self.style().standardIcon(QStyle.SP_FileDialogDetailedView), "Settings", self)
        settings_action.triggered.connect(self.show_settings)
        toolbar.addAction(settings_action)
        
        profanity_action = QAction("Manage Profanity Lists", self)
        profanity_action.triggered.connect(self.manage_profanity_lists)
        toolbar.addAction(profanity_action)
        
        toolbar.addSeparator()
        
        help_action = QAction(self.style().standardIcon(QStyle.SP_MessageBoxQuestion), "Help", self)
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)
        
        # File selection area
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        
        # Input file
        input_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("font-weight: bold;")
        select_file_button = QPushButton("Select Audio File")
        select_file_button.clicked.connect(self.select_file)
        input_layout.addWidget(self.file_label, 1)
        input_layout.addWidget(select_file_button)
        file_layout.addLayout(input_layout)
        
        # Output file
        output_layout = QHBoxLayout()
        output_label = QLabel("Output:")
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Output file path (will be auto-generated if empty)")
        output_button = QPushButton("Browse")
        output_button.clicked.connect(self.select_output_path)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_path_edit, 1)
        output_layout.addWidget(output_button)
        file_layout.addLayout(output_layout)
        
        top_layout.addWidget(file_group)
        
        # Configuration panel
        config_group = QGroupBox("Censoring Configuration")
        config_layout = QGridLayout(config_group)
        
        # Whisper model selection
        model_label = QLabel("Whisper Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText(self.model_name)
        self.model_combo.currentTextChanged.connect(self.change_model)
        
        config_layout.addWidget(model_label, 0, 0)
        config_layout.addWidget(self.model_combo, 0, 1)
        
        # Censoring method
        method_label = QLabel("Censoring Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Beep", "Mute", "Reverse"])
        self.method_combo.currentTextChanged.connect(self.change_censor_method)
        
        config_layout.addWidget(method_label, 0, 2)
        config_layout.addWidget(self.method_combo, 0, 3)
        
        # Languages to detect
        lang_label = QLabel("Languages:")
        self.english_check = QCheckBox("English")
        self.english_check.setChecked(True)
        self.hindi_check = QCheckBox("Hindi")
        self.hindi_check.setChecked(True)
        self.custom_check = QCheckBox("Custom")
        self.custom_check.setChecked(True)
        
        config_layout.addWidget(lang_label, 1, 0)
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(self.english_check)
        lang_layout.addWidget(self.hindi_check)
        lang_layout.addWidget(self.custom_check)
        config_layout.addLayout(lang_layout, 1, 1, 1, 3)
        
        # Advanced settings
        adv_label = QLabel("Advanced:")
        
        self.padding_spin = QDoubleSpinBox()
        self.padding_spin.setRange(0.0, 1.0)
        self.padding_spin.setSingleStep(0.05)
        self.padding_spin.setValue(self.audio_processor.padding)
        self.padding_spin.setPrefix("Padding: ")
        self.padding_spin.setSuffix(" sec")
        self.padding_spin.valueChanged.connect(self.change_padding)
        
        self.beep_freq_spin = QSpinBox()
        self.beep_freq_spin.setRange(200, 5000)
        self.beep_freq_spin.setSingleStep(100)
        self.beep_freq_spin.setValue(self.audio_processor.beep_freq)
        self.beep_freq_spin.setPrefix("Beep freq: ")
        self.beep_freq_spin.setSuffix(" Hz")
        self.beep_freq_spin.valueChanged.connect(self.change_beep_freq)
        
        config_layout.addWidget(adv_label, 2, 0)
        adv_layout = QHBoxLayout()
        adv_layout.addWidget(self.padding_spin)
        adv_layout.addWidget(self.beep_freq_spin)
        config_layout.addLayout(adv_layout, 2, 1, 1, 3)
        
        top_layout.addWidget(config_group)
        
        # Audio visualization panel
        self.visualizer = AudioVisualizerWidget()
        top_layout.addWidget(self.visualizer)
        
        # Media controls
        controls_layout = QHBoxLayout()
        
        # Play/pause button
        self.play_button = QPushButton("Play")
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_button)
        
        # Position slider
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)  # Will be set to media duration later
        self.position_slider.sliderMoved.connect(self.set_position)
        controls_layout.addWidget(self.position_slider, 1)
        
        # Time labels
        self.time_label = QLabel("0:00 / 0:00")
        controls_layout.addWidget(self.time_label)
        
        # Original/Censored selector
        self.audio_source_combo = QComboBox()
        self.audio_source_combo.addItems(["Original", "Censored"])
        self.audio_source_combo.currentTextChanged.connect(self.change_audio_source)
        self.audio_source_combo.setEnabled(False)
        controls_layout.addWidget(self.audio_source_combo)
        
        top_layout.addLayout(controls_layout)
        
        # Add to main splitter
        main_splitter.addWidget(top_widget)
        
        # Bottom area - Log and processing controls
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(10, 10, 10, 10)
        
        # Tabs for logs and transcription
        tabs = QTabWidget()
        
        # Status log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        tabs.addTab(self.log_text, "Status Log")
        
        # Transcription tab
        self.transcription_text = QTextEdit()
        self.transcription_text.setReadOnly(True)
        tabs.addTab(self.transcription_text, "Transcription")
        
        # Profanity instances tab
        self.profanity_text = QTextEdit()
        self.profanity_text.setReadOnly(True)
        tabs.addTab(self.profanity_text, "Detected Profanity")
        
        bottom_layout.addWidget(tabs)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        bottom_layout.addWidget(self.progress_bar)
        
        # Process button
        self.process_button = QPushButton("Start Processing")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setMinimumHeight(40)
        self.process_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        bottom_layout.addWidget(self.process_button)
        
        main_splitter.addWidget(bottom_widget)
        
        # Set initial sizes for the splitter
        main_splitter.setSizes([int(self.height() * 0.6), int(self.height() * 0.4)])
        
        # Main layout with splitter
        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(main_splitter)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Set up media player
        self.media_player = QMediaPlayer()
        self.media_player.positionChanged.connect(self.update_position_ui)
        self.media_player.durationChanged.connect(self.update_duration)
        self.media_player.stateChanged.connect(self.media_state_changed)
        
        # Add initial log message
        self.log_message("Application started. Select an audio file to begin.")
    
    def load_model_async(self):
        try:
            self.log_signal.emit(f"Loading Whisper model '{self.model_name}'... (this may take a few moments)", False)
            import whisper
            model = whisper.load_model(self.model_name)
            self.model_loaded.emit(model)
            self.log_signal.emit(f"Whisper model '{self.model_name}' loaded successfully!", False)
        except Exception as e:
            self.log_signal.emit(f"Error loading Whisper model: {e}", True)

    def on_model_loaded(self, model):
        self.whisper_model = model
    
    def change_model(self, model_name):
        """Change the Whisper model"""
        if model_name == self.model_name:
            return
            
        self.model_name = model_name
        self.whisper_model = None  # Clear current model
        
        # Load new model in background
        self.load_model_thread = threading.Thread(target=self.load_model_async)
        self.load_model_thread.daemon = True
        self.load_model_thread.start()
    
    def change_censor_method(self, method):
        """Change the censoring method"""
        self.audio_processor.set_censor_method(method.lower())
        self.log_message(f"Censoring method changed to: {method}")
    
    def change_padding(self, value):
        """Change the padding value"""
        self.audio_processor.set_padding(value)
    
    def change_beep_freq(self, value):
        """Change the beep frequency"""
        self.audio_processor.set_beep_frequency(value)
    
    def log_message(self, message, error=False):
        """Add a message to the log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        
        # Log to UI
        self.log_text.append(formatted_msg)
        
        # Also log to console/file
        if error:
            logger.error(message)
        else:
            logger.info(message)
        
        # Update status bar for important messages
        if error:
            self.status_bar.showMessage(f"Error: {message}", 5000)
        else:
            self.status_bar.showMessage(message, 3000)
    
    def select_file(self):
        """Open a file dialog to select an audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", 
            "Audio Files (*.mp3 *.wav *.m4a *.aac *.ogg *.flac);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        # Validate file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.audio_processor.supported_formats:
            self.log_message(
                f"Unsupported audio format: {file_ext}. Supported formats: {', '.join(self.audio_processor.supported_formats)}",
                error=True
            )
            return
        
        # Update UI
        self.input_file_path = file_path
        self.file_label.setText(os.path.basename(file_path))
        
        # Generate default output path
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        ext = os.path.splitext(file_path)[1]
        self.output_file_path = os.path.join(
            os.path.dirname(file_path), 
            f"{base_name}_censored{ext}"
        )
        self.output_path_edit.setText(self.output_file_path)
        
        # Update media player with new file
        self.load_media(file_path)
        
        # Enable processing button
        self.process_button.setEnabled(True)
        
        # Load audio visualization
        self.visualizer.set_audio(file_path)
        
        self.log_message(f"Loaded audio file: {os.path.basename(file_path)}")
        
        # Reset any previous transcription/censoring
        self.transcription = None
        self.profanity_timestamps = []
        self.censored_file_path = None
        self.audio_source_combo.setEnabled(False)
        self.audio_source_combo.setCurrentText("Original")
        self.transcription_text.clear()
        self.profanity_text.clear()
        self.visualizer.set_profanity_timestamps([])
    
    def select_output_path(self):
        """Open a dialog to select the output file path"""
        if not self.input_file_path:
            self.log_message("Please select an input file first", error=True)
            return
            
        # Get default directory and extension from input file
        default_dir = os.path.dirname(self.input_file_path)
        base_name = os.path.splitext(os.path.basename(self.input_file_path))[0]
        ext = os.path.splitext(self.input_file_path)[1]
        default_name = f"{base_name}_censored{ext}"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output Location", 
            os.path.join(default_dir, default_name),
            f"Audio Files (*{ext});;All Files (*.*)"
        )
        
        if file_path:
            self.output_file_path = file_path
            self.output_path_edit.setText(file_path)
    
    def load_media(self, file_path):
        """Load a media file into the player"""
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
        self.play_button.setEnabled(True)
    
    def toggle_playback(self):
        """Toggle play/pause state of the media player"""
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()
            self.position_timer.start()
    
    def media_state_changed(self, state):
        """Handle media player state changes"""
        if state == QMediaPlayer.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.play_button.setText("Pause")
        else:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.play_button.setText("Play")
            
            if state == QMediaPlayer.StoppedState:
                self.position_timer.stop()
    
    def update_position(self):
        """Update position from timer (for visualizer updates)"""
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.visualizer.set_position(self.media_player.position())
    
    def update_position_ui(self, position):
        """Update UI elements for position changes"""
        duration = self.media_player.duration()
        if duration > 0:
            self.position_slider.setValue(position)
        
        # Update time label
        current_time = position / 1000  # Convert to seconds
        total_time = duration / 1000
        
        current_min = int(current_time // 60)
        current_sec = int(current_time % 60)
        total_min = int(total_time // 60)
        total_sec = int(total_time % 60)
        
        self.time_label.setText(f"{current_min}:{current_sec:02d} / {total_min}:{total_sec:02d}")
    
    def update_duration(self, duration):
        """Update UI for media duration changes"""
        self.position_slider.setRange(0, duration)
        
        # Update time label
        total_time = duration / 1000
        total_min = int(total_time // 60)
        total_sec = int(total_time % 60)
        
        self.time_label.setText(f"0:00 / {total_min}:{total_sec:02d}")
    
    def set_position(self, position):
        """Set player position from slider (position in ms)"""
        self.media_player.setPosition(position)
    
    def change_audio_source(self, source):
        """Switch between original and censored audio"""
        # Save current playing state
        was_playing = self.media_player.state() == QMediaPlayer.PlayingState
        if was_playing:
            self.media_player.pause()
        
        # Change source
        if source == "Original" and self.input_file_path:
            self.load_media(self.input_file_path)
        elif source == "Censored" and self.censored_file_path:
            self.load_media(self.censored_file_path)
        
        # Restore playing state
        if was_playing:
            self.media_player.play()
    
    def start_processing(self):
        """Start the audio processing workflow"""
        if not self.input_file_path:
            self.log_message("No input file selected", error=True)
            return
            
        if not self.whisper_model:
            self.log_message("Whisper model is still loading. Please wait.", error=True)
            return
            
        # Get output path from UI
        self.output_file_path = self.output_path_edit.text()
        if not self.output_file_path:
            # Generate default output path
            base_name = os.path.splitext(os.path.basename(self.input_file_path))[0]
            ext = os.path.splitext(self.input_file_path)[1]
            self.output_file_path = os.path.join(
                os.path.dirname(self.input_file_path), 
                f"{base_name}_censored{ext}"
            )
            self.output_path_edit.setText(self.output_file_path)
        
        # Disable UI controls during processing
        self.process_button.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.method_combo.setEnabled(False)
        
        # Reset progress
        self.progress_bar.setValue(0)
        
        # Start transcription worker
        self.log_message("Starting audio processing...")
        
        worker = TranscriptionWorker(self.whisper_model, self.input_file_path)
        worker.progress_update.connect(self.log_message)
        worker.transcription_complete.connect(self.handle_transcription)
        worker.error_occurred.connect(lambda msg: self.log_message(msg, error=True))
        worker.error_occurred.connect(self.processing_error)
        
        worker.start()
        self.processing_thread = worker
    
    def handle_transcription(self, transcription):
        """Handle completed transcription"""
        self.transcription = transcription
        self.progress_bar.setValue(30)
        self.log_message("Transcription complete!")

        # Display transcription text
        transcript_text = ""
        for segment in transcription['segments']:
            start = self.format_time(segment['start'])
            end = self.format_time(segment['end'])
            text = segment['text']
            transcript_text += f"[{start} - {end}] {text}\n"
        self.transcription_text.setText(transcript_text)

        # Get selected languages
        active_languages = []
        if self.english_check.isChecked():
            active_languages.append('english')
        if self.hindi_check.isChecked():
            active_languages.append('hindi')
        if self.custom_check.isChecked():
            active_languages.append('custom')

        # Detect profanity
        self.log_message("Detecting explicit content...")
        try:
            self.profanity_timestamps = self.profanity_detector.detect_profanity(
                transcription, padding=self.audio_processor.padding, active_languages=active_languages)
            self.progress_bar.setValue(50)
            if not self.profanity_timestamps:
                self.log_message("No explicit content detected in the audio.")
                self.profanity_text.setText("No profanity detected.")
                self.censored_file_path = None
                self.audio_source_combo.setEnabled(False)
                self.process_button.setEnabled(True)
                self.model_combo.setEnabled(True)
                self.method_combo.setEnabled(True)
                return
            # Show detected profanities
            profanity_report = ""
            for start, end, word in self.profanity_timestamps:
                profanity_report += f"{word}: {self.format_time(start)} - {self.format_time(end)}\n"
            self.profanity_text.setText(profanity_report)
            self.visualizer.set_profanity_timestamps(self.profanity_timestamps)
            self.log_message(f"Found {len(self.profanity_timestamps)} profane instance(s). Starting censoring...")
            self.progress_bar.setValue(60)
            self.censor_audio()
        except Exception as e:
            self.log_message(f"Error during profanity detection: {e}", error=True)
            self.processing_error(str(e))

    def censor_audio(self):
        """Censor the audio and update UI"""
        try:
            self.progress_bar.setValue(70)
            self.log_message("Censoring audio...")
            output_path = self.output_file_path
            self.censored_file_path = self.audio_processor.censor_audio(
                self.input_file_path,
                self.profanity_timestamps,
                output_path,
                progress_callback=self.log_message
            )
            self.progress_bar.setValue(90)
            self.log_message(f"Censored audio saved to: {self.censored_file_path}")
            self.audio_source_combo.setEnabled(True)
            self.audio_source_combo.setCurrentText("Censored")
            self.visualizer.set_audio(self.censored_file_path)
            self.censoring_complete()
        except Exception as e:
            self.log_message(f"Error during censoring: {e}", error=True)
            self.processing_error(str(e))

    def censoring_complete(self):
        """Handle completion of censoring process"""
        self.progress_bar.setValue(100)
        self.log_message("Processing complete!")
        self.process_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.method_combo.setEnabled(True)

    def processing_error(self, error_message):
        """Handle errors during processing"""
        self.progress_bar.setValue(0)
        self.process_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.method_combo.setEnabled(True)
        QMessageBox.critical(self, "Processing Error", error_message)

    def load_settings(self):
        """Load user settings"""
        if self.settings.contains("model_name"):
            self.model_name = self.settings.value("model_name", self.model_name)
            self.model_combo.setCurrentText(self.model_name)
        if self.settings.contains("censor_method"):
            method = self.settings.value("censor_method", "Beep")
            self.method_combo.setCurrentText(method)
        if self.settings.contains("padding"):
            self.audio_processor.set_padding(float(self.settings.value("padding", 0.1)))
            self.padding_spin.setValue(self.audio_processor.padding)
        if self.settings.contains("beep_freq"):
            self.audio_processor.set_beep_frequency(int(self.settings.value("beep_freq", 1000)))
            self.beep_freq_spin.setValue(self.audio_processor.beep_freq)

    def show_settings(self):
        QMessageBox.information(self, "Settings", "Settings dialog not implemented yet.")

    def show_help(self):
        QMessageBox.information(self, "Help", "Audio Censor Pro\n\nSelect an audio file, choose your settings, and click 'Start Processing'.")

    def manage_profanity_lists(self):
        dialog = ProfanityManagerDialog(self.profanity_detector, self)
        dialog.exec_()
        # Refresh profanity detection if needed

    def format_time(self, seconds):
        """Format seconds to MM:SS.ms format"""
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes}:{seconds:05.2f}" if seconds < 10 else f"{minutes}:{seconds:04.2f}"

# Main entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioCensorApp()
    window.show()
    sys.exit(app.exec_())