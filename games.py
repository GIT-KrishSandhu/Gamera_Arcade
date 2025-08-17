import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
from threading import Thread, Lock, RLock
import base64
import json
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import atexit
import logging
import traceback
import sys
from datetime import datetime

class GameState(Enum):
    """Enhanced game state enumeration for robust state management"""
    READY = "ready"
    STARTING = "starting"
    RUNNING = "running"
    GAME_OVER = "game_over"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class GameStateData:
    """Game state data model for thread-safe state management"""
    current_state: GameState
    is_running: bool
    game_over: bool
    stop_requested: bool
    score: int
    lives: int
    game_over_time: Optional[float]
    error_message: Optional[str]

@dataclass
class ResourceState:
    """Resource state tracking for comprehensive cleanup management"""
    camera_initialized: bool = False
    camera_index: Optional[int] = None
    cleanup_in_progress: bool = False
    last_cleanup_time: Optional[float] = None
    opencv_windows_created: bool = False
    mediapipe_initialized: bool = False
    cleanup_attempts: int = 0
    last_cleanup_level: Optional[str] = None

class InvalidStateTransition(Exception):
    """Exception raised for invalid state transitions"""
    def __init__(self, from_state: GameState, to_state: GameState):
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(f"Invalid state transition from {from_state.value} to {to_state.value}")

class GameLogger:
    """Enhanced logging system for game operations with context information"""
    
    def __init__(self, name="FruitNinjaGame"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create console handler if not exists
        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            
            # Create detailed formatter with context
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def debug(self, message, context=None):
        """Log debug message with optional context"""
        self._log_with_context(self.logger.debug, message, context)
    
    def info(self, message, context=None):
        """Log info message with optional context"""
        self._log_with_context(self.logger.info, message, context)
    
    def warning(self, message, context=None):
        """Log warning message with optional context"""
        self._log_with_context(self.logger.warning, message, context)
    
    def error(self, message, context=None, exc_info=None):
        """Log error message with optional context and exception info"""
        if exc_info:
            message += f"\nException: {exc_info}"
            message += f"\nTraceback: {traceback.format_exc()}"
        self._log_with_context(self.logger.error, message, context)
    
    def critical(self, message, context=None, exc_info=None):
        """Log critical message with optional context and exception info"""
        if exc_info:
            message += f"\nException: {exc_info}"
            message += f"\nTraceback: {traceback.format_exc()}"
        self._log_with_context(self.logger.critical, message, context)
    
    def _log_with_context(self, log_func, message, context):
        """Internal method to log with context information"""
        if context:
            context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
            message = f"{message} | Context: {context_str}"
        log_func(message)

class ErrorRecoveryManager:
    """Manages error recovery mechanisms for camera and processing failures"""
    
    def __init__(self, logger):
        self.logger = logger
        self.camera_failure_count = 0
        self.processing_failure_count = 0
        self.last_camera_error = None
        self.last_processing_error = None
        self.max_camera_retries = 5
        self.max_processing_retries = 3
        self.camera_retry_delay = 1.0
        self.processing_retry_delay = 0.5
        
    def handle_camera_error(self, error, context=None):
        """Handle camera-related errors with recovery mechanisms"""
        self.camera_failure_count += 1
        self.last_camera_error = error
        
        error_context = {
            'failure_count': self.camera_failure_count,
            'max_retries': self.max_camera_retries,
            'error_type': type(error).__name__,
            **(context or {})
        }
        
        self.logger.error(f"Camera error occurred: {error}", error_context, exc_info=error)
        
        if self.camera_failure_count <= self.max_camera_retries:
            self.logger.info(f"Attempting camera recovery (attempt {self.camera_failure_count}/{self.max_camera_retries})")
            time.sleep(self.camera_retry_delay)
            return True  # Retry
        else:
            self.logger.critical("Camera recovery failed - maximum retries exceeded", error_context)
            return False  # Give up
    
    def handle_processing_error(self, error, context=None):
        """Handle processing-related errors with recovery mechanisms"""
        self.processing_failure_count += 1
        self.last_processing_error = error
        
        error_context = {
            'failure_count': self.processing_failure_count,
            'max_retries': self.max_processing_retries,
            'error_type': type(error).__name__,
            **(context or {})
        }
        
        self.logger.error(f"Processing error occurred: {error}", error_context, exc_info=error)
        
        if self.processing_failure_count <= self.max_processing_retries:
            self.logger.info(f"Attempting processing recovery (attempt {self.processing_failure_count}/{self.max_processing_retries})")
            time.sleep(self.processing_retry_delay)
            return True  # Retry
        else:
            self.logger.critical("Processing recovery failed - maximum retries exceeded", error_context)
            return False  # Give up
    
    def reset_camera_errors(self):
        """Reset camera error tracking after successful operation"""
        if self.camera_failure_count > 0:
            self.logger.info(f"Camera recovery successful after {self.camera_failure_count} failures")
            self.camera_failure_count = 0
            self.last_camera_error = None
    
    def reset_processing_errors(self):
        """Reset processing error tracking after successful operation"""
        if self.processing_failure_count > 0:
            self.logger.info(f"Processing recovery successful after {self.processing_failure_count} failures")
            self.processing_failure_count = 0
            self.last_processing_error = None
    
    def get_error_status(self):
        """Get current error status for debugging"""
        return {
            'camera_failures': self.camera_failure_count,
            'processing_failures': self.processing_failure_count,
            'last_camera_error': str(self.last_camera_error) if self.last_camera_error else None,
            'last_processing_error': str(self.last_processing_error) if self.last_processing_error else None
        }

class FruitNinjaGame:
    def __init__(self):
        # Initialize logging and error recovery systems
        self.logger = GameLogger("FruitNinjaGame")
        self.error_recovery = ErrorRecoveryManager(self.logger)
        
        self.logger.info("Initializing Fruit Ninja Game")
        
        try:
            # MediaPipe setup with error handling
            self.logger.debug("Setting up MediaPipe hands detection")
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.logger.info("MediaPipe hands detection initialized successfully")
        except Exception as e:
            self.logger.critical("Failed to initialize MediaPipe", exc_info=e)
            raise
        
        # Enhanced state management
        self._state = GameState.READY
        self._state_lock = RLock()  # Reentrant lock for nested state operations
        self._processing_lock = RLock()  # Separate lock for frame processing
        
        # Game state data
        self._state_data = GameStateData(
            current_state=GameState.READY,
            is_running=False,
            game_over=False,
            stop_requested=False,
            score=0,
            lives=3,
            game_over_time=None,
            error_message=None
        )
        
        # Legacy state properties (for backward compatibility)
        self.is_running = False
        self.game_over = False
        self.score = 0
        self.lives = 3
        self.fruits = []
        self.particles = []
        self.hand_positions = []
        self.last_hand_pos = None
        self.swipe_threshold = 50
        self.game_over_time = None
        self.stop_requested = False
        self.game_over_displayed = False
        
        # Game settings
        self.fruit_spawn_rate = 0.02
        self.max_fruits = 8
        self.fruit_speed_range = (2, 6)
        self.fruit_size_range = (30, 60)
        
        # Colors for fruits
        self.fruit_colors = [
            (0, 255, 0),    # Green (Apple)
            (0, 165, 255),  # Orange
            (0, 0, 255),    # Red (Strawberry)
            (255, 255, 0),  # Cyan (Banana)
            (255, 0, 255),  # Magenta (Grape)
            (128, 0, 128),  # Purple (Plum)
        ]
        
        # Thread safety (keeping legacy lock for compatibility)
        self.lock = Lock()
        
        # Camera
        self.cap = None
        self.frame = None
        self.processed_frame = None
        
        # Valid state transitions
        self._valid_transitions = {
            GameState.READY: [GameState.STARTING, GameState.ERROR],
            GameState.STARTING: [GameState.RUNNING, GameState.ERROR, GameState.STOPPING],
            GameState.RUNNING: [GameState.GAME_OVER, GameState.STOPPING, GameState.ERROR],
            GameState.GAME_OVER: [GameState.STOPPING, GameState.ERROR],
            GameState.STOPPING: [GameState.READY, GameState.ERROR],
            GameState.ERROR: [GameState.READY, GameState.STOPPING]
        }
        
        # Resource state tracking
        self._resource_state = ResourceState()
        self._resource_lock = RLock()  # Separate lock for resource operations
        
        # Mark MediaPipe as initialized
        try:
            with self._resource_lock:
                self._resource_state.mediapipe_initialized = True
            self.logger.debug("MediaPipe marked as initialized in resource state")
        except Exception as e:
            self.logger.error("Failed to mark MediaPipe as initialized", exc_info=e)
        
        # Register emergency cleanup on exit
        try:
            atexit.register(self._emergency_cleanup)
            self.logger.debug("Emergency cleanup registered for exit")
        except Exception as e:
            self.logger.error("Failed to register emergency cleanup", exc_info=e)
        
        self.logger.info("Fruit Ninja Game initialization completed successfully")
    
    def _is_valid_transition(self, from_state: GameState, to_state: GameState) -> bool:
        """Check if a state transition is valid"""
        return to_state in self._valid_transitions.get(from_state, [])
    
    def _transition_state(self, new_state: GameState, error_message: Optional[str] = None):
        """Thread-safe state transitions with validation"""
        with self._state_lock:
            if not self._is_valid_transition(self._state, new_state):
                raise InvalidStateTransition(self._state, new_state)
            
            old_state = self._state
            self._state = new_state
            
            # Update state data
            self._state_data.current_state = new_state
            if error_message:
                self._state_data.error_message = error_message
            
            # Update legacy properties for backward compatibility
            if new_state == GameState.RUNNING:
                self._state_data.is_running = True
                self.is_running = True
                self._state_data.game_over = False
                self.game_over = False
            elif new_state == GameState.GAME_OVER:
                self._state_data.game_over = True
                self.game_over = True
                self._state_data.game_over_time = time.time()
                self.game_over_time = self._state_data.game_over_time
            elif new_state == GameState.STOPPING:
                self._state_data.stop_requested = True
                self.stop_requested = True
            elif new_state == GameState.READY:
                self._state_data.is_running = False
                self.is_running = False
                self._state_data.game_over = False
                self.game_over = False
                self._state_data.stop_requested = False
                self.stop_requested = False
                self._state_data.error_message = None
            
            self._on_state_changed(old_state, new_state)
    
    def _on_state_changed(self, old_state: GameState, new_state: GameState):
        """Handle state change events"""
        print(f"🔄 State transition: {old_state.value} → {new_state.value}")
        
        # Perform state-specific actions
        if new_state == GameState.GAME_OVER and old_state == GameState.RUNNING:
            print(f"💀 GAME OVER TRIGGERED! Final Score: {self.score}")
        elif new_state == GameState.STOPPING:
            print("🛑 Game stopping - cleanup will begin")
        elif new_state == GameState.READY:
            print("✅ Game ready for new session")
    
    def _get_current_state(self) -> GameState:
        """Get current state thread-safely"""
        with self._state_lock:
            return self._state
    
    def _should_exit_processing(self) -> bool:
        """Check if processing should exit based on current state"""
        current_state = self._get_current_state()
        return current_state in [GameState.STOPPING, GameState.ERROR] or self.stop_requested
    
    def _reset_game_state(self):
        """Reset all game state variables to initial values"""
        with self._state_lock:
            self._state_data.score = 0
            self._state_data.lives = 3
            self._state_data.game_over_time = None
            self._state_data.error_message = None
            
            # Update legacy properties
            self.score = 0
            self.lives = 3
            self.game_over_time = None
            self.fruits = []
            self.particles = []
            self.hand_positions = []
            self.last_hand_pos = None
            self.game_over_displayed = False
        
    def start_game(self):
        """Initialize and start the game with enhanced state management and robust error handling"""
        self.logger.info("Starting Fruit Ninja game")
        
        try:
            # Transition to STARTING state
            self.logger.debug("Transitioning to STARTING state")
            self._transition_state(GameState.STARTING)
            
            # COMPLETE cleanup first
            self.logger.debug("Performing initial cleanup")
            self._force_cleanup()
            
            # Reset game state
            self.logger.debug("Resetting game state")
            self._reset_game_state()

            # Camera initialization with comprehensive error handling and recovery
            self.logger.info("Scanning for available cameras")
            camera_initialized = False
            camera_errors = []
            
            for cam_index in range(5):  # Try indexes 0 to 4
                try:
                    self.logger.debug(f"Attempting to initialize camera at index {cam_index}")
                    temp_cap = cv2.VideoCapture(cam_index)
                    
                    if temp_cap.isOpened():
                        # Test camera by reading a frame
                        ret, test_frame = temp_cap.read()
                        if ret and test_frame is not None:
                            self.cap = temp_cap
                            # Update resource state tracking
                            with self._resource_lock:
                                self._resource_state.camera_initialized = True
                                self._resource_state.camera_index = cam_index
                            
                            self.logger.info(f"Camera successfully initialized at index {cam_index}")
                            camera_initialized = True
                            self.error_recovery.reset_camera_errors()
                            break
                        else:
                            self.logger.warning(f"Camera at index {cam_index} opened but failed to read frame")
                            temp_cap.release()
                    else:
                        self.logger.debug(f"Camera at index {cam_index} failed to open")
                        temp_cap.release()
                        
                except Exception as e:
                    camera_errors.append((cam_index, e))
                    self.logger.warning(f"Error testing camera at index {cam_index}: {e}")
                    try:
                        temp_cap.release()
                    except:
                        pass

            if not camera_initialized:
                error_msg = "No available camera found"
                context = {
                    'attempted_indices': list(range(5)),
                    'errors': [f"Index {idx}: {err}" for idx, err in camera_errors]
                }
                self.logger.critical(error_msg, context)
                self._transition_state(GameState.ERROR, error_msg)
                return {"success": False, "message": error_msg}

            # Set camera properties with error handling
            try:
                self.logger.debug("Setting camera properties")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.logger.info("Camera properties set successfully")
            except Exception as e:
                self.logger.warning("Could not set camera properties, using defaults", exc_info=e)

            # Transition to RUNNING state
            self.logger.debug("Transitioning to RUNNING state")
            self._transition_state(GameState.RUNNING)
            self.logger.info("Fruit Ninja game started successfully")
            return {"success": True, "message": "Game started with auto-detected camera."}
            
        except Exception as e:
            error_msg = f"Failed to start game: {e}"
            self.logger.critical("Critical error during game startup", exc_info=e)
            
            try:
                self._transition_state(GameState.ERROR, str(e))
            except Exception as transition_error:
                self.logger.error("Failed to transition to error state", exc_info=transition_error)
            
            return {"success": False, "message": error_msg}

    
    def _force_cleanup(self):
        """Enhanced multi-level resource cleanup with comprehensive retry mechanisms"""
        print("🧹 FORCE cleaning up game resources...")
        
        with self._resource_lock:
            # Prevent concurrent cleanup operations
            if self._resource_state.cleanup_in_progress:
                print("⚠️ Cleanup already in progress, skipping duplicate cleanup")
                return
            
            self._resource_state.cleanup_in_progress = True
            self._resource_state.last_cleanup_time = time.time()
            self._resource_state.cleanup_attempts += 1
        
        try:
            # Multi-level cleanup strategy
            cleanup_levels = [
                ("normal", self._normal_cleanup),
                ("aggressive", self._aggressive_cleanup),
                ("emergency", self._emergency_cleanup)
            ]
            
            cleanup_success = False
            for level_name, cleanup_func in cleanup_levels:
                try:
                    print(f"🔧 Attempting {level_name} cleanup (attempt {self._resource_state.cleanup_attempts})")
                    cleanup_func()
                    
                    with self._resource_lock:
                        self._resource_state.last_cleanup_level = level_name
                    
                    cleanup_success = True
                    print(f"✅ {level_name.capitalize()} cleanup completed successfully")
                    break
                    
                except Exception as e:
                    print(f"❌ {level_name.capitalize()} cleanup failed: {e}")
                    if level_name == "emergency":
                        print("⚠️ All cleanup levels failed, forcing resource reset")
                        self._force_resource_reset()
                    continue
            
            if cleanup_success:
                print("🧹 FORCE cleanup completed successfully")
            else:
                print("⚠️ FORCE cleanup completed with errors")
                
        finally:
            # Always reset cleanup flag
            with self._resource_lock:
                self._resource_state.cleanup_in_progress = False
    
    def _normal_cleanup(self):
        """Normal cleanup level - standard resource release"""
        print("🔧 Executing normal cleanup...")
        
        # Release camera with retry mechanism
        self._release_camera_with_retry(max_attempts=3, delay=0.1)
        
        # Standard OpenCV cleanup
        if self._resource_state.opencv_windows_created:
            cv2.destroyAllWindows()
            with self._resource_lock:
                self._resource_state.opencv_windows_created = False
        
        # Clear game state
        self._clear_game_state()
        
        print("✅ Normal cleanup completed")
    
    def _aggressive_cleanup(self):
        """Aggressive cleanup level - more forceful resource release"""
        print("🔧 Executing aggressive cleanup...")
        
        # More aggressive camera release
        self._release_camera_with_retry(max_attempts=5, delay=0.2)
        
        # Force OpenCV cleanup multiple times
        for attempt in range(3):
            try:
                cv2.destroyAllWindows()
                time.sleep(0.1)
            except Exception as e:
                print(f"⚠️ OpenCV cleanup attempt {attempt + 1} failed: {e}")
        
        with self._resource_lock:
            self._resource_state.opencv_windows_created = False
        
        # Clear game state with error handling
        try:
            self._clear_game_state()
        except Exception as e:
            print(f"⚠️ Game state clearing failed: {e}")
            # Force clear critical variables
            self.cap = None
            self.processed_frame = None
            self.frame = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print("✅ Aggressive cleanup completed")
    
    def _emergency_cleanup(self):
        """Emergency cleanup level - last resort cleanup"""
        print("🚨 Executing emergency cleanup...")
        
        # Emergency camera release - ignore all errors
        try:
            if self.cap is not None:
                for attempt in range(10):
                    try:
                        self.cap.release()
                        time.sleep(0.05)
                    except:
                        pass
                self.cap = None
        except:
            pass
        
        # Emergency OpenCV cleanup - ignore all errors
        try:
            for _ in range(5):
                try:
                    cv2.destroyAllWindows()
                    time.sleep(0.05)
                except:
                    pass
        except:
            pass
        
        # Force reset all resource tracking
        self._force_resource_reset()
        
        # Emergency game state reset
        try:
            self._emergency_state_reset()
        except:
            pass
        
        print("🚨 Emergency cleanup completed")
    
    def _release_camera_with_retry(self, max_attempts=3, delay=0.1):
        """Enhanced camera release with comprehensive retry mechanism and detailed error handling"""
        if not self.cap:
            self.logger.debug("No camera to release")
            return True
        
        context = {
            'max_attempts': max_attempts,
            'delay': delay,
            'camera_index': self._resource_state.camera_index
        }
        
        self.logger.info(f"Releasing camera with retry mechanism", context)
        
        for attempt in range(max_attempts):
            attempt_context = {**context, 'attempt': attempt + 1}
            
            try:
                # Check if camera is still valid
                if hasattr(self.cap, 'isOpened'):
                    try:
                        is_opened = self.cap.isOpened()
                        if not is_opened:
                            self.logger.info("Camera already closed", attempt_context)
                            self.cap = None
                            with self._resource_lock:
                                self._resource_state.camera_initialized = False
                                self._resource_state.camera_index = None
                            return True
                    except Exception as check_error:
                        self.logger.warning("Failed to check camera status", attempt_context, exc_info=check_error)
                
                # Attempt to release
                try:
                    self.cap.release()
                    self.logger.debug("Camera release method called", attempt_context)
                except Exception as release_error:
                    self.logger.error("Camera release method failed", attempt_context, exc_info=release_error)
                    raise
                
                # Verify release
                try:
                    if hasattr(self.cap, 'isOpened') and not self.cap.isOpened():
                        self.logger.info("Camera released successfully", attempt_context)
                        self.cap = None
                        with self._resource_lock:
                            self._resource_state.camera_initialized = False
                            self._resource_state.camera_index = None
                        return True
                    else:
                        self.logger.warning("Camera release verification failed", attempt_context)
                except Exception as verify_error:
                    self.logger.error("Camera release verification error", attempt_context, exc_info=verify_error)
                    
            except Exception as e:
                self.logger.error(f"Camera release attempt failed", attempt_context, exc_info=e)
            
            # Wait before retry (except on last attempt)
            if attempt < max_attempts - 1:
                try:
                    time.sleep(delay)
                except Exception as sleep_error:
                    self.logger.error("Sleep during camera release retry failed", exc_info=sleep_error)
        
        # Force set to None even if release failed
        self.logger.warning("Camera release failed after all attempts, forcing cleanup")
        try:
            self.cap = None
            with self._resource_lock:
                self._resource_state.camera_initialized = False
                self._resource_state.camera_index = None
        except Exception as force_error:
            self.logger.error("Failed to force camera cleanup", exc_info=force_error)
        
        return False
    
    def _clear_game_state(self):
        """Clear all game state variables safely"""
        print("🧹 Clearing game state...")
        
        # Clear collections safely
        try:
            self.fruits.clear() if hasattr(self.fruits, 'clear') else setattr(self, 'fruits', [])
            self.particles.clear() if hasattr(self.particles, 'clear') else setattr(self, 'particles', [])
            self.hand_positions.clear() if hasattr(self.hand_positions, 'clear') else setattr(self, 'hand_positions', [])
        except Exception as e:
            print(f"⚠️ Error clearing collections: {e}")
            self.fruits = []
            self.particles = []
            self.hand_positions = []
        
        # Clear other state variables
        self.last_hand_pos = None
        self.game_over_time = None
        self.processed_frame = None
        self.frame = None
        self.game_over_displayed = False
        
        print("✅ Game state cleared")
    
    def _force_resource_reset(self):
        """Force reset all resource state tracking"""
        print("🔄 Force resetting resource state...")
        
        with self._resource_lock:
            self._resource_state.camera_initialized = False
            self._resource_state.camera_index = None
            self._resource_state.opencv_windows_created = False
            self._resource_state.mediapipe_initialized = False
        
        print("✅ Resource state reset")
    
    def _emergency_state_reset(self):
        """Emergency reset of all game state - ignore all errors"""
        try:
            # Force reset all basic attributes
            self.cap = None
            self.frame = None
            self.processed_frame = None
            self.fruits = []
            self.particles = []
            self.hand_positions = []
            self.last_hand_pos = None
            self.game_over_time = None
            self.game_over_displayed = False
            
            # Force reset state data
            with self._state_lock:
                self._state = GameState.READY
                self._state_data.current_state = GameState.READY
                self._state_data.is_running = False
                self._state_data.game_over = False
                self._state_data.stop_requested = False
                self._state_data.error_message = None
                
                # Update legacy properties
                self.is_running = False
                self.game_over = False
                self.stop_requested = False
            
        except Exception as e:
            print(f"⚠️ Emergency state reset error (ignored): {e}")
    
    def get_resource_state(self):
        """Get current resource state for debugging"""
        with self._resource_lock:
            return {
                'camera_initialized': self._resource_state.camera_initialized,
                'camera_index': self._resource_state.camera_index,
                'cleanup_in_progress': self._resource_state.cleanup_in_progress,
                'last_cleanup_time': self._resource_state.last_cleanup_time,
                'cleanup_attempts': self._resource_state.cleanup_attempts,
                'last_cleanup_level': self._resource_state.last_cleanup_level,
                'opencv_windows_created': self._resource_state.opencv_windows_created,
                'mediapipe_initialized': self._resource_state.mediapipe_initialized
            }
    
    def stop_game(self):
        """Stop the game and cleanup with enhanced state management and comprehensive error handling"""
        self.logger.info("Stopping Fruit Ninja game")
        
        final_score = self.score  # Capture score early in case of errors
        
        try:
            # Transition to STOPPING state
            current_state = self._get_current_state()
            context = {
                'current_state': current_state.value,
                'final_score': final_score
            }
            
            if current_state != GameState.READY:
                self.logger.debug("Transitioning to STOPPING state", context)
                self._transition_state(GameState.STOPPING)
            else:
                self.logger.debug("Already in READY state, skipping state transition", context)
            
            # Force cleanup with error handling
            try:
                self.logger.debug("Initiating force cleanup")
                self._force_cleanup()
                self.logger.info("Force cleanup completed successfully")
            except Exception as cleanup_error:
                self.logger.error("Force cleanup failed", context, exc_info=cleanup_error)
                # Continue with stop process even if cleanup fails
            
            # Clear frame references with error handling
            try:
                self.processed_frame = None
                self.frame = None
                self.logger.debug("Frame references cleared")
            except Exception as frame_clear_error:
                self.logger.error("Failed to clear frame references", exc_info=frame_clear_error)
            
            # Transition to READY state
            try:
                self.logger.debug("Transitioning to READY state")
                self._transition_state(GameState.READY)
                self.logger.info(f"Fruit Ninja game stopped successfully! Final Score: {final_score}")
            except Exception as transition_error:
                self.logger.error("Failed to transition to READY state", context, exc_info=transition_error)
                # Force transition as fallback
                self._force_ready_state()
            
            return {"success": True, "score": final_score, "message": "Game stopped!"}
            
        except Exception as e:
            error_context = {
                'final_score': final_score,
                'operation': 'stop_game'
            }
            self.logger.critical("Critical error stopping game", error_context, exc_info=e)
            
            # Comprehensive error recovery
            try:
                self.logger.info("Attempting error recovery during game stop")
                self._transition_state(GameState.READY)
            except Exception as recovery_error:
                self.logger.error("State transition recovery failed", exc_info=recovery_error)
                # Last resort: force reset
                self._force_ready_state()
            
            return {"success": True, "score": final_score, "message": "Game force stopped"}
    
    def _force_ready_state(self):
        """Force the game to READY state as last resort recovery"""
        try:
            self.logger.warning("Forcing READY state as last resort")
            with self._state_lock:
                self._state = GameState.READY
                self._reset_game_state()
            self.logger.info("Successfully forced READY state")
        except Exception as force_error:
            self.logger.critical("Failed to force READY state", exc_info=force_error)
    
    def get_frame(self):
        """Capture and process a single frame with enhanced state management and robust error handling"""
        # Check stop conditions using enhanced state management
        if self._should_exit_processing() or not self.cap:
            return None
    
        try:
            # Camera frame capture with error handling
            try:
                ret, frame = self.cap.read()
                if not ret:
                    error_msg = "Failed to read frame from camera"
                    context = {
                        'camera_index': self._resource_state.camera_index,
                        'camera_initialized': self._resource_state.camera_initialized
                    }
                    self.logger.warning(error_msg, context)
                    
                    # Attempt camera recovery
                    if self.error_recovery.handle_camera_error(Exception(error_msg), context):
                        return None  # Retry on next call
                    else:
                        # Camera recovery failed, transition to error state
                        self._transition_state(GameState.ERROR, "Camera failure - recovery failed")
                        return None
                
                # Reset camera error count on successful read
                self.error_recovery.reset_camera_errors()
                
            except Exception as camera_error:
                context = {
                    'camera_index': self._resource_state.camera_index,
                    'operation': 'frame_capture'
                }
                
                if self.error_recovery.handle_camera_error(camera_error, context):
                    return None  # Retry on next call
                else:
                    # Camera recovery failed, transition to error state
                    self._transition_state(GameState.ERROR, f"Camera error: {camera_error}")
                    return None
            
            # Frame processing with error handling
            try:
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                self.frame = frame.copy()
                
                # Process the frame
                self.process_frame(frame)

                if self.processed_frame is None:
                    self.logger.warning("No processed frame available")
                    return None
                
                # Reset processing error count on successful processing
                self.error_recovery.reset_processing_errors()
                
            except Exception as processing_error:
                context = {
                    'frame_shape': frame.shape if 'frame' in locals() else 'unknown',
                    'operation': 'frame_processing'
                }
                
                if self.error_recovery.handle_processing_error(processing_error, context):
                    return None  # Retry on next call
                else:
                    # Processing recovery failed, but continue with degraded functionality
                    self.logger.error("Frame processing failed, continuing with degraded functionality")
                    return None
            
            # Frame encoding with error handling
            try:
                _, buffer = cv2.imencode('.jpg', self.processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                return frame_base64
                
            except Exception as encoding_error:
                context = {
                    'processed_frame_shape': self.processed_frame.shape if self.processed_frame is not None else 'None',
                    'operation': 'frame_encoding'
                }
                self.logger.error("Frame encoding failed", context, exc_info=encoding_error)
                return None
            
        except Exception as e:
            # Catch-all for any unexpected errors
            context = {
                'method': 'get_frame',
                'current_state': self._get_current_state().value
            }
            self.logger.error("Unexpected error in get_frame", context, exc_info=e)
            return None
    
    def process_frame(self, frame):
        """Process frame with enhanced state management and comprehensive error handling"""
        # Comprehensive early exit checks using enhanced state management
        if self._should_exit_processing() or self.stop_requested or self.game_over:
            return
        
        # Additional safety check for camera availability
        if not self.cap or not hasattr(self.cap, 'isOpened') or not self.cap.isOpened():
            return

        try:
            height, width, _ = frame.shape
            current_state = self._get_current_state()
            
            context = {
                'frame_shape': (height, width),
                'current_state': current_state.value,
                'stop_requested': self.stop_requested,
                'game_over': self.game_over
            }
            
            self.logger.debug("Processing frame", context)

        except Exception as frame_info_error:
            self.logger.error("Failed to get frame information", exc_info=frame_info_error)
            return

        try:
            with self._processing_lock:
                # Check stop conditions again after acquiring lock
                if self._should_exit_processing() or self.stop_requested or self.game_over:
                    return
                
                # Process based on current state with individual error handling
                try:
                    if current_state == GameState.RUNNING:
                        # Additional check before game logic processing
                        if not (self._should_exit_processing() or self.stop_requested or self.game_over):
                            self._process_game_logic(frame, width, height)
                except Exception as game_logic_error:
                    context = {
                        'operation': 'game_logic_processing',
                        'current_state': current_state.value
                    }
                    self.logger.error("Error in game logic processing", context, exc_info=game_logic_error)
                    # Continue processing other elements even if game logic fails
                
                try:
                    if current_state == GameState.GAME_OVER:
                        # Additional check before game over processing
                        if not (self._should_exit_processing() or self.stop_requested):
                            self._process_game_over(frame)
                except Exception as game_over_error:
                    context = {
                        'operation': 'game_over_processing',
                        'current_state': current_state.value
                    }
                    self.logger.error("Error in game over processing", context, exc_info=game_over_error)
                    # Continue processing other elements even if game over processing fails

                # UI drawing with error handling
                try:
                    # Check stop conditions before drawing UI
                    if not (self._should_exit_processing() or self.stop_requested):
                        self.draw_ui(frame)
                except Exception as ui_error:
                    context = {
                        'operation': 'ui_drawing',
                        'current_state': current_state.value
                    }
                    self.logger.error("Error in UI drawing", context, exc_info=ui_error)
                    # Continue even if UI drawing fails

                # Final frame assignment with comprehensive safety checks
                try:
                    if not self._should_exit_processing() and not self.stop_requested and self.cap and hasattr(self.cap, 'isOpened') and self.cap.isOpened():
                        self.processed_frame = frame
                        self.logger.debug("Frame processing completed successfully")
                except Exception as assignment_error:
                    self.logger.error("Error assigning processed frame", exc_info=assignment_error)

        except Exception as processing_error:
            context = {
                'method': 'process_frame',
                'current_state': current_state.value if 'current_state' in locals() else 'unknown'
            }
            self.logger.error("Critical error in process_frame", context, exc_info=processing_error)
            self._handle_processing_error(processing_error)
    
    def _process_game_logic(self, frame, width, height):
        """Process game logic during RUNNING state with comprehensive error handling"""
        # Early exit check at the start of game logic processing
        if self._should_exit_processing() or self.stop_requested or self.game_over:
            return
        
        # Check current state is still RUNNING
        current_state = self._get_current_state()
        if current_state != GameState.RUNNING:
            return
        
        context = {
            'frame_dimensions': (width, height),
            'current_state': current_state.value
        }
        
        try:
            # Color conversion with error handling
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as color_error:
                self.logger.error("Failed to convert frame color space", context, exc_info=color_error)
                return

            # Hand detection with error handling
            try:
                if self._should_exit_processing() or self.stop_requested or self.game_over:
                    return
                
                results = self.hands.process(rgb_frame)
                self.logger.debug("Hand detection completed", context)
            except Exception as hand_error:
                self.logger.error("Hand detection failed", context, exc_info=hand_error)
                results = None  # Continue without hand detection

            # Update game logic with individual error handling for each component
            try:
                if not (self._should_exit_processing() or self.stop_requested or self.game_over):
                    self.update_fruits(width, height)
            except Exception as fruit_error:
                self.logger.error("Fruit update failed", context, exc_info=fruit_error)
                # Continue processing other elements
            
            # Check if game over was triggered during fruit update
            if self.game_over or self._get_current_state() == GameState.GAME_OVER:
                return
            
            try:
                if not (self._should_exit_processing() or self.stop_requested or self.game_over):
                    self.update_particles()
            except Exception as particle_error:
                self.logger.error("Particle update failed", context, exc_info=particle_error)
                # Continue processing other elements

            # Draw game elements with individual error handling
            try:
                if not (self._should_exit_processing() or self.stop_requested or self.game_over):
                    self.draw_fruits(frame)
            except Exception as draw_fruits_error:
                self.logger.error("Drawing fruits failed", context, exc_info=draw_fruits_error)
                # Continue processing other elements
            
            try:
                if not (self._should_exit_processing() or self.stop_requested or self.game_over):
                    self.draw_particles(frame)
            except Exception as draw_particles_error:
                self.logger.error("Drawing particles failed", context, exc_info=draw_particles_error)
                # Continue processing other elements

            # Process hand landmarks with comprehensive error handling
            try:
                if results and results.multi_hand_landmarks and not (self._should_exit_processing() or self.stop_requested or self.game_over):
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Check stop conditions within the loop
                        if self._should_exit_processing() or self.stop_requested or self.game_over:
                            break
                        
                        try:
                            # Draw hand landmarks
                            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                            # Get finger position
                            index_tip = hand_landmarks.landmark[8]
                            finger_x = int(index_tip.x * width)
                            finger_y = int(index_tip.y * height)
                            
                            # Draw finger indicator
                            cv2.circle(frame, (finger_x, finger_y), 10, (0, 255, 255), -1)

                            # Track hand movement
                            if not (self._should_exit_processing() or self.stop_requested or self.game_over):
                                self.track_hand_movement(finger_x, finger_y)
                                
                        except Exception as landmark_error:
                            hand_context = {**context, 'hand_index': len([h for h in results.multi_hand_landmarks if h == hand_landmarks])}
                            self.logger.error("Hand landmark processing failed", hand_context, exc_info=landmark_error)
                            continue  # Continue with next hand
                            
            except Exception as hand_processing_error:
                self.logger.error("Hand processing failed", context, exc_info=hand_processing_error)
                # Continue without hand processing
                
        except Exception as game_logic_error:
            self.logger.error("Critical error in game logic processing", context, exc_info=game_logic_error)
            # Don't re-raise, let the game continue with degraded functionality
    
    def _handle_processing_error(self, error):
        """Handle errors that occur during frame processing with comprehensive recovery"""
        context = {
            'error_type': type(error).__name__,
            'current_state': self._get_current_state().value,
            'processing_failures': self.error_recovery.processing_failure_count
        }
        
        self.logger.error("Processing error occurred", context, exc_info=error)
        
        try:
            # Attempt error recovery
            if self.error_recovery.handle_processing_error(error, context):
                self.logger.info("Processing error recovery initiated")
                return  # Recovery will be attempted
            
            # If recovery failed, transition to error state
            current_state = self._get_current_state()
            if current_state not in [GameState.STOPPING, GameState.ERROR]:
                self.logger.warning("Transitioning to error state due to processing failure")
                self._transition_state(GameState.ERROR, str(error))
            else:
                self.logger.debug(f"Not transitioning to error state - already in {current_state.value}")
                
        except Exception as handling_error:
            # Critical error in error handling itself
            self.logger.critical("Critical error in processing error handler", exc_info=handling_error)
            
            # Force error state as last resort
            try:
                with self._state_lock:
                    self._state = GameState.ERROR
                    self._state_data.error_message = f"Processing error: {error}, Handler error: {handling_error}"
            except Exception as force_error:
                self.logger.critical("Failed to force error state", exc_info=force_error)
    
    def _process_game_over(self, frame):
        """Process game over state with enhanced automatic cleanup after 3 seconds"""
        # Always draw the game over overlay
        self.draw_game_over_overlay(frame)

        # Check for automatic cleanup after 3-second delay
        current_time = time.time()
        if self.game_over_time and (current_time - self.game_over_time >= 3.0):
            print("🛑 Auto-stopping game after 3-second Game Over delay...")
            try:
                # Ensure we only trigger cleanup once and we're still in game over state
                current_state = self._get_current_state()
                if current_state == GameState.GAME_OVER:
                    self.stop_game()
                else:
                    print(f"⚠️ State changed during auto-cleanup delay: {current_state.value}")
            except Exception as e:
                print(f"❌ Error during auto-cleanup: {e}")
                # Force cleanup even on error to prevent hanging
                self._force_cleanup()
                try:
                    self._transition_state(GameState.READY)
                except:
                    # Last resort: force reset to ready state
                    with self._state_lock:
                        self._state = GameState.READY
                        self._reset_game_state()
    
    def _handle_processing_error(self, error: Exception):
        """Handle processing errors with state management"""
        print(f"❌ Processing error: {error}")
        try:
            self._transition_state(GameState.ERROR, str(error))
        except Exception as transition_error:
            print(f"❌ Failed to transition to error state: {transition_error}")
            # Force error state
            with self._state_lock:
                self._state = GameState.ERROR
                self._state_data.error_message = str(error)

    
    def draw_game_over_overlay(self, frame):
        """Draw ENHANCED game over overlay"""
        height, width, _ = frame.shape
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Game Over text - ENHANCED
        game_over_text = "GAME OVER!"
        text_size = cv2.getTextSize(game_over_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 4)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height // 2 - 80
        
        # Draw text with thick outline for visibility
        cv2.putText(frame, game_over_text, (text_x-3, text_y-3), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8)
        cv2.putText(frame, game_over_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
        
        # You Lost :( text - ENHANCED
        lost_text = "You Lost :("
        lost_size = cv2.getTextSize(lost_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        lost_x = (width - lost_size[0]) // 2
        lost_y = text_y + 80
        
        cv2.putText(frame, lost_text, (lost_x-2, lost_y-2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6)
        cv2.putText(frame, lost_text, (lost_x, lost_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Final Score - ENHANCED
        score_text = f"Final Score: {self.score}"
        score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        score_x = (width - score_size[0]) // 2
        score_y = lost_y + 60
        
        cv2.putText(frame, score_text, (score_x-2, score_y-2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
        cv2.putText(frame, score_text, (score_x, score_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
        
        # Instructions - ENHANCED
        restart_text = "Click 'Stop Game' to restart"
        restart_size = cv2.getTextSize(restart_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        restart_x = (width - restart_size[0]) // 2
        restart_y = score_y + 50
        
        cv2.putText(frame, restart_text, (restart_x-1, restart_y-1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(frame, restart_text, (restart_x, restart_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        
        # Mark overlay as displayed
        if not self.game_over_displayed:
            self.game_over_displayed = True
            print("💀 GAME OVER OVERLAY DISPLAYED!")
    
    def track_hand_movement(self, x, y):
        """Track hand movement and detect swipes"""
        current_state = self._get_current_state()
        # Comprehensive stop condition checks
        if current_state != GameState.RUNNING or self._should_exit_processing() or self.stop_requested or self.game_over:
            return
            
        current_pos = (x, y)
        
        # Check stop conditions before modifying hand positions
        if self._should_exit_processing() or self.stop_requested or self.game_over:
            return
        
        # Add to position history
        self.hand_positions.append(current_pos)
        if len(self.hand_positions) > 5:
            self.hand_positions.pop(0)
        
        # Check stop conditions before swipe detection
        if self._should_exit_processing() or self.stop_requested or self.game_over:
            return
        
        # Check for swipe if we have enough positions
        if len(self.hand_positions) >= 3:
            start_pos = self.hand_positions[0]
            end_pos = self.hand_positions[-1]
        
            distance = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
            if distance > self.swipe_threshold:
                # Final check before collision detection
                if not (self._should_exit_processing() or self.stop_requested or self.game_over):
                    self.detect_fruit_collision(current_pos)
                    self.hand_positions = []
    
    def detect_fruit_collision(self, hand_pos):
        """Check if hand position collides with any fruits"""
        current_state = self._get_current_state()
        # Comprehensive stop condition checks
        if current_state != GameState.RUNNING or self._should_exit_processing() or self.stop_requested or self.game_over:
            return
            
        with self.lock:
            # Check stop conditions again after acquiring lock
            if self._should_exit_processing() or self.stop_requested or self.game_over:
                return
            
            for i, fruit in enumerate(self.fruits):
                # Check stop conditions within the loop
                if self._should_exit_processing() or self.stop_requested or self.game_over:
                    break
                
                fruit_center = (int(fruit['x']), int(fruit['y']))
                distance = math.sqrt((hand_pos[0] - fruit_center[0])**2 + (hand_pos[1] - fruit_center[1])**2)
                
                if distance < fruit['size']:
                    # Final check before slicing fruit
                    if not (self._should_exit_processing() or self.stop_requested or self.game_over):
                        self.slice_fruit(i, fruit)
                    break
    
    def slice_fruit(self, fruit_index, fruit):
        """Handle fruit slicing"""
        current_state = self._get_current_state()
        # Comprehensive stop condition checks
        if current_state != GameState.RUNNING or self._should_exit_processing() or self.stop_requested or self.game_over:
            return
        
        # Check stop conditions before modifying game state
        if self._should_exit_processing() or self.stop_requested or self.game_over:
            return
            
        # Remove fruit
        self.fruits.pop(fruit_index)
        
        # Add score
        self.score += 10
        
        # Check stop conditions before creating particle effect
        if self._should_exit_processing() or self.stop_requested or self.game_over:
            return
        
        # Create particle effect
        self.create_slice_particles(fruit['x'], fruit['y'], fruit['color'])
        
        print(f"🎯 Fruit sliced! Score: {self.score}")
    
    def create_slice_particles(self, x, y, color):
        """Create particle explosion effect"""
        for _ in range(8):
            particle = {
                'x': x,
                'y': y,
                'vx': random.uniform(-5, 5),
                'vy': random.uniform(-8, -2),
                'color': color,
                'life': 30,
                'max_life': 30
            }
            self.particles.append(particle)
    
    def update_fruits(self, width, height):
        """Update fruit positions and spawn new fruits"""
        current_state = self._get_current_state()
        # Comprehensive stop condition checks
        if current_state != GameState.RUNNING or self._should_exit_processing() or self.stop_requested or self.game_over:
            return
        
        with self.lock:
            # Check stop conditions again after acquiring lock
            if self._should_exit_processing() or self.stop_requested or self.game_over:
                return
            
            # Spawn new fruits - check conditions before spawning
            if len(self.fruits) < self.max_fruits and random.random() < self.fruit_spawn_rate:
                # Additional check before spawning
                if not (self._should_exit_processing() or self.stop_requested or self.game_over):
                    self.spawn_fruit(width, height)
            
            # Check stop conditions before updating existing fruits
            if self._should_exit_processing() or self.stop_requested or self.game_over:
                return
            
            # Update existing fruits
            fruits_to_remove = []
            for i, fruit in enumerate(self.fruits):
                # Check stop conditions within the loop
                if self._should_exit_processing() or self.stop_requested or self.game_over:
                    break
                
                # Update position
                fruit['y'] += fruit['vy']
                fruit['x'] += fruit['vx']
                fruit['vy'] += 0.3  # Gravity
                
                # Remove fruits that are off screen
                if fruit['y'] > height + 50 or fruit['x'] < -50 or fruit['x'] > width + 50:
                    fruits_to_remove.append(i)
                    # Lose life if fruit falls off bottom
                    if fruit['y'] > height + 50:
                        self.lives -= 1
                        print(f"💔 Life lost! Lives remaining: {self.lives}")
                    
                        if self.lives <= 0 and not self.game_over:
                            print("💀 All lives lost - TRIGGERING GAME OVER!")
                            self.trigger_game_over()
                            # Exit immediately after triggering game over
                            return
            
            # Check stop conditions before removing fruits
            if self._should_exit_processing() or self.stop_requested or self.game_over:
                return
        
            # Remove fruits that are off screen
            for i in reversed(fruits_to_remove):
                # Check stop conditions within removal loop
                if self._should_exit_processing() or self.stop_requested or self.game_over:
                    break
                self.fruits.pop(i)
            
            # Final comprehensive check: if game over was triggered during processing, stop immediately
            if self.game_over or self._get_current_state() == GameState.GAME_OVER or self._should_exit_processing() or self.stop_requested:
                return
    
    def trigger_game_over(self):
        """Trigger game over state with enhanced atomic state management"""
        try:
            with self._state_lock:
                current_state = self._state
                if current_state == GameState.RUNNING:
                    # Atomically set all game over flags
                    self._state = GameState.GAME_OVER
                    self._state_data.current_state = GameState.GAME_OVER
                    self._state_data.game_over = True
                    self._state_data.is_running = False
                    self._state_data.game_over_time = time.time()
                    
                    # Update legacy properties atomically
                    self.game_over = True
                    self.is_running = False
                    self.game_over_time = self._state_data.game_over_time
                    self.game_over_displayed = False  # Reset display flag
                    
                    print("💀 GAME OVER TRIGGERED ATOMICALLY - OVERLAY WILL BE DISPLAYED!")
                    self._on_state_changed(current_state, GameState.GAME_OVER)
                else:
                    print(f"⚠️ Cannot trigger game over from state: {current_state.value}")
        except Exception as e:
            print(f"❌ Error triggering game over: {e}")
            # Force game over state atomically
            with self._state_lock:
                self._state = GameState.GAME_OVER
                self._state_data.current_state = GameState.GAME_OVER
                self._state_data.game_over = True
                self._state_data.is_running = False
                self._state_data.game_over_time = time.time()
                self.game_over = True
                self.is_running = False
                self.game_over_time = self._state_data.game_over_time
    
    def spawn_fruit(self, width, height):
        """Spawn a new fruit"""
        current_state = self._get_current_state()
        # Comprehensive stop condition checks
        if current_state != GameState.RUNNING or self._should_exit_processing() or self.stop_requested or self.game_over:
            return
        
        # Additional check before creating fruit object
        if self._should_exit_processing() or self.stop_requested or self.game_over:
            return
            
        fruit = {
            'x': random.uniform(50, width - 50),
            'y': height + 20,
            'vx': random.uniform(-2, 2),
            'vy': random.uniform(-12, -6),
            'size': random.uniform(*self.fruit_size_range),
            'color': random.choice(self.fruit_colors),
            'type': random.choice(['apple', 'orange', 'banana', 'grape'])
        }
        
        # Final check before adding fruit to list
        if not (self._should_exit_processing() or self.stop_requested or self.game_over):
            self.fruits.append(fruit)
    
    def update_particles(self):
        """Update particle effects"""
        current_state = self._get_current_state()
        # Comprehensive stop condition checks
        if current_state not in [GameState.RUNNING, GameState.GAME_OVER] or self._should_exit_processing() or self.stop_requested:
            return
        
        # Additional check for game over state - particles should continue during game over for visual effect
        if current_state == GameState.GAME_OVER and self.game_over:
            # Allow particles to continue during game over for visual effect, but check stop conditions
            if self._should_exit_processing() or self.stop_requested:
                return
            
        particles_to_remove = []
        for i, particle in enumerate(self.particles):
            # Check stop conditions within the loop
            if self._should_exit_processing() or self.stop_requested:
                break
            
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['vy'] += 0.2  # Gravity
            particle['life'] -= 1
            
            if particle['life'] <= 0:
                particles_to_remove.append(i)
        
        # Check stop conditions before removing particles
        if not (self._should_exit_processing() or self.stop_requested):
            for i in reversed(particles_to_remove):
                # Check stop conditions within removal loop
                if self._should_exit_processing() or self.stop_requested:
                    break
                self.particles.pop(i)
    
    def draw_fruits(self, frame):
        """Draw all fruits on the frame"""
        current_state = self._get_current_state()
        if current_state not in [GameState.RUNNING] or self._should_exit_processing() or self.stop_requested or self.game_over:
            return
            
        for fruit in self.fruits:
            # Check stop conditions within the loop
            if self._should_exit_processing() or self.stop_requested or self.game_over:
                break
                
            center = (int(fruit['x']), int(fruit['y']))
            radius = int(fruit['size'])
            color = fruit['color']
            
            # Draw fruit with gradient effect
            cv2.circle(frame, center, radius, color, -1)
            # Add highlight
            highlight_center = (center[0] - radius//3, center[1] - radius//3)
            cv2.circle(frame, highlight_center, radius//3, (255, 255, 255), -1)
    
    def draw_particles(self, frame):
        """Draw particle effects"""
        current_state = self._get_current_state()
        if current_state not in [GameState.RUNNING, GameState.GAME_OVER] or self._should_exit_processing():
            return
            
        for particle in self.particles:
            alpha = particle['life'] / particle['max_life']
            center = (int(particle['x']), int(particle['y']))
            color = tuple(int(c * alpha) for c in particle['color'])
            cv2.circle(frame, center, 3, color, -1)
    
    def draw_ui(self, frame):
        """Draw UI elements (score, lives, etc.)"""
        height, width = frame.shape[:2]
        current_state = self._get_current_state()
        
        if current_state == GameState.RUNNING:
            # Draw score with better visibility
            cv2.putText(frame, f"Score: {self.score}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)  # Black outline
            cv2.putText(frame, f"Score: {self.score}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # White text
            
            # Draw lives with better visibility
            cv2.putText(frame, f"Lives: {self.lives}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)  # Black outline
            cv2.putText(frame, f"Lives: {self.lives}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # White text
            
            # Draw instructions with better visibility
            cv2.putText(frame, "Swipe to slice fruits!", (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)  # Black outline
            cv2.putText(frame, "Swipe to slice fruits!", (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # Cyan text
    
    def get_game_state(self):
        """Get current game state with enhanced state management"""
        with self._state_lock:
            return {
                'current_state': self._state.value,
                'is_running': self.is_running,
                'game_over': self.game_over,
                'score': self.score,
                'lives': self.lives,
                'fruits_count': len(self.fruits),
                'particles_count': len(self.particles),
                'game_over_displayed': self.game_over_displayed,
                'stop_requested': self.stop_requested,
                'error_message': self._state_data.error_message
            }

class TRexRunGame:
    def __init__(self):
        self.trex_img = cv2.imread("static/images/trex.png", cv2.IMREAD_UNCHANGED)
        self.cactus_img = cv2.imread("static/images/cactus.png", cv2.IMREAD_UNCHANGED)
        
        # Resize images to prevent drawing errors
        if self.trex_img is not None:
            self.trex_img = cv2.resize(self.trex_img, (80, 80))
        else:
            print("❌ Could not load trex.png")

        if self.cactus_img is not None:
            self.cactus_img = cv2.resize(self.cactus_img, (50, 100))
        else:
            print("❌ Could not load cactus.png")
            
        # Game state
        self.is_running = False
        self.game_over = False
        self.score = 0
        self.trex_y = 0
        self.jump_velocity = 0
        self.gravity = 1.6
        self.is_jumping = False
        self.obstacles = []
        self.frame = None
        self.cap = None
        self.hand_detected = False
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.lock = Lock()
        self.ground_y = 350
        self.jump_power = -20
        self.trex_x = 50
        self.last_spawn_time = 0
        self.spawn_delay = 2.5
        self.obstacle_speed = 8
        self.game_over_time = None
        
    def start_game(self):
        self.is_running = True
        self.game_over = False
        self.score = 0
        self.trex_y = self.ground_y
        self.jump_velocity = 0
        self.is_jumping = False
        self.obstacles = []
        self.cap = cv2.VideoCapture(0)
        self.last_spawn_time = time.time()
        self.obstacle_speed = 8
        self.game_over_time = None
        return {"success": True, "message": "T-Rex Run game started!"}

    def stop_game(self):
        self.is_running = False
        self.game_over = True
        if self.cap:
            self.cap.release()
            self.cap = None
        return {"success": True, "score": self.score}

    def get_frame(self):
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        
        # Only update game logic if the game is running and not over
        if self.is_running and not self.game_over:
            self.detect_hand(frame)
            self.update_game_logic(frame.shape[1], frame.shape[0])
            
        self.draw_elements(frame)
        
        # Automatically call stop_game after a 3-second game over delay
        if self.game_over and (self.game_over_time is not None and time.time() - self.game_over_time > 3):
            self.stop_game()
            
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64

    def detect_hand(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        self.hand_detected = bool(results.multi_hand_landmarks)

    def update_game_logic(self, width, height):
        # Jump logic
        if self.hand_detected and not self.is_jumping:
            self.is_jumping = True
            self.jump_velocity = self.jump_power

        if self.is_jumping:
            self.trex_y += self.jump_velocity
            self.jump_velocity += self.gravity
            if self.trex_y >= self.ground_y:
                self.trex_y = self.ground_y
                self.is_jumping = False
                
        # Increase speed gradually
        self.obstacle_speed += 0.005

        # Spawn obstacles
        current_time = time.time()
        if current_time - self.last_spawn_time > self.spawn_delay:
            self.obstacles.append({'x': width, 'width': 50, 'height': 100})
            self.last_spawn_time = current_time
            self.spawn_delay = max(1.0, self.spawn_delay - 0.01)

        # Move obstacles
        for obs in self.obstacles:
            obs['x'] -= self.obstacle_speed

        # Collision detection
        trex_rect = (self.trex_x, int(self.trex_y - 80), 80, 80)
        for obs in self.obstacles:
            cactus_rect = (obs['x'], self.ground_y - 100, 50, 100)
            if self.check_collision(trex_rect, cactus_rect):
                self.game_over = True
                self.game_over_time = time.time()
                
        # Update score only if the game is still running
        if not self.game_over:
            self.score += 1
            
        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs['x'] > -obs['width']]

    def check_collision(self, rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        # Add a small buffer to make collision more forgiving
        buffer = 10
        if (x1 + buffer < x2 + w2 - buffer and
            x1 + w1 - buffer > x2 + buffer and
            y1 + buffer < y2 + h2 - buffer and
            y1 + h1 - buffer > y2 + buffer):
            return True
        return False

    def draw_elements(self, frame):
        # Draw ground
        cv2.line(frame, (0, self.ground_y), (frame.shape[1], self.ground_y), (100, 100, 100), 2)

        # Draw T-Rex and obstacles
        if self.trex_img is not None:
            self.overlay_image_alpha(frame, self.trex_img, (self.trex_x, int(self.trex_y - 80)))
        if self.cactus_img is not None:
            for obs in self.obstacles:
                self.overlay_image_alpha(frame, self.cactus_img, (int(obs['x']), self.ground_y - 100))

        # Draw Score
        cv2.putText(frame, f"Score: {self.score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(frame, f"Score: {self.score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw Game Over screen
        if self.game_over:
            text = "GAME OVER"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            
            cv2.putText(frame, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    def overlay_image_alpha(self, img, img_overlay, pos):
        x, y = pos
        h_overlay, w_overlay = img_overlay.shape[:2]

        h_img, w_img = img.shape[:2]

        if x >= w_img or y >= h_img or x + w_overlay <= 0 or y + h_overlay <= 0:
            return

        x_start = max(0, -x)
        y_start = max(0, -y)
        x_end = min(w_overlay, w_img - x)
        y_end = min(h_overlay, h_img - y)

        if x_end <= x_start or y_end <= y_start:
            return

        img_overlay_cropped = img_overlay[y_start:y_end, x_start:x_end]

        x_pos = max(0, x)
        y_pos = max(0, y)

        roi = img[y_pos:y_pos + (y_end - y_start), x_pos:x_pos + (x_end - x_start)]
        
        alpha = img_overlay_cropped[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)

        roi[:] = (1.0 - alpha) * roi + alpha * img_overlay_cropped[:, :, :3]

    def get_game_state(self):
        return {
            "is_running": self.is_running,
            "score": self.score,
            "lives": 1 if self.is_running else 0
        }

class RockPaperScissorsGame:
    def __init__(self):
        self.is_running = False
        self.cap = None
        self.lock = Lock()
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.reset_game()

    def reset_game(self):
        self.score = 0
        self.user_move = None
        self.computer_move = None
        self.result = None
        self.round_active = False
        self.timer_start = None
        self.countdown = 0
        self.last_play_time = 0

    def start_game(self):
        self.is_running = True
        self.reset_game()
        self.cap = cv2.VideoCapture(0)
        return {"success": True, "message": "Rock Paper Scissors game started!"}

    def stop_game(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        return {"success": True, "score": self.score}

    def get_frame(self):
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        
        self.detect_gesture_and_play(frame)
        self.draw_overlay(frame)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def detect_gesture_and_play(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        now = time.time()

        if not self.round_active and (now - self.last_play_time) > 2:
            self.round_active = True
            self.timer_start = now
            self.user_move = None
            self.computer_move = None
            self.result = None

        if self.round_active:
            elapsed = now - self.timer_start
            self.countdown = 3 - int(elapsed)

            if self.countdown <= 0:
                self.countdown = 0
                if self.user_move is None:
                    self.computer_move = random.choice(["Rock", "Paper", "Scissors"])
                    
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.user_move = self.get_gesture_from_landmarks(hand_landmarks)
                    else:
                        self.user_move = "No gesture"

                    self.result = self.get_winner(self.user_move, self.computer_move)
                    if self.result == "You Win":
                        self.score += 1
                        
                if elapsed > 5:
                    self.round_active = False
                    self.last_play_time = now

    def get_gesture_from_landmarks(self, hand_landmarks):
        lm = hand_landmarks.landmark
        
        # Check which fingers are extended
        fingers_up = []
        for tip_id, pip_id in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            fingers_up.append(1 if lm[tip_id].y < lm[pip_id].y else 0)
        
        total_fingers = fingers_up.count(1)

        # Scissors: Index and middle finger are up
        if total_fingers == 2 and fingers_up[0] and fingers_up[1]:
            return "Scissors"
        # Paper: All four fingers are up
        elif total_fingers == 4:
            return "Paper"
        # Rock: All four fingers are down
        elif total_fingers == 0:
            return "Rock"
        else:
            return "Invalid"

    def get_winner(self, user, comp):
        if user == "Invalid" or user == "No gesture":
            return "Play your move!"
        if user == comp:
            return "Draw"
        if (user == "Rock" and comp == "Scissors") or \
           (user == "Paper" and comp == "Rock") or \
           (user == "Scissors" and comp == "Paper"):
            return "You Win"
        return "Computer Wins"

    def draw_overlay(self, frame):
        h, w, _ = frame.shape
        
        # Draw Score
        cv2.putText(frame, f"Score: {self.score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
        cv2.putText(frame, f"Score: {self.score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Draw Countdown
        if self.round_active and self.countdown > 0:
            text = str(self.countdown)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4, 10)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10)
        
        # Draw Moves and Result
        if self.user_move:
            cv2.putText(frame, f"You: {self.user_move}", (10, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 4)
            cv2.putText(frame, f"Computer: {self.computer_move}", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 4)
        
        if self.result:
            result_size = cv2.getTextSize(self.result, cv2.FONT_HERSHEY_SIMPLEX, 2, 5)[0]
            result_x = (w - result_size[0]) // 2
            cv2.putText(frame, self.result, (result_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            
    def get_game_state(self):
        return {
            "is_running": self.is_running,
            "score": self.score
        }
    
class GameManager:
    """Manages all games - BULLETPROOF"""
    def __init__(self):
        self.games = {
            'fruit_ninja': FruitNinjaGame(),
            'trex_run': TRexRunGame(),
            'rock_paper_scissors': RockPaperScissorsGame(),
        }
        self.current_game = None
        self.current_game_id = None
        self.lock = Lock()
    
    def start_game(self, game_id):
        """Start a specific game"""
        with self.lock:
            if game_id not in self.games:
                return {"success": False, "message": "Game not found!"}
            
            # FORCE stop current game if running
            if self.current_game:
                print(f"🔄 FORCE stopping current game before starting {game_id}")
                try:
                    self.current_game.stop_game()
                except Exception as e:
                    print(f"⚠️ Error stopping previous game: {e}")
                
                # Wait a moment for cleanup
                time.sleep(0.2)
            
            # Start new game
            self.current_game = self.games[game_id]
            self.current_game_id = game_id
            
            return self.current_game.start_game()
    
    def stop_current_game(self):
        """Stop the currently running game - BULLETPROOF"""
        with self.lock:
            if self.current_game:
                try:
                    result = self.current_game.stop_game()
                    self.current_game = None
                    self.current_game_id = None
                    return result
                except Exception as e:
                    print(f"❌ Error stopping game: {e}")
                    # Force cleanup anyway
                    self.current_game = None
                    self.current_game_id = None
                    return {"success": True, "message": "Game force stopped", "score": 0}
            return {"success": True, "message": "No game running"}
    
    def get_current_frame(self):
        """Get current game frame"""
        if self.current_game:
            try:
                return self.current_game.get_frame()
            except Exception as e:
                print(f"❌ Error getting frame: {e}")
                return None
        return None
    
    def get_current_game_state(self):
        """Get current game state"""
        if self.current_game:
            try:
                state = self.current_game.get_game_state()
                state['game_id'] = self.current_game_id
                return state
            except Exception as e:
                print(f"❌ Error getting game state: {e}")
                return {'is_running': False, 'game_id': None}
        return {'is_running': False, 'game_id': None}


# Global game manager instance
game_manager = GameManager()