"""
Rock Paper Scissors Game Engine with OpenCV and MediaPipe
Gesture-controlled gameplay against AI opponent
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class RPSGameState:
    """Game state data structure with state transition methods"""
    current_round: int = 1
    player_score: int = 0
    ai_score: int = 0
    total_points: int = 0
    game_phase: str = "waiting"  # waiting, countdown, playing, result
    countdown_timer: float = 0.0
    player_choice: Optional[str] = None
    ai_choice: Optional[str] = None
    last_result: Optional[str] = None
    round_start_time: float = 0.0
    high_score: int = 0  # Track personal high score
    tokens_earned_this_session: int = 0  # Track tokens earned in current session
    
    # Phase transition methods
    def transition_to_countdown(self, countdown_duration: float = 3.0):
        """Transition to countdown phase"""
        self.game_phase = "countdown"
        self.countdown_timer = countdown_duration
        self.round_start_time = time.time()
        self.player_choice = None
        self.ai_choice = None
        self.last_result = None
    
    def transition_to_playing(self):
        """Transition to playing phase"""
        self.game_phase = "playing"
        self.round_start_time = time.time()
        self.countdown_timer = 0.0
    
    def transition_to_result(self, player_choice: str, ai_choice: str, result: str):
        """Transition to result phase with choices and result"""
        self.game_phase = "result"
        self.player_choice = player_choice
        self.ai_choice = ai_choice
        self.last_result = result
        self.round_start_time = time.time()
    
    def transition_to_waiting(self):
        """Transition to waiting phase and increment round"""
        self.game_phase = "waiting"
        self.current_round += 1
        self.countdown_timer = 0.0
    
    def reset_state(self):
        """Reset game state to initial values"""
        # Preserve high score and reset session tokens
        high_score = self.high_score
        
        self.current_round = 1
        self.player_score = 0
        self.ai_score = 0
        self.total_points = 0
        self.game_phase = "waiting"
        self.countdown_timer = 0.0
        self.player_choice = None
        self.ai_choice = None
        self.last_result = None
        self.round_start_time = 0.0
        self.high_score = high_score  # Preserve high score
        self.tokens_earned_this_session = 0  # Reset session tokens
    
    def update_scores(self, result: str, points_per_win: int = 10, points_per_tie: int = 2, points_per_loss: int = 0):
        """Update scores based on round result"""
        if result == "win":
            self.player_score += 1
            self.total_points += points_per_win
        elif result == "tie":
            self.total_points += points_per_tie
        elif result == "lose":
            self.ai_score += 1
            self.total_points += points_per_loss
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since current phase started"""
        return time.time() - self.round_start_time
    
    def is_countdown_finished(self) -> bool:
        """Check if countdown phase is finished"""
        return self.game_phase == "countdown" and self.countdown_timer <= 0
    
    def should_transition_from_result(self, result_display_duration: float = 2.0) -> bool:
        """Check if result phase should transition to waiting"""
        return self.game_phase == "result" and self.get_elapsed_time() >= result_display_duration


class GestureDetector:
    """Handles hand gesture detection and classification using MediaPipe"""
    
    # Gesture patterns based on finger states (thumb, index, middle, ring, pinky)
    GESTURE_PATTERNS = {
        "rock": {
            "thumb": False,
            "index": False, 
            "middle": False,
            "ring": False,
            "pinky": False
        },
        "paper": {
            "thumb": True,
            "index": True,
            "middle": True, 
            "ring": True,
            "pinky": True
        },
        "scissors": {
            "thumb": False,
            "index": True,
            "middle": True,
            "ring": False,
            "pinky": False
        }
    }
    
    # MediaPipe hand landmark indices for fingertips and joints
    FINGER_TIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
    FINGER_PIP_IDS = [3, 6, 10, 14, 18]  # finger joints for comparison
    FINGER_MCP_IDS = [2, 5, 9, 13, 17]   # finger base joints for additional validation
    
    def __init__(self, confidence_threshold: float = 0.8):
        """Initialize gesture detector with MediaPipe and error handling"""
        self.mp_hands = mp.solutions.hands
        self.hands = None
        self.mp_draw = mp.solutions.drawing_utils
        self.confidence_threshold = confidence_threshold
        
        # Error handling and recovery state
        self.mediapipe_initialized = False
        self.initialization_attempts = 0
        self.max_initialization_attempts = 3
        self.last_error = None
        self.error_count = 0
        self.max_consecutive_errors = 10
        self.fallback_mode = False
        
        # Initialize MediaPipe with error handling
        self._initialize_mediapipe()
        
        # Gesture validation and filtering
        self.gesture_history = []
        self.history_size = 5  # Number of recent gestures to track
        self.stability_threshold = 0.6  # Minimum ratio of consistent gestures
        self.min_hand_size = 0.05  # Minimum hand size relative to frame
        self.max_hand_size = 0.8   # Maximum hand size relative to frame
        
        # Enhanced error handling for unclear gestures
        self.unclear_gesture_count = 0
        self.max_unclear_gestures = 5  # Max consecutive unclear gestures before adjusting
        self.unclear_gesture_threshold = 0.6  # Lower threshold for unclear gesture detection
        self.last_unclear_time = 0.0
        self.unclear_gesture_cooldown = 2.0  # Seconds between unclear gesture messages
        
        # Multiple hand detection management
        self.multiple_hands_detected = False
        self.multiple_hands_count = 0
        self.multiple_hands_warning_shown = False
        self.hand_selection_history = []  # Track which hand was selected
        self.max_hand_history = 10
        
        # Dynamic confidence threshold management
        self.base_confidence_threshold = confidence_threshold
        self.adaptive_threshold = confidence_threshold
        self.threshold_adjustment_factor = 0.05
        self.min_threshold = 0.5
        self.max_threshold = 0.95
        self.performance_history = []  # Track detection performance
        self.performance_window = 20  # Number of recent detections to consider
    
    def _initialize_mediapipe(self):
        """Initialize MediaPipe with error handling and recovery"""
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mediapipe_initialized = True
            self.initialization_attempts = 0
            self.error_count = 0
            self.fallback_mode = False
            self.last_error = None
            print("✅ MediaPipe Hands initialized successfully")
            return True
        except Exception as e:
            self.initialization_attempts += 1
            self.last_error = str(e)
            self.mediapipe_initialized = False
            print(f"❌ MediaPipe initialization failed (attempt {self.initialization_attempts}): {e}")
            
            if self.initialization_attempts >= self.max_initialization_attempts:
                print("⚠️ MediaPipe initialization failed after maximum attempts, enabling fallback mode")
                self.fallback_mode = True
            
            return False
    
    def _retry_mediapipe_initialization(self):
        """Retry MediaPipe initialization if it failed"""
        if not self.mediapipe_initialized and not self.fallback_mode:
            if self.initialization_attempts < self.max_initialization_attempts:
                print(f"🔄 Retrying MediaPipe initialization (attempt {self.initialization_attempts + 1})")
                return self._initialize_mediapipe()
        return False
    
    def _handle_mediapipe_error(self, error: Exception):
        """Handle MediaPipe processing errors with recovery strategies"""
        self.error_count += 1
        self.last_error = str(error)
        
        print(f"⚠️ MediaPipe error ({self.error_count}): {error}")
        
        # If too many consecutive errors, try to reinitialize
        if self.error_count >= self.max_consecutive_errors:
            print("🔄 Too many MediaPipe errors, attempting reinitialization")
            self.mediapipe_initialized = False
            self.initialization_attempts = 0
            self._initialize_mediapipe()
        
        return None, 0.0
    
    def get_error_status(self) -> dict:
        """Get current error status and diagnostics"""
        return {
            'mediapipe_initialized': self.mediapipe_initialized,
            'fallback_mode': self.fallback_mode,
            'initialization_attempts': self.initialization_attempts,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'can_retry': (not self.mediapipe_initialized and 
                         not self.fallback_mode and 
                         self.initialization_attempts < self.max_initialization_attempts),
            'unclear_gesture_count': self.unclear_gesture_count,
            'multiple_hands_detected': self.multiple_hands_detected,
            'adaptive_threshold': self.adaptive_threshold,
            'performance_score': self._calculate_performance_score()
        }
    
    def _handle_unclear_gesture(self, confidence: float) -> dict:
        """Handle unclear gesture detection with adaptive feedback"""
        current_time = time.time()
        self.unclear_gesture_count += 1
        
        # Determine the type of unclear gesture issue
        issue_type = "low_confidence"
        guidance_message = "Try making a clearer gesture"
        
        if confidence < 0.3:
            issue_type = "very_unclear"
            guidance_message = "Hold your hand steady and make a clear rock, paper, or scissors"
        elif confidence < 0.5:
            issue_type = "somewhat_unclear"
            guidance_message = "Make your gesture more distinct"
        elif confidence < self.unclear_gesture_threshold:
            issue_type = "borderline"
            guidance_message = "Hold your gesture steady for better recognition"
        
        # Adjust confidence threshold if too many unclear gestures
        if self.unclear_gesture_count >= self.max_unclear_gestures:
            self._adjust_confidence_threshold_down()
            self.unclear_gesture_count = 0
            guidance_message += " (Sensitivity increased)"
        
        # Rate limit unclear gesture messages
        show_message = (current_time - self.last_unclear_time) > self.unclear_gesture_cooldown
        if show_message:
            self.last_unclear_time = current_time
        
        return {
            'type': issue_type,
            'message': guidance_message,
            'show_message': show_message,
            'confidence': confidence,
            'count': self.unclear_gesture_count
        }
    
    def _handle_multiple_hands(self, multi_hand_landmarks, frame_shape) -> dict:
        """Enhanced multiple hand detection with user guidance"""
        hand_count = len(multi_hand_landmarks)
        self.multiple_hands_detected = True
        self.multiple_hands_count = hand_count
        
        # Select the best hand using enhanced criteria
        selected_hand = self._select_best_hand_enhanced(multi_hand_landmarks, frame_shape)
        
        # Track hand selection for consistency
        if selected_hand:
            hand_info = self._get_hand_info(selected_hand, frame_shape)
            self.hand_selection_history.append(hand_info)
            
            # Maintain history size
            if len(self.hand_selection_history) > self.max_hand_history:
                self.hand_selection_history.pop(0)
        
        # Determine guidance message
        guidance_message = f"Multiple hands detected ({hand_count}). Using the most prominent one."
        
        if hand_count > 2:
            guidance_message = f"Too many hands detected ({hand_count}). Please show only one hand."
        elif hand_count == 2:
            guidance_message = "Two hands detected. Please use only one hand for gestures."
        
        return {
            'hand_count': hand_count,
            'selected_hand': selected_hand,
            'guidance_message': guidance_message,
            'hand_info': hand_info if selected_hand else None
        }
    
    def _select_best_hand_enhanced(self, multi_hand_landmarks, frame_shape):
        """Enhanced hand selection with multiple criteria"""
        if len(multi_hand_landmarks) == 1:
            return multi_hand_landmarks[0]
        
        best_hand = None
        best_score = -1
        hand_scores = []
        
        for i, hand_landmarks in enumerate(multi_hand_landmarks):
            score_components = self._calculate_hand_score(hand_landmarks, frame_shape)
            total_score = sum(score_components.values())
            
            hand_scores.append({
                'index': i,
                'hand': hand_landmarks,
                'score': total_score,
                'components': score_components
            })
            
            if total_score > best_score:
                best_score = total_score
                best_hand = hand_landmarks
        
        # Log hand selection decision for debugging
        print(f"🤚 Selected hand with score {best_score:.2f} from {len(multi_hand_landmarks)} hands")
        
        return best_hand
    
    def _calculate_hand_score(self, hand_landmarks, frame_shape) -> dict:
        """Calculate comprehensive score for hand selection"""
        landmarks = hand_landmarks.landmark
        
        # Calculate hand properties
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        hand_width = max(x_coords) - min(x_coords)
        hand_height = max(y_coords) - min(y_coords)
        hand_size = max(hand_width, hand_height)
        
        # Center position
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        # Score components
        scores = {}
        
        # Size score (prefer larger hands, but not too large)
        optimal_size = 0.3
        size_diff = abs(hand_size - optimal_size)
        scores['size'] = max(0, 1.0 - (size_diff / optimal_size)) * 30
        
        # Centrality score (prefer hands closer to center)
        center_distance = math.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
        scores['centrality'] = max(0, 1.0 - (center_distance / 0.7)) * 25
        
        # Stability score (prefer hands that were selected before)
        stability_bonus = 0
        if self.hand_selection_history:
            for hist_hand in self.hand_selection_history[-3:]:  # Check last 3 selections
                if self._hands_similar(landmarks, hist_hand):
                    stability_bonus += 5
        scores['stability'] = min(stability_bonus, 15)
        
        # Completeness score (prefer hands with all landmarks visible)
        visible_landmarks = sum(1 for lm in landmarks if 0 <= lm.x <= 1 and 0 <= lm.y <= 1)
        scores['completeness'] = (visible_landmarks / len(landmarks)) * 20
        
        # Gesture clarity score (prefer hands that form clear gestures)
        try:
            finger_states = self.get_finger_states(landmarks)
            gesture, confidence = self.classify_hand_pose(finger_states)
            scores['clarity'] = confidence * 10 if gesture else 0
        except:
            scores['clarity'] = 0
        
        return scores
    
    def _get_hand_info(self, hand_landmarks, frame_shape) -> dict:
        """Get comprehensive information about a hand"""
        landmarks = hand_landmarks.landmark
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        return {
            'center_x': sum(x_coords) / len(x_coords),
            'center_y': sum(y_coords) / len(y_coords),
            'width': max(x_coords) - min(x_coords),
            'height': max(y_coords) - min(y_coords),
            'timestamp': time.time()
        }
    
    def _hands_similar(self, landmarks1, hand_info2) -> bool:
        """Check if two hands are similar (same hand tracked over time)"""
        x_coords = [lm.x for lm in landmarks1]
        y_coords = [lm.y for lm in landmarks1]
        
        center_x1 = sum(x_coords) / len(x_coords)
        center_y1 = sum(y_coords) / len(y_coords)
        
        # Compare centers (within 10% of frame)
        distance = math.sqrt((center_x1 - hand_info2['center_x'])**2 + 
                           (center_y1 - hand_info2['center_y'])**2)
        
        return distance < 0.1  # 10% of frame diagonal
    
    def _adjust_confidence_threshold_down(self):
        """Lower confidence threshold to be more permissive"""
        old_threshold = self.adaptive_threshold
        self.adaptive_threshold = max(
            self.min_threshold,
            self.adaptive_threshold - self.threshold_adjustment_factor
        )
        
        if self.adaptive_threshold != old_threshold:
            print(f"🎯 Lowered confidence threshold: {old_threshold:.2f} → {self.adaptive_threshold:.2f}")
    
    def _adjust_confidence_threshold_up(self):
        """Raise confidence threshold to be more strict"""
        old_threshold = self.adaptive_threshold
        self.adaptive_threshold = min(
            self.max_threshold,
            self.adaptive_threshold + self.threshold_adjustment_factor
        )
        
        if self.adaptive_threshold != old_threshold:
            print(f"🎯 Raised confidence threshold: {old_threshold:.2f} → {self.adaptive_threshold:.2f}")
    
    def _calculate_performance_score(self) -> float:
        """Calculate recent detection performance score"""
        if not self.performance_history:
            return 0.0
        
        recent_performance = self.performance_history[-self.performance_window:]
        successful_detections = sum(1 for p in recent_performance if p['success'])
        
        return successful_detections / len(recent_performance) if recent_performance else 0.0
    
    def _update_performance_history(self, success: bool, confidence: float = 0.0):
        """Update performance tracking"""
        self.performance_history.append({
            'success': success,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        # Maintain history size
        if len(self.performance_history) > self.performance_window * 2:
            self.performance_history = self.performance_history[-self.performance_window:]
        
        # Adjust threshold based on performance
        performance_score = self._calculate_performance_score()
        
        if len(self.performance_history) >= self.performance_window:
            if performance_score > 0.8:  # High success rate
                self._adjust_confidence_threshold_up()
            elif performance_score < 0.4:  # Low success rate
                self._adjust_confidence_threshold_down()
        
    def detect_gesture(self, frame: np.ndarray) -> Tuple[Optional[str], float, dict]:
        """
        Enhanced gesture detection with comprehensive error handling
        Returns: (gesture_name, confidence_score, error_info)
        """
        error_info = {
            'type': 'none',
            'message': '',
            'show_message': False,
            'multiple_hands': False,
            'hand_count': 0
        }
        
        # Check if MediaPipe is initialized, try to recover if not
        if not self.mediapipe_initialized:
            if not self._retry_mediapipe_initialization():
                if self.fallback_mode:
                    result = self._fallback_gesture_detection(frame)
                    error_info.update({
                        'type': 'fallback_mode',
                        'message': 'Using fallback detection (MediaPipe unavailable)',
                        'show_message': True
                    })
                    return result[0], result[1], error_info
                
                error_info.update({
                    'type': 'mediapipe_error',
                    'message': 'MediaPipe initialization failed',
                    'show_message': True
                })
                self._update_performance_history(False)
                return None, 0.0, error_info
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Reset error count on successful processing
            if self.error_count > 0:
                self.error_count = max(0, self.error_count - 1)
            
            if results.multi_hand_landmarks:
                hand_count = len(results.multi_hand_landmarks)
                
                # Handle multiple hands with enhanced logic
                if hand_count > 1:
                    multiple_hand_info = self._handle_multiple_hands(results.multi_hand_landmarks, frame.shape)
                    best_hand = multiple_hand_info['selected_hand']
                    
                    error_info.update({
                        'multiple_hands': True,
                        'hand_count': hand_count,
                        'type': 'multiple_hands',
                        'message': multiple_hand_info['guidance_message'],
                        'show_message': True
                    })
                else:
                    # Single hand detected
                    best_hand = results.multi_hand_landmarks[0]
                    self.multiple_hands_detected = False
                
                if best_hand:
                    # Validate hand detection quality
                    if not self._validate_hand_detection(best_hand.landmark, frame.shape):
                        error_info.update({
                            'type': 'poor_quality',
                            'message': 'Hand too small or at edge of frame',
                            'show_message': True
                        })
                        self._update_performance_history(False)
                        return None, 0.0, error_info
                    
                    # Draw hand landmarks on frame
                    self.mp_draw.draw_landmarks(
                        frame, best_hand, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Get finger states and classify gesture
                    finger_states = self.get_finger_states(best_hand.landmark)
                    gesture, raw_confidence = self.classify_hand_pose(finger_states)
                    
                    # Apply gesture filtering and stability check
                    filtered_gesture, final_confidence = self._filter_gesture(gesture, raw_confidence)
                    
                    # Use adaptive threshold instead of fixed threshold
                    if filtered_gesture and final_confidence >= self.adaptive_threshold:
                        # Successful detection
                        self.unclear_gesture_count = 0  # Reset unclear count on success
                        self._update_performance_history(True, final_confidence)
                        return filtered_gesture, final_confidence, error_info
                    else:
                        # Handle unclear gesture
                        unclear_info = self._handle_unclear_gesture(final_confidence)
                        error_info.update({
                            'type': 'unclear_gesture',
                            'message': unclear_info['message'],
                            'show_message': unclear_info['show_message'],
                            'confidence': final_confidence,
                            'unclear_count': unclear_info['count']
                        })
                        self._update_performance_history(False, final_confidence)
                        return None, final_confidence, error_info
            
            # No hand detected - clear gesture history and reset counters
            self.gesture_history.clear()
            self.multiple_hands_detected = False
            self.unclear_gesture_count = 0
            
            error_info.update({
                'type': 'no_hand',
                'message': 'Show your hand to the camera',
                'show_message': True
            })
            self._update_performance_history(False)
            return None, 0.0, error_info
            
        except Exception as e:
            error_result = self._handle_mediapipe_error(e)
            error_info.update({
                'type': 'processing_error',
                'message': f'Detection error: {str(e)[:50]}...',
                'show_message': True
            })
            self._update_performance_history(False)
            return error_result[0], error_result[1], error_info
    
    def _select_best_hand(self, multi_hand_landmarks, frame_shape):
        """Select the best hand from multiple detected hands"""
        if len(multi_hand_landmarks) == 1:
            return multi_hand_landmarks[0]
        
        # If multiple hands, select the largest/most prominent one
        best_hand = None
        best_score = 0
        
        for hand_landmarks in multi_hand_landmarks:
            # Calculate hand size and position score
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            hand_width = max(x_coords) - min(x_coords)
            hand_height = max(y_coords) - min(y_coords)
            hand_size = max(hand_width, hand_height)
            
            # Prefer hands closer to center and larger
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            center_distance = abs(center_x - 0.5) + abs(center_y - 0.5)
            
            # Score based on size and centrality
            score = hand_size * 2 - center_distance
            
            if score > best_score:
                best_score = score
                best_hand = hand_landmarks
        
        return best_hand
    
    def _fallback_gesture_detection(self, frame: np.ndarray) -> Tuple[Optional[str], float]:
        """Fallback gesture detection when MediaPipe fails"""
        # Simple fallback - could implement basic computer vision techniques
        # For now, return None to indicate detection failure
        print("⚠️ Using fallback gesture detection (MediaPipe unavailable)")
        return None, 0.0
    
    def get_finger_states(self, landmarks) -> List[bool]:
        """
        Determine which fingers are extended based on landmarks with improved accuracy
        Returns: [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
        """
        finger_states = []
        
        # Check each finger with enhanced logic
        for i in range(5):
            if i == 0:  # Thumb - special case due to different orientation
                # Use multiple reference points for better thumb detection
                tip_x = landmarks[self.FINGER_TIP_IDS[i]].x
                pip_x = landmarks[self.FINGER_PIP_IDS[i]].x
                mcp_x = landmarks[self.FINGER_MCP_IDS[i]].x
                wrist_x = landmarks[0].x
                
                # Thumb extended if tip is further from palm center than pip joint
                palm_center_x = (landmarks[5].x + landmarks[17].x) / 2  # Average of index and pinky base
                
                tip_distance = abs(tip_x - palm_center_x)
                pip_distance = abs(pip_x - palm_center_x)
                
                # Additional check: tip should be further from wrist than mcp
                thumb_extended = (tip_distance > pip_distance * 1.2) and (abs(tip_x - wrist_x) > abs(mcp_x - wrist_x))
                finger_states.append(thumb_extended)
            else:
                # Other fingers - use multiple joint comparisons for accuracy
                tip_y = landmarks[self.FINGER_TIP_IDS[i]].y
                pip_y = landmarks[self.FINGER_PIP_IDS[i]].y
                mcp_y = landmarks[self.FINGER_MCP_IDS[i]].y
                
                # Finger extended if tip is significantly above pip joint
                # and tip is above mcp joint (to avoid false positives from bent fingers)
                finger_extended = (tip_y < pip_y - 0.02) and (tip_y < mcp_y)
                finger_states.append(finger_extended)
        
        return finger_states
    
    def classify_hand_pose(self, finger_states: List[bool]) -> Tuple[Optional[str], float]:
        """
        Classify hand pose into rock/paper/scissors with weighted confidence scoring
        Returns: (gesture_name, confidence_score)
        """
        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        
        # Weighted importance for different fingers in each gesture
        gesture_weights = {
            "rock": [0.15, 0.25, 0.25, 0.20, 0.15],      # All fingers important for fist
            "paper": [0.20, 0.20, 0.20, 0.20, 0.20],     # All fingers equally important
            "scissors": [0.10, 0.40, 0.40, 0.05, 0.05]   # Index and middle most important
        }
        
        best_match = None
        best_confidence = 0.0
        confidence_scores = {}
        
        # Compare against each gesture pattern with weighted scoring
        for gesture_name, pattern in self.GESTURE_PATTERNS.items():
            weighted_score = 0.0
            weights = gesture_weights[gesture_name]
            
            # Calculate weighted confidence
            for i, finger_name in enumerate(finger_names):
                if finger_states[i] == pattern[finger_name]:
                    weighted_score += weights[i]
            
            confidence_scores[gesture_name] = weighted_score
            
            # Update best match if this is better
            if weighted_score > best_confidence:
                best_confidence = weighted_score
                best_match = gesture_name
        
        # Additional validation: check for gesture-specific requirements
        if best_match:
            best_confidence = self._validate_gesture_specifics(best_match, finger_states, best_confidence)
        
        # Only return gesture if confidence meets adaptive threshold
        if best_confidence >= self.adaptive_threshold:
            return best_match, best_confidence
        else:
            return None, best_confidence
    
    def _validate_hand_detection(self, landmarks, frame_shape) -> bool:
        """
        Validate the quality of hand detection
        Returns: True if hand detection is reliable
        """
        # Calculate hand bounding box
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        hand_width = max(x_coords) - min(x_coords)
        hand_height = max(y_coords) - min(y_coords)
        hand_size = max(hand_width, hand_height)
        
        # Check if hand size is within reasonable bounds
        if hand_size < self.min_hand_size or hand_size > self.max_hand_size:
            return False
        
        # Check if hand is reasonably centered (not cut off at edges)
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        margin = 0.1  # 10% margin from edges
        if (center_x < margin or center_x > 1 - margin or 
            center_y < margin or center_y > 1 - margin):
            return False
        
        return True
    
    def _validate_gesture_specifics(self, gesture: str, finger_states: List[bool], base_confidence: float) -> float:
        """
        Apply gesture-specific validation rules
        Returns: Adjusted confidence score
        """
        thumb, index, middle, ring, pinky = finger_states
        
        if gesture == "rock":
            # Rock should have all fingers closed
            if any(finger_states):
                base_confidence *= 0.8  # Reduce confidence if any finger is extended
        
        elif gesture == "paper":
            # Paper should have all fingers extended
            if not all(finger_states):
                base_confidence *= 0.8  # Reduce confidence if any finger is closed
        
        elif gesture == "scissors":
            # Scissors should have exactly index and middle extended
            expected_extended = sum([index, middle])
            expected_closed = sum([thumb, ring, pinky])
            
            if expected_extended != 2:
                base_confidence *= 0.7  # Heavily penalize if not exactly 2 fingers
            if expected_closed != 0:
                base_confidence *= 0.9  # Slightly penalize if other fingers extended
        
        return base_confidence
    
    def _filter_gesture(self, gesture: Optional[str], confidence: float) -> Tuple[Optional[str], float]:
        """
        Apply temporal filtering to reduce noise and improve stability
        Returns: (filtered_gesture, filtered_confidence)
        """
        # Add current gesture to history
        if gesture:
            self.gesture_history.append((gesture, confidence))
        else:
            self.gesture_history.append((None, 0.0))
        
        # Maintain history size
        if len(self.gesture_history) > self.history_size:
            self.gesture_history.pop(0)
        
        # If we don't have enough history, return current gesture
        if len(self.gesture_history) < 3:
            return gesture, confidence
        
        # Count occurrences of each gesture in recent history
        gesture_counts = {}
        total_confidence = 0.0
        valid_detections = 0
        
        for hist_gesture, hist_confidence in self.gesture_history:
            if hist_gesture:
                gesture_counts[hist_gesture] = gesture_counts.get(hist_gesture, 0) + 1
                total_confidence += hist_confidence
                valid_detections += 1
        
        # If no valid detections in history, return None
        if valid_detections == 0:
            return None, 0.0
        
        # Find most frequent gesture
        most_frequent_gesture = max(gesture_counts.items(), key=lambda x: x[1])[0] if gesture_counts else None
        
        # Check stability - gesture should appear in at least stability_threshold of recent frames
        if most_frequent_gesture:
            stability_ratio = gesture_counts[most_frequent_gesture] / len(self.gesture_history)
            
            if stability_ratio >= self.stability_threshold:
                # Return stable gesture with average confidence
                avg_confidence = total_confidence / valid_detections
                return most_frequent_gesture, min(avg_confidence, 1.0)
        
        # Not stable enough, return None
        return None, confidence
    
    def get_gesture_diagnostics(self) -> dict:
        """Get comprehensive gesture detection diagnostics"""
        return {
            'adaptive_threshold': self.adaptive_threshold,
            'base_threshold': self.base_confidence_threshold,
            'unclear_gesture_count': self.unclear_gesture_count,
            'multiple_hands_detected': self.multiple_hands_detected,
            'performance_score': self._calculate_performance_score(),
            'error_status': self.get_error_status(),
            'recent_performance': self.performance_history[-5:] if self.performance_history else [],
            'hand_selection_consistency': len(set(h.get('center_x', 0) for h in self.hand_selection_history[-3:])) <= 1 if len(self.hand_selection_history) >= 3 else True
        }
    
    def reset_error_counters(self):
        """Reset all error counters and adaptive settings"""
        self.unclear_gesture_count = 0
        self.multiple_hands_detected = False
        self.multiple_hands_count = 0
        self.adaptive_threshold = self.base_confidence_threshold
        self.performance_history.clear()
        self.hand_selection_history.clear()
        print("🔄 Gesture detection error counters reset")
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'hands'):
            self.hands.close()

class AIOpponent:
    """AI opponent system with fair randomization and choice display"""
    
    def __init__(self):
        """Initialize AI opponent with fair randomization"""
        self.choices = ["rock", "paper", "scissors"]
        
        # Fair randomization system
        self.choice_history = []
        self.max_history_size = 10  # Track last 10 choices for fairness
        self.min_choice_gap = 2  # Minimum rounds between same choice
        
        # Choice timing and display
        self.choice_made_time = 0.0
        self.choice_display_delay = 0.5  # Delay before showing AI choice
        self.current_choice = None
        self.choice_visible = False
        
        # Randomization seed management
        self.reseed_interval = 50  # Reseed random generator every 50 rounds
        self.rounds_since_reseed = 0
        
        print("🤖 AI Opponent initialized with fair randomization")
    
    def make_choice(self) -> str:
        """Generate AI choice using fair randomization algorithm"""
        # Reseed periodically for better randomness
        if self.rounds_since_reseed >= self.reseed_interval:
            random.seed()
            self.rounds_since_reseed = 0
            print("🔄 AI randomization reseeded")
        
        # Get available choices (excluding recent repeats for fairness)
        available_choices = self._get_fair_choices()
        
        # Make random choice from available options
        choice = random.choice(available_choices)
        
        # Update choice history and timing
        self.choice_history.append(choice)
        if len(self.choice_history) > self.max_history_size:
            self.choice_history.pop(0)
        
        self.current_choice = choice
        self.choice_made_time = time.time()
        self.choice_visible = False
        self.rounds_since_reseed += 1
        
        print(f"🤖 AI chose: {choice}")
        return choice
    
    def _get_fair_choices(self) -> List[str]:
        """Get available choices ensuring fairness and avoiding patterns"""
        if len(self.choice_history) < self.min_choice_gap:
            return self.choices.copy()
        
        # Check recent history to avoid immediate repeats
        recent_choices = self.choice_history[-self.min_choice_gap:]
        
        # If all recent choices are the same, force different choice
        if len(set(recent_choices)) == 1 and len(recent_choices) >= self.min_choice_gap:
            excluded_choice = recent_choices[0]
            available = [choice for choice in self.choices if choice != excluded_choice]
            print(f"🎯 AI avoiding repeat: excluding {excluded_choice}")
            return available
        
        # Normal case - all choices available
        return self.choices.copy()
    
    def should_display_choice(self) -> bool:
        """Check if AI choice should be displayed based on timing"""
        if not self.current_choice or self.choice_visible:
            return self.choice_visible
        
        # Display choice after delay
        if time.time() - self.choice_made_time >= self.choice_display_delay:
            self.choice_visible = True
            return True
        
        return False
    
    def get_choice_display_info(self) -> dict:
        """Get information about AI choice display state"""
        return {
            'choice': self.current_choice,
            'visible': self.choice_visible,
            'time_since_choice': time.time() - self.choice_made_time if self.current_choice else 0,
            'display_delay': self.choice_display_delay
        }
    
    def reset_choice(self):
        """Reset current choice state"""
        self.current_choice = None
        self.choice_visible = False
        self.choice_made_time = 0.0
    
    def get_choice_statistics(self) -> dict:
        """Get statistics about AI choice patterns for debugging"""
        if not self.choice_history:
            return {'total_choices': 0, 'distribution': {}}
        
        distribution = {}
        for choice in self.choices:
            count = self.choice_history.count(choice)
            distribution[choice] = {
                'count': count,
                'percentage': (count / len(self.choice_history)) * 100
            }
        
        return {
            'total_choices': len(self.choice_history),
            'distribution': distribution,
            'recent_choices': self.choice_history[-5:],  # Last 5 choices
            'rounds_since_reseed': self.rounds_since_reseed
        }


class RockPaperScissorsEngine:
    """Rock Paper Scissors game engine with gesture recognition"""
    
    def __init__(self):
        """Initialize the Rock Paper Scissors game engine"""
        # Initialize gesture detector
        self.gesture_detector = GestureDetector(confidence_threshold=0.8)
        
        # Initialize AI opponent
        self.ai_opponent = AIOpponent()
        
        # Game state
        self.game_state = RPSGameState()
        
        # Gesture detection settings
        self.gesture_confidence_threshold = 0.8
        self.last_detected_gesture = None
        self.gesture_detection_time = 0.0
        self.last_gesture_error_info = {'type': 'none', 'message': '', 'show_message': False}
        
        # Game timing
        self.countdown_duration = 3.0  # 3 seconds countdown
        self.result_display_duration = 2.0  # 2 seconds to show result
        self.auto_start_delay = 1.0  # 1 second delay before auto-starting next round
        
        # Game choices
        self.choices = ["rock", "paper", "scissors"]
        
        # Scoring system
        self.points_per_win = 10
        self.points_per_tie = 2
        self.points_per_loss = 0
        self.tokens_per_milestone = 50
        self.milestone_points = 100
        
        # Round management
        self.max_rounds = None  # None for unlimited rounds
        self.auto_start_rounds = True  # Automatically start next round
        self.round_end_callbacks = []  # Callbacks for round end events
        
        # Camera and processing error handling
        self.camera_error_count = 0
        self.max_camera_errors = 10
        self.processing_error_count = 0
        self.max_processing_errors = 5
        self.last_camera_error = None
        self.last_processing_error = None
        self.camera_recovery_mode = False
        self.processing_recovery_mode = False
        self.error_display_time = 0.0
        self.error_message = None
        self.frame_skip_count = 0
        self.max_frame_skips = 3  # Skip frames during recovery
        
        print("✂️ Rock Paper Scissors Engine initialized!")
    
    def reset_game(self):
        """Reset game to initial state"""
        self.game_state.reset_state()
        self.ai_opponent.reset_choice()
        self.last_detected_gesture = None
        self.gesture_detection_time = 0.0
        print("🔄 Rock Paper Scissors game reset")
    
    def start_round(self):
        """Start a new round with countdown"""
        if self.game_state.game_phase == "waiting":
            # Check if we've reached max rounds
            if self.max_rounds and self.game_state.current_round > self.max_rounds:
                self.end_game()
                return False
            
            self.game_state.transition_to_countdown(self.countdown_duration)
            print(f"🎮 Starting round {self.game_state.current_round}")
            return True
        return False
    
    def end_round(self, player_choice: str, ai_choice: str, result: str):
        """End current round with results and transition to result phase"""
        if self.game_state.game_phase == "playing":
            # Store previous milestone count for comparison
            previous_milestones = self.game_state.total_points // self.milestone_points
            
            # Update scores
            self.game_state.update_scores(result, self.points_per_win, self.points_per_tie, self.points_per_loss)
            
            # Transition to result phase
            self.game_state.transition_to_result(player_choice, ai_choice, result)
            
            # Check for token milestones and award tokens
            current_milestones = self.game_state.total_points // self.milestone_points
            if current_milestones > previous_milestones:
                self._check_token_milestones()
            
            # Call round end callbacks
            for callback in self.round_end_callbacks:
                callback(self.game_state.current_round, result, self.game_state)
            
            print(f"🏁 Round {self.game_state.current_round} ended: {result.upper()}")
            print(f"📊 Score - Player: {self.game_state.player_score}, AI: {self.game_state.ai_score}, Points: {self.game_state.total_points}")
            return True
        return False
    
    def can_start_round(self) -> bool:
        """Check if a new round can be started"""
        return (self.game_state.game_phase == "waiting" and 
                (not self.max_rounds or self.game_state.current_round <= self.max_rounds))
    
    def is_game_finished(self) -> bool:
        """Check if the game is finished (reached max rounds)"""
        return self.max_rounds and self.game_state.current_round > self.max_rounds
    
    def end_game(self, user_id: int = None):
        """End the game and show final results"""
        total_rounds = self.game_state.current_round - 1
        print(f"🎯 Game finished after {total_rounds} rounds!")
        print(f"Final Score - Player: {self.game_state.player_score}, AI: {self.game_state.ai_score}")
        print(f"Total Points Earned: {self.game_state.total_points}")
        
        # Save high score if user is logged in
        if user_id:
            self.save_high_score(user_id)
        
        # Get final scoring summary
        summary = self.get_scoring_summary()
        print(f"🏆 Win Rate: {summary['win_rate']}%")
        if summary['tokens_earned'] > 0:
            print(f"🪙 Total Tokens Earned: {summary['tokens_earned']}")
        
        # Transition to game over state
        self.game_state.game_phase = "game_over"
        
        return summary
    
    def add_round_end_callback(self, callback):
        """Add callback function to be called when round ends"""
        self.round_end_callbacks.append(callback)
    
    def set_max_rounds(self, max_rounds: Optional[int]):
        """Set maximum number of rounds (None for unlimited)"""
        self.max_rounds = max_rounds
    
    def set_auto_start_rounds(self, auto_start: bool):
        """Set whether rounds should auto-start after result display"""
        self.auto_start_rounds = auto_start
    
    def process_frame(self, frame):
        """Enhanced frame processing pipeline with comprehensive error handling"""
        try:
            # Validate input frame
            if frame is None:
                raise ValueError("Received None frame from camera")
            
            if frame.size == 0:
                raise ValueError("Received empty frame from camera")
            
            # Check if we should skip this frame for recovery
            if self.should_skip_frame():
                self._draw_recovery_screen(frame, "Recovering from camera issues...")
                return frame
            
            # Update game timing
            self._update_game_timing()
            
            # Apply frame preprocessing with error handling
            try:
                processed_frame = self._preprocess_frame(frame)
            except Exception as preprocess_error:
                print(f"⚠️ Frame preprocessing failed: {preprocess_error}")
                processed_frame = frame  # Use original frame as fallback
            
            # Detect gestures and get hand landmarks with error handling
            gesture_info = None
            try:
                gesture_info = self._detect_gesture_with_landmarks(processed_frame)
                
                # Reset processing errors on successful detection
                if gesture_info and self.processing_error_count > 0:
                    self.reset_processing_errors()
                    
            except Exception as detection_error:
                print(f"⚠️ Gesture detection failed: {detection_error}")
                
                # Handle detection error
                if not self.handle_processing_error(detection_error, {'phase': 'gesture_detection'}):
                    # If we can't continue, create empty gesture info
                    gesture_info = self._create_empty_gesture_info()
                else:
                    # Try fallback detection
                    gesture_info = self._create_empty_gesture_info()
            
            # Ensure we have valid gesture info
            if gesture_info is None:
                gesture_info = self._create_empty_gesture_info()
            
            # Draw hand landmarks and gesture overlays with error handling
            try:
                self._draw_gesture_overlays(frame, gesture_info)
            except Exception as overlay_error:
                print(f"⚠️ Gesture overlay drawing failed: {overlay_error}")
                # Continue without overlays
            
            # Process frame based on current game phase with error handling
            try:
                if self.game_state.game_phase == "waiting":
                    self._draw_waiting_screen(frame)
                elif self.game_state.game_phase == "countdown":
                    self._draw_countdown_screen(frame)
                elif self.game_state.game_phase == "playing":
                    self._process_playing_phase_enhanced(frame, gesture_info)
                elif self.game_state.game_phase == "result":
                    self._draw_result_screen(frame)
            except Exception as phase_error:
                print(f"⚠️ Game phase processing failed: {phase_error}")
                self.handle_processing_error(phase_error, {'phase': self.game_state.game_phase})
                # Draw basic error screen
                self._draw_error_screen(frame, f"Game phase error: {str(phase_error)[:30]}...")
            
            # Draw real-time feedback display with error handling
            try:
                self._draw_realtime_feedback(frame, gesture_info)
            except Exception as feedback_error:
                print(f"⚠️ Feedback display failed: {feedback_error}")
                # Continue without feedback display
            
            # Always draw UI elements with error handling
            try:
                self._draw_ui(frame)
            except Exception as ui_error:
                print(f"⚠️ UI drawing failed: {ui_error}")
                # Draw minimal UI
                self._draw_minimal_ui(frame)
            
            # Draw error messages if any
            self._draw_error_messages(frame)
            
            # Reset camera errors on successful frame processing
            if self.camera_error_count > 0:
                self.reset_camera_errors()
            
            return frame
            
        except Exception as frame_error:
            print(f"❌ Critical frame processing error: {frame_error}")
            
            # Handle critical frame processing error
            if not self.handle_camera_error(frame_error, {'frame_shape': getattr(frame, 'shape', 'unknown')}):
                # If we can't recover, draw error screen
                try:
                    self._draw_critical_error_screen(frame, str(frame_error))
                except:
                    # If even error screen fails, return original frame
                    pass
            
            return frame
    
    def get_game_state(self):
        """Return current game state for API"""
        return {
            'round': self.game_state.current_round,
            'player_score': self.game_state.player_score,
            'ai_score': self.game_state.ai_score,
            'total_points': self.game_state.total_points,
            'high_score': self.game_state.high_score,
            'tokens_earned_this_session': self.game_state.tokens_earned_this_session,
            'is_new_high_score': self.is_new_high_score(),
            'game_phase': self.game_state.game_phase,
            'countdown_timer': self.game_state.countdown_timer,
            'player_choice': self.game_state.player_choice,
            'ai_choice': self.game_state.ai_choice,
            'last_result': self.game_state.last_result,
            'max_rounds': self.max_rounds,
            'auto_start_rounds': self.auto_start_rounds,
            'can_start_round': self.can_start_round(),
            'is_game_finished': self.is_game_finished(),
            'elapsed_time': self.game_state.get_elapsed_time(),
            'ai_opponent': self.ai_opponent.get_choice_display_info(),
            'ai_statistics': self.ai_opponent.get_choice_statistics(),
            'scoring_summary': self.get_scoring_summary(),
            'gesture_diagnostics': self.gesture_detector.get_gesture_diagnostics(),
            'last_error_info': getattr(self, 'last_gesture_error_info', {})
        }
    
    def handle_key_input(self, key: str) -> bool:
        """Handle keyboard input for game control"""
        if key.lower() == 's':  # Start round
            return self.start_round()
        elif key.lower() == 'r':  # Reset game
            self.reset_game()
            return True
        elif key.lower() == 'a':  # Toggle auto-start
            self.auto_start_rounds = not self.auto_start_rounds
            print(f"Auto-start rounds: {'ON' if self.auto_start_rounds else 'OFF'}")
            return True
        elif key.lower() == 'd':  # Debug AI statistics
            self._print_ai_statistics()
            return True
        return False
    
    def _print_ai_statistics(self):
        """Print AI opponent statistics for debugging"""
        stats = self.ai_opponent.get_choice_statistics()
        print("\n🤖 AI Opponent Statistics:")
        print(f"Total choices made: {stats['total_choices']}")
        print(f"Rounds since reseed: {stats['rounds_since_reseed']}")
        
        if stats['distribution']:
            print("Choice distribution:")
            for choice, data in stats['distribution'].items():
                print(f"  {choice}: {data['count']} ({data['percentage']:.1f}%)")
        
        if stats['recent_choices']:
            print(f"Recent choices: {' -> '.join(stats['recent_choices'])}")
        print()
    
    def handle_camera_error(self, error: Exception, frame_data: dict = None) -> bool:
        """
        Handle camera-related errors with recovery strategies
        Returns: True if should retry, False if should give up
        """
        self.camera_error_count += 1
        self.last_camera_error = str(error)
        
        print(f"📷 Camera error ({self.camera_error_count}): {error}")
        
        # Store error for display
        self.error_message = f"Camera Error: {str(error)[:50]}..."
        self.error_display_time = time.time()
        
        # Check if we should enter recovery mode
        if self.camera_error_count >= 3 and not self.camera_recovery_mode:
            self.camera_recovery_mode = True
            print("🔧 Entering camera recovery mode")
        
        # If too many errors, give up
        if self.camera_error_count >= self.max_camera_errors:
            print("❌ Camera recovery failed - too many errors")
            self.error_message = "Camera failed - please restart game"
            return False
        
        # Try recovery strategies based on error count
        if self.camera_error_count <= 3:
            # Light recovery - just retry
            return True
        elif self.camera_error_count <= 6:
            # Medium recovery - skip frames and reduce processing
            self.frame_skip_count = 2
            return True
        else:
            # Heavy recovery - try MediaPipe reinitialization
            print("🔄 Attempting MediaPipe reinitialization")
            if hasattr(self.gesture_detector, '_initialize_mediapipe'):
                self.gesture_detector._initialize_mediapipe()
            return True
    
    def handle_processing_error(self, error: Exception, context: dict = None) -> bool:
        """
        Handle frame processing errors with graceful degradation
        Returns: True if should continue, False if should stop
        """
        self.processing_error_count += 1
        self.last_processing_error = str(error)
        
        print(f"⚙️ Processing error ({self.processing_error_count}): {error}")
        
        # Store error for display
        self.error_message = f"Processing Error: {str(error)[:50]}..."
        self.error_display_time = time.time()
        
        # Check if we should enter recovery mode
        if self.processing_error_count >= 2 and not self.processing_recovery_mode:
            self.processing_recovery_mode = True
            print("🔧 Entering processing recovery mode")
        
        # If too many errors, enable fallback mode
        if self.processing_error_count >= self.max_processing_errors:
            print("⚠️ Too many processing errors - enabling fallback mode")
            if hasattr(self.gesture_detector, 'fallback_mode'):
                self.gesture_detector.fallback_mode = True
            return True
        
        # Continue with reduced processing
        return True
    
    def reset_camera_errors(self):
        """Reset camera error tracking after successful operation"""
        if self.camera_error_count > 0:
            print(f"✅ Camera recovery successful after {self.camera_error_count} errors")
            self.camera_error_count = 0
            self.last_camera_error = None
            self.camera_recovery_mode = False
            self.frame_skip_count = 0
            self.error_message = None
    
    def reset_processing_errors(self):
        """Reset processing error tracking after successful operation"""
        if self.processing_error_count > 0:
            print(f"✅ Processing recovery successful after {self.processing_error_count} errors")
            self.processing_error_count = 0
            self.last_processing_error = None
            self.processing_recovery_mode = False
            self.error_message = None
    
    def get_error_status(self) -> dict:
        """Get current error status for diagnostics"""
        return {
            'camera_errors': self.camera_error_count,
            'processing_errors': self.processing_error_count,
            'camera_recovery_mode': self.camera_recovery_mode,
            'processing_recovery_mode': self.processing_recovery_mode,
            'last_camera_error': self.last_camera_error,
            'last_processing_error': self.last_processing_error,
            'gesture_detector_status': self.gesture_detector.get_error_status() if hasattr(self.gesture_detector, 'get_error_status') else None,
            'current_error_message': self.error_message,
            'frame_skip_count': self.frame_skip_count
        }
    
    def should_skip_frame(self) -> bool:
        """Check if current frame should be skipped for recovery"""
        if self.frame_skip_count > 0:
            self.frame_skip_count -= 1
            return True
        return False
    
    def _update_game_timing(self):
        """Update game timing and phase transitions"""
        if self.game_state.game_phase == "countdown":
            elapsed = self.game_state.get_elapsed_time()
            self.game_state.countdown_timer = max(0, self.countdown_duration - elapsed)
            
            if self.game_state.is_countdown_finished():
                self.game_state.transition_to_playing()
                print(f"⚡ Round {self.game_state.current_round} - Make your move!")
        
        elif self.game_state.game_phase == "result":
            if self.game_state.should_transition_from_result(self.result_display_duration):
                self.game_state.transition_to_waiting()
                self.ai_opponent.reset_choice()  # Reset AI choice for next round
                
                # Auto-start next round if enabled and game not finished
                if self.auto_start_rounds and not self.is_game_finished():
                    # Add small delay before auto-starting
                    if self.game_state.get_elapsed_time() >= self.result_display_duration + self.auto_start_delay:
                        self.start_round()
        
        elif self.game_state.game_phase == "waiting":
            # Check if we should auto-start the next round
            if (self.auto_start_rounds and 
                self.can_start_round() and 
                self.game_state.get_elapsed_time() >= self.auto_start_delay):
                self.start_round()
    
    def _process_playing_phase(self, frame):
        """Process the playing phase with gesture detection"""
        # Detect hand gestures
        gesture, confidence = self._detect_gesture(frame)
        
        if gesture and confidence > self.gesture_confidence_threshold:
            # AI makes its choice using fair randomization
            ai_choice = self.ai_opponent.make_choice()
            
            # Determine winner and end round
            result = self._determine_winner(gesture, ai_choice)
            self.end_round(gesture, ai_choice, result)
        
        # Draw playing phase UI
        self._draw_playing_screen(frame, gesture, confidence)
    
    def _detect_gesture(self, frame):
        """Detect hand gesture from frame using GestureDetector"""
        gesture, confidence, error_info = self.gesture_detector.detect_gesture(frame)
        
        # Store error info for UI display
        self.last_gesture_error_info = error_info
        
        return gesture, confidence
    
    def _preprocess_frame(self, frame):
        """Preprocess frame for optimal MediaPipe performance"""
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Apply slight blur to reduce noise for better gesture detection
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # Enhance contrast for better hand detection in various lighting
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced_frame = cv2.merge([l, a, b])
        enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_LAB2BGR)
        
        return enhanced_frame
    
    def _detect_gesture_with_landmarks(self, frame):
        """Enhanced gesture detection with comprehensive error handling"""
        gesture_info = {
            'gesture': None,
            'confidence': 0.0,
            'landmarks': None,
            'hand_present': False,
            'hand_bbox': None,
            'finger_states': None,
            'detection_quality': 'none'
        }
        
        try:
            # Check if MediaPipe is available and initialized
            if not hasattr(self.gesture_detector, 'hands') or self.gesture_detector.hands is None:
                print("⚠️ MediaPipe hands not initialized")
                gesture_info['detection_quality'] = 'mediapipe_error'
                return gesture_info
            
            # Check MediaPipe initialization status
            if hasattr(self.gesture_detector, 'mediapipe_initialized') and not self.gesture_detector.mediapipe_initialized:
                print("⚠️ MediaPipe not properly initialized")
                gesture_info['detection_quality'] = 'mediapipe_error'
                
                # Try to reinitialize if possible
                if hasattr(self.gesture_detector, '_retry_mediapipe_initialization'):
                    self.gesture_detector._retry_mediapipe_initialization()
                
                return gesture_info
            
            # Convert BGR to RGB for MediaPipe with error handling
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as color_error:
                print(f"⚠️ Color conversion failed: {color_error}")
                gesture_info['detection_quality'] = 'frame_error'
                return gesture_info
            
            # Process frame with MediaPipe with timeout protection
            try:
                results = self.gesture_detector.hands.process(rgb_frame)
            except Exception as process_error:
                print(f"⚠️ MediaPipe processing failed: {process_error}")
                gesture_info['detection_quality'] = 'processing_error'
                
                # Handle MediaPipe processing error
                if hasattr(self.gesture_detector, '_handle_mediapipe_error'):
                    self.gesture_detector._handle_mediapipe_error(process_error)
                
                return gesture_info
            
            # Process results if available
            if results and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
                try:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Validate hand detection quality
                        try:
                            if self.gesture_detector._validate_hand_detection(hand_landmarks.landmark, frame.shape):
                                gesture_info['hand_present'] = True
                                gesture_info['landmarks'] = hand_landmarks
                                gesture_info['detection_quality'] = 'good'
                                
                                # Get finger states with error handling
                                try:
                                    finger_states = self.gesture_detector.get_finger_states(hand_landmarks.landmark)
                                    gesture_info['finger_states'] = finger_states
                                    
                                    # Classify gesture with error handling
                                    try:
                                        gesture, confidence = self.gesture_detector.classify_hand_pose(finger_states)
                                        gesture_info['gesture'] = gesture
                                        gesture_info['confidence'] = confidence
                                    except Exception as classify_error:
                                        print(f"⚠️ Gesture classification failed: {classify_error}")
                                        gesture_info['detection_quality'] = 'classification_error'
                                    
                                    # Calculate hand bounding box with error handling
                                    try:
                                        gesture_info['hand_bbox'] = self._calculate_hand_bbox(hand_landmarks.landmark, frame.shape)
                                    except Exception as bbox_error:
                                        print(f"⚠️ Bounding box calculation failed: {bbox_error}")
                                    
                                except Exception as finger_error:
                                    print(f"⚠️ Finger state detection failed: {finger_error}")
                                    gesture_info['detection_quality'] = 'finger_error'
                                
                                break  # Use first valid hand
                            else:
                                gesture_info['hand_present'] = True
                                gesture_info['detection_quality'] = 'poor'
                        except Exception as validation_error:
                            print(f"⚠️ Hand validation failed: {validation_error}")
                            gesture_info['detection_quality'] = 'validation_error'
                            
                except Exception as landmarks_error:
                    print(f"⚠️ Landmark processing failed: {landmarks_error}")
                    gesture_info['detection_quality'] = 'landmarks_error'
            
            return gesture_info
            
        except Exception as detection_error:
            print(f"❌ Critical gesture detection error: {detection_error}")
            gesture_info['detection_quality'] = 'critical_error'
            
            # Try to handle the error through gesture detector if possible
            if hasattr(self.gesture_detector, '_handle_mediapipe_error'):
                self.gesture_detector._handle_mediapipe_error(detection_error)
            
            return gesture_info
    
    def _calculate_hand_bbox(self, landmarks, frame_shape):
        """Calculate bounding box for hand landmarks"""
        h, w, _ = frame_shape
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        return (x_min, y_min, x_max, y_max)
    
    def _draw_gesture_overlays(self, frame, gesture_info):
        """Draw gesture visualization overlays on frame"""
        if not gesture_info['hand_present']:
            return
        
        # Draw hand landmarks if present
        if gesture_info['landmarks']:
            self.gesture_detector.mp_draw.draw_landmarks(
                frame, 
                gesture_info['landmarks'], 
                self.gesture_detector.mp_hands.HAND_CONNECTIONS,
                self.gesture_detector.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.gesture_detector.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        
        # Draw hand bounding box
        if gesture_info['hand_bbox']:
            x_min, y_min, x_max, y_max = gesture_info['hand_bbox']
            
            # Color based on detection quality
            if gesture_info['detection_quality'] == 'good':
                bbox_color = (0, 255, 0)  # Green for good detection
            else:
                bbox_color = (0, 255, 255)  # Yellow for poor detection
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), bbox_color, 2)
        
        # Draw finger state indicators
        if gesture_info['finger_states']:
            self._draw_finger_indicators(frame, gesture_info)
    
    def _draw_finger_indicators(self, frame, gesture_info):
        """Draw finger state indicators for debugging"""
        if not gesture_info['finger_states'] or not gesture_info['hand_bbox']:
            return
        
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        x_min, y_min, x_max, y_max = gesture_info['hand_bbox']
        
        # Draw finger state indicators next to hand
        for i, (finger_name, is_extended) in enumerate(zip(finger_names, gesture_info['finger_states'])):
            y_pos = y_min + (i * 25)
            color = (0, 255, 0) if is_extended else (0, 0, 255)
            status = "UP" if is_extended else "DOWN"
            
            cv2.putText(frame, f"{finger_name}: {status}", (x_max + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_realtime_feedback(self, frame, gesture_info):
        """Draw real-time feedback display for gesture detection"""
        h, w, _ = frame.shape
        
        # Feedback panel background
        panel_height = 120
        panel_y = h - panel_height
        cv2.rectangle(frame, (0, panel_y), (w, h), (0, 0, 0), -1)  # Black background
        cv2.rectangle(frame, (0, panel_y), (w, h), (255, 255, 255), 2)  # White border
        
        # Hand detection status
        if gesture_info['hand_present']:
            if gesture_info['detection_quality'] == 'good':
                status_text = "✓ Hand Detected"
                status_color = (0, 255, 0)
            else:
                status_text = "⚠ Hand Detected (Poor Quality)"
                status_color = (0, 255, 255)
        else:
            status_text = "✗ No Hand Detected"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Gesture recognition feedback
        if gesture_info['gesture'] and gesture_info['confidence'] > 0.5:
            gesture_text = f"Gesture: {gesture_info['gesture'].upper()}"
            confidence_text = f"Confidence: {gesture_info['confidence']:.2f}"
            
            # Color based on confidence
            if gesture_info['confidence'] >= self.gesture_confidence_threshold:
                gesture_color = (0, 255, 0)  # Green for high confidence
            else:
                gesture_color = (0, 255, 255)  # Yellow for medium confidence
            
            cv2.putText(frame, gesture_text, (10, panel_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, gesture_color, 2)
            cv2.putText(frame, confidence_text, (10, panel_y + 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        elif gesture_info['hand_present']:
            # Show enhanced unclear gesture feedback
            error_info = getattr(self, 'last_gesture_error_info', {})
            
            if error_info.get('type') == 'unclear_gesture':
                cv2.putText(frame, "Gesture: Unclear", (10, panel_y + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)  # Orange
                
                if error_info.get('show_message', False):
                    cv2.putText(frame, error_info.get('message', 'Try making a clearer gesture'), 
                               (10, panel_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Show confidence level
                    confidence = error_info.get('confidence', 0.0)
                    cv2.putText(frame, f"Confidence: {confidence:.1%}", (10, panel_y + 95), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            elif error_info.get('type') == 'multiple_hands':
                cv2.putText(frame, f"Multiple Hands ({error_info.get('hand_count', 0)})", 
                           (10, panel_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, "Use only one hand", (10, panel_y + 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(frame, "Gesture: Unclear", (10, panel_y + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                cv2.putText(frame, "Try making a clearer gesture", (10, panel_y + 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # Show enhanced no-hand feedback
            error_info = getattr(self, 'last_gesture_error_info', {})
            message = error_info.get('message', 'Show your hand to the camera')
            cv2.putText(frame, message, (10, panel_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Performance indicator
        fps_text = f"Processing: {30}fps"  # Placeholder for actual FPS
        cv2.putText(frame, fps_text, (w - 150, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _process_playing_phase_enhanced(self, frame, gesture_info):
        """Enhanced playing phase processing with improved gesture handling"""
        # Use enhanced gesture detection results
        gesture = gesture_info['gesture']
        confidence = gesture_info['confidence']
        
        if gesture and confidence > self.gesture_confidence_threshold:
            # AI makes its choice using fair randomization
            ai_choice = self.ai_opponent.make_choice()
            
            # Determine winner and end round
            result = self._determine_winner(gesture, ai_choice)
            self.end_round(gesture, ai_choice, result)
        
        # Draw enhanced playing phase UI
        self._draw_playing_screen_enhanced(frame, gesture_info)
    
    def _determine_winner(self, player_choice, ai_choice):
        """Determine winner using standard rock-paper-scissors rules"""
        if player_choice == ai_choice:
            return "tie"
        elif (player_choice == "rock" and ai_choice == "scissors" or
              player_choice == "scissors" and ai_choice == "paper" or
              player_choice == "paper" and ai_choice == "rock"):
            return "win"
        else:
            return "lose"
    
    def _check_token_milestones(self):
        """Check for token milestones and award tokens"""
        current_milestones = self.game_state.total_points // self.milestone_points
        if current_milestones > 0:
            # Calculate new tokens earned (only for new milestones)
            total_tokens_possible = current_milestones * self.tokens_per_milestone
            new_tokens = total_tokens_possible - self.game_state.tokens_earned_this_session
            
            if new_tokens > 0:
                self.game_state.tokens_earned_this_session = total_tokens_possible
                print(f"🪙 Milestone reached! Earned {new_tokens} tokens (Total: {total_tokens_possible})")
                
                # Update user token balance if callback is available
                if hasattr(self, 'token_callback') and self.token_callback:
                    self.token_callback(new_tokens)
    
    def save_high_score(self, user_id: int, game_name: str = "rock_paper_scissors"):
        """Save high score to database if callback is available"""
        if hasattr(self, 'score_callback') and self.score_callback:
            score_data = {
                'user_id': user_id,
                'game_name': game_name,
                'score': self.game_state.total_points,
                'player_wins': self.game_state.player_score,
                'ai_wins': self.game_state.ai_score,
                'total_rounds': self.game_state.current_round - 1
            }
            return self.score_callback(score_data)
        return False
    
    def set_score_callback(self, callback):
        """Set callback function for saving scores to database"""
        self.score_callback = callback
    
    def set_token_callback(self, callback):
        """Set callback function for updating user token balance"""
        self.token_callback = callback
    
    def set_high_score(self, high_score: int):
        """Set the current high score for comparison"""
        self.game_state.high_score = high_score
        print(f"🏆 High score set to: {high_score}")
    
    def is_new_high_score(self) -> bool:
        """Check if current score is a new high score"""
        return self.game_state.total_points > self.game_state.high_score
    
    def update_high_score_if_needed(self):
        """Update high score if current score is higher"""
        if self.is_new_high_score():
            old_high_score = self.game_state.high_score
            self.game_state.high_score = self.game_state.total_points
            print(f"🎉 NEW HIGH SCORE! {old_high_score} → {self.game_state.high_score}")
            return True
        return False
    
    def get_scoring_summary(self) -> dict:
        """Get comprehensive scoring summary for the current game"""
        total_rounds = self.game_state.current_round - 1
        win_rate = (self.game_state.player_score / total_rounds * 100) if total_rounds > 0 else 0
        
        # Calculate tokens earned from milestones
        tokens_from_milestones = (self.game_state.total_points // self.milestone_points) * self.tokens_per_milestone
        
        return {
            'total_points': self.game_state.total_points,
            'player_wins': self.game_state.player_score,
            'ai_wins': self.game_state.ai_score,
            'total_rounds': total_rounds,
            'win_rate': round(win_rate, 1),
            'tokens_earned': tokens_from_milestones,
            'points_breakdown': {
                'wins': self.game_state.player_score * self.points_per_win,
                'ties': (total_rounds - self.game_state.player_score - self.game_state.ai_score) * self.points_per_tie,
                'losses': self.game_state.ai_score * self.points_per_loss
            }
        }
    
    def _draw_waiting_screen(self, frame):
        """Draw waiting screen with round information"""
        h, w, _ = frame.shape
        
        # Game title
        cv2.putText(frame, "Rock Paper Scissors", (w//2 - 150, h//2 - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Round information
        if self.game_state.current_round == 1:
            cv2.putText(frame, "Ready to start!", (w//2 - 80, h//2 - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Ready for Round {self.game_state.current_round}", (w//2 - 120, h//2 - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Instructions
        if self.auto_start_rounds and self.game_state.current_round > 1:
            cv2.putText(frame, "Next round starting soon...", (w//2 - 120, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Show your hand to start!", (w//2 - 120, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(frame, "Press 'S' to start round", (w//2 - 100, h//2 + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Max rounds information
        if self.max_rounds:
            cv2.putText(frame, f"Game: {self.game_state.current_round}/{self.max_rounds} rounds", 
                       (w//2 - 100, h//2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    def _draw_countdown_screen(self, frame):
        """Enhanced countdown screen with animated timer display"""
        h, w, _ = frame.shape
        
        # Calculate countdown display value
        countdown_display = int(self.game_state.countdown_timer) + 1
        countdown_progress = self.game_state.countdown_timer - int(self.game_state.countdown_timer)
        
        # Change color based on countdown value with pulsing effect
        base_colors = {
            3: (0, 255, 0),    # Green for 3
            2: (0, 255, 255),  # Yellow for 2
            1: (0, 0, 255),    # Red for 1
            0: (255, 255, 255) # White for GO
        }
        
        color = base_colors.get(countdown_display, (255, 255, 255))
        
        # Add pulsing effect based on countdown progress
        pulse_factor = 0.7 + 0.3 * abs(np.sin(countdown_progress * np.pi * 4))
        color = tuple(int(c * pulse_factor) for c in color)
        
        # Draw countdown circle background
        center = (w//2, h//2)
        circle_radius = 100
        circle_thickness = 8
        
        # Draw progress circle
        if countdown_display > 0:
            progress_angle = int(360 * (1 - countdown_progress))
            cv2.ellipse(frame, center, (circle_radius, circle_radius), 
                       -90, 0, progress_angle, color, circle_thickness)
        
        # Draw large countdown number with shadow effect
        countdown_text = str(countdown_display) if countdown_display > 0 else "GO!"
        text_size = 4 if countdown_display > 0 else 2.5
        
        # Calculate text position for centering
        text_size_info = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, text_size, 6)
        text_width, text_height = text_size_info[0]
        text_x = w//2 - text_width//2
        text_y = h//2 + text_height//2
        
        # Draw shadow
        cv2.putText(frame, countdown_text, (text_x + 3, text_y + 3), 
                   cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), 6)
        
        # Draw main text
        cv2.putText(frame, countdown_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, text_size, color, 6)
        
        # Draw instruction text with animation
        if countdown_display > 0:
            instruction = "Get ready!"
            instruction_color = (255, 255, 255)
        else:
            instruction = "Make your move!"
            # Blinking effect for "GO!" phase
            blink_factor = 0.5 + 0.5 * abs(np.sin(time.time() * 8))
            instruction_color = tuple(int(255 * blink_factor) for _ in range(3))
        
        instruction_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        instruction_x = w//2 - instruction_size[0][0]//2
        cv2.putText(frame, instruction, (instruction_x, h//2 + 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, instruction_color, 2)
        
        # Draw round information with enhanced styling
        round_text = f"Round {self.game_state.current_round}"
        if self.max_rounds:
            round_text += f" of {self.max_rounds}"
        
        round_size = cv2.getTextSize(round_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        round_x = w//2 - round_size[0][0]//2
        cv2.putText(frame, round_text, (round_x, h//2 - 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw gesture guide icons
        self._draw_gesture_guide_icons(frame, h//2 + 200)
    
    def _draw_playing_screen(self, frame, detected_gesture, confidence):
        """Draw playing phase screen"""
        h, w, _ = frame.shape
        cv2.putText(frame, "Make your choice!", (w//2 - 120, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if detected_gesture:
            cv2.putText(frame, f"Detected: {detected_gesture}", (10, h - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, h - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def _draw_playing_screen_enhanced(self, frame, gesture_info):
        """Enhanced playing phase screen with better gesture feedback"""
        h, w, _ = frame.shape
        
        # Main instruction
        cv2.putText(frame, "Make your choice!", (w//2 - 120, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Enhanced gesture guidance based on error handling
        error_info = getattr(self, 'last_gesture_error_info', {})
        
        if error_info.get('type') == 'no_hand':
            cv2.putText(frame, error_info.get('message', 'Show your hand to the camera'), 
                       (w//2 - 140, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        elif error_info.get('type') == 'multiple_hands':
            cv2.putText(frame, f"Multiple hands detected - use only one", 
                       (w//2 - 180, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        elif error_info.get('type') == 'unclear_gesture':
            if error_info.get('show_message', False):
                cv2.putText(frame, error_info.get('message', 'Make a clearer gesture'), 
                           (w//2 - 160, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                # Show adaptive threshold info
                threshold = getattr(self.gesture_detector, 'adaptive_threshold', 0.8)
                cv2.putText(frame, f"Sensitivity: {threshold:.1%}", (w//2 - 80, 115), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        elif error_info.get('type') == 'poor_quality':
            cv2.putText(frame, "Move your hand to center of screen", (w//2 - 160, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        elif gesture_info['gesture'] and gesture_info['confidence'] >= self.gesture_confidence_threshold:
            cv2.putText(frame, f"Ready to play: {gesture_info['gesture'].upper()}", (w//2 - 120, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif not gesture_info['hand_present']:
            cv2.putText(frame, "Show your hand to the camera", (w//2 - 140, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        elif gesture_info['gesture']:
            cv2.putText(frame, f"Almost there: {gesture_info['gesture']} ({gesture_info['confidence']:.2f})", 
                       (w//2 - 140, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Make a clear rock, paper, or scissors gesture", (w//2 - 180, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _draw_result_screen(self, frame):
        """Enhanced result screen with visual choice representations and animations"""
        h, w, _ = frame.shape
        
        # Draw result panel background
        panel_y = h//2 - 120
        panel_height = 240
        cv2.rectangle(frame, (0, panel_y), (w, panel_y + panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, panel_y), (w, panel_y + panel_height), (255, 255, 255), 2)
        
        # Draw "VS" in center
        vs_text = "VS"
        vs_size = cv2.getTextSize(vs_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
        vs_x = w//2 - vs_size[0][0]//2
        cv2.putText(frame, vs_text, (vs_x, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Player choice visualization (left side)
        player_x = w//4
        self._draw_choice_visualization(frame, player_x, h//2, self.game_state.player_choice, 
                                      "YOU", (0, 255, 0), True)
        
        # AI choice visualization (right side) with timing
        ai_x = 3 * w//4
        ai_display_info = self.ai_opponent.get_choice_display_info()
        
        if ai_display_info['visible']:
            self._draw_choice_visualization(frame, ai_x, h//2, self.game_state.ai_choice, 
                                          "AI", (0, 0, 255), True)
        else:
            # Show thinking animation
            self._draw_thinking_animation(frame, ai_x, h//2, ai_display_info['time_since_choice'])
        
        # Show result only after AI choice is visible
        if ai_display_info['visible']:
            self._draw_result_announcement(frame, h//2 + 80)
        else:
            # Show waiting message
            waiting_text = "Calculating result..."
            waiting_size = cv2.getTextSize(waiting_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            waiting_x = w//2 - waiting_size[0][0]//2
            cv2.putText(frame, waiting_text, (waiting_x, h//2 + 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw score update animation if this is a recent result
        if ai_display_info['visible'] and self.game_state.get_elapsed_time() < 1.0:
            self._draw_score_update_animation(frame)
    
    def _draw_choice_visualization(self, frame, center_x, center_y, choice, label, color, show_choice):
        """Draw visual representation of rock/paper/scissors choice"""
        if not show_choice or not choice:
            return
        
        # Draw choice icon/symbol
        icon_size = 60
        
        if choice == "rock":
            # Draw rock as filled circle
            cv2.circle(frame, (center_x, center_y - 20), icon_size//2, color, -1)
            cv2.circle(frame, (center_x, center_y - 20), icon_size//2, (255, 255, 255), 3)
        elif choice == "paper":
            # Draw paper as rectangle
            rect_size = icon_size
            cv2.rectangle(frame, (center_x - rect_size//2, center_y - 20 - rect_size//2), 
                         (center_x + rect_size//2, center_y - 20 + rect_size//2), color, -1)
            cv2.rectangle(frame, (center_x - rect_size//2, center_y - 20 - rect_size//2), 
                         (center_x + rect_size//2, center_y - 20 + rect_size//2), (255, 255, 255), 3)
        elif choice == "scissors":
            # Draw scissors as two lines
            line_length = icon_size//2
            cv2.line(frame, (center_x - line_length//2, center_y - 20 - line_length//2), 
                    (center_x + line_length//2, center_y - 20 + line_length//2), color, 8)
            cv2.line(frame, (center_x + line_length//2, center_y - 20 - line_length//2), 
                    (center_x - line_length//2, center_y - 20 + line_length//2), color, 8)
        
        # Draw choice text
        choice_text = choice.upper()
        choice_size = cv2.getTextSize(choice_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        choice_x = center_x - choice_size[0][0]//2
        cv2.putText(frame, choice_text, (choice_x, center_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_x = center_x - label_size[0][0]//2
        cv2.putText(frame, label, (label_x, center_y - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _draw_thinking_animation(self, frame, center_x, center_y, elapsed_time):
        """Draw AI thinking animation"""
        # Rotating dots animation
        num_dots = 8
        radius = 40
        dot_radius = 5
        
        for i in range(num_dots):
            angle = (i * 2 * np.pi / num_dots) + (elapsed_time * 4)  # Rotate based on time
            dot_x = int(center_x + radius * np.cos(angle))
            dot_y = int(center_y - 20 + radius * np.sin(angle))
            
            # Fade dots based on position
            alpha = 0.3 + 0.7 * (np.sin(angle) + 1) / 2
            dot_color = tuple(int(128 * alpha) for _ in range(3))
            
            cv2.circle(frame, (dot_x, dot_y), dot_radius, dot_color, -1)
        
        # "AI THINKING" text
        thinking_text = "AI THINKING"
        thinking_size = cv2.getTextSize(thinking_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        thinking_x = center_x - thinking_size[0][0]//2
        cv2.putText(frame, thinking_text, (thinking_x, center_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # Label
        cv2.putText(frame, "AI", (center_x - 10, center_y - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _draw_result_announcement(self, frame, y_pos):
        """Draw animated result announcement"""
        h, w, _ = frame.shape
        
        result_text = self.game_state.last_result.upper()
        
        # Result colors and messages
        if self.game_state.last_result == "win":
            color = (0, 255, 0)
            message = "YOU WIN!"
            emoji = "🎉"
        elif self.game_state.last_result == "tie":
            color = (255, 255, 0)
            message = "IT'S A TIE!"
            emoji = "🤝"
        else:
            color = (0, 0, 255)
            message = "YOU LOSE!"
            emoji = "😔"
        
        # Pulsing effect for result text
        elapsed = self.game_state.get_elapsed_time()
        pulse_factor = 0.8 + 0.2 * abs(np.sin(elapsed * 6))
        text_size = 1.5 * pulse_factor
        
        # Draw result message
        result_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, text_size, 3)
        result_x = w//2 - int(result_size[0][0])//2
        
        # Shadow effect
        cv2.putText(frame, message, (result_x + 2, y_pos + 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), 3)
        cv2.putText(frame, message, (result_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, text_size, color, 3)
    
    def _draw_score_update_animation(self, frame):
        """Draw score update animation for recent results"""
        h, w, _ = frame.shape
        
        # Points earned this round
        points_earned = 0
        if self.game_state.last_result == "win":
            points_earned = self.points_per_win
        elif self.game_state.last_result == "tie":
            points_earned = self.points_per_tie
        
        if points_earned > 0:
            # Animate points floating up
            elapsed = self.game_state.get_elapsed_time()
            y_offset = int(elapsed * 50)  # Float upward
            alpha = max(0, 1 - elapsed)  # Fade out
            
            points_text = f"+{points_earned} points"
            points_color = tuple(int(255 * alpha) for _ in range(3))
            
            cv2.putText(frame, points_text, (w//2 - 60, h//2 + 120 - y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, points_color, 2)
    
    def _draw_ui(self, frame):
        """Enhanced game UI elements with better layout and styling"""
        h, w, _ = frame.shape
        
        # Draw main UI panel background (top-left)
        panel_width = 250
        panel_height = 200
        cv2.rectangle(frame, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (panel_width, panel_height), (255, 255, 255), 2)
        
        # Game title
        cv2.putText(frame, "ROCK PAPER SCISSORS", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Round display with progress
        round_text = f"Round: {self.game_state.current_round}"
        if self.max_rounds:
            round_text += f"/{self.max_rounds}"
        cv2.putText(frame, round_text, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Score display with better formatting
        cv2.putText(frame, "SCORE", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Player: {self.game_state.player_score}", (10, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"AI: {self.game_state.ai_score}", (130, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Points display with high score indicator
        points_color = (0, 255, 255) if self.is_new_high_score() else (255, 255, 0)
        cv2.putText(frame, f"Points: {self.game_state.total_points}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, points_color, 2)
        
        # High score display
        if self.game_state.high_score > 0:
            cv2.putText(frame, f"Best: {self.game_state.high_score}", (10, 145), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 215, 0), 2)
        
        # Tokens earned this session
        if self.game_state.tokens_earned_this_session > 0:
            cv2.putText(frame, f"Tokens: +{self.game_state.tokens_earned_this_session}", (10, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # New high score indicator with animation
        if self.is_new_high_score() and self.game_state.total_points > 0:
            # Blinking effect
            blink_factor = 0.5 + 0.5 * abs(np.sin(time.time() * 4))
            highlight_color = tuple(int(255 * blink_factor) for _ in range(3))
            cv2.putText(frame, "NEW HIGH SCORE!", (w//2 - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, highlight_color, 2)
        
        # Game phase indicator (top-right)
        phase_text = self.game_state.game_phase.upper().replace("_", " ")
        phase_size = cv2.getTextSize(phase_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        phase_x = w - phase_size[0][0] - 10
        cv2.putText(frame, phase_text, (phase_x, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw gesture guide icons (bottom)
        self._draw_gesture_guide_icons(frame, h - 40)
    
    def _draw_gesture_guide_icons(self, frame, y_pos):
        """Draw gesture guide icons at the bottom of the screen"""
        h, w, _ = frame.shape
        
        # Background for gesture guide
        guide_height = 60
        cv2.rectangle(frame, (0, y_pos - 20), (w, y_pos + guide_height - 20), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, y_pos - 20), (w, y_pos + guide_height - 20), (100, 100, 100), 1)
        
        # Draw three gesture icons
        icon_spacing = w // 4
        icon_y = y_pos + 10
        
        gestures = [
            ("ROCK", "Closed Fist", (0, 0, 255)),
            ("PAPER", "Open Hand", (0, 255, 0)),
            ("SCISSORS", "Peace Sign", (255, 0, 0))
        ]
        
        for i, (gesture, description, color) in enumerate(gestures):
            icon_x = icon_spacing * (i + 1)
            
            # Draw simple icon representation
            if gesture == "ROCK":
                cv2.circle(frame, (icon_x, icon_y), 15, color, -1)
            elif gesture == "PAPER":
                cv2.rectangle(frame, (icon_x - 15, icon_y - 15), (icon_x + 15, icon_y + 15), color, -1)
            elif gesture == "SCISSORS":
                cv2.line(frame, (icon_x - 10, icon_y - 10), (icon_x + 10, icon_y + 10), color, 4)
                cv2.line(frame, (icon_x + 10, icon_y - 10), (icon_x - 10, icon_y + 10), color, 4)
            
            # Draw gesture name
            gesture_size = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            gesture_x = icon_x - gesture_size[0][0] // 2
            cv2.putText(frame, gesture, (gesture_x, icon_y + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'gesture_detector'):
            del self.gesture_detector
    
    def _create_empty_gesture_info(self) -> dict:
        """Create empty gesture info structure for error recovery"""
        return {
            'gesture': None,
            'confidence': 0.0,
            'landmarks': None,
            'hand_present': False,
            'hand_bbox': None,
            'finger_states': None,
            'detection_quality': 'error'
        }
    
    def _draw_recovery_screen(self, frame, message: str):
        """Draw recovery screen during error handling"""
        h, w, _ = frame.shape
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Recovery message
        cv2.putText(frame, "RECOVERY MODE", (w//2 - 100, h//2 - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, message, (w//2 - len(message)*6, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Please wait...", (w//2 - 60, h//2 + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def _draw_error_screen(self, frame, error_message: str):
        """Draw error screen for non-critical errors"""
        h, w, _ = frame.shape
        
        # Error panel
        panel_height = 100
        panel_y = h - panel_height - 20
        cv2.rectangle(frame, (10, panel_y), (w - 10, panel_y + panel_height), (0, 0, 100), -1)
        cv2.rectangle(frame, (10, panel_y), (w - 10, panel_y + panel_height), (0, 0, 255), 2)
        
        # Error message
        cv2.putText(frame, "⚠ ERROR", (20, panel_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, error_message, (20, panel_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Game will continue with reduced functionality", (20, panel_y + 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _draw_critical_error_screen(self, frame, error_message: str):
        """Draw critical error screen for severe failures"""
        try:
            h, w, _ = frame.shape
            
            # Full screen error overlay
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 50), -1)
            
            # Critical error message
            cv2.putText(frame, "CRITICAL ERROR", (w//2 - 120, h//2 - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(frame, error_message[:60], (w//2 - 200, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Please restart the game", (w//2 - 120, h//2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        except:
            # If even this fails, just fill with error color
            try:
                frame[:] = (0, 0, 100)  # Dark red
            except:
                pass
    
    def _draw_minimal_ui(self, frame):
        """Draw minimal UI when full UI fails"""
        try:
            h, w, _ = frame.shape
            
            # Basic score display
            score_text = f"Score: {self.game_state.total_points}"
            cv2.putText(frame, score_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Basic round display
            round_text = f"Round: {self.game_state.current_round}"
            cv2.putText(frame, round_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Basic phase display
            phase_text = f"Phase: {self.game_state.game_phase}"
            cv2.putText(frame, phase_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except:
            pass
    
    def _draw_error_messages(self, frame):
        """Draw current error messages on frame"""
        if self.error_message and time.time() - self.error_display_time < 5.0:
            try:
                h, w, _ = frame.shape
                
                # Error message background
                msg_width = len(self.error_message) * 8
                cv2.rectangle(frame, (w - msg_width - 20, 10), (w - 10, 50), (0, 0, 100), -1)
                cv2.rectangle(frame, (w - msg_width - 20, 10), (w - 10, 50), (0, 0, 255), 1)
                
                # Error message text
                cv2.putText(frame, self.error_message, (w - msg_width - 15, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except:
                pass