import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
from threading import Thread, Lock
import base64
import json

class FruitNinjaGame:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Game state
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
        
        # Thread safety
        self.lock = Lock()
        
        # Camera
        self.cap = None
        self.frame = None
        self.processed_frame = None
        
    def start_game(self):
        """Initialize and start the game - AUTO-DETECT CAMERA"""
        print("🎮 Starting Fruit Ninja game...")

        # COMPLETE cleanup first
        self._force_cleanup()

        with self.lock:
            self.is_running = True
            self.game_over = False
            self.stop_requested = False
            self.game_over_displayed = False
            self.score = 0
            self.lives = 3
            self.fruits = []
            self.particles = []
            self.hand_positions = []
            self.last_hand_pos = None
            self.game_over_time = None

        # 🔍 Try to find first working camera index (0–5)
        print("🔍 Scanning for available cameras...")
        self.cap = None
        for cam_index in range(5):  # Try indexes 0 to 4
            temp_cap = cv2.VideoCapture(cam_index)
            if temp_cap.isOpened():
                self.cap = temp_cap
                print(f"📹 Camera found at index {cam_index}")
                break
            else:
                temp_cap.release()

        if self.cap is None or not self.cap.isOpened():
            print("❌ No available camera found.")
            self.is_running = False
            return {"success": False, "message": "No available camera found."}

        # Set camera properties
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print("📸 Camera properties set.")
        except:
            print("⚠️ Could not set camera properties, using defaults.")

        print("🍎 Fruit Ninja game started successfully!")
        return {"success": True, "message": "Game started with auto-detected camera."}

    
    def _force_cleanup(self):
        """BULLETPROOF resource cleanup"""
        print("🧹 FORCE cleaning up game resources...")
        
        # Release camera with multiple attempts
        if self.cap:
            for attempt in range(3):
                try:
                    self.cap.release()
                    print(f"📹 Camera released successfully (attempt {attempt + 1})")
                    break
                except Exception as e:
                    print(f"⚠️ Camera release attempt {attempt + 1} failed: {e}")
                    time.sleep(0.1)
                
            self.cap = None
        
        # Force OpenCV cleanup
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        # Clear all game state
        self.fruits = []
        self.particles = []
        self.hand_positions = []
        self.last_hand_pos = None
        self.game_over_time = None
        self.processed_frame = None
        self.frame = None
        
        print("🧹 FORCE cleanup completed")
    
    def stop_game(self):
        """Stop the game and cleanup - BULLETPROOF"""
        print("🛑 Stopping Fruit Ninja game...")
        
        # Set stop flags IMMEDIATELY
        with self.lock:
            self.stop_requested = True
            self.is_running = False
            self.game_over = False
            self.game_over_displayed = False
        
        # Force cleanup
        final_score = self.score
        self._force_cleanup()
        
        self.processed_frame =None
        self.frame=None

        print(f"🛑 Fruit Ninja game stopped! Final Score: {final_score}")
        return {"success": True, "score": final_score, "message": "Game stopped!"}
    
    def get_frame(self):
        """Capture and process a single frame - BULLETPROOF"""
        # Check stop conditions
        if self.stop_requested or not self.cap:
            return None
    
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ Failed to read frame from camera")
                return None
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            self.frame = frame.copy()
            
            # Process the frame
            self.process_frame(frame)

            if self.processed_frame is None:
                print("⚠️ No processed frame available")
                return None
            
            # Convert to base64 for web transmission
            _, buffer = cv2.imencode('.jpg', self.processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return frame_base64
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return None
    
    def process_frame(self, frame):
        """Process frame for hand detection and game logic - BULLETPROOF"""
        if self.stop_requested:
            return

        height, width, _ = frame.shape

        try:
            # Only process game logic if not game over and not stopping
            if not self.game_over and not self.stop_requested and self.is_running:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Hand detection
                results = self.hands.process(rgb_frame)

                # Update game logic
                self.update_fruits(width, height)
                self.update_particles()

                # Draw game elements
                self.draw_fruits(frame)
                self.draw_particles(frame)

                # Process hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        index_tip = hand_landmarks.landmark[8]
                        finger_x = int(index_tip.x * width)
                        finger_y = int(index_tip.y * height)
                        cv2.circle(frame, (finger_x, finger_y), 10, (0, 255, 255), -1)

                        self.track_hand_movement(finger_x, finger_y)

            # Always draw UI
            self.draw_ui(frame)

            # Handle Game Over overlay and auto-stop
            if self.game_over and not self.stop_requested:
                self.draw_game_over_overlay(frame)

                if self.game_over_time and (time.time() - self.game_over_time > 3):
                    print("🛑 Auto-stopping game after Game Over delay...")
                    self.stop_game()
                    self.processed_frame = None
                    return  # Exit immediately to avoid frame assignment after cleanup

            # Final safety check before assigning
            if self.stop_requested or not self.cap:
                return

            self.processed_frame = frame

        except Exception as e:
            print(f"❌ Error in process_frame: {e}")

    
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
        if self.game_over or self.stop_requested:
            return
            
        current_pos = (x, y)
        
        # Add to position history
        self.hand_positions.append(current_pos)
        if len(self.hand_positions) > 5:
            self.hand_positions.pop(0)
        
        # Check for swipe if we have enough positions
        if len(self.hand_positions) >= 3:
            start_pos = self.hand_positions[0]
            end_pos = self.hand_positions[-1]
        
            distance = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
            if distance > self.swipe_threshold:
                self.detect_fruit_collision(current_pos)
                self.hand_positions = []
    
    def detect_fruit_collision(self, hand_pos):
        """Check if hand position collides with any fruits"""
        if self.game_over or self.stop_requested:
            return
            
        with self.lock:
            for i, fruit in enumerate(self.fruits):
                fruit_center = (int(fruit['x']), int(fruit['y']))
                distance = math.sqrt((hand_pos[0] - fruit_center[0])**2 + (hand_pos[1] - fruit_center[1])**2)
                
                if distance < fruit['size']:
                    self.slice_fruit(i, fruit)
                    break
    
    def slice_fruit(self, fruit_index, fruit):
        """Handle fruit slicing"""
        if self.game_over or self.stop_requested:
            return
            
        # Remove fruit
        self.fruits.pop(fruit_index)
        
        # Add score
        self.score += 10
        
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
        if not self.is_running or self.game_over or self.stop_requested:
            return
        
        with self.lock:
            # Spawn new fruits
            if len(self.fruits) < self.max_fruits and random.random() < self.fruit_spawn_rate:
                self.spawn_fruit(width, height)
            
            # Update existing fruits
            fruits_to_remove = []
            for i, fruit in enumerate(self.fruits):
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
                            
        
            # Remove fruits that are off screen
            for i in reversed(fruits_to_remove):
                self.fruits.pop(i)
    
    def trigger_game_over(self):
        """Trigger game over state"""
        print(f"🎮 GAME OVER TRIGGERED! Final Score: {self.score}")
        with self.lock:
            self.game_over = True
            self.game_over_time = time.time()
            self.game_over_displayed = False  # Reset display flag
        
        print("💀 GAME OVER - OVERLAY WILL BE DISPLAYED!")
    
    def spawn_fruit(self, width, height):
        """Spawn a new fruit"""
        if self.game_over or self.stop_requested:
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
        self.fruits.append(fruit)
    
    def update_particles(self):
        """Update particle effects"""
        if self.game_over or self.stop_requested:
            return
            
        particles_to_remove = []
        for i, particle in enumerate(self.particles):
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['vy'] += 0.2  # Gravity
            particle['life'] -= 1
            
            if particle['life'] <= 0:
                particles_to_remove.append(i)
        
        for i in reversed(particles_to_remove):
            self.particles.pop(i)
    
    def draw_fruits(self, frame):
        """Draw all fruits on the frame"""
        if self.game_over or self.stop_requested:
            return
            
        for fruit in self.fruits:
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
        for particle in self.particles:
            alpha = particle['life'] / particle['max_life']
            center = (int(particle['x']), int(particle['y']))
            color = tuple(int(c * alpha) for c in particle['color'])
            cv2.circle(frame, center, 3, color, -1)
    
    def draw_ui(self, frame):
        """Draw UI elements (score, lives, etc.)"""
        height, width = frame.shape[:2]
        
        if not self.game_over and not self.stop_requested:
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
        """Get current game state"""
        with self.lock:
            return {
                'is_running': self.is_running,
                'game_over': self.game_over,
                'score': self.score,
                'lives': self.lives,
                'fruits_count': len(self.fruits),
                'particles_count': len(self.particles),
                'game_over_displayed': self.game_over_displayed
            }

class TRexRunGame:
    def __init__(self):

        self.trex_img = cv2.imread("static/images/trex.png", cv2.IMREAD_UNCHANGED)
        self.cactus_img = cv2.imread("static/images/cactus.png", cv2.IMREAD_UNCHANGED)
        
        if self.trex_img is None:
            print("❌ Could not load trex.png")
        if self.cactus_img is None:
            print("❌ Could not load cactus.png")
            
        # Game state
        self.is_running = False
        self.score = 0
        self.lives = 1
        self.trex_y = 0
        self.jump_velocity = 0
        self.gravity = 1
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
        self.ground_y = 300  # ground level
        self.jump_power = -12
        self.trex_x = 50
        self.last_spawn_time = 0
        self.spawn_delay = 2
        self.obstacle_speed = 6

    def start_game(self):
        self.is_running = True
        self.score = 0
        self.trex_y = self.ground_y
        self.jump_velocity = 0
        self.is_jumping = False
        self.obstacles = []
        self.cap = cv2.VideoCapture(0)
        return {"success": True, "message": "T-Rex Run game started!"}

    def stop_game(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        return {"success": True, "score": self.score}

    def get_frame(self):
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        self.detect_hand(frame)
        self.update_game_logic(width, height)
        self.draw_elements(frame)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64

    def detect_hand(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        self.hand_detected = bool(results.multi_hand_landmarks)

    def update_game_logic(self, width, height):
        if not self.is_running:
            return

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

        # Spawn obstacles
        current_time = time.time()
        if current_time - self.last_spawn_time > self.spawn_delay:
            self.obstacles.append({'x': width + 50})
            self.last_spawn_time = current_time

        # Move obstacles
        for obs in self.obstacles:
            obs['x'] -= self.obstacle_speed

        # Remove off-screen obstacles
        self.obstacles = [obs for obs in self.obstacles if obs['x'] > -50]

        # Collision detection
        for obs in self.obstacles:
            if obs['x'] < self.trex_x + 30 and obs['x'] > self.trex_x:
                if self.trex_y + 50 >= self.ground_y:
                    self.is_running = False
                    self.lives = 0

        if self.is_running:
            self.score += 1

    def draw_elements(self, frame):
        # Draw ground
        cv2.line(frame, (0, self.ground_y + 50), (frame.shape[1], self.ground_y + 50), (100, 100, 100), 2)

        # Draw T-Rex
        cv2.rectangle(frame, (self.trex_x, int(self.trex_y)), (self.trex_x + 30, int(self.trex_y) + 50), (0, 255, 0), -1)

        # Draw obstacles
        for obs in self.obstacles:
            cv2.rectangle(frame, (obs['x'], self.ground_y + 20), (obs['x'] + 20, self.ground_y + 50), (0, 0, 255), -1)

        # Draw Score
        cv2.putText(frame, f"Score: {self.score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Game Over
        if not self.is_running:
            cv2.putText(frame, "GAME OVER", (frame.shape[1]//2 - 150, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    def get_game_state(self):
        return {
            "is_running": self.is_running,
            "score": self.score,
            "lives": self.lives
        }


class RockPaperScissorsGame:
    def __init__(self):
        self.timer_duration = 3  # seconds
        self.timer_start = None
        self.countdown_text = ""
        self.round_active = False
        self.is_running = False
        self.user_move = None
        self.computer_move = None
        self.result = None
        self.cap = None
        self.last_play_time = 0
        self.play_interval = 3  # seconds

        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.lock = Lock()

    def start_game(self):
        self.is_running = True
        self.user_move = ""
        self.computer_move = ""
        self.result = ""
        self.last_play_time = time.time()
        self.cap = cv2.VideoCapture(0)
        return {"success": True, "message": "Rock Paper Scissors game started!"}

    def stop_game(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        return {"success": True, "result": self.result}

    def get_frame(self):
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        self.detect_gesture_and_play(frame)
        self.draw_overlay(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def detect_gesture_and_play(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        now = time.time()

        # Start new round if enough time has passed
        if not self.round_active and (now - self.last_play_time) > 5:  # 5 sec gap between rounds
            self.timer_start = now
            self.round_active = True
            self.countdown_text = "Get Ready..."

        if self.round_active:
            elapsed = now - self.timer_start
            remaining = int(self.timer_duration - elapsed)

        if remaining > 0:
            self.countdown_text = str(remaining + 1)  # Show 3,2,1
        elif 0 <= elapsed < self.timer_duration + 1:
            self.countdown_text = "GO!"
            # Detect gesture during this moment
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    fingers = []
                    lm = hand_landmarks.landmark

                    # Thumb
                    fingers.append(1 if lm[4].x < lm[3].x else 0)

                    # Other fingers
                    for tip in [8, 12, 16, 20]:
                        fingers.append(1 if lm[tip].y < lm[tip - 2].y else 0)

                    self.user_move = self.get_gesture_from_fingers(fingers)

            self.computer_move = random.choice(["Rock", "Paper", "Scissors"])
            self.result = self.get_winner(self.user_move, self.computer_move)

        else:
            # Reset round
            self.last_play_time = now
            self.round_active = False
            self.countdown_text = ""

    def get_gesture_from_fingers(self, fingers):
        if fingers == [0, 0, 0, 0, 0]:
            return "Rock"
        elif fingers == [1, 1, 1, 1, 1] or fingers == [0, 1, 1, 1, 1]:
            return "Paper"
        elif fingers == [0, 1, 1, 0, 0]:
            return "Scissors"
        return "Unknown"

    def get_winner(self, user, comp):
        if user == comp:
            return "Draw"
        if (user == "Rock" and comp == "Scissors") or \
           (user == "Paper" and comp == "Rock") or \
           (user == "Scissors" and comp == "Paper"):
            return "You Win"
        return "Computer Wins"

    def draw_overlay(self, frame):
        h, w, _ = frame.shape
        cv2.putText(frame, f"You: {self.user_move}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"Computer: {self.computer_move}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        cv2.putText(frame, f"Result: {self.result}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Countdown in center
        if self.countdown_text:
            cv2.putText(frame, self.countdown_text, (frame.shape[1]//2 - 50, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 4)

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
