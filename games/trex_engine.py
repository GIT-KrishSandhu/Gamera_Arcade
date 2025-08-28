"""
T-Rex Run Game Engine with OpenCV and MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import time

class TRexEngine:
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
        self.score = 0
        self.game_over = False
        self.game_speed = 5
        
        # T-Rex properties
        self.trex_x = 100
        self.trex_y = 300
        self.trex_width = 40
        self.trex_height = 60
        self.trex_ground_y = 300
        self.trex_jump_velocity = 0
        self.trex_is_jumping = False
        self.gravity = 1
        self.jump_strength = -15
        
        # Obstacles
        self.obstacles = []
        self.obstacle_spawn_rate = 0.01
        
        # Ground
        self.ground_y = 350
        
        # Jump detection
        self.hand_raised = False
        self.last_jump_time = 0
        self.jump_cooldown = 0.5  # seconds
        
        print("🦕 T-Rex Run Engine initialized!")
    
    def reset_game(self):
        """Reset game state"""
        self.score = 0
        self.game_over = False
        self.game_speed = 5
        self.trex_y = self.trex_ground_y
        self.trex_jump_velocity = 0
        self.trex_is_jumping = False
        self.obstacles = []
        self.hand_raised = False
        self.last_jump_time = 0
        print("🔄 T-Rex Run game reset")
    
    def detect_jump_gesture(self, frame):
        """Detect hand-up gesture for jumping"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        jump_detected = False
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand landmarks
                landmarks = []
                h, w, _ = frame.shape
                
                for lm in hand_landmarks.landmark:
                    landmarks.append([int(lm.x * w), int(lm.y * h)])
                
                # Check if hand is raised (wrist below middle finger)
                wrist_y = landmarks[0][1]
                middle_finger_y = landmarks[12][1]
                
                if wrist_y > middle_finger_y + 50:  # Hand is raised
                    jump_detected = True
                    
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Visual feedback
                    cv2.putText(frame, "JUMP!", (landmarks[0][0] - 30, landmarks[0][1] - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return jump_detected
    
    def update_trex(self):
        """Update T-Rex position and physics"""
        if self.trex_is_jumping:
            self.trex_y += self.trex_jump_velocity
            self.trex_jump_velocity += self.gravity
            
            # Land on ground
            if self.trex_y >= self.trex_ground_y:
                self.trex_y = self.trex_ground_y
                self.trex_is_jumping = False
                self.trex_jump_velocity = 0
    
    def jump(self):
        """Make T-Rex jump"""
        current_time = time.time()
        if not self.trex_is_jumping and (current_time - self.last_jump_time) > self.jump_cooldown:
            self.trex_is_jumping = True
            self.trex_jump_velocity = self.jump_strength
            self.last_jump_time = current_time
            print("🦕 T-Rex jumped!")
    
    def spawn_obstacle(self, frame_width):
        """Spawn obstacles"""
        if random.random() < self.obstacle_spawn_rate:
            obstacle = {
                'x': frame_width,
                'y': self.ground_y - 30,
                'width': 20,
                'height': 30,
                'type': 'cactus'
            }
            self.obstacles.append(obstacle)
    
    def update_obstacles(self):
        """Update obstacle positions"""
        for obstacle in self.obstacles[:]:
            obstacle['x'] -= self.game_speed
            
            # Remove obstacles that are off screen
            if obstacle['x'] + obstacle['width'] < 0:
                self.obstacles.remove(obstacle)
                self.score += 10
                
                # Increase game speed gradually
                if self.score % 100 == 0:
                    self.game_speed += 0.5
    
    def check_collisions(self):
        """Check for collisions between T-Rex and obstacles"""
        trex_rect = {
            'x': self.trex_x,
            'y': self.trex_y,
            'width': self.trex_width,
            'height': self.trex_height
        }
        
        for obstacle in self.obstacles:
            # Simple rectangle collision detection
            if (trex_rect['x'] < obstacle['x'] + obstacle['width'] and
                trex_rect['x'] + trex_rect['width'] > obstacle['x'] and
                trex_rect['y'] < obstacle['y'] + obstacle['height'] and
                trex_rect['y'] + trex_rect['height'] > obstacle['y']):
                
                self.game_over = True
                print(f"💥 Game Over! Final Score: {self.score}")
    
    def draw_ground(self, frame):
        """Draw ground line"""
        h, w, _ = frame.shape
        cv2.line(frame, (0, self.ground_y), (w, self.ground_y), (255, 255, 255), 2)
    
    def draw_trex(self, frame):
        """Draw T-Rex"""
        # Simple T-Rex representation
        color = (0, 255, 0) if not self.game_over else (0, 0, 255)
        
        # Body
        cv2.rectangle(frame, 
                     (self.trex_x, int(self.trex_y)), 
                     (self.trex_x + self.trex_width, int(self.trex_y) + self.trex_height), 
                     color, -1)
        
        # Head
        cv2.circle(frame, (self.trex_x + 20, int(self.trex_y) + 10), 15, color, -1)
        
        # Eye
        cv2.circle(frame, (self.trex_x + 25, int(self.trex_y) + 5), 3, (255, 255, 255), -1)
        
        # Legs (simple lines)
        if not self.trex_is_jumping:
            cv2.line(frame, (self.trex_x + 10, int(self.trex_y) + self.trex_height), 
                    (self.trex_x + 10, int(self.trex_y) + self.trex_height + 10), color, 3)
            cv2.line(frame, (self.trex_x + 30, int(self.trex_y) + self.trex_height), 
                    (self.trex_x + 30, int(self.trex_y) + self.trex_height + 10), color, 3)
    
    def draw_obstacles(self, frame):
        """Draw obstacles"""
        for obstacle in self.obstacles:
            if obstacle['type'] == 'cactus':
                # Draw cactus
                cv2.rectangle(frame, 
                             (int(obstacle['x']), int(obstacle['y'])), 
                             (int(obstacle['x']) + obstacle['width'], int(obstacle['y']) + obstacle['height']), 
                             (0, 128, 0), -1)
                
                # Cactus spikes
                cv2.line(frame, (int(obstacle['x']) + 5, int(obstacle['y']) + 5), 
                        (int(obstacle['x']) - 5, int(obstacle['y']) + 10), (0, 128, 0), 2)
                cv2.line(frame, (int(obstacle['x']) + 15, int(obstacle['y']) + 8), 
                        (int(obstacle['x']) + 25, int(obstacle['y']) + 12), (0, 128, 0), 2)
    
    def draw_ui(self, frame):
        """Draw game UI"""
        h, w, _ = frame.shape
        
        # Score
        cv2.putText(frame, f"Score: {self.score}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Speed
        cv2.putText(frame, f"Speed: {self.game_speed:.1f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Raise your hand to jump!", (10, h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Game over screen
        if self.game_over:
            cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 0, 0), -1)
            cv2.putText(frame, "GAME OVER!", (w//4 + 20, h//2 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame, f"Final Score: {self.score}", (w//4 + 20, h//2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def process_frame(self, frame):
        """Process a single frame"""
        if not self.game_over:
            h, w, _ = frame.shape
            
            # Detect jump gesture
            jump_detected = self.detect_jump_gesture(frame)
            
            # Handle jumping
            if jump_detected and not self.hand_raised:
                self.jump()
            self.hand_raised = jump_detected
            
            # Update game objects
            self.update_trex()
            self.spawn_obstacle(w)
            self.update_obstacles()
            self.check_collisions()
        
        # Draw everything
        self.draw_ground(frame)
        self.draw_trex(frame)
        self.draw_obstacles(frame)
        self.draw_ui(frame)
        
        return frame
    
    def get_game_state(self):
        """Get current game state"""
        return {
            'score': self.score,
            'game_over': self.game_over,
            'speed': self.game_speed,
            'is_jumping': self.trex_is_jumping,
            'obstacles_count': len(self.obstacles)
        }