"""
Fruit Ninja Game Engine with OpenCV and MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math

class FruitNinjaEngine:
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
        self.lives = 3
        self.game_over = False
        self.fruits = []
        self.slashes = []
        self.last_hand_pos = None
        self.slash_trail = []
        
        # Game settings
        self.fruit_spawn_rate = 0.02
        self.fruit_types = ['apple', 'banana', 'orange', 'watermelon', 'bomb']
        self.fruit_colors = {
            'apple': (0, 255, 0),
            'banana': (0, 255, 255),
            'orange': (0, 165, 255),
            'watermelon': (0, 128, 0),
            'bomb': (0, 0, 255)
        }
        
        print("🍎 Fruit Ninja Engine initialized!")
    
    def reset_game(self):
        """Reset game state"""
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.fruits = []
        self.slashes = []
        self.last_hand_pos = None
        self.slash_trail = []
        print("🔄 Fruit Ninja game reset")
    
    def spawn_fruit(self, frame_width, frame_height):
        """Spawn a new fruit"""
        if random.random() < self.fruit_spawn_rate:
            fruit_type = random.choice(self.fruit_types)
            x = random.randint(50, frame_width - 50)
            y = frame_height - 50
            
            # Random velocity
            vx = random.randint(-3, 3)
            vy = random.randint(-15, -8)
            
            fruit = {
                'type': fruit_type,
                'x': x,
                'y': y,
                'vx': vx,
                'vy': vy,
                'radius': 30,
                'sliced': False
            }
            self.fruits.append(fruit)
    
    def update_fruits(self, frame_height):
        """Update fruit positions"""
        for fruit in self.fruits[:]:
            # Apply gravity
            fruit['vy'] += 0.5
            fruit['x'] += fruit['vx']
            fruit['y'] += fruit['vy']
            
            # Remove fruits that fall off screen
            if fruit['y'] > frame_height + 50:
                if not fruit['sliced'] and fruit['type'] != 'bomb':
                    self.lives -= 1
                    if self.lives <= 0:
                        self.game_over = True
                self.fruits.remove(fruit)
    
    def detect_hand_gesture(self, frame):
        """Detect hand gestures using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_pos = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get index finger tip position
                index_tip = hand_landmarks.landmark[8]
                h, w, _ = frame.shape
                hand_pos = (int(index_tip.x * w), int(index_tip.y * h))
                
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Draw finger tip
                cv2.circle(frame, hand_pos, 10, (255, 0, 255), -1)
        
        return hand_pos
    
    def detect_slashes(self, current_pos):
        """Detect slashing gestures"""
        if self.last_hand_pos and current_pos:
            # Calculate distance moved
            distance = math.sqrt(
                (current_pos[0] - self.last_hand_pos[0])**2 + 
                (current_pos[1] - self.last_hand_pos[1])**2
            )
            
            # If hand moved fast enough, it's a slash
            if distance > 30:
                self.slash_trail.append(current_pos)
                if len(self.slash_trail) > 10:
                    self.slash_trail.pop(0)
                
                # Check for fruit collisions
                self.check_fruit_collisions(current_pos)
        
        self.last_hand_pos = current_pos
    
    def check_fruit_collisions(self, slash_pos):
        """Check if slash hits any fruits"""
        for fruit in self.fruits:
            if not fruit['sliced']:
                distance = math.sqrt(
                    (slash_pos[0] - fruit['x'])**2 + 
                    (slash_pos[1] - fruit['y'])**2
                )
                
                if distance < fruit['radius']:
                    fruit['sliced'] = True
                    
                    if fruit['type'] == 'bomb':
                        self.lives -= 1
                        if self.lives <= 0:
                            self.game_over = True
                    else:
                        self.score += 10
                    
                    print(f"🎯 {fruit['type']} sliced! Score: {self.score}")
    
    def draw_fruits(self, frame):
        """Draw fruits on frame"""
        for fruit in self.fruits:
            if not fruit['sliced']:
                color = self.fruit_colors[fruit['type']]
                cv2.circle(frame, (int(fruit['x']), int(fruit['y'])), fruit['radius'], color, -1)
                cv2.circle(frame, (int(fruit['x']), int(fruit['y'])), fruit['radius'], (255, 255, 255), 2)
                
                # Draw fruit type text
                cv2.putText(frame, fruit['type'], 
                           (int(fruit['x'] - 20), int(fruit['y'] + 5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_slash_trail(self, frame):
        """Draw slash trail"""
        if len(self.slash_trail) > 1:
            for i in range(1, len(self.slash_trail)):
                cv2.line(frame, self.slash_trail[i-1], self.slash_trail[i], (255, 255, 0), 5)
    
    def draw_ui(self, frame):
        """Draw game UI"""
        h, w, _ = frame.shape
        
        # Score
        cv2.putText(frame, f"Score: {self.score}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Lives
        cv2.putText(frame, f"Lives: {self.lives}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Slice fruits with your finger!", (10, h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, "Avoid bombs!", (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
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
            
            # Spawn fruits
            self.spawn_fruit(w, h)
            
            # Update fruit positions
            self.update_fruits(h)
            
            # Detect hand gestures
            hand_pos = self.detect_hand_gesture(frame)
            
            # Detect slashes
            if hand_pos:
                self.detect_slashes(hand_pos)
        
        # Draw everything
        self.draw_fruits(frame)
        self.draw_slash_trail(frame)
        self.draw_ui(frame)
        
        return frame
    
    def get_game_state(self):
        """Get current game state"""
        return {
            'score': self.score,
            'lives': self.lives,
            'game_over': self.game_over,
            'fruits_count': len(self.fruits)
        }