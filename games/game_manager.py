"""
Game Manager for Gamera Arcade
Handles game initialization and state management with OpenCV integration
"""

import cv2
import base64
import numpy as np
from threading import Lock
from .fruit_ninja_engine import FruitNinjaEngine
from .trex_engine import TRexEngine
from .rock_paper_scissors_engine import RockPaperScissorsEngine

class GameManager:
    def __init__(self):
        self.current_game_id = None
        self.current_game_engine = None
        self.camera = None
        self.lock = Lock()
        self.is_running = False
        
        # Game engines
        self.game_engines = {
            'fruit_ninja': FruitNinjaEngine,
            'trex_run': TRexEngine,
            'rock_paper_scissors': RockPaperScissorsEngine,
        }
        
    def start_game(self, game_id):
        """Start a specific game"""
        try:
            with self.lock:
                print(f"🎮 Starting game: {game_id}")
                
                # Stop current game if running
                if self.current_game_engine:
                    self.stop_current_game()
                
                # Initialize camera if not already done
                if self.camera is None:
                    self.camera = cv2.VideoCapture(0)
                    if not self.camera.isOpened():
                        return {'success': False, 'message': 'Camera not available'}
                
                # Initialize game engine
                if game_id in self.game_engines and self.game_engines[game_id]:
                    self.current_game_engine = self.game_engines[game_id]()
                    self.current_game_id = game_id
                    self.is_running = True
                    print(f"✅ Game engine {game_id} initialized successfully")
                    return {'success': True, 'message': f'Game {game_id} started'}
                else:
                    return {'success': False, 'message': f'Game {game_id} not implemented yet'}
                    
        except Exception as e:
            print(f"❌ Error starting game: {e}")
            return {'success': False, 'message': str(e)}
    
    def stop_current_game(self):
        """Stop the current game"""
        try:
            with self.lock:
                if self.current_game_engine:
                    print(f"🛑 Stopping game: {self.current_game_id}")
                    self.current_game_engine = None
                    self.current_game_id = None
                    self.is_running = False
                    return {'success': True, 'message': 'Game stopped'}
                return {'success': False, 'message': 'No active game'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def reset_current_game(self):
        """Reset the current game"""
        try:
            if self.current_game_engine and hasattr(self.current_game_engine, 'reset_game'):
                self.current_game_engine.reset_game()
                return {'success': True, 'message': 'Game reset'}
            return {'success': False, 'message': 'No active game to reset'}
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    def get_current_game_state(self):
        """Get current game state"""
        if self.current_game_engine and hasattr(self.current_game_engine, 'get_game_state'):
            game_state = self.current_game_engine.get_game_state()
            game_state['game_id'] = self.current_game_id
            game_state['is_running'] = self.is_running
            return game_state
        
        return {
            'game_id': self.current_game_id,
            'is_running': self.is_running,
            'score': 0
        }
    
    def get_current_frame(self):
        """Get current camera frame with game overlay as base64"""
        try:
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    # Flip frame horizontally for mirror effect (so user's right is camera's right)
                    frame = cv2.flip(frame, 1)
                    
                    # Resize frame for better performance
                    frame = cv2.resize(frame, (640, 480))
                    
                    # Process frame with game engine if running
                    if self.is_running and self.current_game_engine:
                        if hasattr(self.current_game_engine, 'process_frame'):
                            frame = self.current_game_engine.process_frame(frame)
                        else:
                            # Fallback: just add game name
                            cv2.putText(frame, f"Playing: {self.current_game_id}", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        # Show "Ready to play" message
                        cv2.putText(frame, "Ready to play! Start a game.", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(frame, "Camera is working!", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Encode frame to base64
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    return frame_base64
            return None
        except Exception as e:
            print(f"❌ Frame capture error: {e}")
            return None
    
    @property
    def current_game(self):
        """Compatibility property"""
        return self.current_game_engine
    
    def __del__(self):
        """Cleanup resources"""
        if self.camera:
            self.camera.release()

# Global game manager instance
game_manager = GameManager()