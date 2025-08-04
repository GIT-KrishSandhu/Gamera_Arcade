from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import json
import cv2
import base64
import numpy as np
from games import game_manager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///gamera_arcade.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    wallet_address = db.Column(db.String(42), nullable=True)  # Web3 wallet address
    tokens = db.Column(db.Integer, default=1000)  # Game tokens earned
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    scores = db.relationship('Score', backref='user', lazy=True)

class Score(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    game_name = db.Column(db.String(50), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    tokens_earned = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Game Configuration
GAMES_CONFIG = {
    'trex_run': {
        'name': 'T-Rex Run',
        'description': 'Jump over obstacles with hand-up gesture!',
        'thumbnail': '/static/images/trex_thumb.png',
        'status': 'active',
        'engine': 'trex_engine'
    },
    'rock_paper_scissors': {
        'name': 'Rock Paper Scissors',
        'description': 'Battle the computer with real gestures!',
        'thumbnail': '/static/images/rps_thumb.png',
        'status': 'active',
        'engine': 'rps_engine'
    },
    'fruit_ninja': {
        'name': 'Fruit Ninja',
        'description': 'Slice fruits with hand gestures!',
        'thumbnail': '/static/images/fruit_ninja_thumb.png',
        'status': 'active',
        'engine': 'fruit_ninja_engine'
    },
    'breakout': {
        'name': 'Gesture Breakout',
        'description': 'Control the paddle with your hand!',
        'thumbnail': '/static/images/breakout_thumb.jpg',
        'status': 'coming_soon',
        'engine': 'breakout_engine'
    },
    'space_invaders': {
        'name': 'Space Invaders',
        'description': 'Defend Earth with gesture controls!',
        'thumbnail': '/static/images/space_thumb.jpg',
        'status': 'coming_soon',
        'engine': 'space_engine'
    },
    'Snake_Game': {
        'name': 'Snake Game',
        'description': 'Navigate from a small worm to COBRA, eat as many apples as you can and avoid eating yourself!!',
        'thumbnail': '/static/images/snake_game.jpg',
        'status': 'coming_soon',
        'engine': 'snake_engine'
    }
}

# Routes
@app.route('/')
def index():
    users = User.query.order_by(User.tokens.desc()).limit(10).all()
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    return render_template('index.html', games=GAMES_CONFIG, users=users, user=user)

@app.route('/game/<game_id>')
def game(game_id):
    if game_id not in GAMES_CONFIG:
        flash('Game not found!', 'error')
        return redirect(url_for('index'))
    
    game_config = GAMES_CONFIG[game_id]
    
    # Get leaderboard for this game
    leaderboard = db.session.query(Score, User.username)\
        .join(User)\
        .filter(Score.game_name == game_id)\
        .order_by(Score.score.desc())\
        .limit(10)\
        .all()
    
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        
    return render_template('game.html', 
                         game_id=game_id, 
                         game_config=game_config,
                         leaderboard=leaderboard,
                         user=user)

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('auth'))
    
    user = User.query.get(session['user_id'])
    user_scores = Score.query.filter_by(user_id=user.id)\
        .order_by(Score.created_at.desc())\
        .limit(20)\
        .all()
    
    return render_template('profile.html', user=user, scores=user_scores)

@app.route('/auth', methods=['GET', 'POST'])
def auth():
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'login':
            username = request.form.get('username')
            password = request.form.get('password')
            
            user = User.query.filter_by(username=username).first()
            
            if user and check_password_hash(user.password_hash, password):
                session['user_id'] = user.id
                session['username'] = user.username
                session['tokens'] = user.tokens
                flash('Login successful!', 'success')
                return redirect(url_for('index'))
            else:
                flash('Invalid credentials!', 'error')
        
        elif action == 'register':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            
            if User.query.filter_by(username=username).first():
                flash('Username already exists!', 'error')
            elif User.query.filter_by(email=email).first():
                flash('Email already registered!', 'error')
            else:
                user = User(
                    username=username,
                    email=email,
                    password_hash=generate_password_hash(password),
                    tokens=1000
                )
                db.session.add(user)
                db.session.commit()
                
                session['user_id'] = user.id
                session['username'] = user.username
                session['tokens'] = user.tokens
                flash('Registration successful! You have been awarded 1000 coins.', 'success')
                return redirect(url_for('index'))
    
    return render_template('auth.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

# API to get user coins
@app.route('/api/user/coins')
def get_user_coins():
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        return jsonify({'coins': user.tokens})
    return jsonify({'coins': 0})

# Game API Routes
@app.route('/api/game/start/<game_id>', methods=['POST'])
def start_game_api(game_id):
    """Start a game"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'You must be logged in to play.'})

    user = User.query.get(session['user_id'])
    if user.tokens < 20:
        return jsonify({'success': False, 'message': 'You do not have enough coins to play.'})

    user.tokens -= 20
    db.session.commit()
    session['tokens'] = user.tokens


    try:
        print(f"🎮 API: Starting game {game_id}")
        result = game_manager.start_game(game_id)
        print(f"🎮 API: Game start result: {result}")
        return jsonify(result)
    except Exception as e:
        print(f"❌ API: Error starting game: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/game/stop', methods=['POST'])
def stop_game_api():
    """Stop current game"""
    try:
        print(f"🛑 API: Stopping current game")
        result = game_manager.stop_current_game()
        print(f"🛑 API: Game stop result: {result}")
        return jsonify(result)
    except Exception as e:
        print(f"❌ API: Error stopping game: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/game/reset', methods=['POST'])
def reset_game_api():
    """Reset the current game without reloading the page"""
    try:
        print("🔄 API: Resetting current game...")
        if game_manager.current_game:
            game_manager.current_game.reset_game()
            return jsonify({'success': True, 'message': 'Game reset'})
        else:
            return jsonify({'success': False, 'message': 'No active game to reset'}), 400
    except Exception as e:
        print(f"❌ API: Error resetting game: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/game/state')
def get_game_state():
    """Get current game state"""
    try:
        state = game_manager.get_current_game_state()
        return jsonify(state)
    except Exception as e:
        print(f"❌ API: Error getting game state: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/game/video_feed')
def video_feed():
    """Video streaming route - FIXED"""
    def generate():
        print("📹 Starting video feed generator...")
        frame_count = 0
        
        try:
            while True:
                try:
                    frame_base64 = game_manager.get_current_frame()
                    if frame_base64:
                        # Decode base64 and send as MJPEG stream
                        frame_bytes = base64.b64decode(frame_base64)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               frame_bytes + b'\r\n')
                        
                        frame_count += 1
                        if frame_count % 60 == 0:  # Log every 60 frames (less frequent)
                            print(f"📹 Video feed active - {frame_count} frames sent")
                    else:
                        # Send empty frame if no game is running
                        empty_frame = create_empty_frame()
                        _, buffer = cv2.imencode('.jpg', empty_frame)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
                    # Small delay to prevent overwhelming
                    import time
                    time.sleep(0.033)  # ~30 FPS
                    
                except GeneratorExit:
                    print("📹 Video feed client disconnected")
                    break
                except Exception as e:
                    print(f"❌ Video feed frame error: {e}")
                    # Continue trying instead of breaking
                    import time
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"❌ Video feed generator error: {e}")
        finally:
            print("📹 Video feed generator ended")
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/game/score')
def get_current_score():
    """Get current game score"""
    try:
        state = game_manager.get_current_game_state()
        return jsonify({'score': state.get('score', 0)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_empty_frame():
    """Create an empty frame when no game is running"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "No game running", (200, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

@app.route('/api/submit_score', methods=['POST'])
def submit_score():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    game_name = data.get('game_name')
    score = data.get('score')
    
    if not game_name or score is None:
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Check for existing high score
    existing_score = Score.query.filter_by(user_id=session['user_id'], game_name=game_name).first()
    
    if existing_score:
        if score > existing_score.score:
            # Update existing score
            existing_score.score = score
            tokens_earned = calculate_tokens(game_name, score)
            
            # Update user tokens
            user = User.query.get(session['user_id'])
            user.tokens += tokens_earned
            session['tokens'] = user.tokens
            
            db.session.commit()
            
            return jsonify({
                'success': True,
                'tokens_earned': tokens_earned,
                'total_tokens': user.tokens,
                'message': 'New high score!'
            })
        else:
            return jsonify({
                'success': True,
                'message': 'Score not higher than personal best.'
            })
    else:
        # Save new score
        tokens_earned = calculate_tokens(game_name, score)
        new_score = Score(
            user_id=session['user_id'],
            game_name=game_name,
            score=score,
            tokens_earned=tokens_earned
        )
        db.session.add(new_score)
        
        # Update user tokens
        user = User.query.get(session['user_id'])
        user.tokens += tokens_earned
        session['tokens'] = user.tokens
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'tokens_earned': tokens_earned,
            'total_tokens': user.tokens
        })

@app.route('/api/connect_wallet', methods=['POST'])
def connect_wallet():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    wallet_address = data.get('wallet_address')
    
    if not wallet_address:
        return jsonify({'error': 'Wallet address required'}), 400
    
    user = User.query.get(session['user_id'])
    user.wallet_address = wallet_address
    db.session.commit()
    
    return jsonify({'success': True})

def calculate_tokens(game_name, score):
    """Calculate tokens earned based on game and score"""
    base_tokens = {
        'fruit_ninja': 10,
        'trex_run': 15,
        'rock_paper_scissors': 5,
        'breakout': 20,
        'space_invaders': 25
    }
    
    multiplier = max(1, score // 100)  # Bonus for higher scores
    return base_tokens.get(game_name, 10) * multiplier

# Initialize database
def create_tables():
    with app.app_context():
        db.create_all()
        print("📊 Database tables created successfully")

if __name__ == '__main__':
    # Create tables before running the app
    create_tables()
    print("🎮 Gamera Arcade Server Starting...")
    print("🍎 Fruit Ninja with OpenCV/MediaPipe integration ready!")
    print("📹 Video streaming enabled")
    
    # FIXED: Disable reloader to prevent duplicate initialization
    app.run(debug=True, threaded=True, use_reloader=False, host='127.0.0.1', port=5000)