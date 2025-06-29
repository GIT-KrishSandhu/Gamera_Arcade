from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
import json

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
    tokens = db.Column(db.Integer, default=0)  # Game tokens earned
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
    'fruit_ninja': {
        'name': 'Fruit Ninja',
        'description': 'Slice fruits with hand gestures!',
        'thumbnail': '/static/images/fruit_ninja_thumb.jpg',
        'status': 'active',
        'engine': 'fruit_ninja_engine'
    },
    'trex_run': {
        'name': 'T-Rex Run',
        'description': 'Jump over obstacles with hand-up gesture!',
        'thumbnail': '/static/images/trex_thumb.jpg',
        'status': 'coming_soon',
        'engine': 'trex_engine'
    },
    'rock_paper_scissors': {
        'name': 'Rock Paper Scissors',
        'description': 'Battle the computer with real gestures!',
        'thumbnail': '/static/images/rps_thumb.jpg',
        'status': 'coming_soon',
        'engine': 'rps_engine'
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
    }
}

# Routes
@app.route('/')
def index():
    return render_template('index.html', games=GAMES_CONFIG)

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
    
    return render_template('game.html', 
                         game_id=game_id, 
                         game_config=game_config,
                         leaderboard=leaderboard)

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
                    password_hash=generate_password_hash(password)
                )
                db.session.add(user)
                db.session.commit()
                
                session['user_id'] = user.id
                session['username'] = user.username
                flash('Registration successful!', 'success')
                return redirect(url_for('index'))
    
    return render_template('auth.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

# API Routes for Game Integration
@app.route('/api/submit_score', methods=['POST'])
def submit_score():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    game_name = data.get('game_name')
    score = data.get('score')
    
    if not game_name or score is None:
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Calculate tokens earned (placeholder logic)
    tokens_earned = calculate_tokens(game_name, score)
    
    # Save score
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
    
    db.session.commit()
    
    # TODO: Web3 Integration - Mint tokens to user's wallet
    # mint_tokens_to_wallet(user.wallet_address, tokens_earned)
    
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
    
    # TODO: Web3 Integration - Verify wallet ownership
    # verify_wallet_ownership(wallet_address)
    
    return jsonify({'success': True})

def calculate_tokens(game_name, score):
    """Calculate tokens earned based on game and score"""
    # TODO: Implement sophisticated token economics
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

if __name__ == '__main__':
    # Create tables before running the app
    create_tables()
    app.run(debug=True)
