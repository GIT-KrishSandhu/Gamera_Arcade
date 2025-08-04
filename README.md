# 🎮 Gamera Arcade - Gesture Gaming Revolution

A revolutionary web-based gaming platform that combines gesture control with Web3 technology. Built with Flask, OpenCV, MediaPipe, and modern web technologies.

## Features

### Core Features
- **Gesture-Controlled Games**: Play games using hand movements detected by AI
- **Web3 Integration**: Earn tokens and NFTs based on gaming performance
- **Real-time Leaderboards**: Compete with players worldwide
- **Modern UI/UX**: Beautiful, responsive design with light/dark themes
- **Cross-Platform**: Works on desktop, tablet, and mobile devices

### Games Available
1. **Fruit Ninja** - Slice fruits with hand gestures (Active)
2. **T-Rex Run** - Jump over obstacles with hand-up gesture (Active)
3. **Rock Paper Scissors** - Battle the computer with real gestures (Active)
4. **Gesture Breakout** - Control the paddle with your hand (Coming Soon)
5. **Space Invaders** - Defend Earth with gesture controls (Coming Soon)

### Enhanced Design Features
- **Dynamic Color Schemes**: Green theme for light mode, Blue theme for dark mode
- **Smooth Animations**: Floating elements, particle effects, and transitions
- **Visual Feedback**: Score animations, token earning effects, and notifications
- **Responsive Layout**: Optimized for all screen sizes

## Quick Start

### Prerequisites
- Python 3.8+
- Webcam/Camera access
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/GIT-KrishSandhu/Gamera_Arcade.git
cd Gamera_Arcade
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Open your browser**
Navigate to `Gamera Arcade`

## How to Play

### Fruit Ninja
1. Click "Start Game" to begin
2. Allow camera access when prompted
3. Show your hand to the camera
4. Make quick swipe gestures to slice falling fruits
5. Earn points and tokens based on your performance!

### Controls
- **Swipe Gesture**: Move your hand quickly across the camera view
- **Hand Detection**: Keep your hand visible and well-lit
- **Scoring**: Each sliced fruit gives you 10 points

## Technical Architecture

### Backend (Flask)
- **User Management**: Registration, login, profiles
- **Game Scoring**: Score tracking and leaderboards
- **API Endpoints**: RESTful APIs for game integration
- **Database**: SQLite with SQLAlchemy ORM

### Frontend (HTML/CSS/JS)
- **Modern CSS**: Custom properties, gradients, animations
- **Responsive Design**: Mobile-first approach
- **Theme System**: Light/dark mode with color switching
- **Game Engine**: Canvas-based rendering with requestAnimationFrame

### Computer Vision (OpenCV + MediaPipe)
- **Hand Tracking**: Real-time hand landmark detection
- **Gesture Recognition**: Swipe and movement detection
- **Camera Integration**: WebRTC for browser camera access

### Web3 Integration (Placeholder)
- **Wallet Connection**: MetaMask integration ready
- **Token System**: Configurable reward economics
- **NFT Support**: Achievement and collectible framework

## Color Schemes

### Light Mode (Green Theme)
- Primary: `#8dd783` (Light Green)
- Secondary: `#89ff76` (Bright Green)  
- Accent: `#399a4b` (Dark Green)
- Background: `#f8fffe` (Off White)
- Surface: `#e8f5e8` (Light Green Tint)

### Dark Mode (Blue Theme)
- Primary: `#6699CC` (Blue Grey)
- Secondary: `#7393B3` (Steel Blue)
- Accent: `#5F9EA0` (Cadet Blue)
- Background: `#1B1212` (Dark Brown)
- Surface: `#36454F` (Charcoal)

## Project Structure

```
gamera-arcade/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── static/
│   ├── css/
│   │   └── style.css     # Enhanced styling
│   ├── js/
│   │   └── main.js       # Game engines & interactions
│   └── images/           # Game thumbnails & assets
├── templates/
│   ├── layout.html       # Base template
│   ├── index.html        # Homepage
│   ├── game.html         # Individual game pages
│   ├── profile.html      # User profile
│   └── auth.html         # Login/Register
└── games/                # Game engine modules (future)
```

## Configuration

### Environment Variables
```bash
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///gamera_arcade.db
```

### Camera Settings
- Resolution: 640x480 (recommended)
- Frame Rate: 30 FPS
- Format: WebRTC compatible

## 🚀 Future Roadmap

### Phase 1 (Current)
- ✅ Enhanced UI with dynamic colors
- ✅ Fixed Fruit Ninja game mechanics
- ✅ Improved theme switching
- ✅ Better visual feedback

### Phase 2 (Next)
- [ ] Complete T-Rex Run game
- [ ] Implement Rock Paper Scissors
- [ ] Add Breakout game
- [ ] Web3 token integration

### Phase 3 (Future)
- [ ] Multiplayer functionality
- [ ] Mobile app development
- [ ] Advanced gesture recognition
- [ ] Tournament system

## 📄 License

This project is licensed under the MIT License.

## 🎉 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Gamera Arcade** - Where Gaming Meets Innovation 🎮✨

*Built with ❤️ using Flask, OpenCV & MediaPipe technologies*
