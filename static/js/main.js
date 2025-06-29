// Gamera Arcade - Enhanced Main JavaScript File

class GameraArcade {
  constructor() {
    this.init()
  }

  init() {
    this.setupThemeToggle()
    this.setupDropdowns()
    this.setupAnimations()
    this.setupWeb3()
    this.setupNotifications()

    // Initialize on DOM ready
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => this.onReady())
    } else {
      this.onReady()
    }
  }

  onReady() {
    this.loadUserPreferences()
    this.initializeAnimations()
    console.log("🎮 Gamera Arcade initialized successfully!")
  }

  // Enhanced Theme Management
  setupThemeToggle() {
    const themeSwitch = document.getElementById("theme-switch")
    if (themeSwitch) {
      themeSwitch.addEventListener("change", (e) => {
        this.toggleTheme(e.target.checked)
      })
    }
  }

  toggleTheme(isDark) {
    const theme = isDark ? "dark" : "light"
    document.documentElement.setAttribute("data-theme", theme)
    localStorage.setItem("theme", theme)

    // Enhanced theme transition with color changes
    document.body.style.transition = "all 0.5s ease"

    // Add visual feedback
    this.showNotification(`Switched to ${theme} mode! ${isDark ? "🌙" : "☀️"}`, "info", 2000)

    setTimeout(() => {
      document.body.style.transition = ""
    }, 500)
  }

  loadUserPreferences() {
    // Load theme preference
    const savedTheme = localStorage.getItem("theme") || "light"
    const themeSwitch = document.getElementById("theme-switch")

    if (themeSwitch) {
      themeSwitch.checked = savedTheme === "dark"
      document.documentElement.setAttribute("data-theme", savedTheme)
    }
  }

  // Dropdown Management
  setupDropdowns() {
    document.addEventListener("click", (e) => {
      // Close dropdowns when clicking outside
      if (!e.target.closest(".profile-dropdown")) {
        this.closeAllDropdowns()
      }
    })
  }

  closeAllDropdowns() {
    const dropdowns = document.querySelectorAll(".dropdown-menu")
    dropdowns.forEach((dropdown) => {
      dropdown.classList.remove("show")
    })
  }

  // Enhanced Animation Setup
  setupAnimations() {
    // Intersection Observer for scroll animations
    this.observeElements()
    // Smooth scrolling for anchor links
    this.setupSmoothScrolling()
    // Add parallax effects
    this.setupParallax()
  }

  observeElements() {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("fade-in")
            // Add stagger effect for grid items
            if (entry.target.classList.contains("game-card") || entry.target.classList.contains("feature-card")) {
              const delay = Array.from(entry.target.parentElement.children).indexOf(entry.target) * 100
              entry.target.style.animationDelay = `${delay}ms`
            }
          }
        })
      },
      {
        threshold: 0.1,
        rootMargin: "0px 0px -50px 0px",
      },
    )

    // Observe elements that should animate on scroll
    const animatedElements = document.querySelectorAll(
      ".game-card, .feature-card, .score-card, .leaderboard-item, .stat-item",
    )

    animatedElements.forEach((el) => observer.observe(el))
  }

  setupSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
      anchor.addEventListener("click", function (e) {
        e.preventDefault()
        const target = document.querySelector(this.getAttribute("href"))
        if (target) {
          target.scrollIntoView({
            behavior: "smooth",
            block: "start",
          })
        }
      })
    })
  }

  setupParallax() {
    window.addEventListener("scroll", () => {
      const scrolled = window.pageYOffset
      const parallaxElements = document.querySelectorAll(".floating-icon")

      parallaxElements.forEach((element, index) => {
        const speed = 0.5 + index * 0.1
        element.style.transform = `translateY(${scrolled * speed}px)`
      })
    })
  }

  initializeAnimations() {
    // Add stagger animation to grid items
    const gridItems = document.querySelectorAll(".games-grid .game-card")
    gridItems.forEach((item, index) => {
      item.style.animationDelay = `${index * 0.15}s`
      item.classList.add("slide-up")
    })

    // Add pulse animation to important elements
    const importantElements = document.querySelectorAll(".cta-btn, .auth-btn")
    importantElements.forEach((element) => {
      element.classList.add("pulse")
    })
  }

  // Web3 Integration (Enhanced)
  setupWeb3() {
    this.web3Manager = new Web3Manager()
  }

  // Enhanced Notification System
  setupNotifications() {
    this.notificationContainer = this.createNotificationContainer()
  }

  createNotificationContainer() {
    let container = document.getElementById("notification-container")
    if (!container) {
      container = document.createElement("div")
      container.id = "notification-container"
      container.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        z-index: 1100;
        display: flex;
        flex-direction: column;
        gap: 10px;
      `
      document.body.appendChild(container)
    }
    return container
  }

  showNotification(message, type = "info", duration = 5000) {
    const notification = document.createElement("div")
    notification.className = `notification ${type}`

    // Enhanced notification with icons
    const icons = {
      success: "✅",
      error: "❌",
      warning: "⚠️",
      info: "ℹ️",
    }

    notification.innerHTML = `
      <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 1.2em;">${icons[type] || icons.info}</span>
        <span>${message}</span>
      </div>
      <button onclick="this.parentElement.remove()" 
              style="background: none; border: none; color: inherit; cursor: pointer; margin-left: 10px; padding: 5px; border-radius: 3px; transition: background-color 0.2s;">
        <i class="fas fa-times"></i>
      </button>
    `

    this.notificationContainer.appendChild(notification)

    // Auto remove after duration
    setTimeout(() => {
      if (notification.parentElement) {
        notification.style.animation = "slideOut 0.3s ease"
        setTimeout(() => notification.remove(), 300)
      }
    }, duration)
  }
}

// Enhanced Web3 Manager Class
class Web3Manager {
  constructor() {
    this.wallet = null
    this.contract = null
    this.init()
  }

  init() {
    // TODO: Web3 Integration - Initialize Web3 providers
    console.log("🔗 Web3Manager initialized (placeholder)")
    this.checkWalletConnection()
  }

  async checkWalletConnection() {
    if (typeof window.ethereum !== "undefined") {
      try {
        const accounts = await window.ethereum.request({ method: "eth_accounts" })
        if (accounts.length > 0) {
          this.wallet = accounts[0]
          console.log("👛 Wallet already connected:", this.wallet)
        }
      } catch (error) {
        console.log("No wallet connected")
      }
    }
  }

  async connectWallet() {
    try {
      if (typeof window.ethereum !== "undefined") {
        // Show loading state
        gameraArcade.showNotification("Connecting wallet...", "info", 2000)

        const accounts = await window.ethereum.request({
          method: "eth_requestAccounts",
        })

        this.wallet = accounts[0]
        await this.saveWalletAddress(this.wallet)

        gameraArcade.showNotification(
          `Wallet connected! ${this.wallet.slice(0, 6)}...${this.wallet.slice(-4)}`,
          "success",
        )

        // Refresh page to update UI
        setTimeout(() => window.location.reload(), 1000)

        return this.wallet
      } else {
        throw new Error("MetaMask not installed. Please install MetaMask to continue.")
      }
    } catch (error) {
      console.error("Wallet connection error:", error)
      gameraArcade.showNotification(`Failed to connect wallet: ${error.message}`, "error")
      throw error
    }
  }

  async saveWalletAddress(address) {
    try {
      const response = await fetch("/api/connect_wallet", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          wallet_address: address,
        }),
      })

      if (!response.ok) {
        throw new Error("Failed to save wallet address")
      }

      return await response.json()
    } catch (error) {
      console.error("Error saving wallet address:", error)
      throw error
    }
  }

  async mintTokens(amount, recipient) {
    // TODO: Web3 Integration - Implement token minting
    console.log(`🪙 Minting ${amount} tokens to ${recipient} (placeholder)`)

    // Enhanced placeholder implementation
    gameraArcade.showNotification(`Minting ${amount} tokens...`, "info", 2000)

    // Simulate blockchain transaction delay
    await new Promise((resolve) => setTimeout(resolve, 2000))

    const result = {
      success: true,
      txHash: "0x" + Math.random().toString(16).substr(2, 64),
      amount: amount,
    }

    gameraArcade.showNotification(`Successfully minted ${amount} tokens! 🎉`, "success")

    return result
  }

  async getTokenBalance(address) {
    // TODO: Web3 Integration - Get token balance from smart contract
    console.log(`💰 Getting token balance for ${address} (placeholder)`)
    return Math.floor(Math.random() * 1000) // Placeholder
  }
}

// Enhanced Game Engine Base Class
class GameEngine {
  constructor(gameId, videoElement) {
    this.gameId = gameId
    this.video = videoElement
    this.canvas = document.getElementById("gameCanvas")
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null
    this.isRunning = false
    this.score = 0
    this.gameLoop = null
    this.lastFrameTime = 0
  }

  async initialize() {
    if (this.canvas && this.video) {
      // Wait for video to load
      await new Promise((resolve) => {
        this.video.addEventListener("loadedmetadata", resolve)
        if (this.video.readyState >= 1) resolve()
      })

      this.canvas.width = this.video.videoWidth || 640
      this.canvas.height = this.video.videoHeight || 480
    }

    console.log(`🎮 Initializing ${this.gameId} engine...`)
  }

  start() {
    if (this.isRunning) return

    this.isRunning = true
    this.lastFrameTime = performance.now()
    this.gameLoop = requestAnimationFrame((time) => this.gameLoopFunction(time))
    console.log(`▶️ ${this.gameId} started`)
  }

  stop() {
    if (!this.isRunning) return

    this.isRunning = false
    if (this.gameLoop) {
      cancelAnimationFrame(this.gameLoop)
      this.gameLoop = null
    }
    console.log(`⏹️ ${this.gameId} stopped`)
  }

  gameLoopFunction(currentTime) {
    if (!this.isRunning) return

    const deltaTime = currentTime - this.lastFrameTime
    this.lastFrameTime = currentTime

    this.update(deltaTime)
    this.gameLoop = requestAnimationFrame((time) => this.gameLoopFunction(time))
  }

  update(deltaTime) {
    // Override in specific game implementations
    if (this.ctx && this.video) {
      this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height)
    }
  }

  updateScore(newScore) {
    this.score = newScore
    const scoreElement = document.getElementById("currentScore")
    if (scoreElement) {
      scoreElement.textContent = newScore
      // Add visual feedback for score increase
      scoreElement.style.transform = "scale(1.2)"
      setTimeout(() => {
        scoreElement.style.transform = "scale(1)"
      }, 200)
    }
  }

  async submitScore() {
    if (this.score <= 0) return

    try {
      gameraArcade.showNotification("Submitting score...", "info", 2000)

      const response = await fetch("/api/submit_score", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          game_name: this.gameId,
          score: this.score,
        }),
      })

      const data = await response.json()

      if (data.success) {
        gameraArcade.showNotification(`Score submitted! Earned ${data.tokens_earned} tokens! 🎉`, "success")

        // Update tokens display
        const tokensElement = document.getElementById("tokensEarned")
        if (tokensElement) {
          tokensElement.textContent = data.total_tokens
        }

        // Enhanced token animation
        this.showTokenAnimation(data.tokens_earned)
      }
    } catch (error) {
      console.error("Error submitting score:", error)
      gameraArcade.showNotification("Failed to submit score", "error")
    }
  }

  showTokenAnimation(tokens) {
    console.log(`🎉 Earned ${tokens} tokens!`)

    // Create multiple floating token animations
    for (let i = 0; i < 3; i++) {
      setTimeout(() => {
        const tokenElement = document.createElement("div")
        tokenElement.innerHTML = `+${Math.ceil(tokens / 3)} 🪙`
        tokenElement.style.cssText = `
          position: fixed;
          top: 50%;
          left: ${45 + i * 5}%;
          transform: translate(-50%, -50%);
          font-size: 2rem;
          font-weight: bold;
          color: var(--primary-green);
          z-index: 1000;
          pointer-events: none;
          animation: tokenFloat 3s ease-out forwards;
          text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        `

        document.body.appendChild(tokenElement)

        setTimeout(() => {
          tokenElement.remove()
        }, 3000)
      }, i * 200)
    }

    // Add enhanced keyframes if not already present
    if (!document.getElementById("enhanced-token-animation-styles")) {
      const style = document.createElement("style")
      style.id = "enhanced-token-animation-styles"
      style.textContent = `
        @keyframes tokenFloat {
          0% {
            opacity: 1;
            transform: translate(-50%, -50%) scale(0.5) rotate(0deg);
          }
          25% {
            opacity: 1;
            transform: translate(-50%, -70%) scale(1.2) rotate(90deg);
          }
          50% {
            opacity: 1;
            transform: translate(-50%, -90%) scale(1) rotate(180deg);
          }
          75% {
            opacity: 0.8;
            transform: translate(-50%, -110%) scale(1.1) rotate(270deg);
          }
          100% {
            opacity: 0;
            transform: translate(-50%, -130%) scale(0.8) rotate(360deg);
          }
        }
      `
      document.head.appendChild(style)
    }
  }
}

// Enhanced Fruit Ninja Engine
class FruitNinjaEngine extends GameEngine {
  constructor(videoElement) {
    super("fruit_ninja", videoElement)
    this.fruits = []
    this.lastSpawnTime = 0
    this.spawnInterval = 1500 // milliseconds
    this.handPosition = { x: 0, y: 0 }
    this.lastHandPosition = { x: 0, y: 0 }
    this.swipeThreshold = 50
  }

  async initialize() {
    await super.initialize()
    console.log("🍎 Enhanced Fruit Ninja engine initialized")
  }

  update(deltaTime) {
    super.update(deltaTime)

    const currentTime = performance.now()

    // Spawn fruits
    this.spawnFruits(currentTime)

    // Update and draw fruits
    this.updateFruits(deltaTime)

    // Detect gestures (placeholder)
    this.detectGestures()

    // Draw UI elements
    this.drawUI()
  }

  spawnFruits(currentTime) {
    if (currentTime - this.lastSpawnTime > this.spawnInterval) {
      const fruit = {
        x: Math.random() * (this.canvas.width - 60) + 30,
        y: this.canvas.height + 30,
        velocity: -(3 + Math.random() * 4), // Upward velocity
        radius: 25 + Math.random() * 15,
        alive: true,
        color: this.getRandomFruitColor(),
        rotation: 0,
        rotationSpeed: (Math.random() - 0.5) * 0.1,
      }

      this.fruits.push(fruit)
      this.lastSpawnTime = currentTime

      // Gradually increase difficulty
      if (this.spawnInterval > 800) {
        this.spawnInterval -= 10
      }
    }
  }

  getRandomFruitColor() {
    const colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"]
    return colors[Math.floor(Math.random() * colors.length)]
  }

  updateFruits(deltaTime) {
    this.fruits = this.fruits.filter((fruit) => {
      if (!fruit.alive) return false

      // Update position
      fruit.y += fruit.velocity * (deltaTime / 16.67) // Normalize to 60fps
      fruit.rotation += fruit.rotationSpeed * deltaTime

      // Remove fruits that fall off screen
      if (fruit.y < -fruit.radius - 50) {
        return false
      }

      // Draw fruit
      this.drawFruit(fruit)

      return true
    })
  }

  drawFruit(fruit) {
    if (!this.ctx) return

    this.ctx.save()

    // Translate to fruit position
    this.ctx.translate(fruit.x, fruit.y)
    this.ctx.rotate(fruit.rotation)

    // Draw fruit shadow
    this.ctx.fillStyle = "rgba(0, 0, 0, 0.2)"
    this.ctx.beginPath()
    this.ctx.arc(2, 2, fruit.radius, 0, 2 * Math.PI)
    this.ctx.fill()

    // Draw fruit
    this.ctx.fillStyle = fruit.color
    this.ctx.beginPath()
    this.ctx.arc(0, 0, fruit.radius, 0, 2 * Math.PI)
    this.ctx.fill()

    // Add fruit highlight
    this.ctx.fillStyle = "rgba(255, 255, 255, 0.3)"
    this.ctx.beginPath()
    this.ctx.arc(-fruit.radius * 0.3, -fruit.radius * 0.3, fruit.radius * 0.4, 0, 2 * Math.PI)
    this.ctx.fill()

    this.ctx.restore()
  }

  detectGestures() {
    // TODO: Implement actual MediaPipe hand detection
    // For now, simulate hand movement and slicing

    // Simulate random hand position (replace with actual hand tracking)
    if (Math.random() > 0.98) {
      this.handPosition.x = Math.random() * this.canvas.width
      this.handPosition.y = Math.random() * this.canvas.height

      // Check for swipe gesture
      const distance = Math.hypot(
        this.handPosition.x - this.lastHandPosition.x,
        this.handPosition.y - this.lastHandPosition.y,
      )

      if (distance > this.swipeThreshold) {
        this.checkFruitCollision()
      }

      this.lastHandPosition = { ...this.handPosition }
    }
  }

  checkFruitCollision() {
    this.fruits.forEach((fruit) => {
      if (fruit.alive) {
        const distance = Math.hypot(fruit.x - this.handPosition.x, fruit.y - this.handPosition.y)

        if (distance < fruit.radius + 20) {
          fruit.alive = false
          this.updateScore(this.score + 10)
          this.createSliceEffect(fruit.x, fruit.y)
        }
      }
    })
  }

  createSliceEffect(x, y) {
    // Create visual slice effect
    if (!this.ctx) return

    this.ctx.save()
    this.ctx.strokeStyle = "#FFD700"
    this.ctx.lineWidth = 3
    this.ctx.lineCap = "round"

    // Draw slice line
    this.ctx.beginPath()
    this.ctx.moveTo(x - 30, y - 15)
    this.ctx.lineTo(x + 30, y + 15)
    this.ctx.stroke()

    this.ctx.restore()

    // Add particles effect
    this.createParticles(x, y)
  }

  createParticles(x, y) {
    // Create particle explosion effect
    for (let i = 0; i < 8; i++) {
      const particle = document.createElement("div")
      particle.style.cssText = `
        position: absolute;
        width: 4px;
        height: 4px;
        background: #FFD700;
        border-radius: 50%;
        pointer-events: none;
        z-index: 1000;
        left: ${x}px;
        top: ${y}px;
        animation: particleExplode 0.8s ease-out forwards;
      `

      // Random direction for each particle
      const angle = (i / 8) * Math.PI * 2
      const distance = 50 + Math.random() * 30
      particle.style.setProperty("--end-x", `${Math.cos(angle) * distance}px`)
      particle.style.setProperty("--end-y", `${Math.sin(angle) * distance}px`)

      this.canvas.parentElement.appendChild(particle)

      setTimeout(() => particle.remove(), 800)
    }

    // Add particle animation if not exists
    if (!document.getElementById("particle-animation-styles")) {
      const style = document.createElement("style")
      style.id = "particle-animation-styles"
      style.textContent = `
        @keyframes particleExplode {
          0% {
            transform: translate(0, 0) scale(1);
            opacity: 1;
          }
          100% {
            transform: translate(var(--end-x), var(--end-y)) scale(0);
            opacity: 0;
          }
        }
      `
      document.head.appendChild(style)
    }
  }

  drawUI() {
    if (!this.ctx) return

    // Draw hand position indicator (for debugging)
    if (this.handPosition.x > 0 && this.handPosition.y > 0) {
      this.ctx.save()
      this.ctx.fillStyle = "rgba(255, 255, 255, 0.8)"
      this.ctx.beginPath()
      this.ctx.arc(this.handPosition.x, this.handPosition.y, 10, 0, 2 * Math.PI)
      this.ctx.fill()

      this.ctx.strokeStyle = "rgba(0, 0, 0, 0.5)"
      this.ctx.lineWidth = 2
      this.ctx.stroke()
      this.ctx.restore()
    }
  }
}

// Global Functions
function toggleDropdown() {
  const dropdown = document.getElementById("profileDropdown")
  if (dropdown) {
    dropdown.classList.toggle("show")
  }
}

function connectWallet() {
  const modal = document.getElementById("walletModal")
  if (modal) {
    modal.classList.add("show")
  }
}

function closeWalletModal() {
  const modal = document.getElementById("walletModal")
  if (modal) {
    modal.classList.remove("show")
  }
}

function connectMetaMask() {
  if (gameraArcade && gameraArcade.web3Manager) {
    gameraArcade.web3Manager.connectWallet()
  }
  closeWalletModal()
}

// Enhanced Game Control Functions
async function startGame() {
  const gameType = window.location.pathname.split("/").pop()

  try {
    // Show loading state
    updateGameStatus("Requesting camera access...")

    // Request camera access with enhanced constraints
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 30 },
      },
    })

    const video = document.getElementById("gameVideo")
    if (video) {
      video.srcObject = stream

      // Wait for video to be ready
      await new Promise((resolve) => {
        video.addEventListener("canplay", resolve)
        if (video.readyState >= 3) resolve()
      })

      // Initialize appropriate game engine
      let gameEngine
      switch (gameType) {
        case "fruit_ninja":
          gameEngine = new FruitNinjaEngine(video)
          break
        default:
          gameEngine = new GameEngine(gameType, video)
      }

      await gameEngine.initialize()
      gameEngine.start()

      // Store reference for stopping
      window.currentGameEngine = gameEngine

      updateGameUI(true)
      updateGameStatus("Game started! Make swipe gestures to slice fruits!")

      // Add visual feedback
      gameraArcade.showNotification("Game started! 🎮", "success", 3000)
    }
  } catch (error) {
    console.error("Error starting game:", error)
    updateGameStatus("Error: Could not access camera. Please check permissions.")
    gameraArcade.showNotification("Could not access camera. Please check permissions and try again.", "error")
  }
}

function stopGame() {
  if (window.currentGameEngine) {
    const finalScore = window.currentGameEngine.score
    window.currentGameEngine.stop()

    // Submit score if logged in and score > 0
    const isLoggedIn = document.querySelector(".profile-btn") !== null
    if (isLoggedIn && finalScore > 0) {
      window.currentGameEngine.submitScore()
    }

    // Stop camera
    const video = document.getElementById("gameVideo")
    if (video && video.srcObject) {
      video.srcObject.getTracks().forEach((track) => track.stop())
      video.srcObject = null
    }

    updateGameUI(false)
    updateGameStatus(`Game stopped! Final score: ${finalScore}`)

    // Show final score notification
    if (finalScore > 0) {
      gameraArcade.showNotification(`Great job! Final score: ${finalScore} points! 🎉`, "success")
    }

    window.currentGameEngine = null
  }
}

function updateGameUI(isRunning) {
  const startBtn = document.getElementById("startGameBtn")
  const stopBtn = document.getElementById("stopGameBtn")

  if (startBtn && stopBtn) {
    if (isRunning) {
      startBtn.style.display = "none"
      stopBtn.style.display = "inline-flex"
      startBtn.classList.remove("pulse")
      stopBtn.classList.add("pulse")
    } else {
      startBtn.style.display = "inline-flex"
      stopBtn.style.display = "none"
      stopBtn.classList.remove("pulse")
      startBtn.classList.add("pulse")
    }
  }
}

function updateGameStatus(status) {
  const statusElement = document.getElementById("gameStatus")
  if (statusElement) {
    statusElement.textContent = status
    // Add visual feedback
    statusElement.style.animation = "none"
    setTimeout(() => {
      statusElement.style.animation = "pulse 2s infinite"
    }, 10)
  }
}

// Initialize the application
const gameraArcade = new GameraArcade()

// Add CSS for additional animations
document.addEventListener("DOMContentLoaded", () => {
  if (!document.getElementById("additional-animation-styles")) {
    const style = document.createElement("style")
    style.id = "additional-animation-styles"
    style.textContent = `
      @keyframes slideOut {
        from {
          transform: translateX(0);
          opacity: 1;
        }
        to {
          transform: translateX(100%);
          opacity: 0;
        }
      }
      
      .slide-up {
        animation: slideUp 0.6s ease-out forwards;
      }
      
      @keyframes slideUp {
        from {
          transform: translateY(30px);
          opacity: 0;
        }
        to {
          transform: translateY(0);
          opacity: 1;
        }
      }
    `
    document.head.appendChild(style)
  }
})
