// Gamera Arcade - Enhanced Main JavaScript File with Python Integration

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
}

// CLEANED UP Python Game Integration Class - NO LOCAL GAME LOGIC
class PythonGameManager {
  constructor() {
    this.currentGame = null
    this.gameState = null
    this.videoElement = null
    this.stateUpdateInterval = null
    this.lastScore = 0
  }

  async startGame(gameId) {
    try {
      console.log(`🎮 Starting Python game: ${gameId}`)
      gameraArcade.showNotification("Starting game...", "info", 2000)

      const response = await fetch(`/api/game/start/${gameId}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      })

      const result = await response.json()
      console.log("🎮 Start game response:", result)

      if (result.success) {
        this.currentGame = gameId
        this.setupVideoStream()
        this.startStateUpdates()
        gameraArcade.showNotification("Game started! 🎮", "success")
        console.log("✅ Python game started successfully")
        return result
      } else {
        gameraArcade.showNotification(result.message, "error")
        console.error("❌ Failed to start game:", result.message)
        return result
      }
    } catch (error) {
      console.error("❌ Error starting game:", error)
      gameraArcade.showNotification("Failed to start game", "error")
      return { success: false, message: error.message }
    }
  }

  async stopGame() {
    try {
      console.log("🛑 Stopping Python game...")

      const response = await fetch("/api/game/stop", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      })

      const result = await response.json()
      console.log("🛑 Stop game response:", result)

      if (result.success) {
        this.currentGame = null
        this.stopVideoStream()
        this.stopStateUpdates()

        // Submit score if user is logged in
        const isLoggedIn = document.querySelector(".profile-btn") !== null
        if (isLoggedIn && result.score > 0) {
          await this.submitScore(result.score)
        }

        gameraArcade.showNotification(`Game stopped! Final score: ${result.score}`, "success")
        console.log("✅ Python game stopped successfully")
        return result
      }
    } catch (error) {
      console.error("❌ Error stopping game:", error)
      gameraArcade.showNotification("Failed to stop game", "error")
    }
  }

  setupVideoStream() {
    console.log("📹 Setting up video stream...")
    this.videoElement = document.getElementById("gameVideo")
    if (this.videoElement) {
      this.videoElement.src = "/api/game/video_feed"
      this.videoElement.style.display = "block"
      this.videoElement.style.width = "100%"
      this.videoElement.style.height = "100%"
      this.videoElement.style.objectFit = "cover"

      // Handle video load events
      this.videoElement.onload = () => {
        console.log("✅ Video stream loaded successfully")
      }

      this.videoElement.onerror = (e) => {
        console.error("❌ Video stream error:", e)
        updateGameStatus("Error: Could not load video stream")
      }

      console.log("📹 Video element configured, src set to:", this.videoElement.src)
    } else {
      console.error("❌ Video element not found!")
    }
  }

  stopVideoStream() {
    console.log("📹 Stopping video stream...")
    if (this.videoElement) {
      this.videoElement.src = ""
      this.videoElement.style.display = "none"
    }
  }

  startStateUpdates() {
    console.log("🔄 Starting state updates...")
    this.stateUpdateInterval = setInterval(async () => {
      try {
        const response = await fetch("/api/game/state")
        const state = await response.json()
        this.updateGameUI(state)

        // Also get real-time score
        const scoreResponse = await fetch("/api/game/score")
        const scoreData = await scoreResponse.json()
        if (scoreData.score !== undefined) {
          this.updateScore(scoreData.score)
        }
      } catch (error) {
        console.error("❌ Error getting game state:", error)
      }
    }, 500) // Update every 500ms for more responsive score updates
  }

  stopStateUpdates() {
    console.log("🔄 Stopping state updates...")
    if (this.stateUpdateInterval) {
      clearInterval(this.stateUpdateInterval)
      this.stateUpdateInterval = null
    }
  }

  updateGameUI(state) {
    // Update score
    const scoreElement = document.getElementById("currentScore")
    if (scoreElement && state.score !== undefined) {
      scoreElement.textContent = state.score
    }

    // Update game status
    const statusElement = document.getElementById("gameStatus")
    if (statusElement) {
      if (state.is_running) {
        statusElement.textContent = `Playing ${state.game_id || "game"}... Score: ${state.score || 0}`
      } else {
        statusElement.textContent = "Game stopped"
      }
    }
  }

  async submitScore(score) {
    try {
      console.log("📊 Submitting score:", score)

      const response = await fetch("/api/submit_score", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          game_name: this.currentGame,
          score: score,
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
        console.log("✅ Score submitted successfully")
      }
    } catch (error) {
      console.error("❌ Error submitting score:", error)
      gameraArcade.showNotification("Failed to submit score", "error")
    }
  }

  updateScore(score) {
    const scoreElement = document.getElementById("currentScore")
    if (scoreElement && score !== this.lastScore) {
      scoreElement.textContent = score
      this.lastScore = score

      // Add visual feedback when score increases
      if (score > (this.lastScore || 0)) {
        scoreElement.style.transform = "scale(1.2)"
        scoreElement.style.color = "#4caf50"
        setTimeout(() => {
          scoreElement.style.transform = "scale(1)"
          scoreElement.style.color = ""
        }, 300)
      }
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

// Enhanced Game Control Functions with Python Integration
async function startGame() {
  const gameType = window.location.pathname.split("/").pop()

  try {
    console.log(`🎮 Starting game: ${gameType}`)
    const result = await pythonGameManager.startGame(gameType)

    if (result.success) {
      updateGameUI(true)
      updateGameStatus("Game started! Make swipe gestures to slice fruits!")
    } else {
      updateGameStatus(`Error: ${result.message}`)
    }
  } catch (error) {
    console.error("❌ Error starting game:", error)
    updateGameStatus("Error: Could not start game. Please try again.")
    gameraArcade.showNotification("Could not start game. Please try again.", "error")
  }
}

async function stopGame() {
  try {
    console.log("🛑 Stopping game...")
    const result = await pythonGameManager.stopGame()
    updateGameUI(false)
    updateGameStatus(`Game stopped! Final score: ${result.score || 0}`)
  } catch (error) {
    console.error("❌ Error stopping game:", error)
    updateGameStatus("Error stopping game")
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
    console.log("📝 Status updated:", status)
  }
}

// Initialize the application
const gameraArcade = new GameraArcade()
const pythonGameManager = new PythonGameManager()

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

  console.log("🎮 Gamera Arcade fully loaded!")
  console.log("🐍 Python backend integration ready!")
  console.log("📹 Video streaming should work now!")
})
