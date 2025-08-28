/*
==============================================
Gamera Arcade - Theme & Functionality
==============================================
*/

// Theme Management
class GameraTheme {
  constructor() {
    this.currentTheme = localStorage.getItem("gamera-theme") || "dark";
    this.init();
  }

  init() {
    this.applyTheme(this.currentTheme);
    this.bindEvents();
    this.updateCoins();
  }

  applyTheme(theme) {
    if (theme === "light") {
      document.documentElement.setAttribute("data-theme", "light");
      document.getElementById("theme-switch").checked = true;
    } else {
      document.documentElement.removeAttribute("data-theme");
      document.getElementById("theme-switch").checked = false;
    }
    this.currentTheme = theme;
    localStorage.setItem("gamera-theme", theme);
  }

  toggleTheme() {
    const newTheme = this.currentTheme === "dark" ? "light" : "dark";
    this.applyTheme(newTheme);

    // Add smooth transition effect
    document.body.style.transition = "all 0.3s ease";
    setTimeout(() => {
      document.body.style.transition = "";
    }, 300);
  }

  bindEvents() {
    const themeSwitch = document.getElementById("theme-switch");
    if (themeSwitch) {
      themeSwitch.addEventListener("change", () => {
        this.toggleTheme();
      });
    }
  }

  // Update user coins display
  updateCoins() {
    const coinsElement = document.getElementById("user-coins");
    if (coinsElement) {
      fetch("/api/user/coins")
        .then((response) => response.json())
        .then((data) => {
          coinsElement.textContent = data.coins;
        })
        .catch((error) => console.log("Could not update coins:", error));
    }
  }
}

// Wallet Connection
function connectWallet() {
  const modal = document.getElementById("walletModal");
  if (modal) {
    modal.style.display = "block";
    modal.classList.add("show");
  }
}

function closeWalletModal() {
  const modal = document.getElementById("walletModal");
  if (modal) {
    modal.style.display = "none";
    modal.classList.remove("show");
  }
}

function connectMetaMask() {
  if (typeof window.ethereum !== "undefined") {
    window.ethereum
      .request({ method: "eth_requestAccounts" })
      .then((accounts) => {
        const walletAddress = accounts[0];

        // Send wallet address to backend
        fetch("/api/connect_wallet", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ wallet_address: walletAddress }),
        })
          .then((response) => response.json())
          .then((data) => {
            if (data.success) {
              showFlashMessage("Wallet connected successfully!", "success");
              closeWalletModal();
            } else {
              showFlashMessage("Failed to connect wallet.", "error");
            }
          })
          .catch((error) => {
            console.error("Error:", error);
            showFlashMessage("Error connecting wallet.", "error");
          });
      })
      .catch((error) => {
        console.error("Error connecting to MetaMask:", error);
        showFlashMessage(
          "Please install MetaMask to connect your wallet.",
          "error"
        );
      });
  } else {
    showFlashMessage(
      "Please install MetaMask to connect your wallet.",
      "error"
    );
  }
}

// Flash Message System
function showFlashMessage(message, category = "success") {
  const flashContainer =
    document.querySelector(".flash-messages-cyborg") || createFlashContainer();

  const flashDiv = document.createElement("div");
  flashDiv.className = `flash-message flash-${category}`;
  flashDiv.innerHTML = `
        <div class="container">
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" class="flash-close">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;

  flashContainer.appendChild(flashDiv);

  // Auto-remove after 5 seconds
  setTimeout(() => {
    if (flashDiv.parentNode) {
      flashDiv.remove();
    }
  }, 5000);
}

function createFlashContainer() {
  const container = document.createElement("div");
  container.className = "flash-messages-cyborg";
  document.body.appendChild(container);
  return container;
}

// Enhanced Game Card Interactions
function enhanceGameCards() {
  const gameCards = document.querySelectorAll(".item");
  gameCards.forEach((card) => {
    card.addEventListener("mouseenter", function () {
      this.style.transform = "translateY(-5px)";
      this.style.transition = "all 0.3s ease";
    });

    card.addEventListener("mouseleave", function () {
      this.style.transform = "translateY(0)";
    });
  });
}

// Smooth Scrolling for Navigation
function initSmoothScrolling() {
  const navLinks = document.querySelectorAll('a[href^="#"]');
  navLinks.forEach((link) => {
    link.addEventListener("click", function (e) {
      e.preventDefault();
      const targetId = this.getAttribute("href").substring(1);
      const targetElement = document.getElementById(targetId);

      if (targetElement) {
        targetElement.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    });
  });
}

// Performance Monitoring for OpenCV
function monitorPerformance() {
  if (window.performance && window.performance.memory) {
    const memory = window.performance.memory;
    console.log("Memory Usage:", {
      used: Math.round(memory.usedJSHeapSize / 1048576) + " MB",
      total: Math.round(memory.totalJSHeapSize / 1048576) + " MB",
      limit: Math.round(memory.jsHeapSizeLimit / 1048576) + " MB",
    });
  }
}

// Game Integration Functions
function startGame(gameId) {
  fetch(`/api/game/start/${gameId}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        showFlashMessage(`Game ${gameId} started successfully!`, "success");
        // Redirect to game page
        window.location.href = `/game/${gameId}`;
      } else {
        showFlashMessage(data.message || "Failed to start game", "error");
      }
    })
    .catch((error) => {
      console.error("Error starting game:", error);
      showFlashMessage("Error starting game", "error");
    });
}

function stopGame() {
  fetch("/api/game/stop", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.success) {
        showFlashMessage("Game stopped", "success");
      } else {
        showFlashMessage(data.message || "Failed to stop game", "error");
      }
    })
    .catch((error) => {
      console.error("Error stopping game:", error);
      showFlashMessage("Error stopping game", "error");
    });
}

// Fix Cyborg Header Scroll Behavior
function fixHeaderBehavior() {
  // Override the cyborg scroll behavior to keep header always visible
  if (typeof $ !== 'undefined') {
    $(window).off('scroll'); // Remove existing scroll handlers
    
    $(window).scroll(function() {
      var scroll = $(window).scrollTop();
      
      if (scroll >= 100) {
        $("header").addClass("background-header");
      } else {
        $("header").removeClass("background-header");
      }
    });
  }
}

// Enhanced Navigation Functions
function initEnhancedNavigation() {
  // Custom dropdown functionality (no Bootstrap dependency)
  const profileDropdown = document.getElementById('profileDropdown');
  const dropdownMenu = document.getElementById('profileDropdownMenu');
  
  if (profileDropdown && dropdownMenu) {
    // Close dropdown when clicking outside
    document.addEventListener('click', function(event) {
      if (!profileDropdown.contains(event.target) && !dropdownMenu.contains(event.target)) {
        dropdownMenu.classList.remove('show');
        profileDropdown.setAttribute('aria-expanded', 'false');
      }
    });
  }

  // Enhanced coins click animation
  const coinsElement = document.querySelector('.nav-coins');
  if (coinsElement) {
    coinsElement.addEventListener('click', function() {
      this.style.transform = 'scale(0.95)';
      setTimeout(() => {
        this.style.transform = 'scale(1.05)';
        setTimeout(() => {
          this.style.transform = 'scale(1)';
        }, 150);
      }, 100);
    });
  }

  // Mobile menu toggle
  const menuTrigger = document.querySelector('.menu-trigger');
  if (menuTrigger) {
    menuTrigger.addEventListener('click', toggleMobileMenu);
  }

  // Close dropdowns when clicking outside
  document.addEventListener('click', function(event) {
    const dropdowns = document.querySelectorAll('.dropdown-menu.show');
    dropdowns.forEach(dropdown => {
      if (!dropdown.closest('.dropdown').contains(event.target)) {
        const dropdownInstance = bootstrap.Dropdown.getInstance(dropdown.previousElementSibling);
        if (dropdownInstance) {
          dropdownInstance.hide();
        }
      }
    });
  });
}

function toggleMobileMenu() {
  const mobileMenu = document.querySelector('.mobile-menu');
  const overlay = document.querySelector('.mobile-menu-overlay');
  
  if (mobileMenu && overlay) {
    mobileMenu.classList.toggle('active');
    overlay.style.display = mobileMenu.classList.contains('active') ? 'block' : 'none';
  }
}

// Profile Dropdown Toggle Function
function toggleProfileDropdown(event) {
  event.preventDefault();
  event.stopPropagation();
  
  const dropdownMenu = document.getElementById('profileDropdownMenu');
  const profileDropdown = document.getElementById('profileDropdown');
  
  if (dropdownMenu && profileDropdown) {
    const isOpen = dropdownMenu.classList.contains('show');
    
    if (isOpen) {
      dropdownMenu.classList.remove('show');
      profileDropdown.setAttribute('aria-expanded', 'false');
    } else {
      dropdownMenu.classList.add('show');
      profileDropdown.setAttribute('aria-expanded', 'true');
    }
  }
}

// Enhanced Profile Functions
function updateProfileInfo() {
  fetch('/api/user/profile')
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        const profileImg = document.querySelector('.profile-img');
        const profileName = document.querySelector('.profile-name');
        
        if (profileImg && data.profile_image) {
          profileImg.src = data.profile_image;
        }
        
        if (profileName && data.username) {
          profileName.textContent = data.username;
        }
      }
    })
    .catch(error => console.log('Could not update profile info:', error));
}

// Initialize everything when DOM is loaded
document.addEventListener("DOMContentLoaded", function () {
  // Initialize theme system
  window.gameraTheme = new GameraTheme();

  // Initialize enhanced navigation
  initEnhancedNavigation();

  // Update profile info
  updateProfileInfo();

  // Fix header behavior after cyborg JS loads
  setTimeout(() => {
    fixHeaderBehavior();
  }, 1000);

  // Enhance UI interactions
  enhanceGameCards();
  initSmoothScrolling();

  // Monitor performance every 30 seconds
  setInterval(monitorPerformance, 30000);

  // Close modal when clicking outside
  window.addEventListener("click", function (event) {
    const modal = document.getElementById("walletModal");
    if (event.target === modal) {
      closeWalletModal();
    }
  });

  // Initialize cyborg preloader
  setTimeout(() => {
    const preloader = document.getElementById("js-preloader");
    if (preloader) {
      preloader.classList.add("loaded");
    }
  }, 1000);

  console.log("🎮 Gamera Arcade - Enhanced Navigation Initialized");
});

// Export functions for global access
window.connectWallet = connectWallet;
window.closeWalletModal = closeWalletModal;
window.connectMetaMask = connectMetaMask;
window.showFlashMessage = showFlashMessage;
window.startGame = startGame;
window.stopGame = stopGame;
window.toggleMobileMenu = toggleMobileMenu;
window.updateProfileInfo = updateProfileInfo;
window.toggleProfileDropdown = toggleProfileDropdown;
