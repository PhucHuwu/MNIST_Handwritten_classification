// Slide navigation system
let currentSlide = 1;
const totalSlides = 15;

// Initialize slide dots
function initializeSlideDots() {
    const dotsContainer = document.getElementById('slide-dots');
    for (let i = 1; i <= totalSlides; i++) {
        const dot = document.createElement('span');
        dot.className = 'dot';
        dot.onclick = () => goToSlide(i);
        dotsContainer.appendChild(dot);
    }
    updateSlideIndicators();
}

// Update slide display
function updateSlide() {
    const iframe = document.getElementById('slide-frame');
    iframe.src = `${currentSlide}.html`;
    
    document.getElementById('current-slide').textContent = currentSlide;
    updateSlideIndicators();
    updateNavigationButtons();
}

// Update slide indicators (dots)
function updateSlideIndicators() {
    const dots = document.querySelectorAll('.dot');
    dots.forEach((dot, index) => {
        if (index + 1 === currentSlide) {
            dot.classList.add('active');
        } else {
            dot.classList.remove('active');
        }
    });
}

// Update navigation button states
function updateNavigationButtons() {
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    
    prevBtn.disabled = currentSlide === 1;
    nextBtn.disabled = currentSlide === totalSlides;
}

// Navigation functions
function nextSlide() {
    if (currentSlide < totalSlides) {
        currentSlide++;
        updateSlide();
    }
}

function previousSlide() {
    if (currentSlide > 1) {
        currentSlide--;
        updateSlide();
    }
}

function goToSlide(slideNumber) {
    if (slideNumber >= 1 && slideNumber <= totalSlides) {
        currentSlide = slideNumber;
        updateSlide();
    }
}

// Keyboard navigation
document.addEventListener('keydown', (event) => {
    switch(event.key) {
        case 'ArrowRight':
        case 'PageDown':
        case ' ': // Space bar
            event.preventDefault();
            nextSlide();
            break;
        case 'ArrowLeft':
        case 'PageUp':
            event.preventDefault();
            previousSlide();
            break;
        case 'Home':
            event.preventDefault();
            goToSlide(1);
            break;
        case 'End':
            event.preventDefault();
            goToSlide(totalSlides);
            break;
    }
});

// Initialize on page load
window.addEventListener('load', () => {
    initializeSlideDots();
    updateSlide();
});