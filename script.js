// DOM elements
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const feeRange = document.getElementById('fee-range');
const currentFee = document.getElementById('current-fee');
const specializationSearch = document.getElementById('specialization-search');
const symptomChips = document.querySelectorAll('.symptom-chip');

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    updateFeeValue();
});

// Update fee value display
feeRange.addEventListener('input', updateFeeValue);

function updateFeeValue() {
    currentFee.textContent = `৳${feeRange.value}`;
}

// Send message on button click
sendButton.addEventListener('click', handleUserMessage);

// Send message on Enter key
userInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        handleUserMessage();
    }
});

// Quick symptom selection
symptomChips.forEach(chip => {
    chip.addEventListener('click', function() {
        userInput.value = this.getAttribute('data-symptom');
        handleUserMessage();
    });
});

// Search doctors by specialization
specializationSearch.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        searchDoctorsBySpecialization(this.value);
    }
});

// Handle user message
async function handleUserMessage() {
    const message = userInput.value.trim();
    if (message === '') return;

    // Add user message to chat
    addMessageToChat(message, 'user');
    
    // Clear input
    userInput.value = '';
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        // Send message to backend for processing
        const response = await fetch('http://localhost:5000/analyze-symptoms', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                max_fee: parseInt(feeRange.value)
            })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator();
        
        if (data.error) {
            addMessageToChat(data.error, 'bot');
        } else {
            // Add bot response
            addMessageToChat(data.response, 'bot');
            
            // Display doctor cards if available
            if (data.doctors && data.doctors.length > 0) {
                showDoctorCards(data.doctors);
            }
        }
    } catch (error) {
        console.error('Error:', error);
        removeTypingIndicator();
        addMessageToChat("I'm having trouble connecting to the server. Please try again later.", 'bot');
    }
}

// Display doctor cards
function showDoctorCards(doctorsList) {
    const cardsContainer = document.createElement('div');
    cardsContainer.className = 'doctor-cards';
    
    doctorsList.forEach(doctor => {
        const card = document.createElement('div');
        card.className = 'doctor-card';
        card.innerHTML = `
            <h4>${doctor.name}</h4>
            <p><i class="fas fa-stethoscope"></i> ${doctor.specialization}</p>
            <p><i class="fas fa-hospital"></i> ${doctor.hospital}</p>
            <p><i class="fas fa-star"></i> Rating: ${doctor.rating}/5</p>
            <p><i class="fas fa-briefcase"></i> Experience: ${doctor.experience}</p>
            <p class="fee"><i class="fas fa-money-bill-wave"></i> Fee: ৳${doctor.fee}</p>
        `;
        cardsContainer.appendChild(card);
    });
    
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.appendChild(cardsContainer);
    
    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Show typing indicator
function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message';
    typingDiv.id = 'typing-indicator';
    
    const typingContent = document.createElement('div');
    typingContent.className = 'typing-indicator';
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'typing-dot';
        typingContent.appendChild(dot);
    }
    
    typingDiv.appendChild(typingContent);
    chatMessages.appendChild(typingDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Remove typing indicator
function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Add message to chat
function addMessageToChat(message, sender) {
    // Remove typing indicator if present
    removeTypingIndicator();
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = `<p>${message}</p>`;
    
    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}