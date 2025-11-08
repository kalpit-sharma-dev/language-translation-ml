// Main JavaScript for translation UI

const sourceTextarea = document.getElementById('source-text');
const targetTextarea = document.getElementById('target-text');
const translateBtn = document.getElementById('translate-btn');
const clearBtn = document.getElementById('clear-btn');
const beamSizeSelect = document.getElementById('beam-size');
const statusDiv = document.getElementById('status');
const exampleItems = document.querySelectorAll('.example-item');

// Set status message
function setStatus(message, type) {
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
    if (type === 'loading') {
        statusDiv.style.display = 'block';
    }
}

// Clear status
function clearStatus() {
    statusDiv.style.display = 'none';
    statusDiv.className = 'status';
}

// Translate function
async function translate() {
    const text = sourceTextarea.value.trim();
    
    if (!text) {
        setStatus('Please enter some text to translate', 'error');
        return;
    }
    
    setStatus('Translating...', 'loading');
    translateBtn.disabled = true;
    
    try {
        const response = await fetch('/translate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sentence: text,
                beam_size: parseInt(beamSizeSelect.value)
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            targetTextarea.value = data.translation;
            setStatus('Translation completed!', 'success');
            setTimeout(clearStatus, 3000);
        } else {
            setStatus(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        setStatus(`Error: ${error.message}`, 'error');
    } finally {
        translateBtn.disabled = false;
    }
}

// Clear function
function clear() {
    sourceTextarea.value = '';
    targetTextarea.value = '';
    clearStatus();
}

// Event listeners
translateBtn.addEventListener('click', translate);
clearBtn.addEventListener('click', clear);

// Enter key to translate (Ctrl+Enter or Cmd+Enter)
sourceTextarea.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        translate();
    }
});

// Example item click handler
exampleItems.forEach(item => {
    item.addEventListener('click', () => {
        const text = item.getAttribute('data-text');
        sourceTextarea.value = text;
        translate();
    });
});

// Check if model is loaded
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (!data.model_loaded) {
            setStatus('Warning: Model not loaded. Please check server configuration.', 'error');
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// Check health on page load
checkHealth();

