// Main JavaScript for translation UI

const sourceTextarea = document.getElementById('source-text');
const targetTextarea = document.getElementById('target-text');
const translateBtn = document.getElementById('translate-btn');
const clearBtn = document.getElementById('clear-btn');
const beamSizeSelect = document.getElementById('beam-size');
const languageSelect = document.getElementById('language-select');
const targetLabel = document.getElementById('target-label');
const statusDiv = document.getElementById('status');
const exampleItems = document.querySelectorAll('.example-item');

let currentLanguage = null;
let availableLanguages = [];

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
    const selectedLanguage = languageSelect.value;  // Get current selection
    
    if (!text) {
        setStatus('Please enter some text to translate', 'error');
        return;
    }
    
    if (!selectedLanguage) {
        setStatus('Please select a target language', 'error');
        return;
    }
    
    // Ensure language is switched before translating
    if (selectedLanguage !== currentLanguage) {
        const switched = await switchLanguage(selectedLanguage);
        if (!switched) {
            setStatus('Failed to switch language. Please try again.', 'error');
            translateBtn.disabled = false;
            return;
        }
    }
    
    // Double-check we're using the correct language from dropdown
    const languageToUse = languageSelect.value || selectedLanguage;
    console.log('Translating with language:', languageToUse);
    
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
                beam_size: parseInt(beamSizeSelect.value),
                language: languageToUse  // Always use the selected language from dropdown
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            targetTextarea.value = data.translation;
            currentLanguage = data.language;
            updateTargetLabel();
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

// Update target label based on selected language
function updateTargetLabel() {
    if (currentLanguage && availableLanguages.length > 0) {
        const lang = availableLanguages.find(l => l.code === currentLanguage);
        if (lang) {
            targetLabel.textContent = `${lang.flag} ${lang.name} Translation:`;
        } else {
            targetLabel.textContent = 'Translation:';
        }
    } else {
        targetLabel.textContent = 'Translation:';
    }
}

// Load available languages
async function loadLanguages() {
    try {
        const response = await fetch('/languages');
        const data = await response.json();
        
        availableLanguages = data.languages;
        currentLanguage = data.current;
        
        // Update language selector
        languageSelect.innerHTML = '';
        if (availableLanguages.length > 0) {
            availableLanguages.forEach(lang => {
                const option = document.createElement('option');
                option.value = lang.code;
                option.textContent = `${lang.flag} ${lang.name}`;
                if (lang.code === currentLanguage) {
                    option.selected = true;
                }
                languageSelect.appendChild(option);
            });
            updateTargetLabel();
        } else {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No languages available';
            languageSelect.appendChild(option);
            setStatus('No models loaded. Please configure models first.', 'error');
        }
    } catch (error) {
        console.error('Failed to load languages:', error);
    }
}

// Switch language
async function switchLanguage(languageCode) {
    if (!languageCode) return false;
    
    // If already on this language, no need to switch
    if (currentLanguage === languageCode && availableLanguages.find(l => l.code === languageCode)?.loaded) {
        return true;
    }
    
    try {
        const response = await fetch('/switch_language', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                language: languageCode
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentLanguage = data.language;
            updateTargetLabel();
            // Clear previous translation
            targetTextarea.value = '';
            setStatus(`Switched to ${data.name}`, 'success');
            setTimeout(clearStatus, 2000);
            return true;
        } else {
            let errorMsg = data.error || 'Unknown error';
            if (data.help) {
                errorMsg += `\n${data.help}`;
            }
            setStatus(errorMsg, 'error');
            return false;
        }
    } catch (error) {
        setStatus(`Error: ${error.message}`, 'error');
        return false;
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

// Language selector change
languageSelect.addEventListener('change', (e) => {
    const selectedLang = e.target.value;
    if (selectedLang) {
        switchLanguage(selectedLang);
    }
});

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
        
        if (data.models_loaded && data.models_loaded.length === 0) {
            setStatus('Warning: No models loaded. Please check server configuration.', 'error');
        }
    } catch (error) {
        console.error('Health check failed:', error);
    }
}

// Initialize on page load
async function initialize() {
    await loadLanguages();
    checkHealth();
}

// Initialize when page loads
initialize();

