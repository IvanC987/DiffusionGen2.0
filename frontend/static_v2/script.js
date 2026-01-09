// Light/Dark Mode Toggle
document.getElementById('theme-toggle').addEventListener('click', function() {
  document.body.classList.toggle('dark-mode');
  localStorage.setItem('theme', document.body.classList.contains('dark-mode') ? 'dark' : 'light');
});

// Load saved theme preference
window.addEventListener('load', function() {
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme === 'dark') {
    document.body.classList.add('dark-mode');
  }
});

// Toggle advanced features with side drawer
const hamburgerButton = document.getElementById('hamburger-button');
const advancedPanel = document.getElementById('advancedPanel');

hamburgerButton.addEventListener('click', () => {
  document.body.classList.toggle('show-advanced');
});

// Close drawer when clicking outside
document.addEventListener('click', function(event) {
  if (!event.target.closest('.hamburger') && !event.target.closest('.advanced-panel')) {
    document.body.classList.remove('show-advanced');
  }
});

// No validation needed for batch_size - it's a select with predefined values

// Check health once on page load
async function checkHealthOnce() {
  try {
    const response = await fetch('/health');
    const data = await response.json();
    console.log('Backend health check:', data.status);
  } catch (e) {
    console.error("Backend not reachable");
  }
}

window.addEventListener('load', checkHealthOnce);

// Convert file to base64
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      // Remove data:image/... prefix
      const b64 = reader.result.split(",")[1];
      resolve(b64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Progress polling function
async function pollProgress(progressBar, progressText) {
  try {
    const res = await fetch("/progress");
    const data = await res.json();
    const pct = data.progress;

    progressBar.value = pct;
    progressText.textContent = pct + "%";

    if (pct < 100) {
      setTimeout(() => pollProgress(progressBar, progressText), 500);
    }
  } catch (e) {
    console.error("Progress polling failed", e);
  }
}

// Preview polling variables and functions (independent from progress polling)
let previewPolling = false;

async function pollPreview(resultImage, previewStepLabel) {
  if (!previewPolling) return;

  try {
    const res = await fetch("/preview");
    const data = await res.json();

    if (data.image) {
      resultImage.src = `data:image/png;base64,${data.image}`;
      resultImage.style.display = "block";
    }

    if (data.step !== null && data.step !== undefined && data.total_steps !== null && data.total_steps !== undefined) {
      previewStepLabel.textContent = `Denoising step ${data.step} / ${data.total_steps}`;
      previewStepLabel.style.display = "block";
    }
  } catch (err) {
    console.error("Preview polling error:", err);
  }

  if (previewPolling) {
    setTimeout(() => pollPreview(resultImage, previewStepLabel), 500);
  }
}

function startPreviewPolling(resultImage, previewStepLabel) {
  previewPolling = true;
  pollPreview(resultImage, previewStepLabel);
}

function stopPreviewPolling() {
  previewPolling = false;
}

// Global variable to store mask from editor
let pendingMaskFromEditor = null;

// Function called by editor window to pass mask data
window.receiveMaskFromEditor = async function(maskDataUrl) {
    pendingMaskFromEditor = maskDataUrl;
    console.log('Mask received from editor');

    // Populate the inpaint_mask file input with the mask
    try {
        const response = await fetch(maskDataUrl);
        const blob = await response.blob();
        const file = new File([blob], 'inpaint_mask.png', { type: 'image/png' });

        // Create a DataTransfer object to set the file input
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);

        // Set the file input's files
        document.getElementById('inpaint_mask').files = dataTransfer.files;
        console.log('Mask file populated in input field');
    } catch (error) {
        console.error('Error setting mask file:', error);
    }
};

// Listen for localStorage changes from editor window
window.addEventListener('storage', function(event) {
    if (event.key === 'pendingInpaintMask' && event.newValue) {
        pendingMaskFromEditor = event.newValue;
        console.log('Mask loaded from editor via localStorage');
    }
});

// Prompt Matching Controls
document.getElementById('use_prompt_matching').addEventListener('change', function() {
  const trainBtn = document.getElementById('prompt_match_train_btn');
  const valBtn = document.getElementById('prompt_match_val_btn');
  const trainInput = document.getElementById('prompt_match_train');
  const valInput = document.getElementById('prompt_match_val');

  if (this.checked) {
    // Enable buttons
    trainBtn.disabled = false;
    valBtn.disabled = false;
    trainBtn.classList.remove('prompt-match-disabled');
    valBtn.classList.remove('prompt-match-disabled');
    trainBtn.classList.add('prompt-match-enabled', 'prompt-match-active');
    valBtn.classList.add('prompt-match-enabled');

    // Set Training Set as checked by default
    trainInput.value = 'true';
    valInput.value = 'false';
  } else {
    // Disable buttons and reset
    trainBtn.disabled = true;
    valBtn.disabled = true;
    trainBtn.classList.remove('prompt-match-enabled', 'prompt-match-active');
    valBtn.classList.remove('prompt-match-enabled', 'prompt-match-active');
    trainBtn.classList.add('prompt-match-disabled');
    valBtn.classList.add('prompt-match-disabled');
    trainInput.value = 'false';
    valInput.value = 'false';
  }
});

// Training Set button
document.getElementById('prompt_match_train_btn').addEventListener('click', function(e) {
  e.preventDefault();
  if (document.getElementById('use_prompt_matching').checked) {
    const trainInput = document.getElementById('prompt_match_train');
    const valInput = document.getElementById('prompt_match_val');
    const trainBtn = document.getElementById('prompt_match_train_btn');
    const valBtn = document.getElementById('prompt_match_val_btn');

    // Set Training Set as selected
    trainInput.value = 'true';
    valInput.value = 'false';

    // Update button classes
    trainBtn.classList.add('prompt-match-active');
    valBtn.classList.remove('prompt-match-active');
  }
});

// Validation Set button
document.getElementById('prompt_match_val_btn').addEventListener('click', function(e) {
  e.preventDefault();
  if (document.getElementById('use_prompt_matching').checked) {
    const trainInput = document.getElementById('prompt_match_train');
    const valInput = document.getElementById('prompt_match_val');
    const trainBtn = document.getElementById('prompt_match_train_btn');
    const valBtn = document.getElementById('prompt_match_val_btn');

    // Set Validation Set as selected
    trainInput.value = 'false';
    valInput.value = 'true';

    // Update button classes
    valBtn.classList.add('prompt-match-active');
    trainBtn.classList.remove('prompt-match-active');
  }
});

// Mutual exclusivity: img2img and in-painting
// If user uploads to img2img, clear in-painting
document.getElementById('init_image').addEventListener('change', function() {
  if (this.files.length > 0) {
    // Clear in-painting inputs
    document.getElementById('inpaint_image').value = '';
    document.getElementById('inpaint_mask').value = '';
    // Clear the editor mask since using img2img instead
    pendingMaskFromEditor = null;
    localStorage.removeItem('pendingInpaintMask');
  }
  const strengthSlider = document.getElementById('img2img_strength');
  strengthSlider.disabled = this.files.length === 0;
});

// Random Prompt Generation Controls
document.getElementById('generate_random_prompt_btn').addEventListener('click', async function(e) {
  e.preventDefault();

  const split = document.getElementById('random_prompt_split').value;
  const category = document.getElementById('random_prompt_category').value;
  const promptInput = document.getElementById('prompt');

  try {
    const response = await fetch(`/random_prompt?split=${split}&category=${encodeURIComponent(category)}`);

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    // Populate the prompt input with the returned prompt
    if (data.prompt) {
      promptInput.value = data.prompt;
      console.log('Random prompt loaded:', data.prompt);
    }
  } catch (error) {
    console.error('Error fetching random prompt:', error);
    alert('Error generating random prompt: ' + error.message);
  }
});

// Training Set button for random prompt
document.getElementById('random_prompt_train_btn').addEventListener('click', function(e) {
  e.preventDefault();
  const splitInput = document.getElementById('random_prompt_split');
  const trainBtn = document.getElementById('random_prompt_train_btn');
  const valBtn = document.getElementById('random_prompt_val_btn');

  splitInput.value = 'train';

  trainBtn.classList.add('random-prompt-active');
  valBtn.classList.remove('random-prompt-active');
});

// Validation Set button for random prompt
document.getElementById('random_prompt_val_btn').addEventListener('click', function(e) {
  e.preventDefault();
  const splitInput = document.getElementById('random_prompt_split');
  const trainBtn = document.getElementById('random_prompt_train_btn');
  const valBtn = document.getElementById('random_prompt_val_btn');

  splitInput.value = 'val';

  valBtn.classList.add('random-prompt-active');
  trainBtn.classList.remove('random-prompt-active');
});

// If user uploads to in-painting image, clear img2img
document.getElementById('inpaint_image').addEventListener('change', function() {
  if (this.files.length > 0) {
    // Clear img2img inputs
    document.getElementById('init_image').value = '';
    document.getElementById('img2img_strength').disabled = true;
    // Keep the editor mask since user is setting up in-painting workflow
  }
});

// If user uploads to in-painting mask, clear img2img and editor mask
document.getElementById('inpaint_mask').addEventListener('change', function() {
  if (this.files.length > 0) {
    // Clear img2img inputs
    document.getElementById('init_image').value = '';
    document.getElementById('img2img_strength').disabled = true;
    // Clear the editor mask since user is manually uploading
    pendingMaskFromEditor = null;
    localStorage.removeItem('pendingInpaintMask');
  }
});

// Generate Form Submission
document.getElementById('generateForm').addEventListener('submit', async function(event) {
  event.preventDefault();

  const generateBtn = document.querySelector('.generate-btn');

  // Disable button during generation
  generateBtn.disabled = true;
  generateBtn.textContent = 'Generating...';

  // Show progress bar
  const progressBar = document.getElementById('progressBar');
  const progressText = document.getElementById('progressText');
  const previewStepLabel = document.getElementById('previewStepLabel');
  const resultImage = document.getElementById('resultImage');
  const usedPromptContainer = document.getElementById('usedPromptContainer');
  const realTimeDenoisingEnabled = document.getElementById('real_time_denoising').checked;

  progressBar.style.display = 'block';
  progressText.style.display = 'block';
  progressBar.value = 0;
  progressText.textContent = '0%';
  previewStepLabel.style.display = 'none';
  previewStepLabel.textContent = '';
  usedPromptContainer.style.display = 'none';

  try {
    // Handle img2img image upload
    const initImageInput = document.getElementById('init_image');
    let b64Image = null;

    if (initImageInput.files.length > 0) {
      b64Image = await fileToBase64(initImageInput.files[0]);
    }

    // Handle in-painting uploads
    const inpaintImageInput = document.getElementById('inpaint_image');
    const inpaintMaskInput = document.getElementById('inpaint_mask');
    let b64InpaintImage = null;
    let b64InpaintMask = null;

    // Validate in-painting: both image and mask must be provided together
    const hasInpaintImage = inpaintImageInput.files.length > 0;
    const hasInpaintMask = inpaintMaskInput.files.length > 0;

    if (hasInpaintImage && !hasInpaintMask) {
      generateBtn.disabled = false;
      generateBtn.textContent = 'Generate';
      progressBar.style.display = 'none';
      progressText.style.display = 'none';
      alert('Please upload both an in-painting image AND a mask, or provide neither.');
      return;
    }

    if (hasInpaintMask && !hasInpaintImage) {
      generateBtn.disabled = false;
      generateBtn.textContent = 'Generate';
      progressBar.style.display = 'none';
      progressText.style.display = 'none';
      alert('Please upload both an in-painting image AND a mask, or provide neither.');
      return;
    }

    if (inpaintImageInput.files.length > 0) {
      b64InpaintImage = await fileToBase64(inpaintImageInput.files[0]);
    }

    if (inpaintMaskInput.files.length > 0) {
      b64InpaintMask = await fileToBase64(inpaintMaskInput.files[0]);
    } else if (pendingMaskFromEditor) {
      // Use the mask created in the editor if no file was manually uploaded
      b64InpaintMask = pendingMaskFromEditor.split(',')[1];
    }

    // Build JSON payload
    const payload = {
      prompt: document.getElementById('prompt').value,
      second_prompt: document.getElementById('second_prompt').value || '',
      negative_prompt: document.getElementById('negative_prompt').value,
      steps: Number(document.getElementById('steps').value),
      cfg: Number(document.getElementById('cfg').value),
      seed: document.getElementById('seed').value === '' ? null : Number(document.getElementById('seed').value),
      batch_size: Number(document.getElementById('batch_size').value),
      use_ema: document.getElementById('use_ema').checked,
      use_ddpm: document.getElementById('sampling_method').value === 'ddpm',
      use_real_esrgan: document.getElementById('use_real_esrgan').checked,
      real_time_denoising: realTimeDenoisingEnabled,
      preview_interval: Number(document.getElementById('preview_interval').value),
      b64_image: b64Image,
      img2img_strength: Number(document.getElementById('img2img_strength').value),
      b64_inpaint_image: b64InpaintImage,
      b64_inpaint_mask: b64InpaintMask,
      use_prompt_matching: document.getElementById('use_prompt_matching').checked,
      prompt_match_train: document.getElementById('prompt_match_train').value === 'true',
      prompt_match_val: document.getElementById('prompt_match_val').value === 'true'
    };

    // --- FIX: Start polling BEFORE the blocking await ---
    // This ensures the progress bar updates while the fetch is waiting
    pollProgress(progressBar, progressText);

    // Start preview polling if real-time denoising is enabled
    if (realTimeDenoisingEnabled) {
      startPreviewPolling(resultImage, previewStepLabel);
    }
    // -----------------------------------------------------------

    // Send request to backend (This blocks JS execution until done)
    const response = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();

    const resultImage = document.getElementById('resultImage');
    const placeholder = document.getElementById('imagePlaceholder');
    const downloadButton = document.getElementById('downloadButton');
    const prevButton = document.getElementById('prevButton');
    const nextButton = document.getElementById('nextButton');
    const imageCounter = document.getElementById('imageCounter');
    const usedPromptContainer = document.getElementById('usedPromptContainer');
    const usedPromptText = document.getElementById('usedPromptText');

    // Display the used prompt (if prompt matching was used)
    if (result.prompt) {
      usedPromptText.textContent = result.prompt;
      usedPromptContainer.style.display = 'block';
    } else {
      usedPromptContainer.style.display = 'none';
    }

    // Hide placeholder and show generated image
    placeholder.style.display = 'none';

    // Store all images and set current index
    if (result.images && Array.isArray(result.images)) {
      resultImage.dataset.allImages = JSON.stringify(result.images);
      resultImage.dataset.currentIndex = '0';

      // Display first image
      resultImage.src = `data:image/png;base64,${result.images[0]}`;
      resultImage.style.display = 'block';

      // Setup navigation
      if (result.images.length > 1) {
        imageCounter.textContent = `1/${result.images.length}`;
        prevButton.classList.add('active');
        nextButton.classList.add('active');
      } else {
        imageCounter.textContent = '';
        prevButton.classList.remove('active');
        nextButton.classList.remove('active');
      }
    }

    // Show download button
    downloadButton.style.display = 'block';

    // Hide progress
    progressBar.style.display = 'none';
    progressText.style.display = 'none';

    // Stop preview polling
    stopPreviewPolling();

  } catch (error) {
    console.error('Error generating image:', error);
    alert('Error generating image: ' + error.message);

    // Hide progress on error
    progressBar.style.display = 'none';
    progressText.style.display = 'none';

    // Stop preview polling on error
    stopPreviewPolling();
  } finally {
    // Re-enable button
    generateBtn.disabled = false;
    generateBtn.textContent = 'Generate';
  }
});

// Download button functionality
document.getElementById('downloadButton').addEventListener('click', function() {
  const resultImage = document.getElementById('resultImage');
  const allImagesStr = resultImage.dataset.allImages;
  const currentIndex = parseInt(resultImage.dataset.currentIndex);

  if (!allImagesStr) {
    alert('No images to download');
    return;
  }

  const allImages = JSON.parse(allImagesStr);
  const currentImage = allImages[currentIndex];

  const a = document.createElement('a');
  a.href = `data:image/png;base64,${currentImage}`;
  a.download = `generated_image_${currentIndex + 1}.png`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
});

// Navigation arrow buttons
document.getElementById('prevButton').addEventListener('click', function() {
  const resultImage = document.getElementById('resultImage');
  const allImagesStr = resultImage.dataset.allImages;
  let currentIndex = parseInt(resultImage.dataset.currentIndex);

  if (!allImagesStr) return;

  const allImages = JSON.parse(allImagesStr);
  currentIndex = (currentIndex - 1 + allImages.length) % allImages.length;

  resultImage.src = `data:image/png;base64,${allImages[currentIndex]}`;
  resultImage.dataset.currentIndex = currentIndex;

  document.getElementById('imageCounter').textContent = `${currentIndex + 1}/${allImages.length}`;
});

document.getElementById('nextButton').addEventListener('click', function() {
  const resultImage = document.getElementById('resultImage');
  const allImagesStr = resultImage.dataset.allImages;
  let currentIndex = parseInt(resultImage.dataset.currentIndex);

  if (!allImagesStr) return;

  const allImages = JSON.parse(allImagesStr);
  currentIndex = (currentIndex + 1) % allImages.length;

  resultImage.src = `data:image/png;base64,${allImages[currentIndex]}`;
  resultImage.dataset.currentIndex = currentIndex;

  document.getElementById('imageCounter').textContent = `${currentIndex + 1}/${allImages.length}`;
});

// Seed input - allow empty for random seed
document.getElementById('seed').addEventListener('input', function() {
  const val = this.value;
  if (val === '') {
    this.value = '';  // Allow empty
  }
});

// CFG Scale validation
document.getElementById('cfg').addEventListener('input', function() {
  let val = parseFloat(this.value);
  if (val < 0) this.value = 0;
  if (val > 20) this.value = 20;
});

// Steps validation
document.getElementById('steps').addEventListener('input', function() {
  let val = parseInt(this.value);
  if (val < 1) this.value = 1;
  if (val > 1000) this.value = 1000;
});

// Batch size validation
document.getElementById('batch_size').addEventListener('input', function() {
  let val = parseInt(this.value);
  if (val < 1) this.value = 1;
  if (val > 10) this.value = 10;
});

// Real time denoising toggle - enable/disable preview interval input
document.getElementById('real_time_denoising').addEventListener('change', function() {
  const previewIntervalInput = document.getElementById('preview_interval');
  previewIntervalInput.disabled = !this.checked;
});

// Initialize preview interval state on page load
window.addEventListener('load', function() {
  const realTimeDenoising = document.getElementById('real_time_denoising');
  const previewIntervalInput = document.getElementById('preview_interval');
  previewIntervalInput.disabled = !realTimeDenoising.checked;

  // Initialize img2img strength slider state
  const initImageInput = document.getElementById('init_image');
  const strengthSlider = document.getElementById('img2img_strength');
  strengthSlider.disabled = initImageInput.files.length === 0;
});

// Image upload handler - enable/disable strength slider
document.getElementById('init_image').addEventListener('change', function() {
  const strengthSlider = document.getElementById('img2img_strength');
  strengthSlider.disabled = this.files.length === 0;
});

// Update strength value display
document.getElementById('img2img_strength').addEventListener('input', function() {
  document.getElementById('strengthValue').textContent = this.value;
});
