document.addEventListener('DOMContentLoaded', function() {
    // File input validation and preview
    const videoInput = document.getElementById('video-input');
    const videoPreview = document.getElementById('video-preview');
    const videoPreviewContainer = document.getElementById('video-preview-container');
    const videoForm = document.getElementById('upload-form');
    const uploadButton = document.getElementById('upload-button');
    const fileNameDisplay = document.getElementById('file-name');
    const progressBar = document.getElementById('upload-progress');
    const progressContainer = document.getElementById('progress-container');

    if (videoInput) {
        videoInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (!file) {
                return;
            }
            
            // Display file name
            if (fileNameDisplay) {
                fileNameDisplay.textContent = file.name;
            }
            
            // Check file type
            const fileType = file.type;
            if (!fileType.startsWith('video/')) {
                alert('Please select a video file.');
                videoInput.value = '';
                return;
            }
            
            // Check file size (max 50MB)
            const maxSize = 50 * 1024 * 1024; // 50MB in bytes
            if (file.size > maxSize) {
                alert('File size exceeds 50MB limit.');
                videoInput.value = '';
                return;
            }
            
            // Display video preview
            if (videoPreview && videoPreviewContainer) {
                const videoURL = URL.createObjectURL(file);
                videoPreview.src = videoURL;
                videoPreviewContainer.classList.remove('d-none');
                
                // Enable upload button if it exists
                if (uploadButton) {
                    uploadButton.disabled = false;
                }
            }
        });
    }
    
    // Handle form submission with progress
    if (videoForm) {
        videoForm.addEventListener('submit', function(event) {
            const file = videoInput.files[0];
            if (!file) {
                event.preventDefault();
                alert('Please select a video file first.');
                return;
            }
            
            // Show progress bar
            if (progressContainer) {
                progressContainer.classList.remove('d-none');
            }
            
            // Disable upload button to prevent multiple submissions
            if (uploadButton) {
                uploadButton.disabled = true;
                uploadButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Uploading...';
            }
            
            // Note: In a real implementation, you might want to use fetch with FormData to track upload progress
            // For simplicity, we'll just let the form submit normally
        });
    }
    
    // Results page visualization
    const resultCanvas = document.getElementById('result-chart');
    if (resultCanvas) {
        const realProb = parseFloat(resultCanvas.getAttribute('data-real')) || 0;
        const fakeProb = parseFloat(resultCanvas.getAttribute('data-fake')) || 0;
        
        // Create a chart using Chart.js
        const ctx = resultCanvas.getContext('2d');
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Real', 'Fake'],
                datasets: [{
                    data: [realProb * 100, fakeProb * 100],
                    backgroundColor: [
                        'rgba(75, 192, 192, 0.7)',
                        'rgba(255, 99, 132, 0.7)'
                    ],
                    borderColor: [
                        'rgba(75, 192, 192, 1)',
                        'rgba(255, 99, 132, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                cutout: '70%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#ffffff'
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.parsed.toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Enable tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    if (tooltipTriggerList.length > 0) {
        const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    }
});
