document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.querySelector('.dropzone');
    const fileInput = document.querySelector('#video');
    const uploadForm = document.querySelector('#upload-form');
    const progressBarContainer = document.querySelector('.progress');

    dropzone.addEventListener('dragover', (event) => {
        event.preventDefault();
        dropzone.classList.add('dragging');
    });

    dropzone.addEventListener('dragleave', (event) => {
        event.preventDefault();
        dropzone.classList.remove('dragging');
    });

    dropzone.addEventListener('drop', (event) => {
        event.preventDefault();
        dropzone.classList.remove('dragging');
        fileInput.files = event.dataTransfer.files;
    });

    fileInput.addEventListener('change', () => {
        const filename = fileInput.files[0].name;
        const label = document.querySelector('.custom-file-label');
        label.textContent = filename;
    });

    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        progressBarContainer.style.display = 'block';
        const formData = new FormData(uploadForm);
        const response = await fetch(uploadForm.action, {
            method: 'POST',
            body: formData,
        });
        const result = await response.text();
        window.location.href = result;
    });
});
