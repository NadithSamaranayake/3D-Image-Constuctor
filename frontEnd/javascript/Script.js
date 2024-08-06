alert('js is working');
document.getElementById('uploadForm').addEventListener('submit', async function (event) {
    event.preventDefault(); // Prevents the page from reloading

    const formData = new FormData();
    const imageInput = document.getElementById('imageInput');

    if (imageInput.files.length > 0) {
        formData.append('image', imageInput.files[0]);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                const imgElement = document.createElement('img');
                imgElement.src = result.imageUrl;
                document.getElementById('imageDisplay').appendChild(imgElement);
            } else {
                console.error('Failed to upload image.');
                alert('Failed to upload image. Please try again.');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while uploading the image. Please try again.');
        }
    } else {
        alert('Please select an image to upload.');
    }
});
