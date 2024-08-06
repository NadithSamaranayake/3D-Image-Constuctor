from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['PROCESSED_FOLDER'] = 'static/processed/'

# Function definitions for image processing
def adjust_gamma(image, gamma=1):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def histogram_equalization(image):
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    elif len(image.shape) == 3:  # Color image
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def process_image(image_path):
    negative_image = cv2.imread(image_path)
    positive_image = cv2.bitwise_not(negative_image)
    blurred_image = cv2.GaussianBlur(positive_image, (1, 1), 0)
    bilateral_filtered_image = cv2.bilateralFilter(blurred_image, d=7, sigmaColor=75, sigmaSpace=75)
    gaussian_blurred = cv2.GaussianBlur(bilateral_filtered_image, (0, 0), 3)
    unsharp_image = cv2.addWeighted(bilateral_filtered_image, 1.5, gaussian_blurred, -0.5, 0)
    hsv_image = cv2.cvtColor(unsharp_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s = cv2.add(s, 7)
    v = cv2.add(v, 10)
    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)
    adjusted_hsv_image = cv2.merge([h, s, v])
    color_corrected_image = cv2.cvtColor(adjusted_hsv_image, cv2.COLOR_HSV2BGR)
    gamma_corrected_image = adjust_gamma(color_corrected_image, gamma=0.5)
    median_filtered_image = cv2.medianBlur(gamma_corrected_image, 3)
    alpha = 1.3
    beta = 30
    exposed_image = cv2.convertScaleAbs(median_filtered_image, alpha=alpha, beta=beta)
    lab_image = cv2.cvtColor(exposed_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    equalized_color_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    height, width = equalized_color_image.shape[:2]
    interpolated_image = cv2.resize(equalized_color_image, (2 * width, 2 * height), interpolation=cv2.INTER_LINEAR)
    final_sharpened_image = sharpen_image(interpolated_image)
    equalized_image = histogram_equalization(final_sharpened_image)
    return equalized_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            processed_image = process_image(filepath)
            processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
            cv2.imwrite(processed_filepath, processed_image)
            return render_template('index.html', original_image=filepath, processed_image=processed_filepath)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
