from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        r = int(request.form.get('r', 0))
        g = int(request.form.get('g', 0))
        b = int(request.form.get('b', 0))
        flip_option = int(request.form.get('flip', -1))  # -1 for no flip, 0 for vertical, 1 for horizontal

        # Apply color filter
        img = cv2.imread(filepath)
        img = cv2.resize(img, (1280, 700))
        if flip_option in [0, 1]:
            img = cv2.flip(img, flip_option)
        colored_img = apply_color_filter(img, (b, g, r))

        processed_filepath = os.path.join(PROCESSED_FOLDER, f"processed_{file.filename}")
        cv2.imwrite(processed_filepath, colored_img)

        return send_file(processed_filepath, as_attachment=True)

    return "Something went wrong"

def apply_color_filter(image, color):
    color_filter = np.zeros_like(image)
    color_filter[:, :] = color
    filtered_image = cv2.addWeighted(image, 0.5, color_filter, 0.5, 0)
    return filtered_image

if __name__ == '__main__':
    app.run(debug=True)