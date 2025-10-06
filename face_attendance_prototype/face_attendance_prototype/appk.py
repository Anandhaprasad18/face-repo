from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for
import cv2
import os
import face_recognition
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for flashing messages

# Paths
DATA_DIR = "data"
UPLOAD_DIR = "static/uploads"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------- Utility Functions ----------

def pose_score(image):
    """Return a score for face pose (higher = more frontal & clear)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks_list = face_recognition.face_landmarks(rgb)
    if not landmarks_list:
        return -1, "No face detected"

    lm = landmarks_list[0]

    # Check for key facial features
    required_landmarks = ['left_eye', 'right_eye', 'nose_bridge']
    for landmark in required_landmarks:
        if landmark not in lm:
            return -1, f"Missing {landmark}"

    # 1. Eye alignment (horizontal balance)
    left_eye = np.mean(lm['left_eye'], axis=0)
    right_eye = np.mean(lm['right_eye'], axis=0)
    eye_diff_y = abs(left_eye[1] - right_eye[1])
    
    # 2. Head tilt (angle between eyes)
    eye_angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    # 3. Nose position (centrality)
    nose_bridge = lm['nose_bridge']
    nose_tip = nose_bridge[len(nose_bridge) - 1]
    face_center_x = (left_eye[0] + right_eye[0]) / 2
    nose_deviation = abs(nose_tip[0] - face_center_x)

    # 4. Image clarity (blurriness detection)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:  # Threshold for blurriness
        return -1, "Image too blurry"

    # 5. Face size (ensure face is not too small)
    face_locations = face_recognition.face_locations(rgb)
    if not face_locations:
        return -1, "No face detected"
    top, right, bottom, left = face_locations[0]
    face_area = (right - left) * (bottom - top)
    if face_area < 5000:  # Threshold for face size
        return -1, "Face too small"

    # Calculate score (lower is better for these metrics)
    # We negate the score so that a higher final score is better
    score = -(eye_diff_y + abs(eye_angle) + nose_deviation) + laplacian_var + face_area
    
    return score, "Good"


def select_best_pose(images):
    """Pick the best pose image from a list of images."""
    best_img = None
    best_score = -float('inf')
    
    for img in images:
        score, reason = pose_score(img)
        if score > best_score:
            best_score = score
            best_img = img
            
    return best_img, best_score


# ---------- Flask Routes ----------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload_best_pose', methods=['POST'])
def upload_best_pose():
    if 'photos' not in request.files:
        flash('No file part')
        return redirect(request.url)

    uploaded_files = request.files.getlist("photos")
    images = []

    if len(uploaded_files) < 5:
        flash("Please upload at least 5 images.")
        return redirect(url_for('home'))

    for file in uploaded_files:
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        # Read image
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    if len(images) == 0:
        flash("No valid images were uploaded. Please try again.")
        return redirect(url_for('home'))

    # Select best pose
    best_img, best_score = select_best_pose(images)

    if best_img is None:
        flash("Could not determine the best pose. Please try with clearer images.")
        return redirect(url_for('home'))

    # Save best image
    person_id = "person_" + datetime.now().strftime("%Y%m%d%H%M%S")
    person_dir = os.path.join(DATA_DIR, person_id)
    os.makedirs(person_dir, exist_ok=True)

    best_filename = f"best_pose_{person_id}.jpg"
    best_path = os.path.join(UPLOAD_DIR, best_filename)
    cv2.imwrite(best_path, best_img)

    flash(f"Successfully selected the best pose and saved as {best_filename}")
    return render_template("result.html", best_image=best_filename, score=f"{best_score:.2f}")


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
