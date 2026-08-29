import os
import werkzeug
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from detector import analyze_video, VIDEO_ALIASES

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
SAMPLE_FOLDER = os.path.join(BASE_DIR, 'sample_videos')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "FORMIQ Biomechanical Analyzer API"})

@app.route('/api/exercises', methods=['GET'])
def get_exercises():
    exercises = list(VIDEO_ALIASES.keys())
    return jsonify({
        "supported_exercises": exercises,
        "aliases": VIDEO_ALIASES
    })

@app.route('/api/samples', methods=['GET'])
def get_sample_videos():
    samples = []
    if os.path.exists(SAMPLE_FOLDER):
        for f in os.listdir(SAMPLE_FOLDER):
            if f.endswith(('.mp4', '.mov', '.avi', '.MP4', '.MOV')):
                samples.append(f)
    return jsonify({"samples": sorted(samples)})

@app.route('/api/sample_video/<filename>', methods=['GET'])
def serve_sample_video(filename):
    filename = werkzeug.utils.secure_filename(filename)
    return send_from_directory(SAMPLE_FOLDER, filename)

@app.route('/api/upload_video/<filename>', methods=['GET'])
def serve_uploaded_video(filename):
    filename = werkzeug.utils.secure_filename(filename)
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    video_path = None

    # Check if user uploaded a file
    if 'video' in request.files and request.files['video'].filename != '':
        file = request.files['video']
        filename = werkzeug.utils.secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        video_url = f"/api/upload_video/{filename}"

    # Or if user selected a sample video
    elif 'sample_name' in request.form and request.form['sample_name'] != '':
        sample_name = werkzeug.utils.secure_filename(request.form['sample_name'])
        video_path = os.path.join(SAMPLE_FOLDER, sample_name)
        video_url = f"/api/sample_video/{sample_name}"

    elif request.is_json and 'sample_name' in request.json:
        sample_name = werkzeug.utils.secure_filename(request.json['sample_name'])
        video_path = os.path.join(SAMPLE_FOLDER, sample_name)
        video_url = f"/api/sample_video/{sample_name}"

    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "No valid video file or sample name provided"}), 400

    try:
        report = analyze_video(video_path)
        if "error" in report:
            return jsonify(report), 400
        
        report["video_url"] = video_url
        return jsonify(report), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n========================================================")
    print("  FORMIQ — Biomechanical Analyzer API Server Running")
    print("  URL: http://127.0.0.1:5000")
    print("========================================================\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
