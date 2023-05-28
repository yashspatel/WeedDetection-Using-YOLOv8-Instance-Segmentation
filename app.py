from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_bootstrap import Bootstrap
from flask_uploads import UploadSet, configure_uploads, ALL
from models.your_ml_model import detect_objects
import os
import requirement_filling
from flask import send_file

from pathlib import Path
# Add this function to your app.py to define the custom 'zip' filter
def zip_lists(a, b):
    return zip(a, b)



app = Flask(__name__)
Bootstrap(app)
app.jinja_env.filters['zip'] = zip_lists
videos = UploadSet('videos', ALL)
app.config['UPLOADED_VIDEOS_DEST'] = 'static/uploads'
configure_uploads(app, videos)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'video' in request.files:
        video = request.files['video']
        filename = videos.save(video)

        # Process the video with the object_detection function
        input_video = os.path.join(app.config['UPLOADED_VIDEOS_DEST'], filename)
        output_video = os.path.join('static', 'predict', filename)
        detect_objects(input_video, output_video)

        return redirect(url_for('download', filename=filename))
    return render_template('upload.html')

@app.route('/download/<filename>')
def download(filename):
    return render_template('download.html', filename=filename)

# @app.route('/download_file/<filename>')
# def download_file(filename):
#     return send_from_directory('processed', filename, as_attachment=True)


@app.route('/download_file/<folder>/<filename>')
def download_file(folder, filename):
    # if folder == "uploads":
    folder = os.path.join('static', folder)
    return send_from_directory(folder, filename, as_attachment=True)


@app.route('/history')
def history():
    raw_videos = sorted(Path('static/uploads/').glob('*.mp4'), reverse=True)
    processed_videos = sorted(Path('static/predict/').glob('*.mp4'), reverse=True)
    return render_template('history.html', raw_videos=raw_videos, processed_videos=processed_videos)

# @app.route('/play_processed_video/<filename>')
# def play_processed_video(filename):
#     return send_from_directory('.', os.path.join('processed', filename))
# @app.route('/processed/<filename>')
# def processed(filename):
#     return send_file(os.path.join('processed', filename), as_attachment=False)


if __name__ == '__main__':
    app.run(debug=True)
