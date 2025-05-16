from flask import Flask, send_from_directory
from flask_cors import CORS

from utils.constants import *

app = Flask(__name__)
CORS(app)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_ROOT, filename)

@app.route('/texts/<path:filename>')
def serve_text(filename):
    return send_from_directory(ANNOTATION_ROOT, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9091)
