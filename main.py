from flask import Flask, render_template, Response
from faceProejct import gen
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('ip' ,help="input mobile streaming server ip. ex)192.168.0.1:5000")
args = parser.parse_args()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/video_feed')
def video_feed():
    return Response(gen(args.ip),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
