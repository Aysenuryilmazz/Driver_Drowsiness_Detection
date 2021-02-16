import dlib
from flask import Flask, Response
import cv2
from flask_socketio import SocketIO, send
from landmark_detection import dlib_detect, process_video, get_global_variable
from time import sleep
import pickle 
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
socketIo = SocketIO(app, cors_allowed_origins="*")

app.host = 'http://127.0.0.1'

video = cv2.VideoCapture(0)

dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor("input/shape_predictor_68_face_landmarks.dat")

scaler = pickle.load(open('input/scaler.sav', 'rb'))
subject_wise_scaler = StandardScaler()
model = keras.models.load_model('input/lstm_vanilla.h5')
# sk_model = joblib.load(open('input/extra_best_estimator_compressed.pkl', 'rb'))

# buffer = [[0.3, 0.3, 0.3, 0.3, 0.3,0.3,0.3,0.3],[0.3, 0.3, 0.3, 0.3, 0.3,0.3,0.3,0.3]]
# subject_wise_scaler.fit(np.array(buffer))
# subject_scaled = subject_wise_scaler.transform(np.array([0.3, 0.3, 0.3, 0.3, 0.3,0.3,0.3,0.3]).reshape(1,8))

@app.route('/')
def index():
    return "Default Message"
def gen(vs):

    dlib_detect(vs)

    # while True:
    #     success, image = vs.read()
    #
    #
    #
    #     ret, jpeg = cv2.imencode('.jpg', image)
    #     frame = jpeg.tobytes()
    #     yield (b'--frame\r\n'
    #            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/video_feed')
def video_feed():
    global video
    return Response(process_video(vs=video, detector=dlib_detector, predictor=dlib_predictor, scaler=scaler, subject_wise_scaler=subject_wise_scaler, model=model,  ear_th=0.21, consec_th=3),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@socketIo.on('message')
def handle_json(json):

    print(json)

    while 1:

        dict = get_global_variable()

        print(dict)

        sleep(0.2)

        send(dict, broadcast=True)

    return None

if __name__ == '__main__':
    socketIo.run(app)