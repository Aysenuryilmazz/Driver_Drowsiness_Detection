import math

from imutils import face_utils
import time
import dlib
import cv2
from numpy import mean
import numpy as np
from scipy.spatial import distance as dist
import tensorflow as tf


# define three constants.
# You can later experiment with these constants by changing them to adaptive variables.
EAR_THRESHOLD = 0.21 # eye aspect ratio to indicate blink
EAR_CONSEC_FRAMES = 3 # number of consecutive frames the eye must be below the threshold
SKIP_FIRST_FRAMES = 0 # how many frames we should skip at the beggining

# initialize output structures
scores_string = ""
#perclos var
FRAME_COUNTER = 0
EYE_CLOSED_COUNTER = 0


features_dict = {
    "EAR": "",
    "PERCLOS": "",
    "MAR": "",
    "MOE": "",
    "EC": "",
    "LEB": "",
    "SOP": "",
    "EARLYDETECTION":"",
    "DROWSINESS":""
                 }


# define ear function
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

def mouth_aspect_ration(mouth):
    A = dist.euclidean(mouth[0], mouth[1])
    B = dist.euclidean(mouth[2], mouth[3])
    mar = A / B
    return mar

def compute_perclos(perclosList):

    avg = mean(perclosList)
    perclos_percentage = avg * 100

    return perclos_percentage

def eye_circularity(eye):

    A = dist.euclidean(eye[0], eye[1])
    B = dist.euclidean(eye[1], eye[2])
    C = dist.euclidean(eye[2], eye[3])
    D = dist.euclidean(eye[3], eye[4])
    E = dist.euclidean(eye[4], eye[5])
    F = dist.euclidean(eye[5], eye[0])
    eye_perimeter = A + B + C + D + E + F

    diameter = dist.euclidean(eye[1], eye[4])

    pupil_area = ((diameter * diameter) / 4) * math.pi

    eye_circ = (4 * pupil_area * math.pi) / (eye_perimeter * eye_perimeter)

    return eye_circ

def level_of_eyebrows(eye):

    A = dist.euclidean(eye[0], eye[2])
    B = dist.euclidean(eye[1], eye[2])

    leb = (A + B) / 2
    return  leb

def size_of_pupil(eye):
    A = dist.euclidean(eye[1], eye[4])
    B = dist.euclidean(eye[0], eye[3])

    sop = A / B
    return sop


def get_global_variable():
    global features_dict

    return features_dict

def logit(x):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    if (x<=0.5):
        return float(0)
    if (x>=0.880797):
        return float(2)
    return float(- tf.math.log(1. / x - 1.))


# process a given video file
def process_video(vs, detector, predictor, scaler, subject_wise_scaler, model,
                  lStart=42, lEnd=48, rStart=36, rEnd=42, ear_th=0.21, consec_th=3, up_to=None):
    # define necessary variables
    global FRAME_COUNTER, EYE_CLOSED_COUNTER, features_dict
    COUNTER = 0
    TOTAL = 0
    current_frame = 1
    blink_start = 1
    blink_end = 1
    closeness = 0
    output_closeness = []
    output_blinks = []
    blink_info = (0, 0)

    perclos_list = []
    perclos = 0
    
    buffer = [] # for subject-wise calibration
    sequence = [] # for lstm
    

    


    while True:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        grabbed, frame = vs.read()
        if not grabbed:
            break
        height = frame.shape[0]
        weight = frame.shape[1]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        rects = detector(gray, 0)


        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (244, 252, 129), -1)
            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # #mouth aspect ratio
            mouth = [None] * 4
            mouth[0] = shape[62]
            mouth[1] = shape[66]
            mouth[2] = shape[60]
            mouth[3] = shape[64]
            mar = mouth_aspect_ration(mouth)

            #mouth over eye
            moe = mar/ear

            #eye circularity
            leftEC = eye_circularity(leftEye)
            rightEC = eye_circularity(rightEye)

            eye_circ = (leftEC + rightEC) / 2.0

            #level of eyebrows
            left_eye_leb_coordinates = [None] * 3
            left_eye_leb_coordinates = [None] * 3
            left_eye_leb_coordinates[0] = shape[20]
            left_eye_leb_coordinates[1] = shape[21]
            left_eye_leb_coordinates[2] = shape[40]

            right_eye_leb_coordinates = [None] * 3
            right_eye_leb_coordinates[0] = shape[22]
            right_eye_leb_coordinates[1] = shape[23]
            right_eye_leb_coordinates[2] = shape[42]

            leftEyeLEB = level_of_eyebrows(left_eye_leb_coordinates)
            rightEyeLEB = level_of_eyebrows(right_eye_leb_coordinates)

            leb = (rightEyeLEB + leftEyeLEB) / 2

            #size of pupil
            leftEyeSOP = size_of_pupil(leftEye)
            rightEyeSOP = size_of_pupil(rightEye)

            sop = (leftEyeSOP + rightEyeSOP) / 2


            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < ear_th:
                COUNTER += 1
                EYE_CLOSED_COUNTER += 1
                perclos_list.append(1)
                closeness = 1
                output_closeness.append(closeness)
            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                perclos_list.append(0)
                if COUNTER >= consec_th:
                    TOTAL += 1
                    blink_start = current_frame - COUNTER
                    blink_end = current_frame - 1
                    blink_info = (blink_start, blink_end)
                    output_blinks.append(blink_info)
                # reset the eye frame counter
                COUNTER = 0
                closeness = 0
                output_closeness.append(closeness)
                
            features_dict["EAR"] = ear
            features_dict["PERCLOS"] = perclos
            features_dict["MAR"] = mar
            features_dict["MOE"] = moe
            features_dict["EC"] = eye_circ
            features_dict["LEB"] = leb
            features_dict["SOP"] = sop
            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            # cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2.putText(frame, "EAR: {:.2f}".format(ear), (200, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2.putText(frame, "PERCLOS: {:.2f}".format(perclos), (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2.putText(frame, "MAR: {:.2f}".format(mar), (10, 70),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2.putText(frame, "MOE: {:.2f}".format(moe), (10, 100),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2.putText(frame, "EC: {:.2f}".format(eye_circ), (10, 130),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # cv2.putText(frame, "LEB: {:.2f}".format(leb), (10, 160),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "FRAME: {:.2f}".format(FRAME_COUNTER), (10, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "DROWSINESS: {}".format(features_dict["DROWSINESS"]), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EARLY DETECTION: {}".format(features_dict["EARLYDETECTION"]), (10, 90),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        FRAME_COUNTER += 1
        if FRAME_COUNTER >= 150:
            perclos = compute_perclos(perclos_list)
            EYE_CLOSED_COUNTER = 0
            perclos_list.pop(0)

        if FRAME_COUNTER < 240:
            features_dict["DROWSINESS"] = "CALIB"
            
        if FRAME_COUNTER < 390:
            features_dict["EARLYDETECTION"] = "CALIB"
            
        if FRAME_COUNTER >= 150 and FRAME_COUNTER<240:
            buffer.append([ear, mar, moe, eye_circ, leb,sop,perclos,closeness])
            
        if FRAME_COUNTER == 240:
            subject_wise_scaler.fit(np.array(buffer))
            subject_scaled = subject_wise_scaler.transform(np.array([ear, mar, moe, eye_circ, leb,sop,perclos,closeness]).reshape(1,8))
            clmn_scaled = scaler.transform(subject_scaled)
            sequence.append(clmn_scaled)
            # yhat_drow = sk_model.predict(clmn_scaled)
            # features_dict["DROWSINESS"] = yhat_drow[0]
            
        if FRAME_COUNTER > 240 and FRAME_COUNTER < 390:
            subject_scaled = subject_wise_scaler.transform(np.array([ear, mar, moe, eye_circ, leb,sop,perclos,closeness]).reshape(1,8))
            clmn_scaled = scaler.transform(subject_scaled)
            # clmn_scaled=subject_scaled
            sequence.append(clmn_scaled)
            # yhat_drow = sk_model.predict(clmn_scaled)
            # features_dict["DROWSINESS"] = yhat_drow[0]
            
        if FRAME_COUNTER >= 390:
            sequence.pop(0)
            subject_scaled = subject_wise_scaler.transform(np.array([ear, mar, moe, eye_circ, leb,sop,perclos,closeness]).reshape(1,8))
            clmn_scaled = scaler.transform(subject_scaled)
            # clmn_scaled=subject_scaled
            sequence.append(clmn_scaled)

            yhat = model.predict(np.array(sequence).reshape(1,150,8))
            yhat_inversed = np.array([logit(x) for x in yhat])
            features_dict["EARLYDETECTION"] = np.round(yhat_inversed[0], decimals=3)
            # yhat_drow = sk_model.predict(clmn_scaled)
            features_dict["DROWSINESS"] = 1 if ear < 0.23 else 0

            

            # features_dict["EAR"] = clmn_scaled[0][0]
            # features_dict["PERCLOS"] = clmn_scaled[0][6]
            # features_dict["MAR"] = clmn_scaled[0][1]
            # features_dict["MOE"] = clmn_scaled[0][2]
            # features_dict["EC"] = clmn_scaled[0][3]
            # features_dict["LEB"] = clmn_scaled[0][4]
            # features_dict["SOP"] = clmn_scaled[0][5]
            
        # append processed frame to list
        current_frame += 1
        if up_to == current_frame:
            break

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame2 = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')






def dlib_detect(vs):

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("input/shape_predictor_68_face_landmarks.dat")

    # initialize the video stream with faster method of imutils than opencv
    print("[INFO] camera sensor warming up...")
    #vs = FileVideoStream('input/yawning.mp4').start()
    time.sleep(1.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream, resize it to
        # have a maximum width of 400 pixels, and convert it to
        # grayscale
        success, frame = vs.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)



        ret, jpeg = cv2.imencode('.jpg', frame)
        frame2 = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')