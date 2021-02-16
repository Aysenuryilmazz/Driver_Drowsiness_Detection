#download snappy whl file with your python version: https://www.lfd.uci.edu/~gohlke/pythonlibs/#python-snappy
#pip install python_snappy-0.5-cp37-cp37m-win_amd64.whl
#pip install pyarrow fastparquet
from scipy.spatial import distance as dist
import numpy as np
import argparse
import time
import dlib
import cv2
import os
import pandas as pd
import statistics
from statistics import mean
import math
import csv


# define three constants.
# You can later experiment with these constants by changing them to adaptive variables.
EAR_THRESHOLD = 0.21 # eye aspect ratio to indicate blink
EAR_CONSEC_FRAMES = 3 # number of consecutive frames the eye must be below the threshold
SKIP_FIRST_FRAMES = 150 # how many frames we should skip at the beggining

# initialize dlib variables
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# initialize output structures
#scores_string = ""

# writing frames to output video
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('sleepyCombination_show_drowsiness.avi', fourcc, 30.0, (640, 480))

# display statistics
# if you want to display test scores set test=True to change headline
def display_stats(closeness_list, blinks_list, video_info=None, skip_n=0, test=False):
    str_out = ""
    # write video info
    if video_info != None:
        str_out += ("Video info\n")
        str_out += ("FPS: {}\n".format(video_info["fps"]))
        str_out += ("FRAME_COUNT: {}\n".format(video_info["frame_count"]))
        str_out += ("DURATION (s): {:.2f}\n".format(video_info["duration(s)"]))
        str_out += ("\n")

    # if you skipped n frames previously
    if skip_n > 0:
        str_out += ("After skipping {} frames,\n".format(skip_n))

        # if you are displaying prediction information
    if test == False:
        str_out += ("Statistics on the prediction set are\n")

    # if you are displaying test information
    if test == True:
        str_out += ("Statistics on the test set are\n")

    str_out += ("TOTAL NUMBER OF FRAMES PROCESSED: {}\n".format(len(closeness_list)))
    str_out += ("NUMBER OF CLOSED FRAMES: {}\n".format(closeness_list.count(1)))
    str_out += ("NUMBER OF BLINKS: {}\n".format(len(blinks_list)))
    str_out += ("\n")

    print(str_out)
    return str_out

def display_stats_for_ds(frame_list_df, closeness_list):

    mar_list = frame_list_df['mar'].values.tolist()
    ear_list = frame_list_df['avg_ear'].values.tolist()
    moe_list = frame_list_df['moe'].values.tolist()
    perclos_list = frame_list_df['perclos'].values.tolist()
    ec_list = frame_list_df['eye_circularity'].values.tolist()
    leb_list = frame_list_df['leb'].values.tolist()
    sop_list = frame_list_df['sop'].values.tolist()
    print(f"EAR: \nMIN: {min(ear_list)} \nMAX: {max(ear_list)} \nAVG: {mean(ear_list)} \nSTDEV: {statistics.stdev(ear_list)}")
    print(f"MAR: \nMIN: {min(mar_list)} \nMAX: {max(mar_list)} \nAVG: {mean(mar_list)} \nSTDEV: {statistics.stdev(mar_list)}")
    print(f"MOE: \nMIN: {min(moe_list)} \nMAX: {max(moe_list)} \nAVG: {mean(moe_list)} \nSTDEV: {statistics.stdev(moe_list)}")
    print(f"PERCLOS: \nMIN: {min(perclos_list)} \nMAX: {max(perclos_list)} \nAVG: {mean(perclos_list)} \nSTDEV: {statistics.stdev(perclos_list)}")
    print(f"LEB: \nMIN: {min(leb_list)} \nMAX: {max(leb_list)} \nAVG: {mean(leb_list)} \nSTDEV: {statistics.stdev(leb_list)}")
    print(f"SOP: \nMIN: {min(sop_list)} \nMAX: {max(sop_list)} \nAVG: {mean(sop_list)} \nSTDEV: {statistics.stdev(sop_list)}")
    print(f"EC: \nMIN: {min(ec_list)} \nMAX: {max(ec_list)} \nAVG: {mean(ec_list)} \nSTDEV: {statistics.stdev(ec_list)}")


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
resols=[]
# process a given video file
def process_video(input_file, fold, subject, videoname, orientation, detector=dlib_detector, predictor=dlib_predictor, \
                  ear_th=0.21, consec_th=3, skip_n=150, up_to=None):
    # define necessary variables
    COUNTER = 0
    TOTAL = 0
    current_frame = 1
    blink_start = 1
    blink_end = 1
    closeness = 0
#    output_closeness = []
#    output_blinks = []
#    blink_info = (0, 0)
#    processed_frames = []
    frame_info_list = []
    video_info_list = []

    perclos_list = []
    perclos = 0

    # define capturing method
    cap = cv2.VideoCapture(input_file)
    time.sleep(1.0)

    # build a dictionary video_info
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    video_info_dict = {
        'fps': fps,
        'frame_count': frame_count,
        'duration(s)': duration,
    }

    while True:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        (grabbed, frame) = cap.read()
        if not grabbed:
            break
         
        rows=np.shape(frame)[0]
        cols = np.shape(frame)[1]
        # To Rotate by 90 degreees
        if orientation=="sol":
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE) 
        if orientation=="sağ":
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        if orientation=="ters":
            frame = cv2.rotate(frame, cv2.cv2.ROTATE_180)

        y = frame.shape[0]
        x = frame.shape[1]
        if y>x:
            frame = cv2.resize(frame, (480, int(480 * (y / x))))
        if x>y:
            frame = cv2.resize(frame, (640, int(640 * (y / x))))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        rects = detector(frame, 0)
        
        if len(rects) == 0:
            frame_info = {
                'fold': fold,
                'subject': subject,
                'videoname': videoname,
                'frame_no': current_frame,
                'face_detected': 0,
#                'face_coordinates': [[rect.tl_corner().x, rect.tl_corner().y],
#                                     [rect.tr_corner().x, rect.tr_corner().y],
#                                     [rect.bl_corner().x, rect.bl_corner().y],
#                                     [rect.br_corner().x, rect.br_corner().y]],
#                'left_eye_coordinates': leftEye,
#                'right_eye_coordinates': rightEye,
#                'mouth_coordinates': mouth,
#                'left_eyebrow_coordinates': leftEyeBrow,
#                'right_eyebrow_coordinates': rightEyeBrow,
                'left_ear': -1,
                'right_ear': -1,
                'avg_ear': -1,
                'mar': -1,
                'moe': -1,
                'left_ec' : -1,
                'right_ec' : -1,
                'avg_ec' : -1,
                'left_leb': -1,
                'right_leb': -1,
                'avg_leb': -1,
                'left_sop': -1,
                'right_sop': -1,
                'avg_sop': -1,
                'closeness': -1,
                'blink_no': -1,
                'blink_start_frame': -1,
                'blink_end_frame': -1,
                'reserved_for_calibration': False,
                'perclos': -1,
                }
            
        if len(rects) > 1:
            frame_info = {
                'fold': fold,
                'subject': subject,
                'videoname': videoname,
                'frame_no': current_frame,
                'face_detected': len(rects),
#                'face_coordinates': [[rect.tl_corner().x, rect.tl_corner().y],
#                                     [rect.tr_corner().x, rect.tr_corner().y],
#                                     [rect.bl_corner().x, rect.bl_corner().y],
#                                     [rect.br_corner().x, rect.br_corner().y]],
#                'left_eye_coordinates': leftEye,
#                'right_eye_coordinates': rightEye,
#                'mouth_coordinates': mouth,
#                'left_eyebrow_coordinates': leftEyeBrow,
#                'right_eyebrow_coordinates': rightEyeBrow,
                'left_ear': -1,
                'right_ear': -1,
                'avg_ear': -1,
                'mar': -1,
                'moe': -1,
                'left_ec' : -1,
                'right_ec' : -1,
                'avg_ec' : -1,
                'left_leb': -1,
                'right_leb': -1,
                'avg_leb': -1,
                'left_sop': -1,
                'right_sop': -1,
                'avg_sop': -1,
                'closeness': -1,
                'blink_no': -1,
                'blink_start_frame': -1,
                'blink_end_frame': -1,
                'reserved_for_calibration': False,
                'perclos': -1,
                }
            
        # loop over the face detections
        if len(rects) == 1:
            rect=rects[0]
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(frame, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])
            
            
            ##For dlib’s 68-point facial landmark detector:
            #FACIAL_LANDMARKS_68_IDXS = OrderedDict([
            #   ("mouth", (48, 68)),
            #   ("inner_mouth", (60, 68)),
            #   ("right_eyebrow", (17, 22)),
            #   ("left_eyebrow", (22, 27)),
            #   ("right_eye", (36, 42)),
            #   ("left_eye", (42, 48)),
            #   ("nose", (27, 36)),
            #   ("jaw", (0, 17))
            #])

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[42:48]
            rightEye = shape[36:42]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

             # mouth aspect ratio
            mouth = [shape[62], shape[66], shape[60], shape[64]]
            mar = mouth_aspect_ration(mouth)

            #mouth over eye
            moe = mar/ear

            #eye circularity
            leftEC = eye_circularity(leftEye)
            rightEC = eye_circularity(rightEye)

            eye_circ = (leftEC + rightEC) / 2.0

            #level of eyebrows
            right_eye_leb_coordinates = [shape[20], shape[21], shape[40]]
            left_eye_leb_coordinates = [shape[22],shape[23],shape[42]] 

            leftEyeLEB = level_of_eyebrows(left_eye_leb_coordinates)
            rightEyeLEB = level_of_eyebrows(right_eye_leb_coordinates)

            leb = (rightEyeLEB + leftEyeLEB) / 2

            #size of pupil
            rightEyeSOP = size_of_pupil(leftEye)
            leftEyeSOP = size_of_pupil(rightEye)

            sop = (leftEyeSOP + rightEyeSOP) / 2

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # draw mouth convex hull
            mouthHull = cv2.convexHull(shape[60:68])
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            
            # draw eyebrow
            leftEyeBrow = shape[22:27]
            rightEyeBrow = shape[17:22]
            cv2.polylines(frame,[leftEyeBrow],False,(0,255,255),1)
            cv2.polylines(frame,[rightEyeBrow],False,(0,255,255),1)
            
            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < ear_th:
                closeness=1
                COUNTER += 1
#                output_closeness.append(closeness)
                perclos_list.append(1)
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
#                    blink_info = (blink_start, blink_end)
#                    output_blinks.append(blink_info)
                # reset the eye frame counter
                COUNTER = 0
                closeness = 0
#                output_closeness.append(closeness)

            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (340, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "PERCLOS: {:.2f}".format(perclos), (140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "MAR: {:.2f}".format(mar), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "MOE: {:.2f}".format(moe), (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EC: {:.2f}".format(eye_circ), (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "LEB: {:.2f}".format(leb), (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "SOP: {:.2f}".format(sop), (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "TIME: {:.1f}".format(current_frame/fps), (10, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # build frame_info dictionary then add to list
            frame_info = {
                'fold': fold,
                'subject': subject,
                'videoname': videoname,
                'frame_no': current_frame,
                'face_detected': 1,
#                'face_coordinates': [[rect.tl_corner().x, rect.tl_corner().y],
#                                     [rect.tr_corner().x, rect.tr_corner().y],
#                                     [rect.bl_corner().x, rect.bl_corner().y],
#                                     [rect.br_corner().x, rect.br_corner().y]],
#                'left_eye_coordinates': leftEye,
#                'right_eye_coordinates': rightEye,
#                'mouth_coordinates': mouth,
#                'left_eyebrow_coordinates': leftEyeBrow,
#                'right_eyebrow_coordinates': rightEyeBrow,
                'left_ear': leftEAR,
                'right_ear': rightEAR,
                'avg_ear': ear,
                'mar': mar,
                'moe': moe,
                'left_ec' : leftEC,
                'right_ec' : rightEC,
                'avg_ec' : eye_circ,
                'left_leb': leftEyeLEB,
                'right_leb': rightEyeLEB,
                'avg_leb': leb,
                'left_sop': leftEyeSOP,
                'right_sop': rightEyeSOP,
                'avg_sop': sop,
                'closeness': closeness,
                'blink_no': TOTAL,
                'blink_start_frame': blink_start,
                'blink_end_frame': blink_end,
                'reserved_for_calibration': False,
                'perclos': -1,
            }

            if current_frame > skip_n:
                perclos = compute_perclos(perclos_list)
                perclos_list.pop(0)
                frame_info['perclos'] = perclos
    
        if current_frame <= skip_n:
            frame_info['reserved_for_calibration'] = True
        frame_info_list.append(frame_info)

        #show the frame (this part doesn't work in online kernel. If you are running on offline jupyter
        #notebook, you can uncomment this part and try displaying video frames)
        cv2.imshow("Frame", frame)
        #out.write(frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("é"):
            resols.append([rows,cols])
            break

        # append processed frame to list
#        processed_frames.append(frame)
        current_frame += 1
        if up_to == current_frame:
            break

    # a bit of clean-up
    cv2.destroyAllWindows()
    cap.release()

    video_info_list.append(video_info_dict)

    frame_info_df = pd.DataFrame(frame_info_list)  # build a dataframe from frame_info_list
    video_info_df = pd.DataFrame(video_info_list)
    
    # print status
    file_name = os.path.basename(input_file)
    output_str = "Processing {} has done.\n\n".format(file_name)
    print(output_str)


    return frame_info_df, video_info_df



######################## MAIN ######################################
hacilar=[]
with open("hacıyatmaz.txt", 'r', encoding='utf-8') as haci:
    read_txt = csv.reader(haci, delimiter=' ')		
    for row in read_txt:
        try:
            hacilar.append(row[0].strip())		
        except:
            continue
        
infos=[]
directory = "./"
files=os.listdir(directory)
i=0
for file in files:
    if file[:4]=="Fold":
        fold = file
        subjects = os.listdir(os.path.join(directory,fold))
        for subject in subjects:
            video_names=os.listdir(os.path.join(directory,fold,subject))
            for video_name in video_names:
                clean_name = os.path.splitext(video_name)[0]
                extension = os.path.splitext(video_name)[1]
                file_path = os.path.join(directory,fold,subject,video_name)
                print(file_path)
                frame_info_df, video_info_dict = process_video(file_path, fold=fold[:5], subject=subject, videoname=clean_name, orientation=hacilar[i], \
                                                                ear_th=EAR_THRESHOLD, consec_th=EAR_CONSEC_FRAMES, skip_n=SKIP_FIRST_FRAMES)
                frame_info_df.to_pickle(os.path.join(directory,'{}_{}_{}_frame_info_df.pkl'.format(fold[:5],subject,clean_name)))
                video_info_dict.to_pickle(os.path.join(directory,'{}_{}_{}_video_info_df.pkl'.format(fold[:5],subject,clean_name)))
                i+=1
                infos.append([subject,clean_name,extension])
                
pd.DataFrame(zip(infos, resols)).to_pickle("rldd_video_resolutions.pkl")