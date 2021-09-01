#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import mediapipe
import copy
import itertools
import csv
import numpy

from model import SIGNLANGUAGE
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5,max_num_hands=1) as hands:
    while (True):
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        debug_frame = copy.deepcopy(frame)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks != None:
            for handLandmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)

            image_width, image_height = frame.shape[1], frame.shape[0]
            landmark_point = []


            for _, landmark in enumerate(handLandmarks.landmark):
                landmark_x = min(int(landmark.x * image_width), image_width - 1)
                landmark_y = min(int(landmark.y * image_height), image_height - 1)

                landmark_point.append([landmark_x, landmark_y])
                landmark_list = landmark_point
            temp_landmark_list = copy.deepcopy(landmark_list)
            # Convert to relative coordinates
            base_x, base_y = 0, 0
            for index, point in enumerate(temp_landmark_list):
                if index == 0:
                    base_x, base_y = point[0], point[1]
                temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
                temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

            temp_landmark_list = list(
                itertools.chain.from_iterable(temp_landmark_list))

            max_value = max(list(map(abs, temp_landmark_list)))
            def normalize_(n):
                return n / max_value

            temp_landmark_list = list(map(normalize_, temp_landmark_list))
            image_width, image_height = frame.shape[1], frame.shape[0]

            landmark_array = numpy.empty((0, 2), int)

            for _, landmark in enumerate(handLandmarks.landmark):
                landmark_x = min(int(landmark.x * image_width), image_width - 1)
                landmark_y = min(int(landmark.y * image_height), image_height - 1)

                landmark_point = [numpy.array((landmark_x, landmark_y))]

                landmark_array = numpy.append(landmark_array, landmark_point, axis=0)

            x, y, w, h = cv2.boundingRect(landmark_array)

            brect = (x, y, x + w, y + h)

            train = SIGNLANGUAGE()

            with open('model/label.csv',
                     encoding='utf-8-sig') as f:
                labels = csv.reader(f)
                labels = [
                    row[0] for row in labels
                ]
            key = cv2.waitKey(1)
            if 97 <= key <= 122:
                csv_path = 'model/train.csv'
                with open(csv_path, 'a', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([key, *temp_landmark_list])
                    print(key)
            cv2.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[3]), (255, 255, 255), 1)
            cv2.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[1] - 22), (255, 255, 255), -1)
            info_text = handedness.classification[0].label[0:]
            hand_sign_id = train(temp_landmark_list)
            if labels[hand_sign_id] != "":
                info_text = info_text + ' ' + labels[hand_sign_id]
            cv2.putText(frame, info_text, (brect[0] + 5, brect[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('Test hand', frame)

        if cv2.waitKey(1) == 27:
            break


cv2.destroyAllWindows()
capture.release()