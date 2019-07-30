# Import required modules
import cv2 as cv
import argparse
from PIL import Image
import os

parser = argparse.ArgumentParser(description='Use this script to run gender recognition using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

# Load network
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

def extractFaces(folder):

    faces = []

    #iterate over the files in the given folder
    for num, file in enumerate(os.listdir(folder)):

        # Open a video file or an image file or a camera stream
        print ("now checkout file " + file)
        path = "data/" + file
        cap = cv.VideoCapture(path if args.input else 0)
        padding = 20

        while cv.waitKey(1) < 0:
            # Read frame
            hasFrame, frame = cap.read()
            if not hasFrame:
                cv.waitKey()
                break

            frameFace, bboxes = getFaceBox(faceNet, frame)
            if not bboxes:
                print("No face Detected, Checking next frame")
                continue

            for numbox, bbox in enumerate(bboxes):
                face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                       max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

                blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

                #save a new image for each face found in the unstructured dataset
                img = Image.open(path)
                cropped_img = img.crop(bbox)
                img_name = "image" + str(num) + "face" + str(numbox) + ".bmp"
                cropped_img.save(img_name)

                faces.append({"gender": gender, "file": img_name})

    return faces

print(extractFaces(args.input))
