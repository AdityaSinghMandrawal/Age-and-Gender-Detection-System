import cv2 as cv

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

# Load the face detection model
faceProto = r"C:\Users\Acer\Desktop\sidraproject\deploy.prototxt.txt"
faceModel = r"C:\Users\Acer\Desktop\sidraproject\res10_300x300_ssd_iter_140000 (1).caffemodel"
faceNet = cv.dnn.readNetFromCaffe(faceProto, faceModel)

# Load the gender classification model
genderProto = r"C:\Users\Acer\Desktop\sidraproject\gender_deploy.prototxt"
genderModel = r"C:\Users\Acer\Desktop\sidraproject\gender_net.caffemodel"
genderNet = cv.dnn.readNet(genderModel, genderProto)
genderList = ['Male', 'Female']

# Load the age estimation model
ageProto = r"C:\Users\Acer\Desktop\sidraproject\age_deploy.prototxt"
ageModel = r"C:\Users\Acer\Desktop\sidraproject\age_net.caffemodel"
ageNet = cv.dnn.readNet(ageModel, ageProto)
ageList = ['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']

# Open a connection to the webcam (usually the default camera, index 0)
cap = cv.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Get face bounding boxes
    frame, face_bboxes = getFaceBox(faceNet, frame)

    for face_bbox in face_bboxes:
        x1, y1, x2, y2 = face_bbox
        face = frame[y1:y2, x1:x2]

        # Run gender classification
        blob = cv.dnn.blobFromImage(face, 1, (227, 227), [78.42633776, 87.76891437, 114.89584775], swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Run age estimation
        blob = cv.dnn.blobFromImage(face, 1, (227, 227), [78.42633776, 87.76891437, 114.89584775], swapRB=False)
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # Draw bounding box and labels on the frame
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = "{}, {}".format(gender, age)
        cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv.LINE_AA)

    # Display the resulting frame
    cv.imshow('Age Gender Demo', frame)

    # Break the loop if 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv.destroyAllWindows()
