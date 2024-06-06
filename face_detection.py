import cv2
# enable system camera
face_cap = cv2.CascadeClassifier("C:/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
video_cap = cv2.VideoCapture(0)
while True:
    ret, video_data = video_cap.read()
    # convert to gray scale
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    # detect faces
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        # draw rectangle around the face
        cv2.rectangle(video_data, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # after reading, we want to show
    cv2.imshow("video_live", video_data)
    # if we press 'a' then we want to break the loop
    if cv2.waitKey(10) == ord('a'):
        break
video_cap.release()

""" video_cap = cv2.VideoCapture(0)
while True:
    ret, video_data = video_cap.read()
    # after reading, we want to show
    cv2.imshow("video_live", video_data)
    # if we press 'a' then we want to break the loop
    if cv2.waitKey(10) == ord('a'):
        break
video_cap.release() """