import pathlib
import cv2

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture (0) #dipende kung ilan camera mo 0 if default.

#camera = cv2.VideoCapture("dancing.mp4")

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    muka = clf.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5, #Higher number less faces will detect.
        minSize = (30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, width, height) in muka :
        cv2.rectangle(frame, (x,y), (x + width, y + height), (255, 255, 0), 2)

    cv2.imshow("Muka", frame)

    if cv2.waitKey(1) == ord("x"): #x is the key to terminate the program
        break
camera.release()
cv2.destroyAllWindows()