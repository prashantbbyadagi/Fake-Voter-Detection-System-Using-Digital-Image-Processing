import numpy as np, cv2


a=input("enter a aadhaar number\n")
v=len(a)
if v==12:
    print(" 1 for vote\n 2 for Other Services")
    b=input(" Please Enter your Service Number\n")
    key = cv2. waitKey(1)
    cap = cv2.VideoCapture(0)
else:

     print("enter a valid aadhaar number")



face_cascade = cv2.CascadeClassifier('C:/Users/PRASHANT/Desktop/harr cascades/haar-cascade-files-master/haar-cascade-files-master/haarcascade_frontalface_default.xml')

rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("C:/Users/PRASHANT/PycharmProjects/final project/facedata/prashant.yml")
id = 0
fontFace = cv2.FONT_HERSHEY_COMPLEX
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, conf = rec.predict(gray[y:y + h, x:x + w])
        if (id == 1):
            id = "prashant"
            print("Face detected you can proceed :")

        cv2.putText(img, str(id), (x, y + h), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()