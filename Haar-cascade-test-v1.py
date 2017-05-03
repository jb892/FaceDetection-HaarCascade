import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

test_cases = ['test.jpg', 'example.jpg', 'face.jpg']

img = cv2.imread(test_cases[0])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

print len(faces)

i = 0
for (x,y,w,h) in faces:
	cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
	i = i + 1
	cv2.putText(img, ('Face_%03d' % i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.CV_AA)
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]
	eyes = eye_cascade.detectMultiScale(roi_gray)
	for (ex,ey,ew,eh) in eyes:
		cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.imwrite('test-result.jpg', img)
cv2.destroyAllWindows()