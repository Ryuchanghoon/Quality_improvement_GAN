import cv2

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel('EDSR_x3.pb')
sr.setModel('edsr', 3)



img = cv2.imread('image_100.png')

result = sr.upsample(img)



cv2.imshow('img', img)
cv2.imshow('result', result)
cv2.waitKey(0)