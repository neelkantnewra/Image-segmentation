import numpy as np
import cv2

img = cv2.imread("/content/CT-kidneys.jpeg")
img2 = img.reshape((-1,3))
img2 =  np.float32(img2)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER,10,1.0)
k = 5

attempt =10
ret,label,center = cv2.kmeans(img2,k,attempt,criteria,10,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imwrite('segmented.jpg',res2)
