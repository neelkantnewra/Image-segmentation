import cv2 
import numpy as np 

img = cv2.imread("OP2_Bild_1.jpeg")
cv2.imshow("BGR Image",img)

#<=========== blank image =================>

blank = np.ones(img.shape[:2],dtype='uint8')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("GRAY",gray)

# <====== Simple thresholding =========>

threshold , thresh = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
cv2.imshow("Threshold Image",thresh)

threshold , threshInv = cv2.threshold(gray,80,255,cv2.THRESH_BINARY_INV)
cv2.imshow("Threshold Image Inverse",threshInv)

# <======= Apply mean on thresholded image========>

mean = cv2.medianBlur(thresh,9)
cv2.imshow("MEAN Blurr",mean)

# <============ Bitwise Operator================>

bitwiseop = cv2.bitwise_and(img,img,mask = mean)
cv2.imshow("oper",bitwiseop)

#<========== Appling K mean =================>
img2 = bitwiseop.reshape((-1,3))
img2 =  np.float32(img2)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER,10,1.0)
k = 3

attempt =10
ret,label,center = cv2.kmeans(img2,k,attempt,criteria,10,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imshow("Final Output",res2)

#<===== meadinan filter on processed image==========>

m_processed = cv2.medianBlur(res2,5)
cv2.imshow("Process median",m_processed)

#<===== creating mask =============>


# cv2.imwrite("processedimage.png",bitwiseop)
threshold , thresh_new = cv2.threshold(m_processed,175,255,cv2.THRESH_BINARY)
cv2.imshow("Threshold Image_new",thresh_new) 


threshold , thresh_new1 = cv2.threshold(m_processed,100,50,cv2.THRESH_BINARY)
cv2.imshow("Threshold Image_new1",thresh_new1) 

# ******color masking**********

mer = cv2.bitwise_or(thresh_new1,thresh_new)
cv2.imshow("Threshold Image_merge",mer) 

#<========Drawing region of interest ========>

draw = cv2.circle(mer,(mer.shape[0]//3 + 80,mer.shape[1]//2),90,(0,255,0),thickness=2)
cv2.imshow("Drawn image",draw)

draw = cv2.circle(draw,(draw.shape[0]-80,draw.shape[1]//2),90,(0,255,0),thickness=2)
cv2.imshow("Drawn image",draw)


cv2.waitKey(0) 


