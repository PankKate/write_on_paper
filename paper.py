# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 01:29:35 2021

@author: PankK
"""
import cv2
import numpy as np
#import matplotlib.pyplot as plt

def processing(image):
    image = image.astype('uint8')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.threshold(gray,147, 255, cv2.THRESH_BINARY)[1]
    return thresh
def get_contour(thresh):
   cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   cnts = cnts[0] if len(cnts) == 2 else cnts[1]
   c = max(cnts, key=cv2.contourArea) 
   return c

def define_size(coord_x,coord_y):
    max_x = max(coord_x)
    min_x = min(coord_x)
    width = max_x - min_x
    max_y = max(coord_y)
    min_y = min(coord_y)
    height = max_y - min_y
    return width,height

def create_image(height,width,text):
    print("here")
    blank_image = np.zeros((height,width,3), np.uint8)
    img = cv2.putText(blank_image, text, (width//5, height//2),cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0),6) 
    return img

def make_png(img):
    tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(img)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    cv2.imwrite("./photo/test.png", dst)
    return dst
    
    
def perspective():
    pass

def define_pts2(coord):
   
    
    keys = list(sorted(coord.keys()))
    if (coord[keys[0]]>coord[keys[1]]):
        lt = [keys[1],coord[keys[1]]]
        rt = [keys[0],coord[keys[0]]]
    else:
        lt = [keys[0],coord[keys[0]]]
        rt = [keys[1],coord[keys[1]]]
        
    if (coord[keys[2]]>coord[keys[3]]):
        rb = [keys[2],coord[keys[2]]]
        lb = [keys[3],coord[keys[3]]]
    else:
        rb = [keys[3],coord[keys[3]]]
        lb = [keys[2],coord[keys[2]]]
    
    print([lt,rt,lb,rb])
    pts = np.float32([lt,rt,lb,rb])
    return pts


img2 = cv2.imread('./photo/paper1.jpg')

thresh = processing(img2)
cnt = get_contour(thresh)

approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
# рисует границу контуров.
cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5) 
# Используется для выравнивания массива, содержащего координаты вершин.
n = approx.ravel() 
i = 0
coord_x = []
coord_y = []
coord = {}
for j in n :
    if(i % 2 == 0):
        x = n[i]
        y = n[i + 1]
        coord_x.append(x)
        coord_y.append(y)
        coord[x] = y
      
 # Строка, содержащая координаты.
        string = str(x) + " " + str(y) 
            # текст по координатам.
        cv2.putText(img2, string, (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0)) 
    i = i + 1

#create text
text = str(input())
width, height = img2.shape[1],img2.shape[0]
#define_size(coord_x, coord_y)
img = create_image(height, width,text)
height, width = img.shape[0], img.shape[1]
print(height,' ', width )
#искривляем текст

pts1 = np.float32([[0,0],[0,height],[width,0],[width,height]])
pts2 = define_pts2(coord)
M = cv2.getPerspectiveTransform(pts1, pts2)
img1 = cv2.warpPerspective(img,M,(width,height)) #разворачиваем 
img1 = make_png(img1)
img1 = cv2.imread('./photo/test.png')

#наложение
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
dst = cv2.addWeighted(img1, 1, img2, 1, 0)


# Отображение окончательного изображения.
img2= cv2.resize(img2, (960, 540))    
cv2.imshow('image2', img2) 
img= cv2.resize(img, (960, 540))
cv2.imshow('img', img) 
dst= cv2.resize(dst, (960, 540))
cv2.imshow("res",dst)
# Выход из окна, если на клавиатуре нажата клавиша «q».

if cv2.waitKey(0) & 0xFF == ord('q'): 
    cv2.destroyAllWindows()

#plt.subplot(121)
#plt.imshow(thresh)
#plt.subplot(122)
#plt.imshow(img)