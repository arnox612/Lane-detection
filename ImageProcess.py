import cv2
import numpy as np

def _blur(img):
    img=cv2.GaussianBlur(img,(3,3),0)
    return img
   

def _sharp(img):
    kernel_sharpen = np.array([
        [-1,-1,-1,-1,-1],
        [-1,2,2,2,-1],
        [-1,2,8,2,-1],
        [-1,2,2,2,-1], 
        [-1,-1,-1,-1,-1]])/8.0

    image=cv2.filter2D(img,-1,kernel_sharpen)
    return image

def _resize(image):
    image=cv2.resize(image,(1024,512))
    return image

'''class preproc():

    def __init__(self,img):
    	self.image=img

    def trans(self):
    	img=_sharp(self.image)
    	image=_resize(img)
    	return image
'''
    
if __name__ == '__main__':
    imggray=cv2.imread('2.jpg',0)
    img=_blur(imggray)
    img=_sharp(img)	
    img=_resize(img)
    cv2.imwrite('final.jpg',img)
    print('Done')
