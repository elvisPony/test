
import cv2
import numpy as np
image = cv2.imread('p4.png')

cv2.imshow('ori',image)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #將彩色影像灰度化處理
#cv2.imshow('gray',image)

image_test = cv2.Sobel(image,-1,0,1)
cv2.imshow('sobel',image_test)

_, result = cv2.threshold(image_test,10,255,cv2.THRESH_BINARY_INV)
cv2.imshow('th',result)


'''
image_canny = cv2.Canny(image_test,50,150)
cv2.imshow('canny',image_canny)

_, result = cv2.threshold(image_canny,100,255,cv2.THRESH_BINARY_INV)
cv2.imshow('sobel + threshold', result)


image_canny = cv2.Canny(image,50,150)
cv2.imshow('canny',image_canny)


img_gradient = cv2.Sobel(image_canny,-1,1,0)
cv2.imshow('canny + sobel',img_gradient)

_, result = cv2.threshold(img_gradient,100,255,cv2.THRESH_BINARY_INV)
cv2.imshow('canny + sobel + threshold', result)



img_gray_gradient = cv2.Sobel(image,-1,1,0)
_, rrrr = cv2.threshold(img_gray_gradient,100,255,cv2.THRESH_BINARY_INV)
cv2.imshow('sobel + threshold', rrrr)
'''


'''
import cv2
import numpy as np
 
def dodgeNaive(image, mask):
    # determine the shape of the input image
    width, height = image.shape[:2]
 
    # prepare output argument with same size as image
    blend = np.zeros((width, height), np.uint8)
 
    for col in range(width):
        for row in range(height):
            # do for every pixel
            if mask[col, row] == 255:
                # avoid division by zero
                blend[col, row] = 255
            else:
                # shift image pixel value by 8 bits
                # divide by the inverse of the mask
                tmp = (image[col, row] << 8) / (255 - mask)
                # print('tmp={}'.format(tmp.shape))
                # make sure resulting value stays within bounds
                if tmp.any() > 255:
                    tmp = 255
                    blend[col, row] = tmp
    return blend
 
def dodgeV2(image, mask):
    return cv2.divide(image, 255 - mask, scale=256)
 
def burnV2(image, mask):
    return 255 - cv2.divide(255 - image, 255 - mask, scale=256)
 
def rgb_to_sketch(src_image_name, dst_image_name):
    img_rgb = cv2.imread(src_image_name)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # 讀取圖片時直接轉換操作
    # img_gray = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)
 
    img_gray_inv = 255 - img_gray
    img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),
                                sigmaX=0, sigmaY=0)
    img_blend = dodgeV2(img_gray, img_blur)
 
    cv2.imshow('original', img_rgb)
    cv2.imshow('gray', img_gray)
    cv2.imshow('gray_inv', img_gray_inv)
    cv2.imshow('gray_blur', img_blur)
    cv2.imshow("pencil sketch", img_blend)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(dst_image_name, img_blend)
 
if __name__ == '__main__':
    src_image_name = 'p2.jpg'
    dst_image_name = 'p2.jpg'
    rgb_to_sketch(src_image_name, dst_image_name)
'''
cv2.waitKey(0)