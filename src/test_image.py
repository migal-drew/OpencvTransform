'''
Created on May 21, 2012

@author: andrew
'''
import numpy as np
import cv2

if __name__ == '__main__':
    img = cv2.imread("3_small.jpg")
    
    print img.shape
    src_center = (img.shape[1]/2., img.shape[0]/2.)
    dsize = (900, 900)
    shift_x, shift_y = 100, 100
    src_center = src_center[0] + shift_x, src_center[1] + shift_y
    print "First center", src_center
    M = np.array([[1, 0, shift_x], [0, 1, shift_y]], np.float32)
    img = cv2.warpAffine(img, M, dsize)
    
    M = np.array([[1, 0, 0], [0.2, 1, 0]], np.float32)
    img = cv2.warpAffine(img, M, dsize)
    
    cntr = np.transpose(np.array([src_center[0], src_center[1], 1]))
    add_row = np.array([0, 0, 1])
    M = np.vstack((M, np.array([0, 0, 1])))
    mat_deform_full = np.vstack((M, add_row))
    cntr_err = np.dot(M, cntr)
    diff = cntr - cntr_err
    print "Diff_", diff
    M = np.array([[1, 0, diff[0]], [0, 1, diff[1]]], np.float32)
    img = cv2.warpAffine(img, M, dsize)
    
    
#    M = np.array([[1, -0.1, 0], [0.1, 1, 0]], np.float32)
#    img = cv2.warpAffine(img, M, dsize)
#    
#    cntr = np.transpose(np.array([src_center[0], src_center[1], 1]))
#    add_row = np.array([0, 0, 1])
#    M = np.vstack((M, np.array([0, 0, 1])))
#    mat_deform_full = np.vstack((M, add_row))
#    cntr_err = np.dot(M, cntr)
#    diff = cntr - cntr_err
#    print "Diff_", diff
#    M = np.array([[1, 0, diff[0]], [0, 1, diff[1]]], np.float32)
#    img = cv2.warpAffine(img, M, dsize)
    
    #M = cv2.getRotationMatrix2D(src_center, 40, 1)
    #img = cv2.warpAffine(img, M, dsize)
    
#----------------------------------------------    
#    img2 = cv2.imread("4_small.jpg")
#        
#    print img2.shape
#    src_center = (img2.shape[1]/2., img2.shape[0]/2.)
#    dsize = (900, 900)
#    shift_x, shift_y = 100, 100
#    src_center = src_center[0] + shift_x, src_center[1] + shift_y
#    print "Second center", src_center
#    M = np.array([[1, 0, shift_x], [0, 1, shift_y]], np.float32)
#    img2 = cv2.warpAffine(img2, M, dsize)
#    
#    M = np.array([[1, 0.04, 0], [0.04, 1, 0]], np.float32)
#    #img2 = cv2.warpAffine(img2, M, dsize)
#    
#    M = cv2.getRotationMatrix2D(src_center, 0, 1)
#    img2 = cv2.warpAffine(img2, M, dsize)
#    
#    img /= 2
#    img2 /=2
#    res = img + img2
    cv2.imshow("TEST.jpg", img)
    0xFF & cv2.waitKey()
    cv2.destroyAllWindows() 