import numpy as np
import cv2
from common import anorm
from functools import partial

import utilities as util

help_message = '''SURF image match 

USAGE: findobj.py [ <image1> <image2> ]
'''

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing

flann_params = dict(algorithm = FLANN_INDEX_KDTREE,
                    trees = 4)

def match_bruteforce(desc1, desc2, r_threshold = 0.75):
    res = []
    for i in xrange(len(desc1)):
        dist = anorm( desc2 - desc1[i] )
        n1, n2 = dist.argsort()[:2]
        r = dist[n1] / dist[n2]
        if r < r_threshold:
            res.append((i, n1))
    return np.array(res)

def match_flann(desc1, desc2, r_threshold = 0.6):
    flann = cv2.flann_Index(desc2, flann_params)
    idx2, dist = flann.knnSearch(desc1, 2, params = {}) # bug: need to provide empty dict
    mask = dist[:,0] / dist[:,1] < r_threshold
    idx1 = np.arange(len(desc1))
    pairs = np.int32( zip(idx1, idx2[:,0]) )
    return pairs[mask]


def gradientDescent(iterations, points_1, points_2, theta_1, theta_2,
                    gamma, lambd, gamma_transl, img1, img2, size, c_x, c_y):
    new_theta_1 = theta_1.copy()
    new_theta_2 = theta_2.copy()
    
    m = points_1.shape[0]
    
    for k in range(iterations):
        #Derivatives - respect to theta_1 and theta_2
        der_1 = np.zeros(theta_1.size).reshape(theta_1.shape)
        der_2 = np.zeros(theta_2.size).reshape(theta_2.shape)
        
        for i in range(m):
            
            tmp_1, tmp_2  = util.derivatives(points_1[i], points_2[i], new_theta_1, new_theta_2)
            #print "deriv_1"
            #print tmp_1
            #print "deriv_2"
            #print tmp_2
            der_1 += tmp_1
            der_2 += tmp_2
            #print der_1
            #print der_2
            
#Visualization=--------------------------------        
#        if (k % 300 == 0):
#            res = util.stitch_for_visualization(img1, img2, new_theta_1, new_theta_2, c_x, c_y, size)
#            winname = "Iteration #" + (str)(k)
#            matrix_1 = composeAffineMatrix(new_theta_1)
#            matrix_2 = composeAffineMatrix(new_theta_2)
#            mosaicing.draw_distance_lines(res, points_1, points_2, new_theta_1, new_theta_2, c_x, c_y)
#            cv2.imshow(winname, res)
#            cv2.moveWindow(winname, 0, 0)
#            0xFF & cv2.waitKey()
#            cv2.destroyAllWindows() 
#Visualization=--------------------------------
        
        #print "Sum of derivatives for 1st parameter without penalty", der_1[0]
        
        
        
        #Penalize
        der_1[3:5] = der_1[3:5] + lambd * new_theta_1[3:5]
        der_2[3:5] = der_2[3:5] + lambd * new_theta_2[3:5]
        
        #print "Sum of derivatives for 1st parameter WITH penalty", der_1[0]
        #print der_1[1:3]
        #print dummy_1
        dummy_1 = lambd * np.abs(np.array([1., 1]) - new_theta_1[1:3])
        dummy_2 = lambd * np.abs(np.array([1., 1]) - new_theta_2[1:3])
        der_1[1:3] = der_1[1:3] + dummy_1#lambd * np.abs([1., 1] - der_1[1:3]) #sx, sy = 1
        der_2[1:3] = der_2[1:3] + dummy_2#lambd * np.abs([1., 1] - der_2[1:3]) #sx, sy = 1
        #Refresh parameters
        #new_theta_1 = new_theta_1 - gamma * der_1 / m #- lambd * new_theta_1 / m
        #new_theta_2 = new_theta_2 - gamma * der_2 / m #- lambd * new_theta_2 / m
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        new_theta_1[0] = new_theta_1[0] - gamma * der_1[0] / m
        new_theta_2[0] = new_theta_2[0] - gamma * der_2[0] / m
        new_theta_1[3:5] = new_theta_1[3:5] - gamma * der_1[3:5] / m
        new_theta_2[3:5] = new_theta_2[3:5] - gamma * der_2[3:5] / m
        new_theta_1[5:7] = new_theta_1[5:7] - gamma_transl * der_1[5:7] / m
        new_theta_2[5:7] = new_theta_2[5:7] - gamma_transl * der_2[5:7] / m

        treshhold = 0.15
        #If Skx > threshhold, rollback         
        if np.abs(new_theta_1[3]) > treshhold or np.abs(new_theta_1[4]) > treshhold:
            new_theta_1[3:5] += gamma * der_1[3:5] / m
        if np.abs(new_theta_2[3]) > treshhold or np.abs(new_theta_2[4]) > treshhold:
            new_theta_2[3:5] += gamma * der_2[3:5] / m
        
        #print "verfic", der_1[0]
        #print "verfic", der_2[0]
        
        treshhold_out = 0.01
        
        print "iteration # ", k
        print util.costFunction(points_1, points_2, new_theta_1, new_theta_2, lambd)
        
        if util.costFunction(points_1, points_2, new_theta_1, new_theta_2, lambd) < treshhold_out:
            return np.array( [new_theta_1, new_theta_2] )
        #print new_theta_1
        #print new_theta_2
        
    return np.array( [new_theta_1, new_theta_2] )

if __name__ == '__main__':
    import sys
    try: fn1, fn2 = sys.argv[1:3]
    except:
        fn1 = '../c/box.png'
        fn2 = '../c/box_in_scene.png'
    print help_message

    img1 = cv2.imread(fn1, 0)
    img2 = cv2.imread(fn2, 0)

    surf = cv2.SURF(1000)
    kp1, desc1 = surf.detect(img1, None, False)
    kp2, desc2 = surf.detect(img2, None, False)
    desc1.shape = (-1, surf.descriptorSize())
    desc2.shape = (-1, surf.descriptorSize())
    print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))

    def match_and_draw(match, r_threshold):
        m = match(desc1, desc2, r_threshold)
        matched_p1 = np.array([kp1[i].pt for i, j in m])
        matched_p2 = np.array([kp2[j].pt for i, j in m])
        H, status = cv2.findHomography(matched_p1, matched_p2, cv2.RANSAC, 5.0)
        print '%d / %d  inliers/matched' % (np.sum(status), len(status))
        
        #print matched_p1
        #print "---------"
        #print matched_p2
                
        #Size of image
        size = (img1.shape[1] * 2, img1.shape[0] * 2)
        #Initial prepare(Shift images far way from corners
        #of resulting mosaic
        print 'center of imges '
        c_y, c_x = (np.asarray(img1.shape[:2]) / 2.).tolist()
        #c_y, c_x = (np.asarray(img2.shape[:2]) / 2.).tolist()
        
        
        #Translate matched points in the middle
        for i in range(matched_p1.shape[0]):
            matched_p1[i] -= np.array([c_x, c_y])
            matched_p2[i] -= np.array([c_x, c_y])
        #print matched_p1
        #print "---------"
        #print matched_p2
        
        #Parameters for Gradient Descent
        iterations = 1000
        gamma = 0.000002
        gamma_transl = 0.05
        #gamma = 10e-10
        lambd = 10e4
        #Intitial parameters
        theta_1 = np.array([0, 1., 1, 0, 0, 0, 0])
        theta_2 = np.array([0, 1., 1, 0, 0, 0, 0])
        
        src = np.array(matched_p1[0:3], np.float32)
        dst = np.array(matched_p2[0:3], np.float32)
        warp_affine = cv2.getAffineTransform(src, dst)
        
        print "warp_affine ", warp_affine
        
        theta_1[0] = -np.cos(warp_affine[0][0])
        theta_1[5] = -warp_affine[0][2]
        theta_1[6] = -warp_affine[1][2]
        
        print "warp_affine.ravel()", warp_affine.ravel()
        #theta_1 = np.concatenate((warp_affine.ravel(), [0, 0, 1]))
        print "theta_1", theta_1
         
        #Run Gradient
        t_1, t_2 = gradientDescent(iterations, matched_p1, matched_p2,
                                       theta_1, theta_2, gamma, lambd, gamma_transl,
                                       img1, img2, size, c_x, c_y)
        
        print 'points after transform'
        for i in range(matched_p1.shape[0]):
            err_1, err_2 = util.transformPoint(matched_p1[i], t_1), util.transformPoint(matched_p2[i], t_2)
            print err_1, err_2, "  --error--  ", np.abs(err_1 - err_2)
			
        print '---------------------------------------------------'
      
        result = util.stitch_for_visualization(img1, img2, t_1, t_2, c_x, c_y, size)
        util.visualize(img1, img2, matched_p1, matched_p2, t_1, t_2)
        
        cv2.imwrite('output.jpg', result)
    
        return None

    #print 'bruteforce match:',
    #vis_brute = match_and_draw( match_bruteforce, 0.75 )
    print 'flann match:',
    vis_flann = match_and_draw( match_flann, 0.2) # flann tends to find more distant second
                                                   # neighbours, so r_threshold is decreased
    #cv2.imshow('find_obj SURF', vis_brute)
    #cv2.imshow('find_obj SURF flann', vis_flann)
    
    #cv2.imwrite('out.jpg', vis_flann)
    #0xFF & cv2.waitKey()
    cv2.destroyAllWindows() 			
