import numpy as np
import cv2
from common import anorm
from functools import partial

import mosaic_cool as mos

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

def draw_match(img1, img2, p1, p2, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv2.polylines(vis, [corners], True, (255, 255, 255))
    
    if status is None:
        status = np.ones(len(p1), np.bool_)
    green = (0, 255, 0)
    red = (0, 0, 255)
    for (x1, y1), (x2, y2), inlier in zip(np.int32(p1), np.int32(p2), status):
        col = [red, green][inlier]
        if inlier:
            cv2.line(vis, (x1, y1), (x2+w1, y2), col)
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2+w1, y2), 2, col, -1)
        else:
            r = 2
            thickness = 3
            cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv2.line(vis, (x2+w1-r, y2-r), (x2+w1+r, y2+r), col, thickness)
            cv2.line(vis, (x2+w1-r, y2+r), (x2+w1+r, y2-r), col, thickness)
    return vis

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
        size = (img1.shape[0] * 2, img1.shape[1] * 2) #(4000, 4000)
        #Initial prepare(Shift images far way from corners
        #of resulting mosaic
        print 'center of img '
        c_y_1, c_x_1 = (np.asarray(img1.shape[:2]) / 2.).tolist()
        c_y_2, c_x_2 = (np.asarray(img2.shape[:2]) / 2.).tolist()
        print c_x_1, c_y_1
        print c_x_2, c_y_2
        
        
        #Translate matched points in the middle
        for i in range(matched_p1.shape[0]):
            matched_p1[i] -= np.array([c_x_1, c_y_1])
            matched_p2[i] -= np.array([c_x_2, c_y_2])
        #print matched_p1
        #print "---------"
        #print matched_p2
        
        #Parameters for Gradient Descent
        iterations = 600
        gamma = 0.0000002
        gamma_transl = 0.05
        lambd = 1000
        #Intitial parameters
        theta_1 = np.array([0, 1., 1, 0, 0, 0, 0])
        theta_2 = np.array([0, 1., 1, 0, 0, 0, 0])
         
        #Run Gradient
        t_1, t_2 = mos.gradientDescent(iterations, matched_p1, matched_p2,
                                       theta_1, theta_2, gamma, lambd, gamma_transl)
        
        #vis = draw_match(img1, img2, matched_p1, matched_p2, status, H)
        #vis = draw_match(cv2.warpPerspective(img1, H, (2272, 1704)), img2, 
        #                 matched_p1, matched_p2, status, H)
        
        print 'points after transform'
        for i in range(matched_p1.shape[0]):
            err_1, err_2 = mos.transformPoint(matched_p1[i], t_1), mos.transformPoint(matched_p2[i], t_2)
            print err_1, err_2, "--error--", np.abs(err_1 - err_2)
			
        print '---------------------------------------------------'
        m1 = mos.composeAffineMatrix(t_1)
        m2 = mos.composeAffineMatrix(t_2)
        #vis = draw_match(cv2.warpAffine(img1, m1, size), cv2.warpAffine(img2, m2, size),
        #                 matched_p1, matched_p2)
     
        print t_1[0], t_2[0]
        a1 = (t_1[0] * (180 / np.pi))
        a2 = (t_2[0] * (180 / np.pi))
        print a1, a2
        
        shift_x, shift_y = (0, 0)
        #Shift images far way from corners
        #of resulting mosaic
        #c_x_1 = c_x_1 + shift_x
        #c_y_1 = c_y_1 + shift_y
        #c_x_2 = c_x_2 + shift_x
        #c_y_2 = c_y_2 + shift_y
        
        print "First center", c_x_1 - t_1[5], c_y_1 - t_1[6]
        print "Second center", c_x_2 - t_2[5], c_y_2 - t_2[6]
        #print "Differemce", c_x_1 - c_x_2, c_y_1 - c_y_2
        
        initPrep_1 = np.array([[1., 0, c_x_1], [0, 1, c_y_1]])
        initPrep_2 = np.array([[1., 0, c_x_2], [0, 1, c_y_2]])
        dummy_1 = cv2.warpAffine(img1, initPrep_1, size)
        dummy_2 = cv2.warpAffine(img2, initPrep_2, size)
            
        print "scale 1", np.array([[t_1[1], t_1[4], 0], [t_1[3], t_1[2], 0]], np.float32)
        print "scale 2", np.array([[t_2[1], t_2[4], 0], [t_2[3], t_2[2], 0]], np.float32)
        
        scale_mat_1 = np.array([[t_1[1], t_1[4], 0], [t_1[3], t_1[2], 0]], np.float32)
        dummy_1 = cv2.warpAffine(dummy_1, scale_mat_1, size)                      
        scale_mat_2 = np.array([[t_2[1], t_2[4], 0], [t_2[3], t_2[2], 0]], np.float32)
        dummy_2 = cv2.warpAffine(dummy_2, scale_mat_2, size)
        
        rotat_mat_1 = cv2.getRotationMatrix2D((c_x_1*2, c_y_1*2), -a1, 1.0)
        dummy_1 = cv2.warpAffine(dummy_1, rotat_mat_1, size)
        rotat_mat_2 = cv2.getRotationMatrix2D((c_x_2*2, c_y_2*2), -a2, 1.0)
        dummy_2 = cv2.warpAffine(dummy_2, rotat_mat_2, size)
        
        #initPrep_1 = np.array([[1., 0, -t_1[5]], [0, 1, -t_1[6]] ])
        #initPrep_2 = np.array([[1., 0, -t_2[5]], [0, 1, -t_2[6]] ])
        #dummy_1 = cv2.warpAffine(img1, initPrep_1, size)
        #dummy_2 = cv2.warpAffine(img2, initPrep_2, size)
        m1[0][2] -= c_x_1
        m1[1][2] -= c_y_1
        m2[0][2] -= c_x_1
        m2[1][2] -= c_y_1
        print 'changing points'
        m1 = np.vstack([m1, [0, 0, 1]])
        m2 = np.vstack([m2, [0, 0, 1]])
        for i in range(matched_p1.shape[0]):
            #print np.zeros([matched_p1.shape[0], 1]) + 1
            matched_p1 = np.column_stack((matched_p1, np.zeros((matched_p1.shape[0], 1))))
            matched_p1[i][2] = 1
            print m1
            print matched_p1[i]
			#print matched_p1[i], np.transpose(matched_p1[i])
            matched_p1[i] = np.dot(m1, matched_p1[i])
            print matched_p1[i]
        print 'end changing points'
        print "This is the first matrix"
        print m1 
        print "This is the second matrix"
        print m2

        dummy_1 /= 2		
        dummy_2 /= 2		
        result = dummy_1 + dummy_2

        #output = np.ndarray(size, np.float64)
        #output += dummy_1
        #output += dummy_2
        #output /= 2
        #output2 = np.ndarray(size, np.uint8)
        #output2[:] = output
        cv2.imwrite('output.jpg', result)
    
        return None

    #print 'bruteforce match:',
    #vis_brute = match_and_draw( match_bruteforce, 0.75 )
    print 'flann match:',
    vis_flann = match_and_draw( match_flann, 0.2 ) # flann tends to find more distant second
                                                   # neighbours, so r_threshold is decreased
    #cv2.imshow('find_obj SURF', vis_brute)
    #cv2.imshow('find_obj SURF flann', vis_flann)
    
    #cv2.imwrite('out.jpg', vis_flann)
    #0xFF & cv2.waitKey()
    cv2.destroyAllWindows() 			
