import numpy as np
import cv2
from common import anorm
import utilities as util
import levenberg_marquardt as lma

help_message = '''SURF image match 

USAGE: [ <image1> <image2> ]
'''




FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing

flann_params = dict(algorithm = FLANN_INDEX_KDTREE,
                    trees = 4)

def match_flann(desc1, desc2, r_threshold = 0.6):
    flann = cv2.flann_Index(desc2, flann_params)
    idx2, dist = flann.knnSearch(desc1, 2, params = {}) # bug: need to provide empty dict
    mask = dist[:,0] / dist[:,1] < r_threshold
    idx1 = np.arange(len(desc1))
    pairs = np.int32( zip(idx1, idx2[:,0]) )
    return pairs[mask]

def stitch_for_visualization(img1, img2, t_1, t_2, c_x, c_y, size):
    new_img_1 = transform_for_opencv(img1, t_1, size, c_x, c_y)
    new_img_2 = transform_for_opencv(img2, t_2, size, c_x, c_y)
    
    new_img_1 /= 2
    new_img_2 /= 2
    
    result = new_img_1 + new_img_2
    
    return result

def draw_distance_lines(img, p_1, p_2, matrix_1, matrix_2, c_x, c_y):
    p1 = p_1.copy()
    p2 = p_2.copy()
    green = (0, 255, 0)
    blue = (255, 0, 0)
    red = (0, 0, 255)
    
    col = (255, 255, 255, 0)
    
    for i in range(p1.shape[0]):
        #p1 = np.hstack((p_1[i], [1]))
        #p2 = np.hstack((p_2[i], [1]))
        print "m1", matrix_1, matrix_1.shape
        print "p1", p1, p1.shape
        
        #newPoint_1 = np.dot(matrix_1, p1)
        #newPoint_2 = np.dot(matrix_2, p2)
        
        newPoint_1 = mos.transformPoint(p1[i], matrix_1)
        newPoint_2 = mos.transformPoint(p2[i], matrix_2)
        
        newPoint_1[0] = newPoint_1[0] + c_x * 2;
        newPoint_1[1] = newPoint_1[1] + c_y * 2;
        newPoint_2[0] = newPoint_2[0] + c_x * 2;
        newPoint_2[1] = newPoint_2[1] + c_y * 2;
    
        print "NewPoint_1", newPoint_1
        (x1, y1) = ((int)(newPoint_1[0]), (int)(newPoint_1[1]))
        (x2, y2) = ((int)(newPoint_2[0]), (int)(newPoint_2[1]))
        
        cv2.circle(img, (x1, y1), 6, col, -1)
        cv2.circle(img, (x2, y2), 6, col, -1)
        
        cv2.line(img, (x1, y1), (x2, y2), col, 1)
    

def transform_for_opencv(img, t, size, c_x, c_y):
    #Rotation angles in degrees
    a1 = (t[0] * (180 / np.pi))
    
    #shift_x, shift_y = (800, 800)
    #Shift images far way from corners
    #of resulting mosaic
    #c_x = c_x + shift_x
    #c_y = c_y + shift_y
    #c_x = c_x + shift_x
    #c_y = c_y + shift_y
    
    initPrep_1 = np.array([[1., 0, c_x], [0, 1, c_y]])
    new_img = cv2.warpAffine(img, initPrep_1, size)
        
    #print "scale 1", np.array([[t[1], t[4], 0], [t[3], t[2], 0]], np.float32)
    #print "scale 2", np.array([[t_2[1], t_2[4], 0], [t_2[3], t_2[2], 0]], np.float32)
    
    #new_img = cv2.warpAffine(new_img, m1, size)
    #dummy_2 = cv2.warpAffine(dummy_2, m2, size)
    
    #Scaling and skewing
    mat_deform_1 = np.array([[t[1], t[4], 0], [t[3], t[2], 0]], np.float32)
    new_img = cv2.warpAffine(new_img, mat_deform_1, size)                      
    
    #Restore images' centers after skewing
    cntr = np.transpose(np.array([c_x, c_y, 1]))
    add_row = np.array([0, 0, 1])
    mat_deform_full_1 = np.vstack((mat_deform_1, add_row))
    cntr_err_1 = np.dot(mat_deform_full_1, cntr)
    diff_1 = (cntr - cntr_err_1) * 2
    mat_restore_1 = np.array([[1, 0, diff_1[0]], [0, 1, diff_1[1]]], np.float32)
    new_img = cv2.warpAffine(new_img, mat_restore_1, size)
    
    #Rotation
    mat_rot = cv2.getRotationMatrix2D((c_x * 2, c_y * 2), -a1, 1.0)
    new_img = cv2.warpAffine(new_img, mat_rot, size)

    #Translation      
    mat_trans_1 = np.array([[1, 0, -t[5]], [0, 1, -t[6]] ], np.float32)
    new_img = cv2.warpAffine(new_img, mat_trans_1, size)
    
    return new_img

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
        
        #Intitial parameters
        theta_1 = np.array([0, 1., 1, 0, 0, 0, 0])
        theta_2 = np.array([0, 1., 1, 0, 0, 0, 0])
         
        #for LMA 
        lam = 10
        penalty = 10e2
        threshold = 0.01
        #Run LEVENBERG-MARQUARDT
        params = lma.levenberg_marquardt(matched_p1,
                                         matched_p2,
                                         theta_1,
                                         theta_2,
                                         lam,
                                         penalty, 
                                         threshold)
        
        t_1 = params[0:params.size / 2] 
        t_2 = params[params.size / 2:]
        
        print "T_1", t_1
        print "T_2", t_2
        
        print 'points after transform'
        for i in range(matched_p1.shape[0]):
            err_1, err_2 = util.transformPoint(matched_p1[i], t_1), util.transformPoint(matched_p2[i], t_2)
            print err_1, err_2, "  --error--  ", np.abs(err_1 - err_2)
			
        print '---------------------------------------------------'
      
        result = stitch_for_visualization(img1, img2, t_1, t_2, c_x, c_y, size)
        
        cv2.imwrite('output.jpg', result)
    
        return None

    #print 'bruteforce match:',
    #vis_brute = match_and_draw( match_bruteforce, 0.75 )
    print 'flann match:',
    vis_flann = match_and_draw( match_flann, 0.25) # flann tends to find more distant second
                                                   # neighbours, so r_threshold is decreased
    #cv2.imshow('find_obj SURF', vis_brute)
    #cv2.imshow('find_obj SURF flann', vis_flann)
    
    #cv2.imwrite('out.jpg', vis_flann)
    #0xFF & cv2.waitKey()
    cv2.destroyAllWindows() 			
