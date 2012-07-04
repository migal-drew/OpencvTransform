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
         
        src = np.array(matched_p1[0:3], np.float32)
        dst = np.array(matched_p2[0:3], np.float32)
        warp_affine = cv2.getAffineTransform(src, dst)
        
        print "warp_affine ", warp_affine
        
        theta_1[0] = -np.cos(warp_affine[0][0])
        theta_1[5] = -warp_affine[0][2]
        theta_1[6] = -warp_affine[1][2] 
         
        #for LMA
        lam = 10.0
        penalty = 10e3
        threshold = 0.001
        #Run LEVENBERG-MARQUARDT
        params = lma.levenberg_marquardt(matched_p1,
                                         matched_p2,
                                         theta_1,
                                         theta_2,
                                         lam,
                                         penalty, 
                                         threshold,
                                         img1,
                                         img2)
        
        t_1 = params[0:params.size / 2] 
        t_2 = params[params.size / 2:]
        
        print "T_1", t_1
        print "T_2", t_2
        
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
    vis_flann = match_and_draw( match_flann, 0.25) # flann tends to find more distant second
                                                   # neighbours, so r_threshold is decreased
    #cv2.imshow('find_obj SURF', vis_brute)
    #cv2.imshow('find_obj SURF flann', vis_flann)
    
    #cv2.imwrite('out.jpg', vis_flann)
    #0xFF & cv2.waitKey()
    cv2.destroyAllWindows() 			
