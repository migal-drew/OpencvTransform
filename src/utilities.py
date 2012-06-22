'''
Created on May 7, 2012

@author: andrew
'''
import numpy as np
from numpy.ma.core import cos, sin

#theta = np.array( [alpha, s_x, s_y, sk_x, sk_y, x_0, y_0] )

#Works lovely well!
#p = np.array( [x, y] )
def transformPoint(p, theta):
    alpha = theta[0]
    rot = np.array( [[cos(alpha), -sin(alpha)],
                     [sin(alpha), cos(alpha)]] )
    
    s_sk = np.array( [[theta[1], theta[4]],
                      [theta[3], theta[2]]] )
    
    trans = np.array( [theta[5], theta[6]] )
    
    #Scaling and skewing 
    ans = np.dot(s_sk, p)
    #Rotation
    ans = np.dot(rot, ans)
    #Translation
    ans = ans - trans
    #print ans
    return ans

def costFunction(points_1, points_2, theta_1, theta_2, lambd):
    m = points_1.shape[0]
    m = 1
    J = 0;
    for i in range(m):
        error = np.sum(np.square( transformPoint(points_1[i], theta_1) -
            transformPoint(points_2[i], theta_2) ))
        J = J + error
        print "Error-----------", error
    
    t_1 = theta_1.copy()
    t_2 = theta_2.copy()
    t_1[1:3] = np.abs([1, 1] - t_1[1:3]) #sx & sy approxim = 1
    t_2[1:3] = np.abs([1, 1] - t_2[1:3]) #sx & sy approxim = 1
    
    regularize = (lambd / m) * (np.sum(np.square(t_1[1:5]) +
                                      np.square(t_2[1:5])))
    #print regularize
    #print (lambd / (2 * m))
    J = J / m + regularize
    return J

def costFunctionOnePair(point_1, point_2, theta_1, theta_2, lambd):
    J = 0;
    print "Points in cost function", point_1, point_2
    error = np.sum(np.square( transformPoint(point_1, theta_1) -
        transformPoint(point_2, theta_2) ))
    J = J + error
    print "Error-----------", error
    
    t_1 = theta_1.copy()
    t_2 = theta_2.copy()
    t_1[1:3] = np.abs([1, 1] - t_1[1:3]) #sx & sy approxim = 1
    t_2[1:3] = np.abs([1, 1] - t_2[1:3]) #sx & sy approxim = 1
    
    regularize = (lambd) * (np.sum(np.square(t_1[1:5]) +
                                      np.square(t_2[1:5])))
    #print regularize
    #print (lambd / (2 * m))
    J = J + regularize
    return J

#Returns 2X3 matrix
def composeAffineMatrix(theta):
    alpha = theta[0]
    rot = np.array( [[cos(alpha), -sin(alpha)],
                     [sin(alpha), cos(alpha)]] )
    
    s_sk = np.array( [[theta[1], theta[4]],
                      [theta[3], theta[2]]] )
    
    trans = np.array( [theta[5], theta[6]] )
    rot_scale = np.dot(s_sk, rot)
    res = np.column_stack((rot_scale, trans))
    res = np.vstack((res, [0, 0, 1]))
    
    return res

def derivatives(p_1, p_2, theta_1, theta_2):
    x_1, y_1 = p_1.copy()
    x_2, y_2 = p_2.copy()
    alpha_1, sx_1, sy_1, skx_1, sky_1, x0_1, y0_1 = theta_1.copy()
    alpha_2, sx_2, sy_2, skx_2, sky_2, x0_2, y0_2 = theta_2.copy()
    
    ksi_1 = sx_1 * x_1 + sky_1 * y_1
    eps_1 = skx_1 * x_1 + sy_1 * y_1
    
    ksi_2 = sx_2 * x_2 + sky_2 * y_2
    eps_2 = skx_2 * x_2 + sy_2 * y_2
    
    XX = ksi_1 * cos(alpha_1) - eps_1 * sin(alpha_1) - x0_1 - ksi_2 * cos(alpha_2) + eps_2 * sin(alpha_2) + x0_2
    YY = ksi_1 * sin(alpha_1) + eps_1 * cos(alpha_1) - y0_1 - ksi_2 * sin(alpha_2) - eps_2 * cos(alpha_2) + y0_2
    
    #Resulting derivatives for 1 parameters
    deriv_1 = np.zeros(theta_1.size).reshape(theta_1.shape)
    #Resulting derivatives for 2 parameters
    deriv_2 = np.zeros(theta_2.size).reshape(theta_2.shape)
    
    #alpha
    deriv_1[0] = XX * (-ksi_1 * sin(alpha_1) - eps_1 * cos(alpha_1)) + \
                 YY * (ksi_1 * cos(alpha_1) - eps_1 * sin(alpha_1))
    #sx
    deriv_1[1] = XX * (x_1 * cos(alpha_1)) + YY * (x_1 * sin(alpha_1))
    #sy
    deriv_1[2] = XX * (-y_1 * sin(alpha_1)) + YY * (y_1 * cos(alpha_1))
    #skx
    deriv_1[3] = XX * (-x_1 * sin(alpha_1)) + YY * (x_1 * cos(alpha_1))
    #sky
    deriv_1[4] = XX * (y_1 * cos(alpha_1)) + YY * (y_1 * sin(alpha_1))
    #x0
    deriv_1[5] = XX * (-1)
    #y0
    deriv_1[6] = YY * (-1)
    
    #alpha
    deriv_2[0] = XX * (ksi_2 * sin(alpha_2) + eps_2 * cos(alpha_2)) + YY * (-ksi_2 * cos(alpha_2) + eps_2 * sin(alpha_2))
    #sx
    deriv_2[1] = XX * (-x_2 * cos(alpha_2)) + YY * (-x_2 * sin(alpha_2))
    #sy
    deriv_2[2] = XX * (y_2 * sin(alpha_2)) + YY * (-y_2 * cos(alpha_2))
    #skx
    deriv_2[3] = XX * (x_2 * sin(alpha_2)) + YY * (-x_2 * cos(alpha_2))
    #sky
    deriv_2[4] = XX * (-y_2 * cos(alpha_2)) + YY * (-y_2 * sin(alpha_2))
    #x0
    deriv_2[5] = XX
    #y0
    deriv_2[6] = YY
    
    #print XX
    #print YY
    return np.array([deriv_1, deriv_2])

#gamma - learning rate
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
            tmp_1, tmp_2  = derivatives(points_1[i], points_2[i], new_theta_1, new_theta_2)
            #print "deriv_1"
            #print tmp_1
            #print "deriv_2"
            #print tmp_2
            der_1 += tmp_1
            der_2 += tmp_2
            #print der_1
            #print der_2
            
#Visualization=--------------------------------        
        #if (k % 300 == 0):
            #res = mosaicing.stitch_for_visualization(img1, img2, new_theta_1, new_theta_2, c_x, c_y, size)
            #winname = "Iteration #" + (str)(k)
            #matrix_1 = composeAffineMatrix(new_theta_1)
            #matrix_2 = composeAffineMatrix(new_theta_2)
            #mosaicing.draw_distance_lines(res, points_1, points_2, new_theta_1, new_theta_2, c_x, c_y)
            #cv2.imshow(winname, res)
            #cv2.moveWindow(winname, 0, 0)
            #0xFF & cv2.waitKey()
            #cv2.destroyAllWindows() 
#Visualization=--------------------------------
        
        #Penalize
        der_1[3:5] = der_1[3:5] + lambd * new_theta_1[3:5]
        der_2[3:5] = der_2[3:5] + lambd * new_theta_2[3:5]
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
        
        print "verfic", der_1[0]
        print "verfic", der_2[0]
        
        treshhold_out = 0.01
        print costFunction(points_1, points_2, new_theta_1, new_theta_2, lambd)
        if costFunction(points_1, points_2, new_theta_1, new_theta_2, lambd) < treshhold_out:
            return np.array( [new_theta_1, new_theta_2] )
        #print new_theta_1
        #print new_theta_2
        
    return np.array( [new_theta_1, new_theta_2] )


        


if __name__ == '__main__':
    theta_1 = np.array([0, 1., 1, 0, 0, 0, 0])
    theta_2 = np.array([0, 1., 1, 0, 0, 0, 0])
#    p1 = np.array( [50, 150] )
#    p2 = np.array( [150, 150] )
#    p3 = np.array( [150, 50] )
#    p4 = np.array( [50, 50] )
    
    # House in center
    p1 = np.array([-50., -50])
    p2 = np.array([50., -50])
    p3 = np.array([50., 50])
    p4 = np.array([0., 100])
    p5 = np.array([-50., 50])
    #print p1.shape
    #print transformPoint(p1, theta)
    #print transformPoint(p2, theta)rcalc
    #print transformPoint(p3, theta)
    #print transformPoint(p4, theta)
    
    po_1 = np.array( [p1, p2, p3, p4, p5] )
    po_2 = po_1.copy()
    distor_1 = np.array([1.5, 1, 1, 0, 0, -1, 2])
    distor_2 = np.array([-1., 1, 1, 0, 0, 3, 0])
    for i in range(po_1.shape[0]):
        po_1[i] = transformPoint(po_1[i], distor_1)
    for i in range(po_1.shape[0]):
        po_2[i] = transformPoint(po_2[i], distor_2)
        
    #print "po_1"
    #s1,s2 = ""
    
    
    #print po_1
    #print po_2 
    gamma = 0.00001
    gamma_transl = 0.1
    #gamma_transl = gamma
    lambd = 1000
    t_1, t_2 = gradientDescent(1500, po_1, po_2, theta_1, theta_2, gamma, lambd, gamma_transl)
    #print po_1
    #print po_2
    s1 = ""
    s2 = ""
    for i in range(po_1.shape[0]):
        s1 += (str)(po_1[i][0]) + "," + (str)(po_1[i][1]) + "," 
        s2 += (str)(po_2[i][0]) + "," + (str)(po_2[i][1]) + ","
    #print "po_1"
    print s1
    #print "po_2"
    print s2
    
    s1 = ""
    s2 = ""
    for i in range(len(t_1)):
        s1 += (str)(t_1[i]) + ","  
        s2 += (str)(t_2[i]) + "," 
    #print "theta_1"
    print s1
    #print "theta_2"
    print s2
    
    print "!!!!!!!!! m1"
    m1 = composeAffineMatrix(t_1)
    print m1;
    print "!!!!!!!!! m2"
    m2 = composeAffineMatrix(t_2)
    print m2
    
    
