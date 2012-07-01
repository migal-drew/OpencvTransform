'''
Created on May 7, 2012

@author: andrew
'''
import numpy as np
from numpy.ma.core import cos, sin
import cv2

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
        #print "Error-----------", error
    
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
    #print "Points in cost function", point_1, point_2
    error = np.sum(np.square( transformPoint(point_1, theta_1) -
        transformPoint(point_2, theta_2) ))
    J = J + error
    #print "Error-----------", error
    
    t_1 = theta_1.copy()
    t_2 = theta_2.copy()
    t_1[1:3] = np.abs([1, 1] - t_1[1:3]) #sx & sy approxim = 1
    t_2[1:3] = np.abs([1, 1] - t_2[1:3]) #sx & sy approxim = 1
    
    regularize = 0
    regularize += np.sum(np.square(t_1[3:5]) +
                                      np.square(t_2[3:5]))
    regularize += np.sum(np.square(t_1[0]) +
                                      np.square(t_2[0]))
    regularize *= lambd
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



def draw_distance_lines(img, p_1, p_2, matrix_1, matrix_2, c_x, c_y):
    p1 = p_1.copy()
    p2 = p_2.copy()
    green = (0, 255, 0)
    blue = (255, 0, 0)
    red = (0, 0, 255)
    
    col = (255, 255, 255, 0)
    
    for i in range(p1.shape[0]):
        
        newPoint_1 = transformPoint(p1[i], matrix_1)
        newPoint_2 = transformPoint(p2[i], matrix_2)
        
        newPoint_1[0] = newPoint_1[0] + c_x * 2;
        newPoint_1[1] = newPoint_1[1] + c_y * 2;
        newPoint_2[0] = newPoint_2[0] + c_x * 2;
        newPoint_2[1] = newPoint_2[1] + c_y * 2;
    
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

def stitch_for_visualization(img1, img2, t_1, t_2, c_x, c_y, size):
    new_img_1 = transform_for_opencv(img1, t_1, size, c_x, c_y)
    new_img_2 = transform_for_opencv(img2, t_2, size, c_x, c_y)
    
    new_img_1 /= 2
    new_img_2 /= 2
    
    result = new_img_1 + new_img_2
    
    return result

def visualize(img1, img2, points_1, points_2, new_theta_1, new_theta_2):
    
    size = (img1.shape[1] * 2, img1.shape[0] * 2)
    c_y, c_x = (np.asarray(img1.shape[:2]) / 2.).tolist()
    res = stitch_for_visualization(img1, img2, new_theta_1, new_theta_2, c_x, c_y, size)
    draw_distance_lines(res, points_1, points_2, new_theta_1, new_theta_2, c_x, c_y)
    cv2.imshow("RESULT", res)
    0xFF & cv2.waitKey()
    cv2.destroyAllWindows()         
