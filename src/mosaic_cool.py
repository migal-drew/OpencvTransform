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
    J = 0;
    for i in range(m):
        error = np.sum(np.square( transformPoint(points_1[i], theta_1) -
            transformPoint(points_2[i], theta_2) ))
        J = J + error
        print error
    
    #s_x, s_y normally = 1 => sum = sum - 4
    norm = 4
    regularize = (lambd / (2 * m)) * (np.sum((np.square(theta_1[1:5]) +
                                      np.square(theta_2[1:5]))) - norm)
    #regularize = regularize - 4 
    J = J / (2 * m) + regularize
    return J

def derivatives(p_1, p_2, theta_1, theta_2):
    x_1, y_1 = p_1
    x_2, y_2 = p_2
    alpha_1, sx_1, sy_1, skx_1, sky_1, x0_1, y0_1 = theta_1
    alpha_2, sx_2, sy_2, skx_2, sky_2, x0_2, y0_2 = theta_2
    
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
    deriv_1[0] = XX * (-ksi_1 * sin(alpha_1) - eps_1 * cos(alpha_1)) + YY * (ksi_1 * cos(alpha_1) - eps_1 * sin(alpha_1))
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
    
    #===========================================================================
    # #alpha
    # deriv_2[0] = XX * (ksi_2 * sin(alpha_2) + eps_2 * cos(alpha_2)) + YY * (-ksi_2 * cos(alpha_2) + eps_2 * sin(alpha_2))
    # #sx
    # deriv_2[1] = XX * (-x_2 * cos(alpha_2)) + YY * (-x_2 * sin(alpha_2))
    # #sy
    # deriv_2[2] = XX * (-y_1 * sin(alpha_1)) + YY * (y_1 * cos(alpha_1))
    # #skx
    # deriv_2[3] = XX * (-x_1 * sin(alpha_1)) + YY * (x_1 * cos(alpha_1))
    # #sky
    # deriv_2[4] = XX * (y_1 * cos(alpha_1)) + YY * (y_1 * sin(alpha_1))
    # #x0
    # deriv_2[5] = XX * (-1)
    # #y0
    # deriv_2[6] = YY * (-1)
    #===========================================================================
    deriv_2 = (-1) * deriv_1.copy()
    
    return np.array([deriv_1, deriv_2])

#def derivatives_2(p_1, p_2, theta_1, theta_2, lambd):
#    return None

#gamma - learning rate
def gradientDescent(points_1, points_2, theta_1, theta_2,
                    gamma, lambd):
    #new_theta_1 = np.zeros(theta_1.size).reshape(theta_1.shape)
    #new_theta_2 = np.zeros(theta_2.size).reshape(theta_2.shape)
    new_theta_1 = theta_1.copy()
    new_theta_2 = theta_2.copy()
    
    m = points_1.shape[0]
    
    for k in range(2000):
        #Derivatives - respect to theta_1 and theta_2
        der_1 = np.zeros(theta_1.size).reshape(theta_1.shape)
        der_2 = np.zeros(theta_2.size).reshape(theta_2.shape)
        
        for i in range(m):
            tmp_1, tmp_2  = derivatives(points_1[i], points_2[i], theta_1, theta_2)
            der_1 += tmp_1
            der_2 += tmp_2
        
        #Penalize
        der_1[1:5] = der_1[1:5] + lambd * new_theta_1[1:5]
        der_2[1:5] = der_2[1:5] + lambd * new_theta_2[1:5]
        #Refresh parameters
        new_theta_1 = new_theta_1 - gamma * der_1 / m
        new_theta_2 = new_theta_2 - gamma * der_2 / m
        
    return np.array( [new_theta_1, new_theta_2] )

if __name__ == '__main__':
    theta_1 = np.array([0, 1, 1, 0, 0, 0, 0])
    theta_2 = np.array([0, 1, 1, 0, 0, 0, 0])
    p1 = np.array( [50, 150] )
    p2 = np.array( [150, 150] )
    p3 = np.array( [150, 50] )
    p4 = np.array( [50, 50] )
    #print p1.shape
    #print transformPoint(p1, theta)
    #print transformPoint(p2, theta)
    #print transformPoint(p3, theta)
    #print transformPoint(p4, theta)
    
    po_1 = po_2 = np.array( [p1, p2, p3, p4] )
    #print po_1
    print costFunction(po_1, po_2, theta_1, theta_2, 0.1)
    