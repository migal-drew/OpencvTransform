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
    print ans
    return ans

def costFunction(points_1, points_2, theta_1, theta_2, lambd):
    m = points_1.shape[0]
    J = 0;
    for i in range(m):
        error = np.sum(np.square( transformPoint(points_1[i], theta_1) -
            transformPoint(points_2[i], theta_2) ))
        J = J + error
        print error
    
    regularize = (lambd / (2 * m)) * (np.sum((np.square(theta_1[1:5]) +
                                      np.square(theta_2[1:5]))) - 4)
                                      #s_x, s_y normally = 1 => sum = 4
    #regularize = regularize - 4 
    J = J / (2 * m) + regularize
    return J

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
    