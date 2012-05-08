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
    return ans

def costFunction(points_1, points_2, theta_1, theta_2):
    
    return None

if __name__ == '__main__':
    theta = np.array([0, 2, 2, 0, 0, 0, 0])
    p1 = np.array( [50, 150] )
    p2 = np.array( [150, 150] )
    p3 = np.array( [150, 50] )
    p4 = np.array( [50, 50] )
    print transformPoint(p1, theta)
    print transformPoint(p2, theta)
    print transformPoint(p3, theta)
    print transformPoint(p4, theta)
    