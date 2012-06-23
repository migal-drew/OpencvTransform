import numpy as np
from numpy.ma.core import cos, sin
import utilities as util

def derivatives_with_penalty(p_1, p_2, theta_1, theta_2, lambd):
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
    
    #Penalize
    #deriv_1[0] = deriv_1[0] + lambd * theta_1[0]
    #deriv_2[0] = deriv_2[0] + lambd * theta_2[0]
    deriv_1[3:5] = deriv_1[3:5] + lambd * theta_1[3:5]
    deriv_2[3:5] = deriv_2[3:5] + lambd * theta_2[3:5]
    
    dummy_1 = lambd * np.abs(np.array([1., 1]) - theta_1[1:3])
    dummy_2 = lambd * np.abs(np.array([1., 1]) - theta_2[1:3])
    deriv_1[1:3] = deriv_1[1:3] + dummy_1#lambd * np.abs([1., 1] - der_1[1:3]) #sx, sy = 1
    deriv_2[1:3] = deriv_2[1:3] + dummy_2#lambd * np.abs([1., 1] - der_2[1:3]) #sx, sy = 1
     
    deriv_1[1:3] = np.array([0, 0]);
    deriv_2[1:3] = np.array([0, 0]);
    
    return np.concatenate((deriv_1, deriv_2))

def getJacobian(points_1, points_2, params, penalty):
    
    #Number of points
    m = points_1.shape[0]
    Jacobian = np.zeros([m, params.size])
    
    #print "params 1 ", params[0:params.size/2]
    for i in range(m):
        Jacobian[i] = derivatives_with_penalty(points_1[i],
                                               points_2[i],
                                               params[0:params.size/2],
                                               params[params.size/2:],
                                               penalty / m)
    
    return Jacobian

def getGradient(Jacobian, points_1, points_2, theta_1, theta_2, penalty):
    n = theta_1.size + theta_2.size
    
    m = points_1.shape[0]
    print "test ", points_1[3]
    
    values = np.zeros((m, 1))
    print values
    
    for i in range(m):
        print "Evaluate gradient, stage ", i
        print points_1[i], points_2[i]
        values[i] = util.costFunctionOnePair(points_1[i], points_2[i], theta_1, theta_2, penalty / m)
    
    return np.dot(np.transpose(Jacobian), values)

def levenberg_marquardt(points_1, points_2, theta_1, theta_2,
                        lambd, penalty, threshold):
    #Number of points
    m = points_1.shape[0]
    #Vector of parameters
    x = np.concatenate((theta_1, theta_2))
    len_x = x.size
    #Init 
    x_old = x + threshold * 2
    delta_x = np.zeros(x.shape)
    #Needful Matrices
    #Jacobian = np.zeros([m, x.size])
    I = np.identity(len_x, np.float32)
    
    iteration = -1
    
    print "start"
    print np.sum(np.abs(x_old - x) / len_x)
    
    flag = True
    while ( (np.sum(np.abs(x_old - x) / len_x) and (flag) ) > threshold):
        iteration += 1
        print "Iteration # ", iteration
        print "Cost function= ", util.costFunction(points_1,
                                              points_2,
                                              x[0:x.size/2],
                                              x[x.size/2:],
                                              penalty)
        #For every point    
        Jacobian = getJacobian(points_1, points_2, x, penalty)
        #print "Jacobian shape", Jacobian
        Hessian = np.matrix(np.dot(np.transpose(Jacobian), Jacobian))
        print 'Hessian shape', Hessian.shape
        print "H det = ", np.linalg.det(Hessian)
        
        print "Sum of derivatives for 1st parameter = ", np.sum(Jacobian[:][0])
        
        gradient = getGradient(Jacobian, points_1, points_2, x[0:x.size/2], x[x.size/2:], penalty)
        
        #print "Jacobian", Jacobian
        
        #tmp = np.matrix(Hessian + lambd * I)
        #print "TMP", tmp
        #delta_x = np.dot(tmp.I, -gradient)
        delta_x = np.linalg.lstsq(Hessian + lambd * I, -gradient)
        
        #print "x", x
        #print"delta_x", delta_x[0]
        
        x_old = x.copy()
        x = x + np.transpose(delta_x[0])
        #print "delta_x[0]", delta_x[0]
        
        print "this is NEW x", x[0:7]
        
        if (util.costFunction(points_1, points_2, x[0:7], x[7:14], lambd) >=
            util.costFunction(points_1, points_2, x_old[0:7], x_old[7:14], lambd)):
            lambd *=2
            flag = True
        else:
            lambd = 10
            flag = False
        
        
        #x[0] = x[0] + np.transpose(delta_x[0][0])
        #x[3:7] = x[3:7] + np.transpose(delta_x[0][3:7])
        #x[8] = x[8] + np.transpose(delta_x[0][8])
        #x[10:14] = x[10:14] + np.transpose(delta_x[0][10:14])
        
        
        
    return x
            