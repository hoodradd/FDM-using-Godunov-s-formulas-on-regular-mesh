#!/usr/bin/env python
# coding: utf-8

# In[120]:


import numpy as np 
import numpy.typing as npt
import matplotlib.pyplot as plt
from numba import njit, prange
from time import time 


# In[100]:


def make_imaginary_condition(condition: npt.ArrayLike) -> npt.ArrayLike: 
    new_shape = np.array(condition.shape) + np.array([2,2,0])
    imc = np.zeros(shape = new_shape, dtype = np.float64)
    imc[1:new_shape[0] - 1, 1:new_shape[1] - 1] = condition
    
    #using the border condition
    imc[1:new_shape[0] - 1, 0, :] = imc[1:new_shape[0] - 1, 1, :]
    imc[1:new_shape[0] - 1, - 1, :] = imc[1:new_shape[0] - 1, -2, :]
    imc[0, 1:new_shape[1] - 1, :] = imc[1, 1:new_shape[1] - 1, :]
    imc[-1, 1:new_shape[1] - 1, :] = imc[-2, 1:new_shape[1] - 1, :]   
        
    # imc[1:new_shape[0] - 1, 0, 0] = -imc[1:new_shape[0] - 1, 1, 0]
    # self.imc[1:new_shape[0] - 1, -1, 1] = -self.imc[1:new_shape[0] - 1, -2, 1]
    # imc[0, 1:new_shape[1] - 1, 1] = -imc[1, 1:new_shape[1] - 1, 1]
    # imc[-1, 1:new_shape[1] - 1, 1] = -imc[-2, 1:new_shape[1] - 1, 1]   
    return imc 


# In[101]:


def update_imc(imc: npt.ArrayLike):
    shape = imc.shape
    imc[1:shape[0] - 1, 0, :] = imc[1:shape[0] - 1, 1, :]
    imc[1:shape[0] - 1, - 1, :] = imc[1:shape[0] - 1, -2, :]
    imc[0, 1:shape[1] - 1, :] = imc[1, 1:shape[1] - 1, :]
    imc[-1, 1:shape[1] - 1, :] = imc[-2, 1:shape[1] - 1, :]   
        
    # imc[1:shape[0] - 1, 0, 0] = -imc[1:shape[0] - 1, 1, 0]
    # self.imc[1:new_shape[0] - 1, -1, 1] = -self.imc[1:new_shape[0] - 1, -2, 1]
    # imc[0, 1:shape[1] - 1, 1] = -imc[1, 1:shape[1] - 1, 1]
    # imc[-1, 1:shape[1] - 1, 1] = -imc[-2, 1:shape[1] - 1, 1]   


# In[102]:


@njit
def f(P, p_k, rho_k, hamma):
    pi_k = P/p_k
    c_k = np.sqrt(hamma*p_k/rho_k)
    if P > p_k or abs(P - p_k) < 1e-10:
        return (P-p_k)/(rho_k*c_k*np.sqrt((hamma + 1)*pi_k/(2*hamma) + (hamma - 1)/(2*hamma)))
    else:
        return 2*c_k/(hamma - 1)*(pi_k**((hamma - 1)/(2*hamma)) - 1)


# In[103]:


@njit
def df(P, p_k, rho_k, hamma):
    pi_k = P/p_k
    c_k = np.sqrt(hamma*p_k/rho_k)
    if P > p_k or abs(P - p_k) < 1e-10:
        return ((hamma+1)*pi_k + (3*hamma - 1))/(4*hamma*rho_k*c_k*np.power((hamma + 1)*pi_k/(2*hamma) + (hamma - 1)/(2*hamma), 1.5))
    else:
        return c_k*pi_k**((hamma - 1)/(2*hamma))/(hamma*P)


# In[104]:


@njit
def riemann_problem_p(hamma1, u1, rho1, p1, hamma2, u2, rho2, p2):
    if p1 > p2:
        hamma_1 = hamma2
        hamma_2 = hamma1
        p_1 = p2
        u_1 = -u2
        rho_1 = rho2
        p_2 = p1
        u_2 = -u1
        rho_2 = rho1
        flag = False

    else:         
        hamma_1 = hamma1
        hamma_2 = hamma2
        p_1 = p1
        u_1 = u1
        rho_1 = rho1
        p_2 = p2
        u_2 = u2
        rho_2 = rho2
        flag = True
        
    c_1 = np.sqrt(hamma_1*(p_1/rho_1))
    c_2 = np.sqrt(hamma_2*(p_2/rho_2))
    #checking what type of gap is it
    U_vac = -2*c_1/(hamma_1 - 1) - 2*c_2/(hamma_2 - 1)
    #checking for vacuum situation
    if U_vac > u_1 - u_2:
        return -1
    else:
        p = p_1
        p_new = p
        #Newton method for finding the density on the gap 
        while True:
            p = p_new
            p_new = p - (f(p,p_1,rho_1,hamma_1) + f(p,p_2,rho_2,hamma_2) - (u_1 - u_2))/(df(p,p_1,rho_1,hamma_1) + df(p,p_2,rho_2,hamma_2))
            if p_new < 0: 
                p_new = 0.5*p
                continue
            else: 
                if np.abs(p - p_new) < 1e-7:
                    break
        #velocity of contact gap
        return p
    


# In[105]:


@njit
def left_syssolve(hamma, u_1, p_1, rho_1, db = 0):
    a_1 = np.sqrt(hamma*p_1/rho_1)
    Y_plus = (u_1 + 2*a_1/(hamma - 1))
    u = (hamma - 1)/(hamma + 1)*Y_plus + 2*db/(hamma + 1)
    a = u - db
    rho = np.power(np.power(a, 2)*np.power(rho_1, hamma)/(hamma*p_1), 1/(hamma - 1))
    p = p_1*np.power(rho, hamma)/np.power(rho_1, hamma)
    return u, rho, p


# In[106]:


#solving system for dot in right rarefactions waves area  
@njit
def right_syssolve(hamma, u_2, p_2, rho_2, db = 0):
    a_2 = np.sqrt(hamma*p_2/rho_2)
    Y_minus = (u_2 - 2*a_2/(hamma - 1))
    u = (hamma - 1)/(hamma + 1)*Y_minus + 2*db/(hamma + 1)
    a = db - u
    rho = np.power(np.power(a, 2)*np.power(rho_2, hamma)/(hamma*p_2), 1/(hamma - 1))
    p = p_2*np.power(rho, hamma)/np.power(rho_2, hamma)
    return u, rho, p


# In[107]:


class VacuumError(ValueError):
    pass

class AxisError(ValueError):
    pass

R = 0.83143e04
R = R*1e4/1e6 


# In[108]:


@njit
def riemprob_for_border_x(u_1, v_1, rho_1, p_1, hamma_1, u_2, v_2, rho_2, p_2, hamma_2):
    type_right: int
    type_left: int
    c_1 = np.sqrt(hamma_1*(p_1/rho_1))
    c_2 = np.sqrt(hamma_2*(p_2/rho_2))

    #finding pressure on contact gap
    p = riemann_problem_p(hamma_1, u_1, rho_1, p_1, hamma_2, u_2, rho_2, p_2)

    alpha_1 = np.sqrt(rho_1*((hamma_1 + 1)*p/2 + (hamma_1 - 1)*p_1/2))
    alpha_2 = np.sqrt(rho_2*((hamma_2 + 1)*p/2 + (hamma_2 - 1)*p_2/2))
    U = (u_1 + u_2 - f(p, p_1, rho_1, hamma_1) + f(p, p_2, rho_2, hamma_2))/2.0
    #shock wave on the left
    if p > p_1:
        R_1 = rho_1*((hamma_1 + 1)*p + (hamma_1 - 1)*p_1)/((hamma_1 - 1)*p + (hamma_1 + 1)*p_1)
        D_1 = u_1 - alpha_1 / rho_1
        type_left = 1
    #rarefraction wave on the left
    else:
        R_1 = rho_1*np.power(p / p_1, 1 / hamma_1) 
        D_1_r1 = u_1 - c_1
        c_1_ = c_1 + (hamma_1 - 1)*(u_1 - U)/2
        D_1_r2 = U - c_1_
        type_left = 2
    #shock wave on the right    
    if p > p_2:
        R_2 = rho_2*((hamma_2 + 1)*p + (hamma_2 - 1)*p_2)/((hamma_2 - 1)*p + (hamma_2 + 1)*p_2)
        D_2 = u_2 + alpha_2 / rho_2
        type_right = 1
    #rarefraction wave on the left
    else:
        R_2 = rho_2*np.power(p / p_2, 1 / hamma_2)
        D_2_r1 = u_2 + c_2
        c_2_ = c_2 - (hamma_2 - 1)*(u_2 - U)/2
        D_2_r2 = U + c_2_
        type_right = 2

    if type_left == 1 and type_right == 1:
        if D_1 > 0: 
            return np.array([u_1, v_1, rho_1, p_1, hamma_1])
        elif U > 0:
            return np.array([rho_1*u_1/R_1, v_1, R_1, p, hamma_1])
        elif D_2 > 0:
            return np.array([rho_2*u_2/R_2, v_2, R_2, p, hamma_2])
        else: 
            return np.array([u_2, v_2, rho_2, p_2, hamma_2])
        
    elif type_left == 1 and type_right == 2:
        if D_1 > 0: 
            return np.array([u_1, v_1, rho_1, p_1, hamma_1])
        elif U > 0:
            return np.array([rho_1*u_1/R_1, v_1, R_1, p, hamma_1])
        elif D_2_r2 > 0: 
            return np.array([U, v_2, R_2, p, hamma_2])
        elif D_2_r1 > 0: 
            U, R, P = right_syssolve(hamma_2, u_2, p_2, rho_2) 
            return np.array([U, v_2, R, P, hamma_2])
        else: 
            return np.array([u_2, v_2, rho_2, p_2, hamma_2])
    elif type_left == 2 and type_right == 1:
        if D_1_r1 > 0: 
            return np.array([u_1, v_1, rho_1, p_1, hamma_1])
        elif D_1_r2 > 0: 
            U, R, P = left_syssolve(hamma_1, u_1, p_1, rho_1) 
            return np.array([U, v_1, R, P, hamma_1])
        elif U > 0: 
            return np.array([U, v_1, R_1, p, hamma_1])
        elif D_2 > 0:
            return np.array([rho_2*u_2/R_2, v_2, R_2, p, hamma_2])
        else: 
            return np.array([u_2, v_2, rho_2, p_2, hamma_2])
    else:
        if D_1_r1 > 0: 
            return np.array([u_1, v_1, rho_1, p_1, hamma_1])
        elif D_1_r2 > 0: 
            U, R, P = left_syssolve(hamma_1, u_1, p_1, rho_1) 
            return np.array([U, v_1, R, P, hamma_1])
        elif U > 0: 
            return np.array([U, v_1, R_1, p, hamma_1])
        elif D_2_r2 > 0: 
            return np.array([U, v_2, R_2, p, hamma_2])
        elif D_2_r1 > 0: 
            U, R, P = right_syssolve(hamma_2, u_2, p_2, rho_2) 
            return np.array([U, v_2, R, P, hamma_2])
        else: 
            return np.array([u_2, v_2, rho_2, p_2, hamma_2])


# In[109]:


# #test 1
# riemprob_for_border_x(0.75, 0.0, 1.0, 1.0, 1.4, 0.0, 0.0, 0.125, 0.1, 1.4)
# #test 2
# riemprob_for_border_x(-2.0, 0.0, 1.0, 0.4, 1.4, 2.0, 0.0, 1.0, 0.4, 1.4)


# In[110]:


@njit
def riemprob_for_border_y(u_1, v_1, rho_1, p_1, hamma_1, u_2, v_2, rho_2, p_2, hamma_2):
    type_right: int
    type_left: int
    c_1 = np.sqrt(hamma_1*(p_1/rho_1))
    c_2 = np.sqrt(hamma_2*(p_2/rho_2))
    
    #finding pressure on contact gap
    p = riemann_problem_p(hamma_1, v_1, rho_1, p_1, hamma_2, v_2, rho_2, p_2)

    alpha_1 = np.sqrt(rho_1*((hamma_1 + 1)*p/2 + (hamma_1 - 1)*p_1/2))
    alpha_2 = np.sqrt(rho_2*((hamma_2 + 1)*p/2 + (hamma_2 - 1)*p_2/2))
    V = (v_1 + v_2 - f(p, p_1, rho_1, hamma_1) + f(p, p_2, rho_2, hamma_2))/2.0
    
    #shock wave on the left
    if p > p_1:
        R_1 = rho_1*((hamma_1 + 1)*p + (hamma_1 - 1)*p_1)/((hamma_1 - 1)*p + (hamma_1 + 1)*p_1)
        D_1 = v_1 - alpha_1 / rho_1
        type_left = 1

    #rarefraction wave on the left
    else:
        R_1 = rho_1*np.power(p / p_1, 1 / hamma_1) 
        D_1_r1 = v_1 - c_1
        c_1_ = c_1 + (hamma_1 - 1)*(v_1 - V)/2
        D_1_r2 = V - c_1_
        type_left = 2

    #shock wave on the right    
    if p > p_2:
        R_2 = rho_2*((hamma_2 + 1)*p + (hamma_2 - 1)*p_2)/((hamma_2 - 1)*p + (hamma_2 + 1)*p_2)
        D_2 = v_2 + alpha_2 / rho_2
        type_right = 1

    #rarefraction wave on the left
    else:
        R_2 = rho_2*np.power(p / p_2, 1 / hamma_2)
        D_2_r1 = v_2 + c_2
        c_2_ = c_2 - (hamma_2 - 1)*(v_2 - V)/2
        D_2_r2 = V + c_2_
        type_right = 2

    if type_left == 1 and type_right == 1:
        if D_1 > 0: 
            return np.array([u_1, v_1, rho_1, p_1, hamma_1])
        elif V > 0:
            return np.array([u_1, rho_1*v_1/R_1, R_1, p, hamma_1])
        elif D_2 > 0:
            return np.array([u_2, rho_2*v_2/R_2, R_2, p, hamma_2])
        else: 
            return np.array([u_2, v_2, rho_2, p_2, hamma_2])
    elif type_left == 1 and type_right == 2:
        if D_1 > 0: 
            return np.array([u_1, v_1, rho_1, p_1, hamma_1])
        elif V > 0:
            return np.array([u_1, rho_1*v_1/R_1, R_1, p, hamma_1])   
        elif D_2_r2 > 0: 
            return np.array([u_2, V, R_2, p, hamma_2])
        elif D_2_r1 > 0: 
            V, R, P = right_syssolve(hamma_2, v_2, p_2, rho_2) 
            return np.array([u_2, V, R, P, hamma_2])
        else: 
            return np.array([u_2, v_2, rho_2, p_2, hamma_2])
    elif type_left == 2 and type_right == 1:
        if D_1_r1 > 0: 
            return np.array([u_1, v_1, rho_1, p_1, hamma_1])
        elif D_1_r2 > 0: 
            V, R, P = left_syssolve(hamma_1, v_1, p_1, rho_1) 
            return np.array([u_1, V, R, P, hamma_1])
        elif V > 0: 
            return np.array([u_1, V, R_1, p, hamma_1])
        elif D_2 > 0:
            return np.array([u_2, rho_2*v_2/R_2, R_2, p, hamma_2])
        else: 
            return np.array([u_2, v_2, rho_2, p_2, hamma_2])
    else:
        if D_1_r1 > 0: 
            return np.array([u_1, v_1, rho_1, p_1, hamma_1])
        elif D_1_r2 > 0: 
            V, R, P = left_syssolve(hamma_1, v_1, p_1, rho_1) 
            return np.array([u_1, V, R, P, hamma_1])
        elif V > 0: 
            return np.array([u_1, V, R_1, p, hamma_1])
        elif D_2_r2 > 0: 
            return np.array([u_2, V, R_2, p, hamma_2])
        elif D_2_r1 > 0: 
            V, R, P = right_syssolve(hamma_2, v_2, p_2, rho_2) 
            return np.array([u_2, V, R, P, hamma_2])
        else: 
            return np.array([u_2, v_2, rho_2, p_2, hamma_2])


# In[111]:


# #test 1
# riemprob_for_border_y(0.0, 0.75, 1.0, 1.0, 1.4, 0.0, 0.0, 0.125, 0.1, 1.4)
# #test 2
# riemprob_for_border_y(0.0, -2.0, 1.0, 0.4, 1.4, 0.0, 2.0, 1.0, 0.4, 1.4)


# @njit
# def riemprob_for_border(u_1, v_1, rho_1, p_1, hamma_1, u_2, v_2, rho_2, p_2, hamma_2, axis):
#     type_right: int
#     type_left: int
#     c_1 = np.sqrt(hamma_1*(p_1/rho_1))
#     c_2 = np.sqrt(hamma_2*(p_2/rho_2))
#     #finding pressure on contact gap
#     if axis == 0:
#         try:
#             p = riemann_problem_p(hamma_1, u_1, rho_1, p_1, hamma_2, u_2, rho_2, p_2)
#             if p < 0: 
#                 raise VacuumError
#         except VacuumError:
#             print('Vacuum - impossible situation')
#     elif axis == 1:
#         try:
#             p = riemann_problem_p(hamma_1, v_1, rho_1, p_1, hamma_2, v_2, rho_2, p_2)
#             if p < 0: 
#                 raise VacuumError
#         except VacuumError:
#             print('Vacuum - impossible situation')
#     else:
#         raise AxisError("Wrong axis")
#     alpha_1 = np.sqrt(rho_1*((hamma_1 + 1)*p/2 + (hamma_1 - 1)*p_1/2))
#     alpha_2 = np.sqrt(rho_2*((hamma_2 + 1)*p/2 + (hamma_2 - 1)*p_2/2))
#     U = (u_1 + u_2 - f(p, p_1, rho_1, hamma_1) + f(p, p_2, rho_2, hamma_2))/2.0
#     #shock wave on the left
#     if p > p_1:
#         R_1 = rho_1*((hamma_1 + 1)*p + (hamma_1 - 1)*p_1)/((hamma_1 - 1)*p + (hamma_1 + 1)*p_1)
#         D_1 = u_1 - alpha_1 / rho_1
#         type_left = 1
#     #rarefraction wave on the left
#     else:
#         R_1 = rho_1*np.power(p / p_1, 1 / hamma_1) 
#         D_1_r1 = u_1 - c_1
#         c_1_ = c_1 + (hamma_1 - 1)*(u_1 - U)/2
#         D_1_r2 = U - c_1_
#         type_left = 2
#     #shock wave on the right    
#     if p > p_2:
#         R_2 = rho_2*((hamma_2 + 1)*p + (hamma_2 - 1)*p_2)/((hamma_2 - 1)*p + (hamma_2 + 1)*p_2)
#         D_2 = u_2 + alpha_2 / rho_2
#         type_right = 1
#     #rarefraction wave on the left
#     else:
#         R_2 = rho_2*np.power(p / p_2, 1 / hamma_2)
#         D_2_r1 = u_2 + c_2
#         c_2_ = c_2 - (hamma_2 - 1)*(u_2 - U)/2
#         D_2_r2 = U + c_2_
#         type_right = 2
#     if type_left == 1 and type_right == 1:
#         if D_1 > 0: 
#             return np.array([u_1, v_1, rho_1, p_1, hamma_1])
#         elif U > 0:
#             if axis == 0:
#                 return np.array([rho_1*u_1/R_1, v_1, R_1, p, hamma_1])
#             elif axis == 1: 
#                 return np.array([u_1, rho_1*v_1/R_1, R_1, p, hamma_1])
#         elif D_2 > 0:
#             if axis == 0:
#                 return np.array([rho_2*u_2/R_2, v_2, R_2, p, hamma_2])
#             elif axis == 1: 
#                 return np.array([u_2, rho_2*v_2/R_2, R_2, p, hamma_2])
#         else: 
#             return np.array([u_2, v_2, rho_2, p_2, hamma_2])
#     elif type_left == 1 and type_right == 2:
#         if D_1 > 0: 
#             return np.array([u_1, v_1, rho_1, p_1, hamma_1])
#         elif U > 0:
#             if axis == 0:
#                 return np.array([rho_1*u_1/R_1, v_1, R_1, p, hamma_1])
#             elif axis == 1: 
#                 return np.array([u_1, rho_1*v_1/R_1, R_1, p, hamma_1])   
#         elif D_2_r2 > 0: 
#             if axis == 0: 
#                 return np.array([U, v_2, R_2, p, hamma_2])
#             elif axis == 1: 
#                 return np.array([u_2, U, R_2, p, hamma_2])
#         elif D_2_r1 > 0: 
#             if axis == 0:
#                 U, R, P = right_syssolve(hamma_2, u_2, p_2, rho_2) 
#                 return np.array([U, v_2, R, P, hamma_2])
#             elif axis == 1: 
#                 V, R, P = right_syssolve(hamma_2, v_2, p_2, rho_2) 
#                 return np.array([u_2, V, R, P, hamma_2])
#         else: 
#             return np.array([u_2, v_2, rho_2, p_2, hamma_2])
#     elif type_left == 2 and type_right == 1:
#         if D_1_r1 > 0: 
#             return np.array([u_1, v_1, rho_1, p_1, hamma_1])
#         elif D_1_r2 > 0: 
#             if axis == 0:
#                 U, R, P = left_syssolve(hamma_1, u_1, p_1, rho_1) 
#                 return np.array([U, v_1, R, P, hamma_1])
#             elif axis == 1: 
#                 V, R, P = left_syssolve(hamma_1, v_1, p_1, rho_1) 
#                 return np.array([u_1, V, R, P, hamma_1])
#         elif U > 0: 
#             if axis == 0: 
#                 return np.array([U, v_1, R_1, p, hamma_1])
#             elif axis == 1: 
#                 return np.array([u_1, U, R_1, p, hamma_1])
#         elif D_2 > 0:
#             if axis == 0:
#                 return np.array([rho_2*u_2/R_2, v_2, R_2, p, hamma_2])
#             elif axis == 1: 
#                 return np.array([u_2, rho_2*v_2/R_2, R_2, p, hamma_2])
#         else: 
#             return np.array([u_2, v_2, rho_2, p_2, hamma_2])
#     else:
#         if D_1_r1 > 0: 
#             return np.array([u_1, v_1, rho_1, p_1, hamma_1])
#         elif D_1_r2 > 0: 
#             if axis == 0:
#                 U, R, P = left_syssolve(hamma_1, u_1, p_1, rho_1) 
#                 return np.array([U, v_1, R, P, hamma_1])
#             elif axis == 1: 
#                 V, R, P = left_syssolve(hamma_1, v_1, p_1, rho_1) 
#                 return np.array([u_1, V, R, P, hamma_1])
#         elif U > 0: 
#             if axis == 0: 
#                 return np.array([U, v_1, R_1, p, hamma_1])
#             elif axis == 1: 
#                 return np.array([u_1, U, R_1, p, hamma_1])
#         elif D_2_r2 > 0: 
#             if axis == 0: 
#                 return np.array([U, v_2, R_2, p, hamma_2])
#             elif axis == 1: 
#                 return np.array([u_2, U, R_2, p, hamma_2])
#         elif D_2_r1 > 0: 
#             if axis == 0:
#                 U, R, P = right_syssolve(hamma_2, u_2, p_2, rho_2) 
#                 return np.array([U, v_2, R, P, hamma_2])
#             elif axis == 1: 
#                 V, R, P = right_syssolve(hamma_2, v_2, p_2, rho_2) 
#                 return np.array([u_2, V, R, P, hamma_2])
#         else: 
#             return np.array([u_2, v_2, rho_2, p_2, hamma_2])

# In[112]:


@njit
def timestep(params_1_1: npt.ArrayLike, 
            Params_n1_1: npt.ArrayLike,
            Params_n2_1: npt.ArrayLike, 
            Params_1_m1: npt.ArrayLike,
            Params_1_m2: npt.ArrayLike):   
        
    E_n1_1 = Params_n1_1[3]/(Params_n1_1[2]*(Params_n1_1[4] - 1.0))
    E_n2_1 = Params_n2_1[3]/(Params_n2_1[2]*(Params_n2_1[4] - 1.0))
    E_1_m1 = Params_1_m1[3]/(Params_1_m1[2]*(Params_1_m1[4] - 1))
    E_1_m2 = Params_1_m2[3]/(Params_1_m2[2]*(Params_1_m2[4] - 1))
    e_down_1_1 = params_1_1[3]/(params_1_1[2]*(params_1_1[4] - 1.0))

    rho_up_1_1 = params_1_1[2] - dt*((Params_n2_1[2]*Params_n2_1[0]) - (Params_n1_1[2]*Params_n1_1[0])) / h_x - dt*((Params_1_m2[2]*Params_1_m2[1]) - (Params_1_m1[2]*Params_1_m1[1])) / h_y
        # rho_up_1_1 = np.abs(params_1_1[2] - self.dt*((Params_n2_1[2]*Params_n2_1[0]) - (Params_n1_1[2]*Params_n1_1[0])) / self.h_x - self.dt*((Params_1_m2[2]*Params_1_m2[1]) - (Params_1_m1[2]*Params_1_m1[1])) / self.h_y)
    u_up_1_1 = (params_1_1[2]*params_1_1[0] - dt*((Params_n2_1[3] + Params_n2_1[2]*np.power(Params_n2_1[0], 2)) - (Params_n1_1[3] + Params_n1_1[2]*np.power(Params_n1_1[0], 2))) / h_x - 
                    dt*((Params_1_m2[2]*Params_1_m2[0]*Params_1_m2[1]) - (Params_1_m1[2]*Params_1_m1[0]*Params_1_m1[1])) / h_y) / rho_up_1_1
    v_up_1_1 = (params_1_1[2]*params_1_1[1] - dt*((Params_n2_1[2]*Params_n2_1[0]*Params_n2_1[1]) - (Params_n1_1[2]*Params_n1_1[0]*Params_n1_1[1])) / h_x - dt*((Params_1_m2[3] + Params_1_m2[2]*np.power(Params_1_m2[1], 2)) - (Params_1_m1[3] + Params_1_m1[2]*np.power(Params_1_m1[1], 2)))/h_y) / rho_up_1_1  
    e_up_1_1 = (params_1_1[2]*(e_down_1_1 + (np.power(params_1_1[0], 2) + np.power(params_1_1[1], 2)) / 2.0) - 
                    dt*(Params_n2_1[2]*Params_n2_1[0]*(E_n2_1 + Params_n2_1[3]/Params_n2_1[2] + (np.power(Params_n2_1[0], 2) + np.power(Params_n2_1[1], 2)) / 2.0) -
                        Params_n1_1[2]*Params_n1_1[0]*(E_n1_1 + Params_n1_1[3]/Params_n1_1[2] + (np.power(Params_n1_1[0],2) + np.power(Params_n1_1[1],2)) / 2.0))/h_x - 
                        dt*(Params_1_m2[2]*Params_1_m2[1]*(E_1_m2 + Params_1_m2[3]/Params_1_m2[2] + (np.power(Params_1_m2[0], 2) + np.power(Params_1_m2[1],2)) / 2.0) - 
                            Params_1_m1[2]*Params_1_m1[1]*(E_1_m1 + Params_1_m1[3]/Params_1_m1[2] + (np.power(Params_1_m1[0], 2) + np.power(Params_1_m1[1],2)) / 2.0)) / h_y ) / rho_up_1_1 - (np.power(u_up_1_1, 2) + np.power(v_up_1_1, 2)) / 2.0
        
    p_up_1_1 = (params_1_1[4] - 1)*e_up_1_1*rho_up_1_1
    return np.array([u_up_1_1, v_up_1_1, rho_up_1_1, p_up_1_1, params_1_1[4]], dtype = np.float64)  


# In[16]:


#1-dim tests
number_of_dots_x = 101
number_of_dots_y = 2

x_min = 0.0 
x_max = 1.0 

#could be any
y_min = 0.0 
y_max = 1.0

x = np.linspace(x_min, x_max, number_of_dots_x)
y = np.linspace(y_min, y_max, number_of_dots_y)
X, Y = np.meshgrid(x, y)

h_x = (x_max - x_min)/(number_of_dots_x - 1)
h_y = (y_max - y_min)/(number_of_dots_y - 1)


# In[17]:


#test 1 
condition = np.empty(shape = (number_of_dots_y - 1, number_of_dots_x - 1, 5), dtype = np.float64)
condition[:, :30, 0], condition[:, 30:, 0] = 0.75, 0.0
condition[:, :, 1] = 0.0
condition[:, :30, 2], condition[:, 30:, 2] = 1.0, 0.125
condition[:, :30, 3], condition[:, 30:, 3] = 1.0, 0.1
condition[:, :, 4] = 1.4
T = 0.2
steps = 500
dt = T/steps


# In[18]:


fig, ax = plt.subplots()
ax.pcolormesh(X, Y, condition[:,:,2])


# In[33]:


#test 2
condition = np.empty(shape = (number_of_dots_y - 1, number_of_dots_x - 1, 5), dtype = np.float64)
condition[:, :50, 0], condition[:, 50:, 0] = -2.0, 2.0
condition[:, :, 1] = 0.0
condition[:, :50, 2], condition[:, 50:, 2] = 1.0, 1.0
condition[:, :50, 3], condition[:, 50:, 3] = 0.4, 0.4
condition[:, :, 4] = 1.4
T = 0.15
steps = 500
dt = T/steps


# In[34]:


fig, ax = plt.subplots()
ax.pcolormesh(X, Y, condition[:,:,0])


# In[121]:


@njit(parallel = True)
def full_time_step(imc):
    #solving riemann problem on edges
    parametres_on_edges_x = np.empty(shape = (number_of_dots_y - 1, number_of_dots_x, 5), dtype = np.float64)

    for i in prange(number_of_dots_y - 1):
        for j in range(number_of_dots_x):
            parametres_on_edges_x[i,j] = riemprob_for_border_x(imc[i+1, j, 0], imc[i+1, j, 1], imc[i+1, j, 2], imc[i+1, j, 3], imc[i+1, j, 4], 
                                                               imc[i+1, j+1, 0], imc[i+1, j+1, 1], imc[i+1, j+1, 2], imc[i+1, j+1, 3], imc[i+1, j+1, 4])

    parametres_on_edges_y = np.empty(shape = (number_of_dots_y, number_of_dots_x - 1, 5), dtype = np.float64)

    for i in prange(number_of_dots_y):
        for j in range(number_of_dots_x - 1):
            parametres_on_edges_y[i,j] = riemprob_for_border_y(imc[i, j+1, 0], imc[i, j+1, 1], imc[i, j+1, 2], imc[i, j+1, 3], imc[i, j+1, 4],
                                                               imc[i+1, j+1, 0], imc[i+1, j+1, 1],imc[i+1, j+1, 2],imc[i+1, j+1, 3], imc[i+1, j+1, 4])

    #gas-dynamic step 
    for i in prange(number_of_dots_y - 1):
        for j in range(number_of_dots_x - 1):
            imc[i+1, j+1] = timestep(imc[i+1, j+1], parametres_on_edges_x[i, j], parametres_on_edges_x[i, j + 1], parametres_on_edges_y[i, j], parametres_on_edges_y[i+1, j]) 


# In[122]:


imc = make_imaginary_condition(condition)

for _ in range(steps):
    full_time_step(imc)     
    update_imc(imc)
    print(_)  


# In[123]:


fig, ax = plt.subplots(2, 1)
ax[0].pcolormesh(X, Y, imc[1:-1,1:-1,2])
ax[1].plot(x, imc[1,:-1,2])


# In[114]:


#2-dim tests
number_of_dots_x = 401
number_of_dots_y = 401

x_min = 0.0
x_max = 1.0

y_min = 0.0
y_max = 1.0

x = np.linspace(x_min, x_max, number_of_dots_x)
y = np.linspace(y_min, y_max, number_of_dots_y)
X, Y = np.meshgrid(x, y)

h_x = (x_max - x_min)/(number_of_dots_x - 1)
h_y = (y_max - y_min)/(number_of_dots_y - 1)


# In[115]:


#test 1
condition = np.empty(shape = (number_of_dots_y - 1, number_of_dots_x - 1, 5), dtype = np.float64)
condition[:200, :200, 0], condition[:200, 200:, 0], condition[200:, :200, 0], condition[200:, 200:, 0] = 1.206, 0.0, 1.206, 0.0
condition[:200, :200, 1], condition[:200, 200:, 1], condition[200:, :200, 1], condition[200:, 200:, 1] = 1.206, 1.206, 0.0, 0.0 
condition[:200, :200, 2], condition[:200, 200:, 2], condition[200:, :200, 2], condition[200:, 200:, 2] = 0.138, 0.5323, 0.5323, 1.5
condition[:200, :200, 3], condition[:200, 200:, 3], condition[200:, :200, 3], condition[200:, 200:, 3] = 0.029, 0.3, 0.3, 1.5
condition[:,:,4] = 1.4
T = 0.3
steps = 1000
dt = T/steps


# In[84]:


fig, ax = plt.subplots()
ax.pcolormesh(X, Y, condition[:,:,2])


# In[116]:


imc = make_imaginary_condition(condition)

for _ in range(steps):
    full_time_step(imc)     
    update_imc(imc)
    print(_)  


# In[86]:


fig, ax = plt.subplots()
ax.pcolormesh(X, Y, imc[1:-1,1:-1,2])
# ax[1].plot(x, imc[1,:-1,2])


# In[124]:


#test 2
condition = np.empty(shape = (number_of_dots_y - 1, number_of_dots_x - 1, 5), dtype = np.float64)
condition[:200, :200, 0], condition[:200, 200:, 0], condition[200:, :200, 0], condition[200:, 200:, 0] = 0.8939, 0.0, 0.8939, 0.0
condition[:200, :200, 1], condition[:200, 200:, 1], condition[200:, :200, 1], condition[200:, 200:, 1] = 0.8939, 0.8939, 0.0, 0.0 
condition[:200, :200, 2], condition[:200, 200:, 2], condition[200:, :200, 2], condition[200:, 200:, 2] = 1.1, 0.5065, 0.5065, 1.1
condition[:200, :200, 3], condition[:200, 200:, 3], condition[200:, :200, 3], condition[200:, 200:, 3] = 1.1, 0.35, 0.35, 1.1
condition[:,:,4] = 1.4
T = 0.25
steps = 1000
dt = T/steps


# In[125]:


imc = make_imaginary_condition(condition)

for _ in range(steps):
    full_time_step(imc)     
    update_imc(imc)
    print(_)  


# In[89]:


fig, ax = plt.subplots()
ax.pcolormesh(X, Y, imc[1:-1,1:-1,2])
# ax[1].plot(x, imc[1,:-1,2])


# In[90]:


#test 3
condition = np.empty(shape = (number_of_dots_y - 1, number_of_dots_x - 1, 5), dtype = np.float64)
condition[:200, :200, 0], condition[:200, 200:, 0], condition[200:, :200, 0], condition[200:, 200:, 0] = -0.75, -0.75, 0.75, 0.75 
condition[:200, :200, 1], condition[:200, 200:, 1], condition[200:, :200, 1], condition[200:, 200:, 1] = 0.5, -0.5, 0.5, -0.5 
condition[:200, :200, 2], condition[:200, 200:, 2], condition[200:, :200, 2], condition[200:, 200:, 2] = 1.0, 3.0, 2.0, 1.0
condition[:200, :200, 3], condition[:200, 200:, 3], condition[200:, :200, 3], condition[200:, 200:, 3] = 1.0, 1.0, 1.0, 1.0
condition[:,:,4] = 1.4
T = 0.3
steps = 1000
dt = T/steps


# In[91]:


imc = make_imaginary_condition(condition)

for _ in range(steps):
    full_time_step(imc)     
    update_imc(imc)
    print(_)  


# In[92]:


fig, ax = plt.subplots()
ax.pcolormesh(X, Y, imc[1:-1,1:-1,2])
# ax[1].plot(x, imc[1,:-1,2])


# In[93]:


#test 4
condition = np.empty(shape = (number_of_dots_y - 1, number_of_dots_x - 1, 5), dtype = np.float64)
condition[:200, :200, 0], condition[:200, 200:, 0], condition[200:, :200, 0], condition[200:, 200:, 0] = 0.0, 0.0, 0.7276, 0.0
condition[:200, :200, 1], condition[:200, 200:, 1], condition[200:, :200, 1], condition[200:, 200:, 1] = 0.0, 0.7276, 0.0, 0.0
condition[:200, :200, 2], condition[:200, 200:, 2], condition[200:, :200, 2], condition[200:, 200:, 2] = 0.8, 1.0, 1.0, 0.5313
condition[:200, :200, 3], condition[:200, 200:, 3], condition[200:, :200, 3], condition[200:, 200:, 3] = 1.0, 1.0, 1.0, 0.4
condition[:,:,4] = 1.4
T = 0.25
steps = 1000
dt = T/steps


# In[94]:


imc = make_imaginary_condition(condition)

for _ in range(steps):
    full_time_step(imc)     
    update_imc(imc)
    print(_)  


# In[95]:


fig, ax = plt.subplots()
ax.pcolormesh(X, Y, imc[1:-1,1:-1,2])
# ax[1].plot(x, imc[1,:-1,2])


# In[96]:


#test 5
condition = np.empty(shape = (number_of_dots_y - 1, number_of_dots_x - 1, 5), dtype = np.float64)
condition[:200, :200, 0], condition[:200, 200:, 0], condition[200:, :200, 0], condition[200:, 200:, 0] = 0.1, 0.1, -0.6259, 0.1
condition[:200, :200, 1], condition[:200, 200:, 1], condition[200:, :200, 1], condition[200:, 200:, 1] = -0.3, 0.4276, -0.3, -0.3 
condition[:200, :200, 2], condition[:200, 200:, 2], condition[200:, :200, 2], condition[200:, 200:, 2] = 0.8, 0.5313, 0.5197, 1.0
condition[:200, :200, 3], condition[:200, 200:, 3], condition[200:, :200, 3], condition[200:, 200:, 3] = 0.4, 0.4, 0.4, 1.0
condition[:,:,4] = 1.4
T = 0.2
steps = 1000
dt = T/steps


# In[97]:


imc = make_imaginary_condition(condition)

for _ in range(steps):
    full_time_step(imc)     
    update_imc(imc)
    print(_)  


# In[98]:


fig, ax = plt.subplots()
ax.pcolormesh(X, Y, imc[1:-1,1:-1,2])
# ax[1].plot(x, imc[1,:-1,2])

