import numpy as np 
import numpy.typing as npt
import matplotlib.pyplot as plt
from numba import njit, prange
from time import time 

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

@njit
def f(P, p_k, rho_k, hamma):
    pi_k = P/p_k
    c_k = np.sqrt(hamma*p_k/rho_k)
    if P > p_k or abs(P - p_k) < 1e-10:
        return (P-p_k)/(rho_k*c_k*np.sqrt((hamma + 1)*pi_k/(2*hamma) + (hamma - 1)/(2*hamma)))
    else:
        return 2*c_k/(hamma - 1)*(pi_k**((hamma - 1)/(2*hamma)) - 1)

@njit
def df(P, p_k, rho_k, hamma):
    pi_k = P/p_k
    c_k = np.sqrt(hamma*p_k/rho_k)
    if P > p_k or abs(P - p_k) < 1e-10:
        return ((hamma+1)*pi_k + (3*hamma - 1))/(4*hamma*rho_k*c_k*np.power((hamma + 1)*pi_k/(2*hamma) + (hamma - 1)/(2*hamma), 1.5))
    else:
        return c_k*pi_k**((hamma - 1)/(2*hamma))/(hamma*P)


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
    

@njit
def left_syssolve(hamma, u_1, p_1, rho_1, db = 0):
    a_1 = np.sqrt(hamma*p_1/rho_1)
    Y_plus = (u_1 + 2*a_1/(hamma - 1))
    u = (hamma - 1)/(hamma + 1)*Y_plus + 2*db/(hamma + 1)
    a = u - db
    rho = np.power(np.power(a, 2)*np.power(rho_1, hamma)/(hamma*p_1), 1/(hamma - 1))
    p = p_1*np.power(rho, hamma)/np.power(rho_1, hamma)
    return u, rho, p

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

R = 0.83143e04
R = R*1e4/1e6 

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