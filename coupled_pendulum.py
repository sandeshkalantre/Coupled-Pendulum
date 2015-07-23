import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#parameters
m1 = 0.2 
m2 = 0.2
d = 0.1
g = 0.5
t0 = 0.0

#variables of the system x,v,th1,w1,th2,w2
y0 = [0.3, 0.0, 0.4, 0.0, 0.3, 0.0, 0.0]


def dx_dt(y):
    return y[1]

def dv_dt(y):
    term1 = m1 * y[0] * (y[3]**2) 
    term2 = -1.0 * m2 * (1 - d - y[0]) * (y[5]**2)
    term3 = m1 * g * np.cos(y[2])
    term4 = -1.0 * m2 * g * np.cos(y[4])
    return (term1 + term2 + term3 + term4) / (m1 + m2)

def dth1_dt(y):
    return y[3]

def dth2_dt(y):
    return y[5]

def dw1_dt(y):
    return -1.0 * g * np.sin(y[2])/y[0]

def dw2_dt(y):
    return -1.0 * g * np.sin(y[4])/(1 - d - y[0])

def dy_dt(y,t):
    return [dx_dt(y),dv_dt(y),dth1_dt(y),dw1_dt(y),dth2_dt(y),dw2_dt(y), 1]

#solving the ode
t = np.linspace(t0,10,100)

y_sol = odeint(dy_dt,y0,t)
y1 = - y_sol[:,0] * np.cos(y_sol[:,2])
x1 = y_sol[:,0] * np.sin(y_sol[:,2])
y2 = - (1 -d - y_sol[:,0]) * np.cos(y_sol[:,4])
x2 = d + (1 - d - y_sol[:,0]) * np.sin(y_sol[:,4])

print y_sol
print x1,y1
print x2,y2

base_x = np.linspace(0,d,10)
base_y = np.linspace(0,0,10)

plt.xlim((-0.5,1.0))
plt.ylim((-0.5,0.5))

plt.plot(base_x,base_y,'r')
plt.plot(x1,y1,'b',x2,y2,'g',markersize = 2)
plt.show()
