# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 17:16:08 2022

@author: AhmetTalha
"""
import casadi as cs
import numpy as np
N = 40  # number of control intervals
sigma_max = 0.3
opti = cs.Opti() # Optimization problem

# ---- decision variables ---------
X = opti.variable(7,N+1) # state trajectory
# pos   = X[0,:]
# speed = X[1,:]
U = opti.variable(3,N)   # control trajectory 
T  = 834.0

# ---- objective          ---------
opti.minimize(X[6,N]) # minimum fuel problem

# ---- dynamic constraints --------
def f(x,u):
    # dx/dt = f(x,u)
    mu = 3.986*10**14
    R = 6793137.0
    omega = mu/R**3
    px = x[0]
    py = x[1]
    pz = x[2]
    vx = x[3]
    vy = x[4]
    vz = x[5]
    sigma = u[0]
    theta = u[1]
    phi = u[2]
    vx_dot = sigma * cs.cos(theta) * cs.cos(phi) + 2*omega*vy
    vy_dot = sigma * cs.cos(theta) * cs.sin(phi) - 2*omega*vx + 3*py*omega**2
    vz_dot = sigma * cs.sin(theta) - pz*omega**2
    w = sigma
    return cs.vertcat(vx, vy, vz, vx_dot, vy_dot, vz_dot,w)

max_change_in_angle = 2*cs.pi 
dt = T/N # length of a control interval
for k in range(N): # loop over control intervals
   # Runge-Kutta 4 integration
   k1 = f(X[:,k],         U[:,k])
   k2 = f(X[:,k]+dt/2*k1, U[:,k])
   k3 = f(X[:,k]+dt/2*k2, U[:,k])
   k4 = f(X[:,k]+dt*k3,   U[:,k])
   x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
   if k>1:
        # constraint: limit input change
        opti.subject_to(U[:,k]-U[:,k-1] <= max_change_in_angle)
   opti.subject_to(X[:,k+1]==x_next) # close the gaps

# ---- path constraints -----------
opti.subject_to(opti.bounded(0,U[0,:],sigma_max)) # control is limited
opti.subject_to(opti.bounded(-5*cs.pi,U[1:3,:],5*cs.pi)) # control is limited


# ---- boundary conditions --------
# initial conditions
opti.subject_to(X[0,0]==30000)
opti.subject_to(X[1,0]==-15000)
opti.subject_to(X[2,0]==7500)
opti.subject_to(X[3,0]==-60)
opti.subject_to(X[4,0]==-15)
opti.subject_to(X[5,0]==-6)
opti.subject_to(X[6,0]==0)

# final conditions
opti.subject_to(X[0,-1]==0)
opti.subject_to(X[1,-1]==0)
opti.subject_to(X[2,-1]==0)
opti.subject_to(X[3,-1]==0)
opti.subject_to(X[4,-1]==0)
opti.subject_to(X[5,-1]==0)

# ---- solve NLP              ------
opts = {}
opts["ipopt"] = dict(max_iter=20000, print_level=0)
opti.solver("ipopt", opts) # set numerical backend
sol = opti.solve()   # actual solve

#%%
# ---- post-processing        ------

import matplotlib.pyplot as plt
t_opt = T
print("Time:", t_opt)
print("Delta V:", sol.value(X[6,N]))
time_array = t_opt*np.arange(0,N+1)/N

optimal_control_input = sol.value(U).T
sigma = optimal_control_input[:,0]
theta = optimal_control_input[:,1]
phi = optimal_control_input[:,2]

sigma_x = sigma * np.cos(theta) * np.cos(phi)
sigma_y = sigma * np.cos(theta) * np.sin(phi)
sigma_z = sigma * np.sin(theta)

plt.figure(1)
plt.clf()
plt.plot(time_array, sol.value(X[0,:]))
plt.plot(time_array, sol.value(X[1,:]))
plt.plot(time_array, sol.value(X[2,:]))
plt.title("Position")
plt.legend(["x","y","z"])
plt.xlabel("Time")
plt.ylabel("Position")
plt.minorticks_on()
plt.grid(visible=True, which="both", linestyle=":")
plt.savefig("figures/P4_position_with_sigma_max_{}_t_final_{}.png".format(sigma_max,T), dpi=300)

plt.figure(2)
plt.clf()
plt.plot(time_array, sol.value(X[3,:]))
plt.plot(time_array, sol.value(X[4,:]))
plt.plot(time_array, sol.value(X[5,:]))
plt.title("Velocity")
plt.legend(["$V_x$","$V_y$","$V_z$"])
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.minorticks_on()
plt.grid(visible=True, which="both", linestyle=":")
plt.savefig("figures/P4_velocity_with_sigma_max_{}_t_final_{}.png".format(sigma_max,T), dpi=300)

plt.figure(3)
time_array = t_opt*np.arange(0,N)/(N-1)
plt.clf()
plt.plot(time_array, sigma_x)
plt.plot(time_array, sigma_y)
plt.plot(time_array, sigma_z)
plt.title("Thrust Acceleration")
plt.legend(["$\sigma_x$","$\sigma_y$","$\sigma_z$"])
plt.xlabel("Time")
plt.ylabel("Thrust Acceleration")
plt.minorticks_on()
plt.grid(visible=True, which="both", linestyle=":")
plt.savefig("figures/P4_thrust_acceleration_with_sigma_max_{}_t_final_{}.png".format(sigma_max,T), dpi=300)
plt.show()