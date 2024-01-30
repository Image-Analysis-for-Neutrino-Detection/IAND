# %%
import math

def MSE_deltaphi(output, target):
    output *= math.pi/180.0
    target *= math.pi/180.0
    dphi = math.atan2(math.sin(output- target), math.cos(output - target))
    dphi *= 180.0/math.pi
    loss = math.mean((dphi)**2)

def delta_phi(output, target):
    output *= math.pi/180.0
    target *= math.pi/180.0
    dphi = math.atan2(math.sin(output- target), math.cos(output - target))
    dphi *= 180.0/math.pi
    return dphi

phi1 = 10.0
phi2 = 12.0

print(phi1,phi2,delta_phi(phi1,phi2))
# %%
phi1 = 10.0
phi2 = -11.0

print(phi1,phi2,delta_phi(phi1,phi2))
# %%
phi1 = -10.0
phi2 = 11.0

print(phi1,phi2,delta_phi(phi1,phi2))
# %%
phi1 = 351.0
phi2 = -11.0

print(phi1,phi2,delta_phi(phi1,phi2))
# %%
phi1 = 1.0
phi2 = 358.0

print(phi1,phi2,delta_phi(phi1,phi2))
# %%
