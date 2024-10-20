#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 01:56:12 2024

@author: arthur
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
import time
start=time.time()

from orbit import orbit
from shadow import shadow


##First, we define the required parameters and initial data:
cSI=299792458; GSI=6.67408e-11; M=4e30; Rs=2*GSI*M/cSI**2; a=0.95; Q=0.3;
##The initial value of the (massive) orbit:
X=[30656,np.pi/2,0,0,1677.2693,2234.125]; tau=0.0075; N=1000; mu=1;
eqnss=["Euler-Lagrange","Hamilton","Carter","Verlet","Stormer-Verlet","Symplectic Euler p","Symplectic Euler q"];
colors=['b','g','r','c','purple','orange','k','m','pink']

#X=[30656,np.pi/2,0,0,0*1677.2693,1.92*2234.125]; tau=0.0075/4; N=1600; mu=1; a=0; Lambda=0;
#eqnss=["Euler-Lagrange","Hamilton","Carter","Verlet","Stormer-Verlet","Symplectic Euler p","Symplectic Euler q","Weierstrass","Polar"];

###We test the functions for Lambda=0 and Lambda<>0
for Lambda in [0,3.3e-4]:
    ##On a new figure, draw the horizon ellipsoide and the orbit using the various formulations:
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100);
    def U(r):
        return (1-Lambda/3*r**2)*(r**2+a**2)-2*r+Q**2
    hor=Rs/2*fsolve(U,2); J=a*GSI*M**2/cSI; A=J/(M*cSI);
    x = np.sqrt(hor**2+A**2) * np.outer(np.cos(u), np.sin(v))
    y = np.sqrt(hor**2+A**2) * np.outer(np.sin(u), np.sin(v))
    z = hor * np.outer(np.ones(np.size(u)), np.cos(v))
    ##The horizon ellipsoide
    ax.plot_surface(x, y, z)
    
    HAMS=np.zeros((0,N-1)); CARS=np.zeros((0,N-1));
    for i in range(len(eqnss)):
        eq=eqnss[i]
        [Vec,HAM,CAR]=orbit(Lambda,M,a,Q,X,eq,tau,N,mu,1,0);
        R=Vec[:,0]; theta=Vec[:,1]; phi=Vec[:,2];
        HAMS=np.vstack([HAMS,np.array(HAM[1:])/HAM[1]]); CARS=np.vstack([CARS,np.array(CAR[1:])/CAR[1]]);
        ax.plot(np.sqrt(R**2+A**2)*np.sin(theta)*np.cos(phi),np.sqrt(R**2+A**2)*np.sin(theta)*np.sin(phi),R*np.cos(theta),colors[i],label=eq);
    
    ax.set_aspect('equal')
    plt.title("Orbit with several methods and $(\Lambda,M,a,Q,\mu)=($"+str(Lambda)+", "+str(M)+", "+str(a)+", "+str(Q)+", "+str(mu)+")")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()

    ##Compare the conservation of the Hamiltonian:
    fig = plt.figure(figsize=(8,8))
    I=cSI*2/Rs*np.linspace(tau/N,tau,num=N-1);
    for i in range(len(eqnss)):
        H=HAMS[i,:]
        plt.plot(I,H,colors[i],label=eqnss[i])
    plt.title("Hamiltonian conservation for different methods ($\mu$="+str(mu)+")")
    plt.xlabel("proper time $[M/c]$")
    plt.ylabel("$H/H_0$ where $2H=g^{ab}p_ap_b$")
    plt.legend()
    plt.show()
    
    ##Compare the conservation of the Carter constant:
    fig = plt.figure(figsize=(8,8))
    for i in range(len(eqnss)):
        C=CARS[i,:]
        plt.plot(I,C,colors[i],label=eqnss[i])
    plt.title("Carter constant conservation for different methods ($\mu$="+str(mu)+")");
    plt.xlabel("proper time $[M/c]$")
    plt.ylabel("$C/C_0$ where $C$ is the Carter constant")
    plt.legend()
    plt.show()    
    
    ##Maximal deviation (Hamilton and Carter) for each method:
    dev_ham=[]; dev_car=[];
    for i in range(1,len(eqnss)):
        dev_ham.append(max(abs(HAMS[i,:]-1)))
        dev_car.append(max(abs(CARS[i,:]-1)))

###"""
    ##Testing the shadowing programs (the reader is invited to un-comment the three lines below to test the effect of the two shifts we introduced):
    Mass=4e30; Kerr=0.95; Newman=0.3; Image='figure32.png';
    Accretion=list([1,np.pi/18,"Blackbody","Doppler+",[1.455,6],[3,15000,100],3800,1]);
    ##Accretion=list([16,np.pi/18," ","Gravitation",[1.455,6],[3],3800,0]);
    ##Accretion=list([16,np.pi/18," ","Doppler",[1.455,6],[3],3800,0]);
    ##Accretion=list([16,np.pi/18," ","Doppler+",[1.455,6],[3],3800,0]);
    shadow(Lambda,Mass,Kerr,Newman,Image,Accretion)




###Comet plot of orbit:
from matplotlib.animation import FuncAnimation

##First, we define the required parameters and initial data:
cSI=299792458; GSI=6.67408e-11; M=4e30; Rs=2*GSI*M/cSI**2; a=0.95; Q=0.3; Lambda=3.3e-4;
##The initial value of the (massive) orbit:
X=[30656,np.pi/2,0,0,1677.2693,2234.125]; tau=0.004; N=120; mu=1;
eqnss=["Euler-Lagrange","Hamilton","Carter","Verlet","Stormer-Verlet","Symplectic Euler p","Symplectic Euler q"];
colors=['b','g','r','c','purple','orange','k','m','pink']

#X=[30656,np.pi/2,0,0,0*1677.2693,1.92*2234.125]; tau=0.0075/4; N=1000; mu=1; a=0; Lambda=0;
#eqnss=["Euler-Lagrange","Hamilton","Carter","Verlet","Stormer-Verlet","Symplectic Euler p","Symplectic Euler q","Weierstrass","Polar"];

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
lines=[]; XX=np.zeros((0,N)); YY=np.zeros((0,N)); ZZ=np.zeros((0,N)); MX=0; MY=0; MZ=0; mX=0; mY=0; mZ=0;

def U(r):
    return (1-Lambda/3*r**2)*(r**2+a**2)-2*r+Q**2
hor=Rs/2*fsolve(U,2); J=a*GSI*M**2/cSI; A=J/(M*cSI);
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100);
x = np.sqrt(hor**2+A**2) * np.outer(np.cos(u), np.sin(v))
y = np.sqrt(hor**2+A**2) * np.outer(np.sin(u), np.sin(v))
z = hor * np.outer(np.ones(np.size(u)), np.cos(v))
##The horizon ellipsoide
ax.plot_surface(x, y, z)


for i in range(len(eqnss)):
    eq=eqnss[i]
    li,=ax.plot([],[],colors[i],animated=True,label=eq)
    lines.append(li)
    [Vec,HAM,CAR]=orbit(Lambda,M,a,Q,X,eq,tau,N,mu,0,0);
    R=Vec[:,0]; theta=Vec[:,1]; phi=Vec[:,2];
    X0=np.sqrt(R**2+A**2)*np.sin(theta)*np.cos(phi); Y0=np.sqrt(R**2+A**2)*np.sin(theta)*np.sin(phi); Z0=R*np.cos(theta);
    #ax.plot(X0,Y0,Z0,label=eq);
    #ax.set_aspect('equal')
    XX=np.vstack([XX,X0]); YY=np.vstack([YY,Y0]); ZZ=np.vstack([ZZ,Z0]);
    MX=max(MX,max(X0)); MY=max(MY,max(Y0)); MZ=max(MZ,max(Z0))
    mX=min(mX,min(X0)); mY=min(mY,min(Y0)); mZ=min(mZ,min(Z0))
    

ax.set(xlim=[mX,MX],ylim=[mY,MY],zlim=[mZ,MZ])
ax.set_aspect('equal')

def init():
    for li in lines:
        li.set_data([],[])
    return lines

def update(frame):
    for li in range(len(eqnss)):
        Xl=XX[li,:frame]; Yl=YY[li,:frame]; Zl=ZZ[li,:frame];
        li=lines[li];
        li.set_data(Xl,Yl)
        li.set_3d_properties(Zl)
    return lines

ani=FuncAnimation(fig, update, frames=range(N), init_func=init, blit=True, interval=2, repeat=False)
plt.legend()
ani.save('comet.gif', writer='ffmpeg',fps=30)#writer=imagemagick
plt.show()




###Creating a gif of an RNdS black hole before a diagonally moving celestial sphere
from gif import DatFile4gif, make_gif_with_DatFile, make_gif

Nimages=240; Name="figure"; Image="figure.png"; Resol=[60,60]; Shifts=[0,0,3.5]; Direction="d2-"; FPS=24;
Lambda=3.3e-4; Mass=0.5*4e30; Kerr=0.95; Newman=0.3; Angle=0;

#make_gif(Nimages,Name,Image,Resol,Shifts,Direction,FPS,Lambda,Mass,Kerr,Newman,Angle)
DatFile4gif(Resol,Lambda,Mass,Kerr,Newman,Angle)
make_gif_with_DatFile(Nimages,Name,Image,Resol,Shifts,Direction,FPS,Lambda,Mass,Kerr,Newman,Angle)


print(time.time()-start)