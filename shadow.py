#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 17:30:26 2024

@author: arthur
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import cmath
from scipy.optimize import newton as nt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

import cv2 as cv

from scipy.integrate import odeint

import warnings

from auxi import *




def shadow(Lambda,Mass,Kerr,Newman,Image,Accretion_data):
    if Kerr!=0: #Carter's equations
        warnings.filterwarnings("ignore");#warnings.filterwarnings("error")
        ##We define the refining parameters for details near the horizon(s)
        ##cft=0.95; itermax=100; N=1500;//N=3000;
        cft=0.85; itermax=50; N=1500;#N=1200;
        ##cft=0.8; itermax=50; N=1200;
        N=1500;
        ##Here again, distinguish between whether Lambda is zero or not (faster if Lambda=0)
        if Lambda!=0:
            ##Initialize: unitless initial conditions  parameters (of the black hole  the accretion disk: size  extremal temperatures if input)
            c=1; G=1; M=1; GSI=6.67408e-11; cSI=299792458; e0=8.854187e-12; sb=5.67e-8;
            Rs=2*GSI*Mass/cSI**2; A=Kerr*Rs/2; alpha=-Accretion_data[1];##J=Kerr*GSI*Mass**2/cSI; A=J/(Mass*cSI); alpha=-Accretion_data[1];
            rq=Newman**2;##Q=Newman*2*Mass*np.sqrt(np.pi*e0*GSI); rq2=Q**2*GSI/(4*np.pi*e0*cSI**4); rq=4*rq2/Rs**2;
            x0=50000; sizee=Accretion_data[4]; rint=sizee[0]*Rs; rext=sizee[1]*Rs; rf=60000;
            rs=2; rg=1; a=Kerr; T_int=1001; T_ext=1001; lam=0.8; chi=1+Lambda*a**2/3;
            Mrate=Accretion_data[5]; Mrate=Mrate[0]*Rs*cSI**2/(Mass*2*GSI); T0=3*cSI**2*Rs*Mrate/(2*np.pi*sb);
            Temp=[];
            if (len(Accretion_data[5])>1 and Accretion_data[5][2]<Accretion_data[5][1]):
                T_int=Accretion_data[5]; T_ext=T_int[2]; T_int=T_int[1];
            
            
            ##Defining the 'ode' functions (distinguishing between rotating  non-rotating black hole)
            if Kerr==0:
                def Carter_ter(V,tau):
                    r=V[0]; th=V[1]; ph=V[2]; pr=V[3]; pth=V[4];
                    Dr=(1-Lambda*r**2/3)*r**2-2*r+rq;
                    Drp=-4/3*Lambda*r**3+2*r-2;
                    Pofr=E*r**2;
                    prp=((2*E*r*Pofr-Drp*k/2)/Dr-Drp*pr**2)/r**2;
                    pthp=(np.cos(th)*np.sin(th)*Lz**2/np.sin(th)**4)/r**2;
                    rp=Dr*pr/r**2;
                    thp=pth/r**2;
                    php=Lz/(r**2*np.sin(th)**2);
                    Y=[rp,thp,php,prp,pthp];
                    return Y
            else:
                def Carter_ter(V,tau):
                    r=V[0]; th=V[1]; ph=V[2]; pr=V[3]; pth=V[4];
                    Dr=(1-Lambda*r**2/3)*(r**2+a**2)-2*r+rq; Dt=1+Lambda*a**2*np.cos(th)**2/3; S=r**2+a**2*np.cos(th)**2;
                    Drp=-2/3*Lambda*a**2*r-4/3*Lambda*r**3+2*r-2; Dtp=-2/3*Lambda*a**2*np.cos(th)*np.sin(th);
                    Pofr=chi*(E*(r**2+a**2)-a*Lz); Wofth=chi*(a*E*np.sin(th)-Lz/np.sin(th));
                    prp=((2*chi*E*r*Pofr-Drp*k/2)/Dr-Drp*pr**2)/S;
                    pthp=((Dtp*k/2-chi**2*np.cos(th)*np.sin(th)*(a**2*E**2-Lz**2/np.sin(th)**4))/Dt-Dtp*pth**2)/S;
                    rp=Dr*pr/S;
                    thp=Dt*pth/S;
                    php=chi/S*(a*Pofr/Dr-Wofth/(Dt*np.sin(th)));
                    Y=[rp,thp,php,prp,pthp];
                    return Y
            
            
            ##Equi-rectangular projection from sphere to tangent plane
            def projtoplane_bis(w):
                if w[2]>rf:
                    wrf=1
                elif w[2]<-rf:
                    wrf=-1
                else:
                    wrf=w[2]/rf
                wp=[-1,np.arctan2(w[1],-w[0]),np.pi/2-np.arccos(wrf)];
                return(wp)
            
            ##From  to BL coordinates (with velocities)
            def BoyerLindquist_bis(R,T,P):
                BB=[np.sqrt(R**2+A**2)*np.sin(T)*np.cos(P),np.sqrt(R**2+A**2)*np.sin(T)*np.sin(P),R*np.cos(T)];
                return BB
            
            def InvBoyerLindquist_bis(x,xp,y,yp,z,zp):
                P=np.arctan2(y,x);
                R=np.sqrt((-A**2+x**2+y**2+z**2+np.sqrt(A**2*(A**2-2*x**2-2*y**2+2*z**2)+(x**2+y**2+z**2)**2))/2);
                T=np.arccos(z/R);
                Rp=R*(x*xp+y*yp+z*zp)/(2*R**2+A**2-x**2-y**2-z**2)+A**2*z*zp/(R*(2*R**2+A**2-x**2-y**2-z**2));
                Tp=(z*Rp-zp*R)/(R*np.sqrt(R**2-z**2));
                Pp=(yp*x-xp*y)/(x**2+y**2);
                XX=[0,1,R,Rp,T,Tp,P,Pp];
                return XX
            
            Xmax=22983; tau=-2*cSI/Rs*0.00042; dtau=tau/N; 
            
            ##Accretion_data[0]<2 means that we want the image with accretion disk  otherwise, we just want the accretion disk  no image is required.
            if Accretion_data[0]<2:
                Img=cv.imread(Image);
                Img=cv.cvtColor(Img,cv.COLOR_BGR2RGB)
                Npix=np.shape(Img)[0]; Npiy=np.shape(Img)[1]; IMG=np.zeros((Npiy,Npix,3));
                for i in range(3):
                    IMG[:,:,i]=np.transpose(Img[:,:,i])/256;
                Npix=np.shape(IMG)[0]; Npiy=np.shape(IMG)[1];
                
                XX=np.linspace(-Xmax,Xmax,Npix); YY=np.linspace(-Xmax*Npiy/Npix,Xmax*Npiy/Npix,Npiy);
                h=x0*Xmax*np.sqrt(1+Npiy**2/Npix**2)/(rf-Xmax*np.sqrt(1+Npiy**2/Npix**2));
            else:
                Npix=Accretion_data[0]; Npiy=Npix;
                XX=np.linspace(-Xmax,Xmax,Npix); YY=np.linspace(-Xmax,Xmax,Npiy);
                h=x0*Xmax*np.sqrt(2)/(rf-Xmax*np.sqrt(2));
            
            
            ##The initial datum from a pixel (y,z) on the screen (a ray leaving it with the appropriate velocity)
            def init_conds_with_angle_bis(y,z):
                v0=[x0,-cSI*h/np.sqrt(h**2+y**2+z**2),y,cSI*y/np.sqrt(h**2+y**2+z**2),z,cSI*z/np.sqrt(h**2+y**2+z**2)];
                matrot=np.array([[np.cos(alpha),0,-np.sin(alpha)],[0,1,0],[np.sin(alpha),0,np.cos(alpha)]]);
                vrot=matrot.dot(np.array([v0[0],v0[2],v0[4]])); vvrot=-matrot.dot(np.array([v0[1],v0[3],v0[5]]));
                Z=InvBoyerLindquist_bis(vrot[0],vvrot[0],vrot[1],vvrot[1],vrot[2],vvrot[2]);
                Z=[Z[2],Z[4],Z[6],Z[3],Z[5],Z[7]];
                Z=[2/Rs*Z[0],Z[1],Z[2],Z[3]/cSI,Z[4]*Rs/(2*cSI),Z[5]*Rs/(2*cSI)];
                return Z
            
            def init_conds_bis(y,z):
                v0=[x0,-cSI*h/np.sqrt(h**2+y**2+z**2),y,cSI*y/np.sqrt(h**2+y**2+z**2),z,cSI*z/np.sqrt(h**2+y**2+z**2)];
                vrot=np.array([v0[0],v0[2],v0[4]]); vvrot=-np.array([v0[1],v0[3],v0[5]]);
                Z=InvBoyerLindquist_bis(vrot[0],vvrot[0],vrot[1],vvrot[1],vrot[2],vvrot[2]);
                Z=[Z[2],Z[4],Z[6],Z[3],Z[5],Z[7]];
                Z=[2/Rs*Z[0],Z[1],Z[2],Z[3]/cSI,Z[4]*Rs/(2*cSI),Z[5]*Rs/(2*cSI)];
                return Z
            
            ##Cartesian velocity from BL velocity
            def velocity(sphvel):
                r=sphvel[0]; th=sphvel[1]; ph=sphvel[2]; rp=sphvel[3]; thp=sphvel[4]; php=sphvel[5];
                vx=np.cos(ph)*np.sin(th)*r*rp/np.sqrt(r**2 + a**2) - np.sqrt(r**2 + a**2)*php*np.sin(ph)*np.sin(th) + np.sqrt(r**2 + a**2)*np.cos(ph)*thp*np.cos(th);
                vy=np.sin(ph)*np.sin(th)*r*rp/np.sqrt(r**2 + a**2) + np.sqrt(r**2 + a**2)*php*np.cos(ph)*np.sin(th) + np.sqrt(r**2 + a**2)*np.sin(ph)*thp*np.cos(th);
                vz=rp*np.cos(th) - r*thp*np.sin(th);
                vel=[vx,vy,vz];
                return vel
            
            ##Function for the color shift
            def doppler_color(dope):
                dil_dope=(dope-1/2)**1;
                if dil_dope<1/4:
                    rcol=[0,4*dil_dope,1];
                elif (1/4<=dil_dope and dil_dope<1/2):
                    rcol=[0,1,2-4*dil_dope];
                elif (1/2<=dil_dope and dil_dope<3/4):
                    rcol=[4*dil_dope-2,1,0];
                else:
                    rcol=[1,4-4*dil_dope,0];
                if (abs(rcol[0])<0.05 and abs(rcol[2])<0.05 and abs(rcol[1]-1)<0.05):
                    rcol=[2,2,2];
                return rcol
            
            ##Depending on what kind of accretion is required, we create a function 'accretion_disk'
            ##which computes the color to be attributed to a pixel, taking into account the various effects.
            ##This function is defined at this point so that we don't need to put a selection process for each pixel:
            ##the color function is chosen once  for all.
            if (Accretion_data[2]=="Blackbody" and Accretion_data[3]=="Doppler"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); Dr=(1-Lambda*r**2/3)*(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2; Dt=1+Lambda*a**2*np.cos(th)**2/3;
                    veloc=cosmo_inv_met_mat([0,r,th,ph],Lambda,a,rq).dot(np.array([pt,V[3],V[4],pph])); al=(a+r**2/np.sqrt(-Lambda*r**4/3+r-rq))/rb;
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc);
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4); T=T/doppler_shift;
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    colou=blackbody[np.where(abs(T-blackbody[:,0])==min(abs(T-blackbody[:,0])))[0][0],range(1,4)];
                    cb=[colou,bright/doppler_shift,T];
                    return cb
            elif (Accretion_data[2]=="Blackbody" and Accretion_data[3]=="Gravitation"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); Dr=(1-Lambda*r**2/3)*(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2; Dt=1+Lambda*a**2*np.cos(th)**2/3;
                    grav_shift=1/np.sqrt(abs(Dr-a**2*np.sin(th)**2*Dt)/(chi**2*S));
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4); T=T/grav_shift;
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    colou=blackbody[np.where(abs(T-blackbody[:,0])==min(abs(T-blackbody[:,0])))[0][0],range(1,4)];
                    cb=[colou,bright/(grav_shift),T];
                    return cb
            elif (Accretion_data[2]=="Blackbody" and Accretion_data[3]=="Doppler+"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); Dr=(1-Lambda*r**2/3)*(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2; Dt=1+Lambda*a**2*np.cos(th)**2/3;
                    veloc=cosmo_inv_met_mat([0,r,th,ph],Lambda,a,rq).dot(np.array([pt,V[3],V[4],pph])); al=(a+r**2/np.sqrt(-Lambda*r**4/3+r-rq))/rb;
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc);
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    grav_shift=1/np.sqrt(abs(Dr-a**2*np.sin(th)**2*Dt)/(chi**2*S));
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4); T=T/(grav_shift*doppler_shift);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    colou=blackbody[np.where(abs(T-blackbody[:,0])==min(abs(T-blackbody[:,0])))[0][0],range(1,4)];
                    cb=[colou,bright/(grav_shift*doppler_shift),T];
                    return cb
            elif (Accretion_data[2]=="Blackbody" and Accretion_data[3]==" "):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    colou=blackbody[np.where(abs(T-blackbody[:,0])==min(abs(T-blackbody[:,0])))[0][0],range(1,4)];
                    cb=[colou,bright,T];
                    return cb
            elif (Accretion_data[2]=="Custom" and Accretion_data[3]=="Doppler"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); Dr=(1-Lambda*r**2/3)*(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2; Dt=1+Lambda*a**2*np.cos(th)**2/3;
                    veloc=cosmo_inv_met_mat([0,r,th,ph],Lambda,a,rq).dot(np.array([pt,V[3],V[4],pph])); al=(a+r**2/np.sqrt(-Lambda*r**4/3+r-rq))/rb;
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc);
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    doppler_coeff=1-np.sqrt((2*rb**2 + a*(a - 4)*rb + 2*a**2)/rb**3);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    cb=[np.array([0,0,0]),bright/doppler_shift,doppler_coeff**2,1/doppler_shift,T];
                    return cb
            elif (Accretion_data[2]=="Custom" and Accretion_data[3]=="Gravitation"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); Dr=(1-Lambda*r**2/3)*(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2; Dt=1+Lambda*a**2*np.cos(th)**2/3;
                    grav_shift=1/np.sqrt(abs(Dr-a**2*np.sin(th)**2*Dt)/(chi**2*S));
                    doppler_coeff=1-np.sqrt((2*rb**2 + a*(a - 4)*rb + 2*a**2)/rb**3);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    cb=[np.array([0,0,0]),bright/grav_shift,doppler_coeff**2,1/grav_shift,T];
                    return cb
            elif (Accretion_data[2]=="Custom" and Accretion_data[3]=="Doppler+"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); Dr=(1-Lambda*r**2/3)*(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2; Dt=1+Lambda*a**2*np.cos(th)**2/3;
                    veloc=cosmo_inv_met_mat([0,r,th,ph],Lambda,a,rq).dot(np.array([pt,V[3],V[4],pph])); al=(a+r**2/np.sqrt(-Lambda*r**4/3+r-rq))/rb;
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc)
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    grav_shift=1/np.sqrt(abs(Dr-a**2*np.sin(th)**2*Dt)/(chi**2*S));
                    doppler_coeff=1-np.sqrt((2*rb**2 + a*(a - 4)*rb + 2*a**2)/rb**3);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    cb=[np.array([0,0,0]),bright/(grav_shift*doppler_shift)**0,doppler_coeff**2,1/(grav_shift*doppler_shift),T];
                    return cb
            elif (Accretion_data[2]=="Custom" and Accretion_data[3]==" "):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2);
                    doppler_coeff=1-np.sqrt((2*rb**2 + a*(a - 4)*rb + 2*a**2)/rb**3);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    cb=[np.array([0,0,0]),bright,doppler_coeff**2,1,T];
                    return cb
            elif (Accretion_data[2]==" " and Accretion_data[3]=="Doppler"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); Dr=(1-Lambda*r**2/3)*(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2; Dt=1+Lambda*a**2*np.cos(th)**2/3;
                    veloc=cosmo_inv_met_mat([0,r,th,ph],Lambda,a,rq).dot(np.array([pt,V[3],V[4],pph])); al=(a+r**2/np.sqrt(-Lambda*r**4/3+r-rq))/rb;
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc)
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    colou=doppler_color(doppler_shift);
                    bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    cb=[np.array(colou),bright];
                    return cb
            elif (Accretion_data[2]==" " and Accretion_data[3]=="Gravitation"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); Dr=(1-Lambda*r**2/3)*(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2; Dt=1+Lambda*a**2*np.cos(th)**2/3;
                    grav_shift=1/np.sqrt(abs(Dr-a**2*np.sin(th)**2*Dt)/(chi**2*S));
                    colou=doppler_color(grav_shift);
                    bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    cb=[np.array(colou),bright];
                    return cb
            elif (Accretion_data[2]==" " and Accretion_data[3]=="Doppler+"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); Dr=(1-Lambda*r**2/3)*(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2; Dt=1+Lambda*a**2*np.cos(th)**2/3;
                    veloc=cosmo_inv_met_mat([0,r,th,ph],Lambda,a,rq).dot(np.array([pt,V[3],V[4],pph])); al=(a+r**2/np.sqrt(-Lambda*r**4/3+r-rq))/rb;
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc);
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    grav_shift=1/np.sqrt(abs(Dr-a**2*np.sin(th)**2*Dt)/(chi**2*S));
                    colou=doppler_color(doppler_shift*grav_shift);
                    bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    cb=[np.array(colou),bright];
                    return cb
            else:
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); lam=0.2;
                    cb=np.ones((1,4)); cb=np.array([255,69,0])/256; cb[2]=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    cb=[cb,1]
                    return cb
            
            Umax=np.pi/2; Vmax=Umax*Npiy/Npix; xred=np.zeros((Npix,Npiy,3));
            npr=np.roots([-Lambda/3,0,1-Lambda*a**2/3,-2,rq+a**2])
            mi=min(abs(npr-np.real(npr)))
            tes=[np.real(npr[rr]) for rr in np.where(abs(npr-np.real(npr))==mi)[0].tolist() if np.real(npr[rr])<2]
            if (mi<1e-8 and len(tes)>0):
                ir=max(tes)
            else:
                ir=-np.Inf

            ##First case where no accretion disk is required: we compute the ray for each pixel (using Carter's equations), 
            ##find if it hits the sphere and compute the associated pixel on the tangent plane.
            ##Using the maximal reach previously calculated, we find its position on the projected image and give it the right color.
            ##When calling the 'ode' function, we eliminate the pixels for which the integration fails (those who die in the black hole).
            if Accretion_data[0]==0:
                for y in XX:
                    i=np.where(XX==y)[0][0];
                    for z in YY:
                        j=np.where(YY==z)[0][0];
                        X=init_conds_with_angle_bis(y,z);
                        r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
                        Dr=(1-Lambda*r**2/3)*(r**2+a**2)-2*r+rq; Dt=1+Lambda*a**2*np.cos(th)**2/3; S=r**2+a**2*np.cos(th)**2;
                        E=np.sqrt(Dr*Dt*php**2*np.sin(th)**2/chi**4+(Dr-Dt*a**2*np.sin(th)**2)/(chi**2*S)*(rp**2*S/Dr+thp**2*S/Dt));
                        Lz=np.sin(th)**2/(chi**2*(Dt*a**2*np.sin(th)**2-Dr))*(Dt*(a*chi**2*E*(r**2+a**2)-php*S*Dr)-a*chi**2*E*Dr);
                        pt=-E; pr=S*rp/Dr; pth=S*thp/Dt; pph=Lz;
                        Q=Dt*pth**2+chi**2*np.cos(th)**2/Dt*(Lz**2/np.sin(th)**2-a**2*(E**2+Lambda**2/3*(a*E-Lz)**2));
                        k=Q+chi**2*(a*E-Lz)**2;
                        X=[r,th,ph,pr,pth];
                        
                        tau0=tau; dtau0=dtau; ite=0;
                        while ite<itermax:
                            ite+=1
                            try:
                                Vec=odeint(Carter_ter,X,np.arange(0,tau0,dtau0));
                            except Warning:
                                tau0*=cft;
                                dtau0*=cft;
                                continue
                            break
                        
                        R=Rs/2*Vec[:,0]; theta=Vec[:,1]; phi=Vec[:,2];
                        wef=np.zeros((3,1)); WEF=[];
                        dWEF=abs(rf-np.sqrt(R**2+A**2*np.sin(theta)**2)); dwef=min(dWEF);
                        if (dwef<2.5e2 and (len(np.where([abs(phii)>=np.Inf*np.pi for phii in phi])[0])==0 and len(np.where([Vec[vv,0]<=1.01*ir for vv in range(N)])[0])==0)):#len(np.where([abs(phii)>=np.Inf*np.pi for phii in phi])[0])==0):#(dwef<2.5e2 and len(np.where([rg<ir*Rs for rg in R])[0])==0):
                            l=np.where(dWEF==dwef)[0][0];
                            wef=BoyerLindquist_bis(R[l],theta[l],phi[l]);
                            wef=np.array([[np.cos(alpha),0,np.sin(alpha)],[0,1,0],[-np.sin(alpha),0,np.cos(alpha)]]).dot(np.array(wef));
                            wef=wef.tolist()
                            wef=projtoplane_bis(wef);
                            s1=np.real(wef[1]+Umax)/(2*Umax); s2=np.real(wef[2]+Vmax)/(2*Vmax);
                            s1=abs(1-abs(1-s1)); s2=abs(1-abs(1-s2));
                            ii=int(max(1,min(Npix,np.ceil(s1*Npix)))); jj=int(max(1,min(Npiy,np.ceil(s2*Npiy))));
                            xred[i,j,0]=IMG[ii-1,jj-1,0]; xred[i,j,1]=IMG[ii-1,jj-1,1]; xred[i,j,2]=IMG[ii-1,jj-1,2];
                        
                    
            
                ##If Accretion_data(1)=1, it means we want to shadow the black hole with a picture and an accretion disk.
                ##We do the same as before, except that we shouldn't eliminate all the dying pixels:
                ##instead, if a ray goes to the horizon (an error is then returned by 'ode'),
                ##then we integrate it on [0,cft*Tau] instead of [0,Tau], where 0<cft<1 and we do it at most itermax times.
                ##These parameters can be tuned at the very first line of the present function. Morally, the smaller the inner radius, 
                ##the bigger cft, itermax and step size N should be.
                ##The first ray that doesn't hit the horizon (if any) is kept, and we test if it crosses the theta=pi/2 plane (with a tolerance of 1e-2).
                ##If so, we check if the crossing happens between the chosen radii rint and rext and if so, this adds a pixel to the accretion disk.
                ##We compute the function accretion_disk for this pixel and obtain the corresponding RGB value.
            elif Accretion_data[0]==1:
                dop_max=np.zeros((Npix,Npiy));
                for y in XX:
                    i=np.where(XX==y)[0][0];
                    for z in YY:
                        j=np.where(YY==z)[0][0];
                        X=init_conds_with_angle_bis(y,z);
                        r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
                        Dr=(1-Lambda*r**2/3)*(r**2+a**2)-2*r+rq; Dt=1+Lambda*a**2*np.cos(th)**2/3; S=r**2+a**2*np.cos(th)**2;
                        E=np.sqrt(Dr*Dt*php**2*np.sin(th)**2/chi**4+(Dr-Dt*a**2*np.sin(th)**2)/(chi**2*S)*(rp**2*S/Dr+thp**2*S/Dt));
                        Lz=np.sin(th)**2/(chi**2*(Dt*a**2*np.sin(th)**2-Dr))*(Dt*(a*chi**2*E*(r**2+a**2)-php*S*Dr)-a*chi**2*E*Dr);
                        pt=-E; pr=S*rp/Dr; pth=S*thp/Dt; pph=Lz;
                        Q=Dt*pth**2+chi**2*np.cos(th)**2/Dt*(Lz**2/np.sin(th)**2-a**2*(E**2+Lambda**2/3*(a*E-Lz)**2));
                        k=Q+chi**2*(a*E-Lz)**2;
                        X=[r,th,ph,pr,pth];                
                        
                        tau0=tau; dtau0=dtau; ite=0;
                        while ite<itermax:
                            ite+=1
                            try:
                                Vec=odeint(Carter_ter,X,np.arange(0,tau0,dtau0));
                            except Warning:
                                tau0*=cft;
                                dtau0*=cft;
                                continue
                            break
                        
                        R=Rs/2*Vec[:,0]; theta=Vec[:,1]; phi=Vec[:,2]; PR=Vec[:,3]; PTH=Vec[:,4];
                        wef=np.zeros((3,1)); WEF=[];
                        dWEF=abs(rf-np.sqrt(R**2+A**2*np.sin(theta)**2)); dwef=min(dWEF);
                        if (dwef<2.5e2 and (len(np.where([abs(phii)>=np.Inf*np.pi for phii in phi])[0])==0 and len(np.where([Vec[vv,0]<=1.01*ir for vv in range(N)])[0])==0)):#len(np.where([abs(phii)>=np.Inf*np.pi for phii in phi])[0])==0):#(dwef<2.5e2 and len(np.where([rg<ir*Rs for rg in R])[0])==0):
                            l=np.where(dWEF==dwef)[0][0];
                            wef=BoyerLindquist_bis(R[l],theta[l],phi[l]);
                            wef=np.array([[np.cos(alpha),0,np.sin(alpha)],[0,1,0],[-np.sin(alpha),0,np.cos(alpha)]]).dot(np.array(wef));
                            wef=wef.tolist()
                            wef=projtoplane_bis(wef);
                            s1=np.real(wef[1]+Umax)/(2*Umax); s2=np.real(wef[2]+Vmax)/(2*Vmax);
                            s1=abs(1-abs(1-s1)); s2=abs(1-abs(1-s2));
                            ii=int(max(1,min(Npix,np.ceil(s1*Npix)))); jj=int(max(1,min(Npiy,np.ceil(s2*Npiy))));
                            xred[i,j,0]=IMG[ii-1,jj-1,0]; xred[i,j,1]=IMG[ii-1,jj-1,1]; xred[i,j,2]=IMG[ii-1,jj-1,2];
                        
                        ll=np.where(abs(theta-np.pi/2)<1/100)[0]; whitness=0;
                        for l in ll:
                            rb=np.sqrt(Vec[l,0]**2+a**2);
                            if (rb*Rs/2>rint and rb*Rs/2<rext and whitness==0 and len(np.where([Vec[vv,0]<=1.05*ir for vv in range(l)])[0])==0):
                                whitness=1;
                                acc=accretion_disk([R[l],theta[l],phi[l],PR[l],PTH[l]]);
                                acc=acc[0].tolist()+acc[1:]
                                vef=BoyerLindquist_bis(R[l],theta[l],phi[l])+acc;
                                Temp.append(vef[-1]);
                                if Accretion_data[2]=="Custom":
                                    xred[i,j,0]=-np.exp(1); xred[i,j,1]=vef[6]; xred[i,j,2]=vef[8]; dop_max[i,j]=vef[7];
                                else:
                                    xred[i,j,0]=vef[6]*vef[3]; xred[i,j,1]=vef[6]*vef[4]; xred[i,j,2]=vef[6]*vef[5];
                                
                            
                        
                    
                
            
                ##In case the inner  outer temperatures are specified (the "Custom" case), we need one more loop on pixels to find the colors of the accretion disk.
                if Accretion_data[2]=="Custom":
                    dp_max=max((dop_max.flatten()).tolist())
                    for i in range(Npix):
                        for j in range(Npiy):
                            if xred[i,j,0]==-np.exp(1):
                                flo=int(np.floor(xred[i,j,2]*(T_int+(T_ext-T_int)*dop_max[i,j]/dp_max)));
                                wef=blackbody[np.where(abs(flo-blackbody[:,0])==min(abs(flo-blackbody[:,0])))[0][0],range(1,4)];
                                xred[i,j,0]=xred[i,j,1]*wef[0]; xred[i,j,2]=xred[i,j,1]*wef[2]; xred[i,j,1]=xred[i,j,1]*wef[1];
                            
                        
                    
                    
                ##If Accretion_data[0]>1,: we don't want the picture, only the shadow of the accretion disk.
            elif Accretion_data[0]>1:
                dop_max=np.zeros((Npix,Npiy));
                for y in XX:
                    i=np.where(XX==y)[0][0];
                    for z in YY:
                        j=np.where(YY==z)[0][0];
                        X=init_conds_with_angle_bis(y,z);
                        r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
                        Dr=(1-Lambda*r**2/3)*(r**2+a**2)-2*r+rq; Dt=1+Lambda*a**2*np.cos(th)**2/3; S=r**2+a**2*np.cos(th)**2;
                        E=np.sqrt(Dr*Dt*php**2*np.sin(th)**2/chi**4+(Dr-Dt*a**2*np.sin(th)**2)/(chi**2*S)*(rp**2*S/Dr+thp**2*S/Dt));
                        Lz=np.sin(th)**2/(chi**2*(Dt*a**2*np.sin(th)**2-Dr))*(Dt*(a*chi**2*E*(r**2+a**2)-php*S*Dr)-a*chi**2*E*Dr);
                        pt=-E; pr=S*rp/Dr; pth=S*thp/Dt; pph=Lz;
                        Q=Dt*pth**2+chi**2*np.cos(th)**2/Dt*(Lz**2/np.sin(th)**2-a**2*(E**2+Lambda**2/3*(a*E-Lz)**2));
                        k=Q+chi**2*(a*E-Lz)**2;
                        X=[r,th,ph,pr,pth];
                        
                        Vec=odeint(Carter_ter,X,np.arange(0,tau,dtau));
                        R=Rs/2*Vec[:,0]; theta=Vec[:,1]; phi=Vec[:,2]; PR=Vec[:,3]; PTH=Vec[:,4];
                        xred[i,j,0]=0; xred[i,j,1]=0; xred[i,j,2]=0;
                        ll=np.where(abs(theta-np.pi/2)<1/100)[0]; whitness=0;
                        for l in ll:
                            rb=np.sqrt(Vec[l,0]**2+a**2);
                            if (rb*Rs/2>rint and rb*Rs/2<rext and whitness==0 and len(np.where([Vec[vv,0]<=1.05*ir for vv in range(l)])[0])==0):
                                whitness=1;
                                acc=accretion_disk([R[l],theta[l],phi[l],PR[l],PTH[l]]);
                                acc=acc[0].tolist()+acc[1:]
                                vef=BoyerLindquist_bis(R[l],theta[l],phi[l])+acc;
                                Temp.append(vef[-1]);
                                if Accretion_data[2]=="Custom":
                                    xred[i,j,0]=-np.exp(1); xred[i,j,1]=vef[6]; xred[i,j,2]=vef[8]; dop_max[i,j]=vef[7];
                                else:
                                    xred[i,j,0]=vef[6]*vef[3]; xred[i,j,1]=vef[6]*vef[4]; xred[i,j,2]=vef[6]*vef[5];
                                
                                
                
            
                if Accretion_data[2]=="Custom":
                    dp_max=max((dop_max.flatten()).tolist())
                    for i in range(Npix):
                        for j in range(Npiy):
                            if xred[i,j,0]==-np.exp(1):
                                flo=int(np.floor(xred[i,j,2]*(T_int+(T_ext-T_int)*dop_max[i,j]/dp_max)));
                                wef=blackbody[np.where(abs(flo-blackbody[:,0])==min(abs(flo-blackbody[:,0])))[0][0],range(1,4)];
                                xred[i,j,0]=xred[i,j,1]*wef[0]; xred[i,j,2]=xred[i,j,1]*wef[2]; xred[i,j,1]=xred[i,j,1]*wef[1];
                            
                        
                    
            

        else:
            c=1; G=1; M=1; GSI=6.67408e-11; cSI=299792458; e0=8.854187e-12; sb=5.67e-8;
            Rs=2*GSI*Mass/cSI**2; A=Kerr*Rs/2; alpha=-Accretion_data[1];##J=Kerr*GSI*Mass**2/cSI; A=J/(Mass*cSI); alpha=-Accretion_data[1];
            rq=Newman**2;##Q=Newman*2*Mass*np.sqrt(np.pi*e0*GSI); rq2=Q**2*GSI/(4*np.pi*e0*cSI**4); rq=4*rq2/Rs**2;
            x0=50000; sizee=Accretion_data[4]; rint=sizee[0]*Rs; rext=sizee[1]*Rs; rf=60000;
            rs=2; rg=1; a=Kerr; T_int=1001; T_ext=1001; lam=0.8;
            Mrate=Accretion_data[5]; Mrate=Mrate[0]*Rs*cSI**2/(Mass*2*GSI); T0=3*cSI**2*Rs*Mrate/(2*np.pi*sb);
            Temp=[];
            if (len(Accretion_data[5])>1 and Accretion_data[5][2]<Accretion_data[5][1]):
                T_int=Accretion_data[5]; T_ext=T_int[2]; T_int=T_int[1];


            if Kerr==0:
                def Carter_ter(V,tau):
                    r=V[0]; th=V[1]; ph=V[2]; pr=V[3]; pth=V[4];
                    Dr=r**2-2*r+rq;
                    Drp=2*r-2;
                    Pofr=E*r**2;
                    prp=((2*E*r*Pofr-Drp*k/2)/Dr-Drp*pr**2)/r**2;
                    pthp=(np.cos(th)*np.sin(th)*Lz**2/np.sin(th)**4)/r**2;
                    rp=Dr*pr/r**2;
                    thp=pth/r**2;
                    php=Lz/(r**2*np.sin(th)**2);
                    Y=[rp,thp,php,prp,pthp];
                    return Y
            else:
                def Carter_ter(V,tau):
                    r=V[0]; th=V[1]; ph=V[2]; pr=V[3]; pth=V[4];
                    Dr=(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2;
                    Drp=2*r-2;
                    Pofr=(E*(r**2+a**2)-a*Lz); Wofth=(a*E*np.sin(th)-Lz/np.sin(th));
                    prp=((2*E*r*Pofr-Drp*k/2)/Dr-Drp*pr**2)/S;
                    pthp=(-np.cos(th)*np.sin(th)*(a**2*E**2-Lz**2/np.sin(th)**4))/S;
                    rp=Dr*pr/S;
                    thp=pth/S;
                    php=1/S*(a*Pofr/Dr-Wofth/np.sin(th));
                    Y=[rp,thp,php,prp,pthp];
                    return Y


            def projtoplane_bis(w):
                if w[2]>rf:
                    wrf=1
                elif w[2]<-rf:
                    wrf=-1
                else:
                    wrf=w[2]/rf
                wp=[-1,np.arctan2(w[1],-w[0]),np.pi/2-np.arccos(wrf)];
                return(wp)

            def BoyerLindquist_bis(R,T,P):
                BB=[np.sqrt(R**2+A**2)*np.sin(T)*np.cos(P),np.sqrt(R**2+A**2)*np.sin(T)*np.sin(P),R*np.cos(T)];
                return BB

            def InvBoyerLindquist_bis(x,xp,y,yp,z,zp):
                P=np.arctan2(y,x);
                R=np.sqrt((-A**2+x**2+y**2+z**2+np.sqrt(A**2*(A**2-2*x**2-2*y**2+2*z**2)+(x**2+y**2+z**2)**2))/2);
                T=np.arccos(z/R);
                Rp=R*(x*xp+y*yp+z*zp)/(2*R**2+A**2-x**2-y**2-z**2)+A**2*z*zp/(R*(2*R**2+A**2-x**2-y**2-z**2));
                Tp=(z*Rp-zp*R)/(R*np.sqrt(R**2-z**2));
                Pp=(yp*x-xp*y)/(x**2+y**2);
                XX=[0,1,R,Rp,T,Tp,P,Pp];
                return XX

            Xmax=22983; tau=-2*cSI/Rs*0.00042; dtau=tau/N; 

            if Accretion_data[0]<2:
                Img=cv.imread(Image);
                Img=cv.cvtColor(Img,cv.COLOR_BGR2RGB)
                Npix=np.shape(Img)[0]; Npiy=np.shape(Img)[1]; IMG=np.zeros((Npiy,Npix,3));
                for i in range(3):
                    IMG[:,:,i]=np.transpose(Img[:,:,i])/256;
                Npix=np.shape(IMG)[0]; Npiy=np.shape(IMG)[1];
                
                XX=np.linspace(-Xmax,Xmax,Npix); YY=np.linspace(-Xmax*Npiy/Npix,Xmax*Npiy/Npix,Npiy);
                h=x0*Xmax*np.sqrt(1+Npiy**2/Npix**2)/(rf-Xmax*np.sqrt(1+Npiy**2/Npix**2));
            else:
                Npix=Accretion_data[0]; Npiy=Npix;
                XX=np.linspace(-Xmax,Xmax,Npix); YY=np.linspace(-Xmax,Xmax,Npiy);
                h=x0*Xmax*np.sqrt(2)/(rf-Xmax*np.sqrt(2));


            def init_conds_with_angle_bis(y,z):
                v0=[x0,-cSI*h/np.sqrt(h**2+y**2+z**2),y,cSI*y/np.sqrt(h**2+y**2+z**2),z,cSI*z/np.sqrt(h**2+y**2+z**2)];
                matrot=np.array([[np.cos(alpha),0,-np.sin(alpha)],[0,1,0],[np.sin(alpha),0,np.cos(alpha)]]);
                vrot=matrot.dot(np.array([v0[0],v0[2],v0[4]])); vvrot=-matrot.dot(np.array([v0[1],v0[3],v0[5]]));
                Z=InvBoyerLindquist_bis(vrot[0],vvrot[0],vrot[1],vvrot[1],vrot[2],vvrot[2]);
                Z=[Z[2],Z[4],Z[6],Z[3],Z[5],Z[7]];
                Z=[2/Rs*Z[0],Z[1],Z[2],Z[3]/cSI,Z[4]*Rs/(2*cSI),Z[5]*Rs/(2*cSI)];
                return Z

            def init_conds_bis(y,z):
                v0=[x0,-cSI*h/np.sqrt(h**2+y**2+z**2),y,cSI*y/np.sqrt(h**2+y**2+z**2),z,cSI*z/np.sqrt(h**2+y**2+z**2)];
                vrot=np.array([v0[0],v0[2],v0[4]]); vvrot=-np.array([v0[1],v0[3],v0[5]]);
                Z=InvBoyerLindquist_bis(vrot[0],vvrot[0],vrot[1],vvrot[1],vrot[2],vvrot[2]);
                Z=[Z[2],Z[4],Z[6],Z[3],Z[5],Z[7]];
                Z=[2/Rs*Z[0],Z[1],Z[2],Z[3]/cSI,Z[4]*Rs/(2*cSI),Z[5]*Rs/(2*cSI)];
                return Z

            def velocity(sphvel):
                r=sphvel[0]; th=sphvel[1]; ph=sphvel[2]; rp=sphvel[3]; thp=sphvel[4]; php=sphvel[5];
                vx=np.cos(ph)*np.sin(th)*r*rp/np.sqrt(r**2 + a**2) - np.sqrt(r**2 + a**2)*php*np.sin(ph)*np.sin(th) + np.sqrt(r**2 + a**2)*np.cos(ph)*thp*np.cos(th);
                vy=np.sin(ph)*np.sin(th)*r*rp/np.sqrt(r**2 + a**2) + np.sqrt(r**2 + a**2)*php*np.cos(ph)*np.sin(th) + np.sqrt(r**2 + a**2)*np.sin(ph)*thp*np.cos(th);
                vz=rp*np.cos(th) - r*thp*np.sin(th);
                vel=[vx,vy,vz];
                return vel

            def doppler_color(dope):
                dil_dope=(dope-1/2)**1;
                if dil_dope<1/4:
                    rcol=[0,4*dil_dope,1];
                elif (1/4<=dil_dope and dil_dope<1/2):
                    rcol=[0,1,2-4*dil_dope];
                elif (1/2<=dil_dope and dil_dope<3/4):
                    rcol=[4*dil_dope-2,1,0];
                else:
                    rcol=[1,4-4*dil_dope,0];
                if (abs(rcol[0])<0.05 and abs(rcol[2])<0.05 and abs(rcol[1]-1)<0.05):
                    rcol=[2,2,2];
                return rcol


            if (Accretion_data[2]=="Blackbody" and Accretion_data[3]=="Doppler"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); Dr=(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2;
                    veloc=inv_met_mat([0,r,th,ph],a,rq).dot(np.array([pt,V[3],V[4],pph])); al=(a+r**2/np.sqrt(r-rq))/rb;
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc);
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4); T=T/doppler_shift;
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    colou=blackbody[np.where(abs(T-blackbody[:,0])==min(abs(T-blackbody[:,0])))[0][0],range(1,4)];
                    cb=[colou,bright/doppler_shift,T];
                    return cb
            elif (Accretion_data[2]=="Blackbody" and Accretion_data[3]=="Gravitation"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); Dr=(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2;
                    grav_shift=1/np.sqrt(abs(Dr-a**2*np.sin(th)**2)/S);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4); T=T/grav_shift;
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    colou=blackbody[np.where(abs(T-blackbody[:,0])==min(abs(T-blackbody[:,0])))[0][0],range(1,4)];
                    cb=[colou,bright/(grav_shift),T];
                    return cb
            elif (Accretion_data[2]=="Blackbody" and Accretion_data[3]=="Doppler+"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); Dr=(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2;
                    veloc=inv_met_mat([0,r,th,ph],a,rq).dot(np.array([pt,V[3],V[4],pph])); al=(a+r**2/np.sqrt(r-rq))/rb;
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc);
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    grav_shift=1/np.sqrt(abs(Dr-a**2*np.sin(th)**2)/S);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4); T=T/(grav_shift*doppler_shift);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    colou=blackbody[np.where(abs(T-blackbody[:,0])==min(abs(T-blackbody[:,0])))[0][0],range(1,4)];
                    cb=[colou,bright/(grav_shift*doppler_shift),T];
                    return cb
            elif (Accretion_data[2]=="Blackbody" and Accretion_data[3]==" "):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    colou=blackbody[np.where(abs(T-blackbody[:,0])==min(abs(T-blackbody[:,0])))[0][0],range(1,4)];
                    cb=[colou,bright,T];
                    return cb
            elif (Accretion_data[2]=="Custom" and Accretion_data[3]=="Doppler"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); Dr=(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2;
                    veloc=inv_met_mat([0,r,th,ph],a,rq).dot(np.array([pt,V[3],V[4],pph])); al=(a+r**2/np.sqrt(r-rq))/rb;
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc);
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    doppler_coeff=1-np.sqrt((2*rb**2 + a*(a - 4)*rb + 2*a**2)/rb**3);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    cb=[np.array([0,0,0]),bright/doppler_shift,doppler_coeff**2,1/doppler_shift,T];
                    return cb
            elif (Accretion_data[2]=="Custom" and Accretion_data[3]=="Gravitation"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); Dr=(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2;
                    grav_shift=1/np.sqrt(abs(Dr-a**2*np.sin(th)**2)/S);
                    doppler_coeff=1-np.sqrt((2*rb**2 + a*(a - 4)*rb + 2*a**2)/rb**3);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    cb=[np.array([0,0,0]),bright/grav_shift,doppler_coeff**2,1/grav_shift,T];
                    return cb
            elif (Accretion_data[2]=="Custom" and Accretion_data[3]=="Doppler+"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); Dr=(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2;
                    veloc=inv_met_mat([0,r,th,ph],a,rq).dot(np.array([pt,V[3],V[4],pph])); al=(a+r**2/np.sqrt(r-rq))/rb;
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc)
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    grav_shift=1/np.sqrt(abs(Dr-a**2*np.sin(th)**2)/S);
                    doppler_coeff=1-np.sqrt((2*rb**2 + a*(a - 4)*rb + 2*a**2)/rb**3);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    cb=[np.array([0,0,0]),bright/(grav_shift*doppler_shift)**0,doppler_coeff**2,1/(grav_shift*doppler_shift),T];
                    return cb
            elif (Accretion_data[2]=="Custom" and Accretion_data[3]==" "):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2);
                    doppler_coeff=1-np.sqrt((2*rb**2 + a*(a - 4)*rb + 2*a**2)/rb**3);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    cb=[np.array([0,0,0]),bright,doppler_coeff**2,1,T];
                    return cb
            elif (Accretion_data[2]==" " and Accretion_data[3]=="Doppler"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); Dr=(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2;
                    veloc=inv_met_mat([0,r,th,ph],a,rq).dot(np.array([pt,V[3],V[4],pph])); al=(a+r**2/np.sqrt(r-rq))/rb;
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc)
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    colou=doppler_color(doppler_shift);
                    bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    cb=[np.array(colou),bright];
                    return cb
            elif (Accretion_data[2]==" " and Accretion_data[3]=="Gravitation"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); Dr=(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2;
                    grav_shift=1/np.sqrt(abs(Dr-a**2*np.sin(th)**2)/S);
                    colou=doppler_color(grav_shift);
                    bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    cb=[np.array(colou),bright];
                    return cb
            elif (Accretion_data[2]==" " and Accretion_data[3]=="Doppler+"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); Dr=(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2;
                    veloc=inv_met_mat([0,r,th,ph],a,rq).dot(np.array([pt,V[3],V[4],pph])); al=(a+r**2/np.sqrt(r-rq))/rb;
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc);
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    grav_shift=1/np.sqrt(abs(Dr-a**2*np.sin(th)**2)/S);
                    colou=doppler_color(doppler_shift*grav_shift);
                    bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    cb=[np.array(colou),bright];
                    return cb
            else:
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); lam=0.2;
                    cb=np.ones((1,4)); cb=np.array([255,69,0])/256; cb[2]=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    cb=[cb,1]
                    return cb

            Umax=np.pi/2; Vmax=Umax*Npiy/Npix; xred=np.zeros((Npix,Npiy,3));
            npr=np.roots([1,-2,rq+a**2])
            mi=min(abs(npr-np.real(npr)))
            tes=[np.real(npr[rr]) for rr in np.where(abs(npr-np.real(npr))==mi)[0].tolist() if np.real(npr[rr])<2]
            if (mi<1e-8 and len(tes)>0):
                ir=max(tes)
            else:
                ir=-np.Inf


            if Accretion_data[0]==0:
                for y in XX:
                    i=np.where(XX==y)[0][0];
                    for z in YY:
                        j=np.where(YY==z)[0][0];
                        X=init_conds_with_angle_bis(y,z);
                        r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
                        Dr=(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2;
                        E=np.sqrt(Dr*php**2*np.sin(th)**2+(Dr-a**2*np.sin(th)**2)/S*(rp**2*S/Dr+thp**2*S));
                        Lz=np.sin(th)**2/(a**2*np.sin(th)**2-Dr)*((a*E*(r**2+a**2)-php*S*Dr)-a*E*Dr);
                        pt=-E; pr=S*rp/Dr; pth=S*thp; pph=Lz;
                        Q=pth**2+np.cos(th)**2*(Lz**2/np.sin(th)**2-a**2*E**2);
                        k=Q+(a*E-Lz)**2;
                        X=[r,th,ph,pr,pth];
                        
                        tau0=tau; dtau0=dtau; ite=0;
                        while ite<itermax:
                            ite+=1
                            try:
                                Vec=odeint(Carter_ter,X,np.arange(0,tau0,dtau0));
                            except Warning:
                                tau0*=cft;
                                dtau0*=cft;
                                continue
                            break
                        
                        R=Rs/2*Vec[:,0]; theta=Vec[:,1]; phi=Vec[:,2];
                        wef=np.zeros((3,1)); WEF=[];
                        dWEF=abs(rf-np.sqrt(R**2+A**2*np.sin(theta)**2)); dwef=min(dWEF);
                        if (dwef<2.5e2 and (len(np.where([abs(phii)>=np.Inf*np.pi for phii in phi])[0])==0 and len(np.where([Vec[vv,0]<=1.01*ir for vv in range(N)])[0])==0)):#len(np.where([abs(phii)>=np.Inf*np.pi for phii in phi])[0])==0):#(dwef<2.5e2 and len(np.where([rg<ir*Rs for rg in R])[0])==0):
                            l=np.where(dWEF==dwef)[0][0];
                            wef=BoyerLindquist_bis(R[l],theta[l],phi[l]);
                            wef=np.array([[np.cos(alpha),0,np.sin(alpha)],[0,1,0],[-np.sin(alpha),0,np.cos(alpha)]]).dot(np.array(wef));
                            wef=wef.tolist()
                            wef=projtoplane_bis(wef);
                            s1=np.real(wef[1]+Umax)/(2*Umax); s2=np.real(wef[2]+Vmax)/(2*Vmax);
                            s1=abs(1-abs(1-s1)); s2=abs(1-abs(1-s2));
                            ii=int(max(1,min(Npix,np.ceil(s1*Npix)))); jj=int(max(1,min(Npiy,np.ceil(s2*Npiy))));
                            xred[i,j,0]=IMG[ii-1,jj-1,0]; xred[i,j,1]=IMG[ii-1,jj-1,1]; xred[i,j,2]=IMG[ii-1,jj-1,2];
                        
                    


            elif Accretion_data[0]==1:
                dop_max=np.zeros((Npix,Npiy));
                for y in XX:
                    i=np.where(XX==y)[0][0];
                    for z in YY:
                        j=np.where(YY==z)[0][0];
                        X=init_conds_with_angle_bis(y,z);
                        r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
                        Dr=(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2;
                        E=np.sqrt(Dr*php**2*np.sin(th)**2+(Dr-a**2*np.sin(th)**2)/S*(rp**2*S/Dr+thp**2*S));
                        Lz=np.sin(th)**2/(a**2*np.sin(th)**2-Dr)*((a*E*(r**2+a**2)-php*S*Dr)-a*E*Dr);
                        pt=-E; pr=S*rp/Dr; pth=S*thp; pph=Lz;
                        Q=pth**2+np.cos(th)**2*(Lz**2/np.sin(th)**2-a**2*E**2);
                        k=Q+(a*E-Lz)**2;
                        X=[r,th,ph,pr,pth];                
                        
                        tau0=tau; dtau0=dtau; ite=0;
                        while ite<itermax:
                            ite+=1
                            try:
                                Vec=odeint(Carter_ter,X,np.arange(0,tau0,dtau0));
                            except Warning:
                                tau0*=cft;
                                dtau0*=cft;
                                continue
                            break
                        
                        R=Rs/2*Vec[:,0]; theta=Vec[:,1]; phi=Vec[:,2]; PR=Vec[:,3]; PTH=Vec[:,4];
                        wef=np.zeros((3,1)); WEF=[];
                        dWEF=abs(rf-np.sqrt(R**2+A**2*np.sin(theta)**2)); dwef=min(dWEF);
                        if (dwef<2.5e2 and (len(np.where([abs(phii)>=np.Inf*np.pi for phii in phi])[0])==0 and len(np.where([Vec[vv,0]<=1.01*ir for vv in range(N)])[0])==0)):#len(np.where([abs(phii)>=np.Inf*np.pi for phii in phi])[0])==0):#(dwef<2.5e2 and len(np.where([rg<ir*Rs for rg in R])[0])==0):
                            l=np.where(dWEF==dwef)[0][0];
                            wef=BoyerLindquist_bis(R[l],theta[l],phi[l]);
                            wef=np.array([[np.cos(alpha),0,np.sin(alpha)],[0,1,0],[-np.sin(alpha),0,np.cos(alpha)]]).dot(np.array(wef));
                            wef=wef.tolist()
                            wef=projtoplane_bis(wef);
                            s1=np.real(wef[1]+Umax)/(2*Umax); s2=np.real(wef[2]+Vmax)/(2*Vmax);
                            s1=abs(1-abs(1-s1)); s2=abs(1-abs(1-s2));
                            ii=int(max(1,min(Npix,np.ceil(s1*Npix)))); jj=int(max(1,min(Npiy,np.ceil(s2*Npiy))));
                            xred[i,j,0]=IMG[ii-1,jj-1,0]; xred[i,j,1]=IMG[ii-1,jj-1,1]; xred[i,j,2]=IMG[ii-1,jj-1,2];
                        
                        ll=np.where(abs(theta-np.pi/2)<1/100)[0]; whitness=0;
                        for l in ll:
                            rb=np.sqrt(Vec[l,0]**2+a**2);
                            if (rb*Rs/2>rint and rb*Rs/2<rext and whitness==0 and len(np.where([Vec[vv,0]<=1.05*ir for vv in range(l)])[0])==0):
                                whitness=1;
                                acc=accretion_disk([R[l],theta[l],phi[l],PR[l],PTH[l]]);
                                acc=acc[0].tolist()+acc[1:]
                                vef=BoyerLindquist_bis(R[l],theta[l],phi[l])+acc;
                                Temp.append(vef[-1]);
                                if Accretion_data[2]=="Custom":
                                    xred[i,j,0]=-np.exp(1); xred[i,j,1]=vef[6]; xred[i,j,2]=vef[8]; dop_max[i,j]=vef[7];
                                else:
                                    xred[i,j,0]=vef[6]*vef[3]; xred[i,j,1]=vef[6]*vef[4]; xred[i,j,2]=vef[6]*vef[5];
                                
                            
                        
                    
            
                if Accretion_data[2]=="Custom":
                    dp_max=max((dop_max.flatten()).tolist())
                    for i in range(Npix):
                        for j in range(Npiy):
                            if xred[i,j,0]==-np.exp(1):
                                flo=int(np.floor(xred[i,j,2]*(T_int+(T_ext-T_int)*dop_max[i,j]/dp_max)));
                                wef=blackbody[np.where(abs(flo-blackbody[:,0])==min(abs(flo-blackbody[:,0])))[0][0],range(1,4)];
                                xred[i,j,0]=xred[i,j,1]*wef[0]; xred[i,j,2]=xred[i,j,1]*wef[2]; xred[i,j,1]=xred[i,j,1]*wef[1];
                            
                        
                    
                    
            elif Accretion_data[0]>1:
                dop_max=np.zeros((Npix,Npiy));
                for y in XX:
                    i=np.where(XX==y)[0][0];
                    for z in YY:
                        j=np.where(YY==z)[0][0];
                        X=init_conds_with_angle_bis(y,z);
                        r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
                        Dr=(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2;
                        E=np.sqrt(Dr*php**2*np.sin(th)**2+(Dr-a**2*np.sin(th)**2)/S*(rp**2*S/Dr+thp**2*S));
                        Lz=np.sin(th)**2/(a**2*np.sin(th)**2-Dr)*((a*E*(r**2+a**2)-php*S*Dr)-a*E*Dr);
                        pt=-E; pr=S*rp/Dr; pth=S*thp; pph=Lz;
                        Q=pth**2+np.cos(th)**2*(Lz**2/np.sin(th)**2-a**2*E**2);
                        k=Q+(a*E-Lz)**2;
                        X=[r,th,ph,pr,pth];
                        
                        tau0=tau; dtau0=dtau; ite=0;
                        while ite<itermax:
                            ite+=1
                            try:
                                Vec=odeint(Carter_ter,X,np.arange(0,tau0,dtau0));
                            except Warning:
                                tau0*=cft;
                                dtau0*=cft;
                                continue
                            break
                        
                        R=Rs/2*Vec[:,0]; theta=Vec[:,1]; phi=Vec[:,2]; PR=Vec[:,3]; PTH=Vec[:,4];
                        xred[i,j,0]=0; xred[i,j,1]=0; xred[i,j,2]=0;
                        ll=np.where(abs(theta-np.pi/2)<1/100)[0]; whitness=0;
                        for l in ll:
                            rb=np.sqrt(Vec[l,0]**2+a**2);
                            if (rb*Rs/2>rint and rb*Rs/2<rext and whitness==0 and len(np.where([Vec[vv,0]<=1.05*ir for vv in range(l)])[0])==0):
                                whitness=1;
                                acc=accretion_disk([R[l],theta[l],phi[l],PR[l],PTH[l]]);
                                acc=acc[0].tolist()+acc[1:]
                                vef=BoyerLindquist_bis(R[l],theta[l],phi[l])+acc;
                                Temp.append(vef[-1]);
                                if Accretion_data[2]=="Custom":
                                    xred[i,j,0]=-np.exp(1); xred[i,j,1]=vef[6]; xred[i,j,2]=vef[8]; dop_max[i,j]=vef[7];
                                else:
                                    xred[i,j,0]=vef[6]*vef[3]; xred[i,j,1]=vef[6]*vef[4]; xred[i,j,2]=vef[6]*vef[5];
                                
                                
                

                if Accretion_data[2]=="Custom":
                    dp_max=max((dop_max.flatten()).tolist())
                    for i in range(Npix):
                        for j in range(Npiy):
                            if xred[i,j,0]==-np.exp(1):
                                flo=int(np.floor(xred[i,j,2]*(T_int+(T_ext-T_int)*dop_max[i,j]/dp_max)));
                                wef=blackbody[np.where(abs(flo-blackbody[:,0])==min(abs(flo-blackbody[:,0])))[0][0],range(1,4)];
                                xred[i,j,0]=xred[i,j,1]*wef[0]; xred[i,j,2]=xred[i,j,1]*wef[2]; xred[i,j,1]=xred[i,j,1]*wef[1];
                            


    
        warnings.filterwarnings("ignore")
        
        
        
        
        
        
    else:
        ##Use Weierstrass' analytic method if a=0:
        ##In the case of a non-rotating black hole (a=0), the spherical symmetry of the space-time and the polar formulation of
        ##the geodesic equation has an analytic solution: the Weierstrass elliptic function \wp.
        ##This allows fast and accurate numerical methods (such as Newton, with suitable initialization) to find if the rays hits
        ##the celestial sphere (or the accretion disk).
        if Lambda!=0:
            ##Initialize contants and some acretion data
            c=1; G=1; M=1; GSI=6.67408e-11; cSI=299792458; e0=8.854187e-12; sb=5.67e-8;
            Rs=2*GSI*Mass/cSI**2; A=Kerr*Rs/2; alpha=-Accretion_data[1]; xi=Accretion_data[1]; txi=np.tan(xi);
            x0=50000; sizee=Accretion_data[4]; rint=sizee[0]*Rs; rext=sizee[1]*Rs; rf=60000; rint_n=2*sizee[0]; rext_n=2*sizee[1];
            rq=Newman**2;##Q=Newman*2*Mass*sqrt(%pi*e0*GSI); rq2=Q^2*GSI/(4*%pi*e0*cSI^4); rq=4*rq2/Rs^2;
            rs=2; rg=1; a=0; T_int=1001; T_ext=1001; lam=0.8;
            Mrate=Accretion_data[5]; Mrate=Mrate[0]*Rs*cSI**2/(Mass*2*GSI); T0=3*cSI**2*Rs*Mrate/(2*np.pi*sb);
            Temp=[];
            if (len(Accretion_data[5])>1 and Accretion_data[5][2]<Accretion_data[5][1]):
                T_int=Accretion_data[5]; T_ext=T_int[2]; T_int=T_int[1];
            
            ##The Carlson algorithm for computing elliptic integrals
            def carlson(x,y,z):
                rtol=1e-10; xn=x; yn=y; zn=z; A0=(xn+yn+zn)/3; m=0;
                Q=np.power(3*rtol,-1/6)*max(abs(A0-xn),abs(A0-yn),abs(A0-zn)); A=A0;
                while Q/(4**m)>abs(A):
                    sqx=cmath.sqrt(xn); sqy=cmath.sqrt(yn); sqz=cmath.sqrt(zn);
                    if np.real(sqx)<0:
                        sqx=-sqx;
                    if np.real(sqy)<0:
                        sqy=-sqy;
                    if np.real(sqz)<0:
                        sqz=-sqz;
                    lm=sqx*sqy+sqx*sqz+sqy*sqz; A=(A+lm)/4;
                    xn=(xn+lm)/4; yn=(yn+lm)/4; zn=(zn+lm)/4; m=m+1;
                X=(A0-x)/(4**m*A); Y=(A0-y)/(4**m*A); Z=-X-Y;
                E2=X*Y-Z**2; E3=X*Y*Z;
                app=(1-E2/10+E3/14+E2**2/24-3*E2*E3/44)/cmath.sqrt(A);
                return(app)
            
            ##From and to spherical coordinates (with velocities), as well as the rotation function
            def From_spherical(R,Rp,T,Tp,P,Pp):
                x=R*np.sin(T)*np.cos(P);
                y=R*np.sin(T)*np.sin(P);
                z=R*np.cos(T);
                xp=(Tp*np.cos(T)*np.cos(P)*R-np.sin(T)*(Pp*np.sin(P)*R-Rp*np.cos(P)));
                yp=(Tp*np.cos(T)*np.sin(P)*R+np.sin(T)*(Pp*np.cos(P)*R+Rp*np.sin(P)));
                zp=Rp*np.cos(T)-R*Tp*np.sin(T);
                BB=[x,xp,y,yp,z,zp];
                return(BB)

            def To_spherical(x,xp,y,yp,z,zp):
                P=np.arctan2(y,x);
                R=np.sqrt(x**2+y**2+z**2);
                T=np.arccos(z/R);
                Rp=(x*xp+y*yp+z*zp)/R;
                Tp=(z*Rp-zp*R)/(R*np.sqrt(R**2-z**2));
                Pp=(yp*x-xp*y)/(x**2+y**2);
                XX=[R,Rp,T,Tp,P,Pp];
                return(XX)

            def rot(axe,theta,u):
                KK=np.array([[0,-axe[2],axe[1]],[axe[2],0,-axe[0]],[-axe[1],axe[0],0]]); KK=KK/lin.norm(axe,2);
                RR=np.identity(3)+np.sin(theta)*KK+(1-np.cos(theta))*(KK.dot(KK));
                v=RR.dot(u);
                return(v)
            
            ##Projection on the tangent plane (equi-rectangular projection)
            def projtoplane_bis(w):
                if w[2]>rf:
                    wrf=1
                elif w[2]<-rf:
                    wrf=-1
                else:
                    wrf=w[2]/rf
                wp=[-1,np.arctan2(w[1],-w[0]),np.pi/2-np.arccos(wrf)];
                return(wp)
            
            Xmax=22983
            
            ##Accretion_data[0]<2 means that we want the image with accretion disk  otherwise, we just want the accretion disk  no image is required.
            if Accretion_data[0]<2:
                Img=cv.imread(Image);
                Img=cv.cvtColor(Img,cv.COLOR_BGR2RGB)
                Npix=np.shape(Img)[0]; Npiy=np.shape(Img)[1]; IMG=np.zeros((Npiy,Npix,3));
                for i in range(3):
                    IMG[:,:,i]=np.transpose(Img[:,:,i])/256;
                Npix=np.shape(IMG)[0]; Npiy=np.shape(IMG)[1];
                
                XX=np.linspace(-Xmax,Xmax,Npix); YY=np.linspace(-Xmax*Npiy/Npix,Xmax*Npiy/Npix,Npiy);
                h=x0*Xmax*np.sqrt(1+Npiy**2/Npix**2)/(rf-Xmax*np.sqrt(1+Npiy**2/Npix**2));
            else:
                Npix=Accretion_data[0]; Npiy=Npix;
                XX=np.linspace(-Xmax,Xmax,Npix); YY=np.linspace(-Xmax,Xmax,Npiy);
                h=x0*Xmax*np.sqrt(2)/(rf-Xmax*np.sqrt(2));
            
            
            ##The initial datum from a pixel (y,z) on the screen (a ray leaving it with the appropriate velocity)
            def init_conds_bis(y,z,alph):
                v0=[x0,-cSI*h/np.sqrt(h**2+y**2+z**2),y,cSI*y/np.sqrt(h**2+y**2+z**2),z,cSI*z/np.sqrt(h**2+y**2+z**2)];
                matrot=np.array([[np.cos(alph),0,-np.sin(alph)],[0,1,0],[np.sin(alph),0,np.cos(alph)]]);
                vrot=matrot.dot(np.array([v0[0],v0[2],v0[4]])); vvrot=-matrot.dot(np.array([v0[1],v0[3],v0[5]]));
                Z=To_spherical(vrot[0],vvrot[0],vrot[1],vvrot[1],vrot[2],vvrot[2]);
                Z=[Z[0],Z[2],Z[4],Z[1],Z[3],Z[5]];
                Z=[2/Rs*Z[0],Z[1],Z[2],Z[3]/cSI,Z[4]*Rs/(2*cSI),Z[5]*Rs/(2*cSI)];
                return Z
            
            ##Cartesian velocity from BL velocity
            def velocity(sphvel):
                r=sphvel[0]; th=sphvel[1]; ph=sphvel[2]; rp=sphvel[3]; thp=sphvel[4]; php=sphvel[5];
                vx=np.cos(ph)*np.sin(th)*r*rp/np.sqrt(r**2 + a**2) - np.sqrt(r**2 + a**2)*php*np.sin(ph)*np.sin(th) + np.sqrt(r**2 + a**2)*np.cos(ph)*thp*np.cos(th);
                vy=np.sin(ph)*np.sin(th)*r*rp/np.sqrt(r**2 + a**2) + np.sqrt(r**2 + a**2)*php*np.cos(ph)*np.sin(th) + np.sqrt(r**2 + a**2)*np.sin(ph)*thp*np.cos(th);
                vz=rp*np.cos(th) - r*thp*np.sin(th);
                vel=[vx,vy,vz];
                return vel
            
            ##Function for the color shift
            def doppler_color(dope):
                dil_dope=(dope-1/2)**1;
                if dil_dope<1/4:
                    rcol=[0,4*dil_dope,1];
                elif (1/4<=dil_dope and dil_dope<1/2):
                    rcol=[0,1,2-4*dil_dope];
                elif (1/2<=dil_dope and dil_dope<3/4):
                    rcol=[4*dil_dope-2,1,0];
                else:
                    rcol=[1,4-4*dil_dope,0];
                if (abs(rcol[0])<0.05 and abs(rcol[2])<0.05 and abs(rcol[1]-1)<0.05):
                    rcol=[2,2,2];
                return rcol
            
            ##Depending on what kind of accretion is required, we create a function 'accretion_disk'
            ##which computes the color to be attributed to a pixel, taking into account the various effects.
            ##This function is defined at this point so that we don't need to put a selection process for each pixel:
            ##the color function is chosen once and for all.
            if (Accretion_data[2]=="Blackbody" and Accretion_data[3]=="Doppler"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r; Dr=r**2*(1-Lambda*r**2/3)-2*r+rq; S=r**2;
                    veloc=cosmo_inv_met_mat([0,r,th,ph],Lambda,0,rq).dot(np.array([pt,V[3],V[4],pph])); al=(r/np.sqrt(-Lambda*r**4/3+r-rq));
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc);
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4); T=T/doppler_shift;
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    colou=blackbody[np.where(abs(T-blackbody[:,0])==min(abs(T-blackbody[:,0])))[0][0],range(1,4)];
                    cb=[colou,bright/doppler_shift,T];
                    return cb
            elif (Accretion_data[2]=="Blackbody" and Accretion_data[3]=="Gravitation"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r; Dr=r**2*(1-Lambda*r**2/3)-2*r+rq; S=r**2;
                    grav_shift=1/np.sqrt(abs(Dr/S));
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4); T=T/grav_shift;
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    colou=blackbody[np.where(abs(T-blackbody[:,0])==min(abs(T-blackbody[:,0])))[0][0],range(1,4)];
                    cb=[colou,bright/(grav_shift),T];
                    return cb
            elif (Accretion_data[2]=="Blackbody" and Accretion_data[3]=="Doppler+"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r; Dr=r**2*(1-Lambda*r**2/3)-2*r+rq; S=r**2;
                    veloc=cosmo_inv_met_mat([0,r,th,ph],Lambda,0,rq).dot(np.array([pt,V[3],V[4],pph])); al=(r/np.sqrt(-Lambda*r**4/3+r-rq));
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc);
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    grav_shift=1/np.sqrt(abs(Dr/S));
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4); T=T/(grav_shift*doppler_shift);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    colou=blackbody[np.where(abs(T-blackbody[:,0])==min(abs(T-blackbody[:,0])))[0][0],range(1,4)];
                    cb=[colou,bright/(grav_shift*doppler_shift),T];
                    return cb
            elif (Accretion_data[2]=="Blackbody" and Accretion_data[3]==" "):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r;
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    colou=blackbody[np.where(abs(T-blackbody[:,0])==min(abs(T-blackbody[:,0])))[0][0],range(1,4)];
                    cb=[colou,bright,T];
                    return cb
            elif (Accretion_data[2]=="Custom" and Accretion_data[3]=="Doppler"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r; Dr=r**2*(1-Lambda*r**2/3)-2*r+rq; S=r**2;
                    veloc=cosmo_inv_met_mat([0,r,th,ph],Lambda,0,rq).dot(np.array([pt,V[3],V[4],pph])); al=(r/np.sqrt(-Lambda*r**4/3+r-rq));
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc);
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    doppler_coeff=1-np.sqrt(2/rb);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    cb=[np.array([0,0,0]),bright/doppler_shift,doppler_coeff**2,1/doppler_shift,T];
                    return cb
            elif (Accretion_data[2]=="Custom" and Accretion_data[3]=="Gravitation"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r; Dr=r**2*(1-Lambda*r**2/3)-2*r+rq; S=r**2;
                    grav_shift=1/np.sqrt(abs(Dr/S));
                    doppler_coeff=1-np.sqrt(2/rb);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    cb=[np.array([0,0,0]),bright/grav_shift,doppler_coeff**2,1/grav_shift,T];
                    return cb
            elif (Accretion_data[2]=="Custom" and Accretion_data[3]=="Doppler+"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r; Dr=r**2*(1-Lambda*r**2/3)-2*r+rq; S=r**2;
                    veloc=cosmo_inv_met_mat([0,r,th,ph],Lambda,0,rq).dot(np.array([pt,V[3],V[4],pph])); al=(r/np.sqrt(-Lambda*r**4/3+r-rq));
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc)
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    grav_shift=1/np.sqrt(abs(Dr/S));
                    doppler_coeff=1-np.sqrt(2/rb);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    cb=[np.array([0,0,0]),bright/(grav_shift*doppler_shift)**0,doppler_coeff**2,1/(grav_shift*doppler_shift),T];
                    return cb
            elif (Accretion_data[2]=="Custom" and Accretion_data[3]==" "):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r;
                    doppler_coeff=1-np.sqrt(2/rb);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    cb=[np.array([0,0,0]),bright,doppler_coeff**2,1,T];
                    return cb
            elif (Accretion_data[2]==" " and Accretion_data[3]=="Doppler"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r; Dr=r**2*(1-Lambda*r**2/3)-2*r+rq; S=r**2;
                    veloc=cosmo_inv_met_mat([0,r,th,ph],Lambda,0,rq).dot(np.array([pt,V[3],V[4],pph])); al=(r/np.sqrt(-Lambda*r**4/3+r-rq));
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc)
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    colou=doppler_color(doppler_shift);
                    bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    cb=[np.array(colou),bright];
                    return cb
            elif (Accretion_data[2]==" " and Accretion_data[3]=="Gravitation"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r; Dr=r**2*(1-Lambda*r**2/3)-2*r+rq; S=r**2;
                    grav_shift=1/np.sqrt(abs(Dr/S));
                    colou=doppler_color(grav_shift);
                    bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    cb=[np.array(colou),bright];
                    return cb
            elif (Accretion_data[2]==" " and Accretion_data[3]=="Doppler+"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r; Dr=r**2*(1-Lambda*r**2/3)-2*r+rq; S=r**2;
                    veloc=cosmo_inv_met_mat([0,r,th,ph],Lambda,0,rq).dot(np.array([pt,V[3],V[4],pph])); al=(r/np.sqrt(-Lambda*r**4/3+r-rq));
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc);
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    grav_shift=1/np.sqrt(abs(Dr/S));
                    colou=doppler_color(doppler_shift*grav_shift);
                    bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    cb=[np.array(colou),bright];
                    return cb
            else:
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); lam=0.2;
                    cb=np.ones((1,4)); cb=np.array([255,69,0])/256; cb[2]=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    cb=[cb,1]
                    return cb
            
            def weierP(g2,g3,z):
                N0=12;
                zz0=z/(2**N0); zz=1/zz0**2+g2/20*zz0**2+g3/28*zz0**4;
                for j in range(N0):
                    zz=-2*zz+(6*zz**2-g2/2)**2/(4*(4*zz**3-g2*zz-g3));
                return(zz)
            
            def newton(g2,g3,Z,t):
                def toanihil(s):
                    return((2*rf/Rs-rbar)*(4*np.real(weierP(g2,g3,Z+s))-beta/3)-alp);
                sgn=np.sign(toanihil(t)); sol=t; step=-0.02;
                while sgn*toanihil(sol)>0:
                    sol=sol+step;
                sol=nt(toanihil,sol);
                return(sol)
            
            
            if Accretion_data[0]==0:
                ##For each pixel in the upper right corner (i.e. y>0, z>0, this is enough by spherical symmetry), we compute its 'theta=pi/2' version (with rotations)
                ##and we find the corresponding pixel on the sphere, which we project on the plane and record its coordinates.
                Xred=np.zeros((0,2));
                AR=[];
                for xx in XX[int(np.floor(Npix/2)):Npix]:
                    for yy in YY[int(np.floor(Npiy/2)):Npiy]:
                        AR.append(np.sqrt(xx**2+yy**2));
                
                AR=np.sort(AR)
                
                for zz in [zu for zu in AR if zu!=0]:
                    X=init_conds_bis(zz,0,0);
                    r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
                    E=np.sqrt(rp**2+php**2*(r**2*(1-Lambda*r**2/3)-2*r+rq)); L=r**2*php; pt=-E; pph=L;
                    ##Find the constants rbar,alpha,beta,gamma,delta,g2,g3 associated to the ray:
                    rpol=np.roots([(E**2)/L**2+Lambda/3,0,-1,2,-rq]);
                    mi=min(abs(rpol-np.real(rpol))); frpol=[np.real(rpol[rr]) for rr in np.where(abs(rpol-np.real(rpol))==mi)][0];
                    rbar=frpol[np.where(abs(frpol)==min(abs(frpol)))[0][0]];
                    delta=(E**2)/L**2+Lambda/3; gamma=2*(2*delta*rbar); beta=-1+3*rbar*(gamma-2*delta*rbar); alp=2+rbar*(2*beta-rbar*(3*gamma-4*delta*rbar));
                    g2=(beta**2/3-alp*gamma)/4; g3=(alp*beta*gamma/6-alp**2*delta/2-beta**3/27)/8;
                    rp2=np.roots([4,0,-g2,-g3]); z0=alp/(4*(r-rbar))+beta/12;
                    if abs(rp)<1e-12:
                        Z0=carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2]);
                    else:
                        Z0=np.sign(-rp)*carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2]);
                    new=newton(g2,g3,Z0,0); P=ph+np.sign(php)*new;
                    if P>np.pi/2:
                        Xred=np.vstack([Xred,np.array([zz,P])]);
                
                ##For each pixel in the upper-right corner, we see if it s on the celestial sphere (using the previously computed data)
                ##then, we put the position of this pixel in the list into a matrix K, which encodes the correspondence
                ##between a pixel on the screen and its final state on the sphere. Besides, we extract the maximal range of coordinates.
                Umax=0; Vmax=0; KK=np.zeros((Npix,Npiy));
                for i in range(int(np.floor(Npix/2)),Npix):
                    x=XX[i];
                    for j in range(int(np.floor(Npiy/2)),Npiy):
                        y=YY[j]; r=np.sqrt(x**2+y**2);
                        mi=min(abs(r-Xred[:,0]));
                        if mi<1e-10:
                            k=np.where(abs(r-Xred[:,0])==mi)[0][0];
                            KK[i,j]=k; KK[i,Npiy-j-1]=k;
                            KK[Npix-i-1,j]=k; KK[Npix-i-1,Npiy-j-1]=k;
                            
                Umax=np.pi/2; Vmax=Umax*Npiy/Npix;
                xred=np.zeros((Npix,Npiy,3));
                ##Attribute RGB value to the pixels
                for i in range(Npix):
                    xx=XX[i];
                    for j in range(Npiy):
                        yy=YY[j];
                        if KK[i,j]!=0:
                            P=Xred[int(KK[i,j]),1];
                            Z=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([np.cos(P),np.sin(P),0]));
                            Z=projtoplane_bis(rf*Z);
                            t1=(Z[1]+Umax)/(2*Umax); t2=(Z[2]+Vmax)/(2*Vmax);
                            s1=abs(1-abs(1-t1)); s2=abs(1-abs(1-t2));
                            ii=int(max(1,min(Npix,np.ceil(s1*Npix)))); jj=int(max(1,min(Npiy,np.ceil(s2*Npiy))));
                            xred[i,j,0]=IMG[ii-1,jj-1,0]; xred[i,j,1]=IMG[ii-1,jj-1,1]; xred[i,j,2]=IMG[ii-1,jj-1,2];0;
                            
                
                
                
            ##If accretion and picture are required:
            elif Accretion_data[0]==1:
                Xred=np.zeros((0,9));
                AR=[]; Yred=np.zeros((0,9))
                for xx in XX[int(np.floor(Npix/2)):Npix]:
                    for yy in YY[int(np.floor(Npiy/2)):Npiy]:
                        AR.append(np.sqrt(xx**2+yy**2));
                
                AR=np.sort(AR)
                
                for zz in [zu for zu in AR if zu!=0]:
                    X=init_conds_bis(zz,0,0);
                    r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
                    E=np.sqrt(rp**2+php**2*(r**2*(1-Lambda*r**2/3)-2*r+rq)); L=r**2*php; pt=-E; pph=L;
                    ##Find the constants rbar,alpha,beta,gamma,delta,g2,g3 associated to the ray:
                    rpol=np.roots([(E**2)/L**2+Lambda/3,0,-1,2,-rq]);
                    mi=min(abs(rpol-np.real(rpol))); frpol=[np.real(rpol[rr]) for rr in np.where(abs(rpol-np.real(rpol))==mi)][0];
                    rbar=frpol[np.where(abs(frpol)==min(abs(frpol)))[0][0]];
                    delta=(E**2)/L**2+Lambda/3; gamma=2*(2*delta*rbar); beta=-1+3*rbar*(gamma-2*delta*rbar); alp=2+rbar*(2*beta-rbar*(3*gamma-4*delta*rbar));
                    g2=(beta**2/3-alp*gamma)/4; g3=(alp*beta*gamma/6-alp**2*delta/2-beta**3/27)/8;
                    rp2=np.roots([4,0,-g2,-g3]); z0=alp/(4*(r-rbar))+beta/12;
                    if abs(rp)<1e-12:
                        Z0=carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2]);
                    else:
                        Z0=np.sign(-rp)*carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2]);
                    new=newton(g2,g3,Z0,0); P=ph+np.sign(php)*new;
                    if P>np.pi/2:
                        Xred=np.vstack([Xred,np.array([zz,P,g2,g3,Z0,ph,alp,beta,rbar])]);
                    else:
                        Yred=np.vstack([Yred,np.array([zz,0,g2,g3,Z0,ph,alp,beta,rbar])]);
                        
                    

                Umax=0; Vmax=0; KK=np.zeros((Npix,Npiy)); LL=np.zeros((Npix,Npiy))
                for i in range(int(np.floor(Npix/2)),Npix):
                    x=XX[i];
                    for j in range(int(np.floor(Npiy/2)),Npiy):
                        y=YY[j]; r=np.sqrt(x**2+y**2);
                        mi=min(abs(r-Xred[:,0]));
                        if mi<1e-10:
                            k=np.where(abs(r-Xred[:,0])==mi)[0][0];
                            KK[i,j]=k+1; KK[i,Npiy-j-1]=k+1;
                            KK[Npix-i-1,j]=k+1; KK[Npix-i-1,Npiy-j-1]=k+1;
                        else:
                            k=np.where(abs(r-Yred[:,0])==min(abs(r-Yred[:,0])))[0][0]
                            LL[i,j]=k; LL[i,Npiy-j-1]=k;
                            LL[Npix-i-1,j]=k; LL[Npix-i-1,Npiy-j-1]=k;
                            
                        
                    
                Umax=np.pi/2; Vmax=Umax*Npiy/Npix; xred=np.zeros((Npix,Npiy,3)); dop_max=np.zeros((Npix,Npiy));

                ##Now, if a pixel hits the accretion disk, we need to find the corresponding gravitational and Doppler effects.
                ##To do this, we need to recover the (conjugate) momenta of the photon when it hits the disk.
                ##This requires an additional loop on pixels, but the local instructions are rather easy and fast to execute and
                ##this doesn't affect the execution time too much, as the whole procedure is fast enough.
                ##We found convenient here to distinguish between the "upper" and "lower" parts of the disk.
                ##Depending on the np.sign of the inclination angle, one part should be computed before the other one.
                ##This is why we introduce a conditional statement on the np.sign of this angle.
                for i in range(Npix):
                    xx=XX[i];
                    for j in range(Npiy):
                        yy=YY[j]; xred[i,j,0]=0; xred[i,j,1]=0; xred[i,j,2]=0;
                        ata=np.arctan2(txi,-np.sign(xi)*yy/np.sqrt(yy**2+xx**2));
                        X=init_conds_bis(xx,yy,xi); r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
                        E=np.sqrt(rp**2+(php**2*np.sin(th)**2+thp**2)*(r**2*(1-Lambda*r**2/3)-2*r+rq)); L=r**2*php*np.sin(th)**2; pt=-E; pph=L;
                        C=r**4*thp**2+L**2/np.tan(th)**2; Pth=np.sqrt(C);
                        if KK[i,j]==0:
                            xe=Yred[int(LL[i,j]),:];
                            Phi=min(np.pi+ata,np.pi-ata);
                            rtest=np.real(xe[6]/(4*np.real(weierP(xe[2],xe[3],xe[4]+(Phi-xe[5])))-xe[7]/3)+xe[8]);
                            if (rint_n<rtest and rtest<rext_n and Phi<np.pi/2):
                                rtesti=rtest; DDr=rtesti**2*(1-Lambda*rtesti**2/3)-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                Cf=From_spherical(Rs*rtest/2,0,np.pi/2,0,Phi,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                cobra=accretion_disk([Rs*rtest/2,np.pi/2,Cf[4],Pr,Pth]);
                                cobra=cobra[0].tolist()+cobra[1:];
                                cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                if Accretion_data[2]=="Custom":
                                    xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                else:
                                    xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                                
                            
                        else:
                            xe=Xred[int(KK[i,j]-1),:]; 
                            Phim=np.pi+np.arctan2(txi,yy/np.sqrt(yy**2+xx**2)); Phip=np.pi-np.arctan2(txi,-yy/np.sqrt(yy**2+xx**2));
                            rtestp=np.real(xe[6]/(4*np.real(weierP(xe[2],xe[3],xe[4]-(Phip-xe[5])))-xe[7]/3)+xe[8]);
                            rtestm=np.real(xe[6]/(4*np.real(weierP(xe[2],xe[3],xe[4]-(Phim-xe[5])))-xe[7]/3)+xe[8]);
                            whit=0;
                            if alpha>=0:
                                if (rint_n<rtestp and rtestp<rext_n):
                                    rtesti=rtestp; DDr=rtesti**2*(1-Lambda*rtesti**2/3)-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                    Cf=From_spherical(Rs*rtestp/2,0,np.pi/2,0,Phip,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                    Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                    cobra=accretion_disk([Rs*rtestp/2,np.pi/2,Cf[4],Pr,Pth]);
                                    cobra=cobra[0].tolist()+cobra[1:];
                                    cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                    if Accretion_data[2]=="Custom":
                                        xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                    else:
                                        xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                                    
                                    whit=1;
                                
                                if (rint_n<rtestm and rtestm<rext_n):
                                    rtesti=rtestm; DDr=rtesti**2*(1-Lambda*rtesti**2/3)-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                    Cf=From_spherical(Rs*rtestm/2,0,np.pi/2,0,Phim,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                    Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                    cobra=accretion_disk([Rs*rtestm/2,np.pi/2,Cf[4],Pr,Pth]);
                                    cobra=cobra[0].tolist()+cobra[1:];
                                    cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                    if Accretion_data[2]=="Custom":
                                        xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                    else:
                                        xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                                    
                                    whit=1;
                                
                            else:
                                if (rint_n<rtestm and rtestm<rext_n):
                                    rtesti=rtestm; DDr=rtesti**2*(1-Lambda*rtesti**2/3)-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                    Cf=From_spherical(Rs*rtestm/2,0,np.pi/2,0,Phim,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                    Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                    cobra=accretion_disk([Rs*rtestm/2,np.pi/2,Cf[4],Pr,Pth]);
                                    cobra=cobra[0].tolist()+cobra[1:];
                                    cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                    if Accretion_data[2]=="Custom":
                                        xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                    else:
                                        xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                                    
                                    whit=1;
                                
                                if (rint_n<rtestp and rtestp<rext_n):
                                    rtesti=rtestp; DDr=rtesti**2*(1-Lambda*rtesti**2/3)-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                    Cf=From_spherical(Rs*rtestp/2,0,np.pi/2,0,Phip,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                    Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                    cobra=accretion_disk([Rs*rtestp/2,np.pi/2,Cf[4],Pr,Pth]);
                                    cobra=cobra[0].tolist()+cobra[1:];
                                    cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                    if Accretion_data[2]=="Custom":
                                        xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                    else:
                                        xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                                    
                                    whit=1;
                                
                            
                            if whit==0:
                                P=xe[1];
                                Z=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([np.cos(P),np.sin(P),0]));
                                Z=projtoplane_bis(rf*np.real(Z));
                                s1=(Z[1]+Umax)/(2*Umax); s2=(Z[2]+Vmax)/(2*Vmax);
                                s1=abs(1-abs(1-s1)); s2=abs(1-abs(1-s2));
                                ii=int(max(1,min(Npix,np.ceil(s1*Npix)))); jj=int(max(1,min(Npiy,np.ceil(s2*Npiy))));
                                xred[i,j,0]=IMG[ii-1,jj-1,0]; xred[i,j,1]=IMG[ii-1,jj-1,1]; xred[i,j,2]=IMG[ii-1,jj-1,2];
                            
                        
                    
                
                if Accretion_data[2]=="Custom":
                    dp_max=max((dop_max.flatten()).tolist())
                    for i in range(Npix):
                        for j in range(Npiy):
                            if xred[i,j,0]==-np.exp(1):
                                flo=np.floor(xred[i,j,2]*(T_int+(T_ext-T_int)*dop_max[i,j]/dp_max));
                                wef=blackbody[np.where(abs(flo-blackbody[:,0])==min(abs(flo-blackbody[:,0])))[0][0],range(1,4)];
                                xred[i,j,0]=xred[i,j,1]*wef[0]; xred[i,j,2]=xred[i,j,1]*wef[2]; xred[i,j,1]=xred[i,j,1]*wef[1];
                                
                    
                
                
                
                
                
            ##If only the accretion disk is required:
            elif Accretion_data[0]>1:
                Xred=np.zeros((0,9));
                AR=[]; Yred=np.zeros((0,9))
                for xx in XX[int(np.floor(Npix/2)):Npix]:
                    for yy in YY[int(np.floor(Npiy/2)):Npiy]:
                        AR.append(np.sqrt(xx**2+yy**2));
                
                AR=np.sort(AR)
                
                for zz in [zu for zu in AR if zu!=0]:
                    X=init_conds_bis(zz,0,0);
                    r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
                    E=np.sqrt(rp**2+php**2*(r**2*(1-Lambda*r**2/3)-2*r+rq)); L=r**2*php; pt=-E; pph=L;
                    ##Find the constants rbar,alpha,beta,gamma,delta,g2,g3 associated to the ray:
                    rpol=np.roots([(E**2)/L**2+Lambda/3,0,-1,2,-rq]);
                    mi=min(abs(rpol-np.real(rpol))); frpol=[np.real(rpol[rr]) for rr in np.where(abs(rpol-np.real(rpol))==mi)][0];
                    rbar=frpol[np.where(abs(frpol)==min(abs(frpol)))[0][0]];
                    delta=(E**2)/L**2+Lambda/3; gamma=2*(2*delta*rbar); beta=-1+3*rbar*(gamma-2*delta*rbar); alp=2+rbar*(2*beta-rbar*(3*gamma-4*delta*rbar));
                    g2=(beta**2/3-alp*gamma)/4; g3=(alp*beta*gamma/6-alp**2*delta/2-beta**3/27)/8;
                    rp2=np.roots([4,0,-g2,-g3]); z0=alp/(4*(r-rbar))+beta/12;
                    if abs(rp)<1e-12:
                        Z0=carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2]);
                    else:
                        Z0=np.sign(-rp)*carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2]);
                    new=newton(g2,g3,Z0,0); P=ph+np.sign(php)*new;
                    if P>np.pi/2:
                        Xred=np.vstack([Xred,np.array([zz,2,g2,g3,Z0,ph,alp,beta,rbar])]);
                    else:
                        Yred=np.vstack([Yred,np.array([zz,0,g2,g3,Z0,ph,alp,beta,rbar])]);
                        
                    

                Umax=0; Vmax=0; KK=np.zeros((Npix,Npiy)); LL=np.zeros((Npix,Npiy))
                for i in range(int(np.floor(Npix/2)),Npix):
                    x=XX[i];
                    for j in range(int(np.floor(Npiy/2)),Npiy):
                        y=YY[j]; r=np.sqrt(x**2+y**2);
                        mi=min(abs(r-Xred[:,0]));
                        if mi<1e-10:
                            k=np.where(abs(r-Xred[:,0])==mi)[0][0];
                            KK[i,j]=k+1; KK[i,Npiy-j-1]=k+1;
                            KK[Npix-i-1,j]=k+1; KK[Npix-i-1,Npiy-j-1]=k+1;
                        else:
                            k=np.where(abs(r-Yred[:,0])==min(abs(r-Yred[:,0])))[0][0]
                            LL[i,j]=k; LL[i,Npiy-j-1]=k;
                            LL[Npix-i-1,j]=k; LL[Npix-i-1,Npiy-j-1]=k;
                            
                        
                    
                Umax=np.pi/2; Vmax=Umax*Npiy/Npix; xred=np.zeros((Npix,Npiy,3)); dop_max=np.zeros((Npix,Npiy));

                for i in range(Npix):
                    xx=XX[i];
                    for j in range(Npiy):
                        yy=YY[j];
                        ata=np.arctan2(txi,-np.sign(xi)*yy/np.sqrt(yy**2+xx**2));
                        X=init_conds_bis(xx,yy,xi); r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
                        E=np.sqrt(rp**2+(php**2*np.sin(th)**2+thp**2)*(r**2*(1-Lambda*r**2/3)-2*r+rq)); L=r**2*php*np.sin(th)**2; pt=-E; pph=L;
                        C=r**4*thp**2+L**2/np.tan(th)**2; Pth=np.sqrt(C);
                        if KK[i,j]==0:
                            xe=Yred[int(LL[i,j]),:];
                            Phi=min(np.pi+ata,np.pi-ata);
                            rtest=np.real(xe[6]/(4*np.real(weierP(xe[2],xe[3],xe[4]+(Phi-xe[5])))-xe[7]/3)+xe[8]);
                            if (rint_n<rtest and rtest<rext_n and Phi<np.pi/2):
                                rtesti=rtest; DDr=rtesti**2*(1-Lambda*rtesti**2/3)-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                Cf=From_spherical(Rs*rtest/2,0,np.pi/2,0,Phi,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                cobra=accretion_disk([Rs*rtest/2,np.pi/2,Cf[4],Pr,Pth]);
                                cobra=cobra[0].tolist()+cobra[1:];
                                cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                if Accretion_data[2]=="Custom":
                                    xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                else:
                                    xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                                
                            
                        else:
                            xe=Xred[int(KK[i,j]-1),:]; 
                            Phim=np.pi+np.arctan2(txi,yy/np.sqrt(yy**2+xx**2)); Phip=np.pi-np.arctan2(txi,-yy/np.sqrt(yy**2+xx**2));
                            rtestp=np.real(xe[6]/(4*np.real(weierP(xe[2],xe[3],xe[4]-(Phip-xe[5])))-xe[7]/3)+xe[8]);
                            rtestm=np.real(xe[6]/(4*np.real(weierP(xe[2],xe[3],xe[4]-(Phim-xe[5])))-xe[7]/3)+xe[8]);
                            #whit=0;
                            if alpha>=0:
                                if (rint_n<rtestp and rtestp<rext_n):
                                    rtesti=rtestp; DDr=rtesti**2*(1-Lambda*rtesti**2/3)-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                    Cf=From_spherical(Rs*rtestp/2,0,np.pi/2,0,Phip,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                    Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                    cobra=accretion_disk([Rs*rtestp/2,np.pi/2,Cf[4],Pr,Pth]);
                                    cobra=cobra[0].tolist()+cobra[1:];
                                    cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                    if Accretion_data[2]=="Custom":
                                        xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                    else:
                                        xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                                    
                                
                                if (rint_n<rtestm and rtestm<rext_n):
                                    rtesti=rtestm; DDr=rtesti**2*(1-Lambda*rtesti**2/3)-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                    Cf=From_spherical(Rs*rtestm/2,0,np.pi/2,0,Phim,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                    Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                    cobra=accretion_disk([Rs*rtestm/2,np.pi/2,Cf[4],Pr,Pth]);
                                    cobra=cobra[0].tolist()+cobra[1:];
                                    cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                    if Accretion_data[2]=="Custom":
                                        xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                    else:
                                        xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                                
                                
                            else:
                                if (rint_n<rtestm and rtestm<rext_n):
                                    rtesti=rtestm; DDr=rtesti**2*(1-Lambda*rtesti**2/3)-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                    Cf=From_spherical(Rs*rtestm/2,0,np.pi/2,0,Phim,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                    Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                    cobra=accretion_disk([Rs*rtestm/2,np.pi/2,Cf[4],Pr,Pth]);
                                    cobra=cobra[0].tolist()+cobra[1:];
                                    cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                    if Accretion_data[2]=="Custom":
                                        xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                    else:
                                        xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                                    
                                
                                if (rint_n<rtestp and rtestp<rext_n):
                                    rtesti=rtestp; DDr=rtesti**2*(1-Lambda*rtesti**2/3)-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                    Cf=From_spherical(Rs*rtestp/2,0,np.pi/2,0,Phip,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                    Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                    cobra=accretion_disk([Rs*rtestp/2,np.pi/2,Cf[4],Pr,Pth]);
                                    cobra=cobra[0].tolist()+cobra[1:];
                                    cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                    if Accretion_data[2]=="Custom":
                                        xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                    else:
                                        xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                        
                    
                
                if Accretion_data[2]=="Custom":
                    dp_max=max((dop_max.flatten()).tolist())
                    for i in range(Npix):
                        for j in range(Npiy):
                            if xred[i,j,0]==-np.exp(1):
                                flo=np.floor(xred[i,j,2]*(T_int+(T_ext-T_int)*dop_max[i,j]/dp_max));
                                wef=blackbody[np.where(abs(flo-blackbody[:,0])==min(abs(flo-blackbody[:,0])))[0][0],range(1,4)];
                                xred[i,j,0]=xred[i,j,1]*wef[0]; xred[i,j,2]=xred[i,j,1]*wef[2]; xred[i,j,1]=xred[i,j,1]*wef[1];
                            
                        
                
                    

        else:
            c=1; G=1; M=1; GSI=6.67408e-11; cSI=299792458; e0=8.854187e-12; sb=5.67e-8;
            Rs=2*GSI*Mass/cSI**2; A=Kerr*Rs/2; alpha=-Accretion_data[1]; xi=Accretion_data[1]; txi=np.tan(xi);
            x0=50000; sizee=Accretion_data[4]; rint=sizee[0]*Rs; rext=sizee[1]*Rs; rf=60000; rint_n=2*sizee[0]; rext_n=2*sizee[1];
            rq=Newman**2;##Q=Newman*2*Mass*sqrt(%pi*e0*GSI); rq2=Q^2*GSI/(4*%pi*e0*cSI^4); rq=4*rq2/Rs^2;
            rs=2; rg=1; a=0; T_int=1001; T_ext=1001; lam=0.8;
            Mrate=Accretion_data[5]; Mrate=Mrate[0]*Rs*cSI**2/(Mass*2*GSI); T0=3*cSI**2*Rs*Mrate/(2*np.pi*sb);
            Temp=[];
            if (len(Accretion_data[5])>1 and Accretion_data[5][2]<Accretion_data[5][1]):
                T_int=Accretion_data[5]; T_ext=T_int[2]; T_int=T_int[1];
            
            
            def carlson(x,y,z):
                rtol=1e-10; xn=x; yn=y; zn=z; A0=(xn+yn+zn)/3; m=0;
                Q=np.power(3*rtol,-1/6)*max(abs(A0-xn),abs(A0-yn),abs(A0-zn)); A=A0;
                while Q/(4**m)>abs(A):
                    sqx=cmath.sqrt(xn); sqy=cmath.sqrt(yn); sqz=cmath.sqrt(zn);
                    if np.real(sqx)<0:
                        sqx=-sqx;
                    if np.real(sqy)<0:
                        sqy=-sqy;
                    if np.real(sqz)<0:
                        sqz=-sqz;
                    lm=sqx*sqy+sqx*sqz+sqy*sqz; A=(A+lm)/4;
                    xn=(xn+lm)/4; yn=(yn+lm)/4; zn=(zn+lm)/4; m=m+1;
                X=(A0-x)/(4**m*A); Y=(A0-y)/(4**m*A); Z=-X-Y;
                E2=X*Y-Z**2; E3=X*Y*Z;
                app=(1-E2/10+E3/14+E2**2/24-3*E2*E3/44)/cmath.sqrt(A);
                return(app)
            
            
            def From_spherical(R,Rp,T,Tp,P,Pp):
                x=R*np.sin(T)*np.cos(P);
                y=R*np.sin(T)*np.sin(P);
                z=R*np.cos(T);
                xp=(Tp*np.cos(T)*np.cos(P)*R-np.sin(T)*(Pp*np.sin(P)*R-Rp*np.cos(P)));
                yp=(Tp*np.cos(T)*np.sin(P)*R+np.sin(T)*(Pp*np.cos(P)*R+Rp*np.sin(P)));
                zp=Rp*np.cos(T)-R*Tp*np.sin(T);
                BB=[x,xp,y,yp,z,zp];
                return(BB)

            def To_spherical(x,xp,y,yp,z,zp):
                P=np.arctan2(y,x);
                R=np.sqrt(x**2+y**2+z**2);
                T=np.arccos(z/R);
                Rp=(x*xp+y*yp+z*zp)/R;
                Tp=(z*Rp-zp*R)/(R*np.sqrt(R**2-z**2));
                Pp=(yp*x-xp*y)/(x**2+y**2);
                XX=[R,Rp,T,Tp,P,Pp];
                return(XX)

            def rot(axe,theta,u):
                KK=np.array([[0,-axe[2],axe[1]],[axe[2],0,-axe[0]],[-axe[1],axe[0],0]]); KK=KK/lin.norm(axe,2);
                RR=np.identity(3)+np.sin(theta)*KK+(1-np.cos(theta))*(KK.dot(KK));
                v=RR.dot(u);
                return(v)
            
            
            def projtoplane_bis(w):
                if w[2]>rf:
                    wrf=1
                elif w[2]<-rf:
                    wrf=-1
                else:
                    wrf=w[2]/rf
                wp=[-1,np.arctan2(w[1],-w[0]),np.pi/2-np.arccos(wrf)];
                return(wp)
            
            Xmax=22983
            
            
            if Accretion_data[0]<2:
                Img=cv.imread(Image);
                Img=cv.cvtColor(Img,cv.COLOR_BGR2RGB)
                Npix=np.shape(Img)[0]; Npiy=np.shape(Img)[1]; IMG=np.zeros((Npiy,Npix,3));
                for i in range(3):
                    IMG[:,:,i]=np.transpose(Img[:,:,i])/256;
                Npix=np.shape(IMG)[0]; Npiy=np.shape(IMG)[1];
                
                XX=np.linspace(-Xmax,Xmax,Npix); YY=np.linspace(-Xmax*Npiy/Npix,Xmax*Npiy/Npix,Npiy);
                h=x0*Xmax*np.sqrt(1+Npiy**2/Npix**2)/(rf-Xmax*np.sqrt(1+Npiy**2/Npix**2));
            else:
                Npix=Accretion_data[0]; Npiy=Npix;
                XX=np.linspace(-Xmax,Xmax,Npix); YY=np.linspace(-Xmax,Xmax,Npiy);
                h=x0*Xmax*np.sqrt(2)/(rf-Xmax*np.sqrt(2));
            
            
            
            def init_conds_bis(y,z,alph):
                v0=[x0,-cSI*h/np.sqrt(h**2+y**2+z**2),y,cSI*y/np.sqrt(h**2+y**2+z**2),z,cSI*z/np.sqrt(h**2+y**2+z**2)];
                matrot=np.array([[np.cos(alph),0,-np.sin(alph)],[0,1,0],[np.sin(alph),0,np.cos(alph)]]);
                vrot=matrot.dot(np.array([v0[0],v0[2],v0[4]])); vvrot=-matrot.dot(np.array([v0[1],v0[3],v0[5]]));
                Z=To_spherical(vrot[0],vvrot[0],vrot[1],vvrot[1],vrot[2],vvrot[2]);
                Z=[Z[0],Z[2],Z[4],Z[1],Z[3],Z[5]];
                Z=[2/Rs*Z[0],Z[1],Z[2],Z[3]/cSI,Z[4]*Rs/(2*cSI),Z[5]*Rs/(2*cSI)];
                return Z
            
            
            def velocity(sphvel):
                r=sphvel[0]; th=sphvel[1]; ph=sphvel[2]; rp=sphvel[3]; thp=sphvel[4]; php=sphvel[5];
                vx=np.cos(ph)*np.sin(th)*r*rp/np.sqrt(r**2 + a**2) - np.sqrt(r**2 + a**2)*php*np.sin(ph)*np.sin(th) + np.sqrt(r**2 + a**2)*np.cos(ph)*thp*np.cos(th);
                vy=np.sin(ph)*np.sin(th)*r*rp/np.sqrt(r**2 + a**2) + np.sqrt(r**2 + a**2)*php*np.cos(ph)*np.sin(th) + np.sqrt(r**2 + a**2)*np.sin(ph)*thp*np.cos(th);
                vz=rp*np.cos(th) - r*thp*np.sin(th);
                vel=[vx,vy,vz];
                return vel
            
            
            def doppler_color(dope):
                dil_dope=(dope-1/2)**1;
                if dil_dope<1/4:
                    rcol=[0,4*dil_dope,1];
                elif (1/4<=dil_dope and dil_dope<1/2):
                    rcol=[0,1,2-4*dil_dope];
                elif (1/2<=dil_dope and dil_dope<3/4):
                    rcol=[4*dil_dope-2,1,0];
                else:
                    rcol=[1,4-4*dil_dope,0];
                if (abs(rcol[0])<0.05 and abs(rcol[2])<0.05 and abs(rcol[1]-1)<0.05):
                    rcol=[2,2,2];
                return rcol
            

            if (Accretion_data[2]=="Blackbody" and Accretion_data[3]=="Doppler"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r; Dr=r**2-2*r+rq; S=r**2;
                    veloc=inv_met_mat([0,r,th,ph],0,rq).dot(np.array([pt,V[3],V[4],pph])); al=(r/np.sqrt(r-rq));
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc);
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4); T=T/doppler_shift;
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    colou=blackbody[np.where(abs(T-blackbody[:,0])==min(abs(T-blackbody[:,0])))[0][0],range(1,4)];
                    cb=[colou,bright/doppler_shift,T];
                    return cb
            elif (Accretion_data[2]=="Blackbody" and Accretion_data[3]=="Gravitation"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r; Dr=r**2-2*r+rq; S=r**2;
                    grav_shift=1/np.sqrt(abs(Dr/S));
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4); T=T/grav_shift;
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    colou=blackbody[np.where(abs(T-blackbody[:,0])==min(abs(T-blackbody[:,0])))[0][0],range(1,4)];
                    cb=[colou,bright/(grav_shift),T];
                    return cb
            elif (Accretion_data[2]=="Blackbody" and Accretion_data[3]=="Doppler+"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r; Dr=r**2-2*r+rq; S=r**2;
                    veloc=inv_met_mat([0,r,th,ph],0,rq).dot(np.array([pt,V[3],V[4],pph])); al=(r/np.sqrt(r-rq));
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc);
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    grav_shift=1/np.sqrt(abs(Dr/S));
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4); T=T/(grav_shift*doppler_shift);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    colou=blackbody[np.where(abs(T-blackbody[:,0])==min(abs(T-blackbody[:,0])))[0][0],range(1,4)];
                    cb=[colou,bright/(grav_shift*doppler_shift),T];
                    return cb
            elif (Accretion_data[2]=="Blackbody" and Accretion_data[3]==" "):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r;
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    colou=blackbody[np.where(abs(T-blackbody[:,0])==min(abs(T-blackbody[:,0])))[0][0],range(1,4)];
                    cb=[colou,bright,T];
                    return cb
            elif (Accretion_data[2]=="Custom" and Accretion_data[3]=="Doppler"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r; Dr=r**2-2*r+rq; S=r**2;
                    veloc=inv_met_mat([0,r,th,ph],0,rq).dot(np.array([pt,V[3],V[4],pph])); al=(r/np.sqrt(r-rq));
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc);
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    doppler_coeff=1-np.sqrt(2/rb);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    cb=[np.array([0,0,0]),bright/doppler_shift,doppler_coeff**2,1/doppler_shift,T];
                    return cb
            elif (Accretion_data[2]=="Custom" and Accretion_data[3]=="Gravitation"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r; Dr=r**2-2*r+rq; S=r**2;
                    grav_shift=1/np.sqrt(abs(Dr/S));
                    doppler_coeff=1-np.sqrt(2/rb);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    cb=[np.array([0,0,0]),bright/grav_shift,doppler_coeff**2,1/grav_shift,T];
                    return cb
            elif (Accretion_data[2]=="Custom" and Accretion_data[3]=="Doppler+"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r; Dr=r**2-2*r+rq; S=r**2;
                    veloc=inv_met_mat([0,r,th,ph],0,rq).dot(np.array([pt,V[3],V[4],pph])); al=(r/np.sqrt(r-rq));
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc)
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    grav_shift=1/np.sqrt(abs(Dr/S));
                    doppler_coeff=1-np.sqrt(2/rb);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    cb=[np.array([0,0,0]),bright/(grav_shift*doppler_shift)**0,doppler_coeff**2,1/(grav_shift*doppler_shift),T];
                    return cb
            elif (Accretion_data[2]=="Custom" and Accretion_data[3]==" "):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r;
                    doppler_coeff=1-np.sqrt(2/rb);
                    T=(T0/(rb*Rs)**3*(1-np.sqrt(2*rint/(rb*Rs))))**(1/4);
                    if Accretion_data[6]!=0:
                        bright=Accretion_data[6]*4.086e-21*T**5;
                    else:
                        bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    
                    cb=[np.array([0,0,0]),bright,doppler_coeff**2,1,T];
                    return cb
            elif (Accretion_data[2]==" " and Accretion_data[3]=="Doppler"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r; Dr=r**2-2*r+rq; S=r**2;
                    veloc=inv_met_mat([0,r,th,ph],0,rq).dot(np.array([pt,V[3],V[4],pph])); al=(r/np.sqrt(r-rq));
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc)
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    colou=doppler_color(doppler_shift);
                    bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    cb=[np.array(colou),bright];
                    return cb
            elif (Accretion_data[2]==" " and Accretion_data[3]=="Gravitation"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r; Dr=r**2-2*r+rq; S=r**2;
                    grav_shift=1/np.sqrt(abs(Dr/S));
                    colou=doppler_color(grav_shift);
                    bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    cb=[np.array(colou),bright];
                    return cb
            elif (Accretion_data[2]==" " and Accretion_data[3]=="Doppler+"):
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=r; Dr=r**2-2*r+rq; S=r**2;
                    veloc=inv_met_mat([0,r,th,ph],0,rq).dot(np.array([pt,V[3],V[4],pph])); al=(r/np.sqrt(r-rq));
                    velockep=np.array([-np.sin(ph),np.cos(ph),0])/al; veloc=velocity([r,th,ph,veloc[1],veloc[2],veloc[3]]); veloc=np.array(veloc);
                    doppler_shift=(1-np.inner(veloc,velockep)/lin.norm(veloc))/np.sqrt(1-1/al**2);
                    grav_shift=1/np.sqrt(abs(Dr/S));
                    colou=doppler_color(doppler_shift*grav_shift);
                    bright=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    cb=[np.array(colou),bright];
                    return cb
            else:
                def accretion_disk(V):
                    r=2*V[0]/Rs; th=V[1]; ph=V[2]; rb=np.sqrt(r**2+a**2); lam=0.2;
                    cb=np.ones((1,4)); cb=np.array([255,69,0])/256; cb[2]=1+(rb*Rs/2-rint)*(lam-1)/(rext-rint);
                    cb=[cb,1]
                    return cb
            
            def weierP(g2,g3,z):
                N0=12;
                zz0=z/(2**N0); zz=1/zz0**2+g2/20*zz0**2+g3/28*zz0**4;
                for j in range(N0):
                    zz=-2*zz+(6*zz**2-g2/2)**2/(4*(4*zz**3-g2*zz-g3));
                return(zz)
            
            def newton(g2,g3,Z,t):
                def toanihil(s):
                    return((2*rf/Rs-rbar)*(4*np.real(weierP(g2,g3,Z+s))-beta/3)-alp);
                sgn=np.sign(toanihil(t)); sol=t; step=-0.02;
                while sgn*toanihil(sol)>0:
                    sol=sol+step;
                sol=nt(toanihil,sol);
                return(sol)
            
            
            if Accretion_data[0]==0:
                Xred=np.zeros((0,2));
                AR=[];
                for xx in XX[int(np.floor(Npix/2)):Npix]:
                    for yy in YY[int(np.floor(Npiy/2)):Npiy]:
                        AR.append(np.sqrt(xx**2+yy**2));
                
                AR=np.sort(AR)
                
                for zz in [zu for zu in AR if zu!=0]:
                    X=init_conds_bis(zz,0,0);
                    r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
                    E=np.sqrt(rp**2+php**2*(r**2-2*r+rq)); L=r**2*php; pt=-E; pph=L;
                    
                    rpol=np.roots([(E**2)/L**2,0,-1,2,-rq]);
                    mi=min(abs(rpol-np.real(rpol))); frpol=[np.real(rpol[rr]) for rr in np.where(abs(rpol-np.real(rpol))==mi)][0];
                    rbar=frpol[np.where(abs(frpol)==min(abs(frpol)))[0][0]];
                    delta=(E**2)/L**2; gamma=2*(2*delta*rbar); beta=-1+3*rbar*(gamma-2*delta*rbar); alp=2+rbar*(2*beta-rbar*(3*gamma-4*delta*rbar));
                    g2=(beta**2/3-alp*gamma)/4; g3=(alp*beta*gamma/6-alp**2*delta/2-beta**3/27)/8;
                    rp2=np.roots([4,0,-g2,-g3]); z0=alp/(4*(r-rbar))+beta/12;
                    if abs(rp)<1e-12:
                        Z0=carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2]);
                    else:
                        Z0=np.sign(-rp)*carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2]);
                    new=newton(g2,g3,Z0,0); P=ph+np.sign(php)*new;
                    if P>np.pi/2:
                        Xred=np.vstack([Xred,np.array([zz,P])]);
                
                
                Umax=0; Vmax=0; KK=np.zeros((Npix,Npiy));
                for i in range(int(np.floor(Npix/2)),Npix):
                    x=XX[i];
                    for j in range(int(np.floor(Npiy/2)),Npiy):
                        y=YY[j]; r=np.sqrt(x**2+y**2);
                        mi=min(abs(r-Xred[:,0]));
                        if mi<1e-10:
                            k=np.where(abs(r-Xred[:,0])==mi)[0][0];
                            KK[i,j]=k; KK[i,Npiy-j-1]=k;
                            KK[Npix-i-1,j]=k; KK[Npix-i-1,Npiy-j-1]=k;
                            
                Umax=np.pi/2; Vmax=Umax*Npiy/Npix;
                xred=np.zeros((Npix,Npiy,3));
                
                for i in range(Npix):
                    xx=XX[i];
                    for j in range(Npiy):
                        yy=YY[j];
                        if KK[i,j]!=0:
                            P=Xred[int(KK[i,j]),1];
                            Z=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([np.cos(P),np.sin(P),0]));
                            Z=projtoplane_bis(rf*Z);
                            t1=(Z[1]+Umax)/(2*Umax); t2=(Z[2]+Vmax)/(2*Vmax);
                            s1=abs(1-abs(1-t1)); s2=abs(1-abs(1-t2));
                            ii=int(max(1,min(Npix,np.ceil(s1*Npix)))); jj=int(max(1,min(Npiy,np.ceil(s2*Npiy))));
                            xred[i,j,0]=IMG[ii-1,jj-1,0]; xred[i,j,1]=IMG[ii-1,jj-1,1]; xred[i,j,2]=IMG[ii-1,jj-1,2];0;
                            
                
                
                
            elif Accretion_data[0]==1:
                Xred=np.zeros((0,9));
                AR=[]; Yred=np.zeros((0,9))
                for xx in XX[int(np.floor(Npix/2)):Npix]:
                    for yy in YY[int(np.floor(Npiy/2)):Npiy]:
                        AR.append(np.sqrt(xx**2+yy**2));
                
                AR=np.sort(AR)
                
                for zz in [zu for zu in AR if zu!=0]:
                    X=init_conds_bis(zz,0,0);
                    r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
                    E=np.sqrt(rp**2+php**2*(r**2-2*r+rq)); L=r**2*php; pt=-E; pph=L;
                    
                    rpol=np.roots([(E**2)/L**2,0,-1,2,-rq]);
                    mi=min(abs(rpol-np.real(rpol))); frpol=[np.real(rpol[rr]) for rr in np.where(abs(rpol-np.real(rpol))==mi)][0];
                    rbar=frpol[np.where(abs(frpol)==min(abs(frpol)))[0][0]];
                    delta=(E**2)/L**2; gamma=2*(2*delta*rbar); beta=-1+3*rbar*(gamma-2*delta*rbar); alp=2+rbar*(2*beta-rbar*(3*gamma-4*delta*rbar));
                    g2=(beta**2/3-alp*gamma)/4; g3=(alp*beta*gamma/6-alp**2*delta/2-beta**3/27)/8;
                    rp2=np.roots([4,0,-g2,-g3]); z0=alp/(4*(r-rbar))+beta/12;
                    if abs(rp)<1e-12:
                        Z0=carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2]);
                    else:
                        Z0=np.sign(-rp)*carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2]);
                    new=newton(g2,g3,Z0,0); P=ph+np.sign(php)*new;
                    if P>np.pi/2:
                        Xred=np.vstack([Xred,np.array([zz,P,g2,g3,Z0,ph,alp,beta,rbar])]);
                    else:
                        Yred=np.vstack([Yred,np.array([zz,0,g2,g3,Z0,ph,alp,beta,rbar])]);
                        
                    

                Umax=0; Vmax=0; KK=np.zeros((Npix,Npiy)); LL=np.zeros((Npix,Npiy))
                for i in range(int(np.floor(Npix/2)),Npix):
                    x=XX[i];
                    for j in range(int(np.floor(Npiy/2)),Npiy):
                        y=YY[j]; r=np.sqrt(x**2+y**2);
                        mi=min(abs(r-Xred[:,0]));
                        if mi<1e-10:
                            k=np.where(abs(r-Xred[:,0])==mi)[0][0];
                            KK[i,j]=k+1; KK[i,Npiy-j-1]=k+1;
                            KK[Npix-i-1,j]=k+1; KK[Npix-i-1,Npiy-j-1]=k+1;
                        else:
                            k=np.where(abs(r-Yred[:,0])==min(abs(r-Yred[:,0])))[0][0]
                            LL[i,j]=k; LL[i,Npiy-j-1]=k;
                            LL[Npix-i-1,j]=k; LL[Npix-i-1,Npiy-j-1]=k;
                            
                        
                    
                Umax=np.pi/2; Vmax=Umax*Npiy/Npix; xred=np.zeros((Npix,Npiy,3)); dop_max=np.zeros((Npix,Npiy));

                
                for i in range(Npix):
                    xx=XX[i];
                    for j in range(Npiy):
                        yy=YY[j]; xred[i,j,0]=0; xred[i,j,1]=0; xred[i,j,2]=0;
                        ata=np.arctan2(txi,-np.sign(xi)*yy/np.sqrt(yy**2+xx**2));
                        X=init_conds_bis(xx,yy,xi); r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
                        E=np.sqrt(rp**2+(php**2*np.sin(th)**2+thp**2)*(r**2-2*r+rq)); L=r**2*php*np.sin(th)**2; pt=-E; pph=L;
                        C=r**4*thp**2+L**2/np.tan(th)**2; Pth=np.sqrt(C);
                        if KK[i,j]==0:
                            xe=Yred[int(LL[i,j]),:];
                            Phi=min(np.pi+ata,np.pi-ata);
                            rtest=np.real(xe[6]/(4*np.real(weierP(xe[2],xe[3],xe[4]+(Phi-xe[5])))-xe[7]/3)+xe[8]);
                            if (rint_n<rtest and rtest<rext_n and Phi<np.pi/2):
                                rtesti=rtest; DDr=rtesti**2-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                Cf=From_spherical(Rs*rtest/2,0,np.pi/2,0,Phi,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                cobra=accretion_disk([Rs*rtest/2,np.pi/2,Cf[4],Pr,Pth]);
                                cobra=cobra[0].tolist()+cobra[1:];
                                cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                if Accretion_data[2]=="Custom":
                                    xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                else:
                                    xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                                
                            
                        else:
                            xe=Xred[int(KK[i,j]-1),:]; 
                            Phim=np.pi+np.arctan2(txi,yy/np.sqrt(yy**2+xx**2)); Phip=np.pi-np.arctan2(txi,-yy/np.sqrt(yy**2+xx**2));
                            rtestp=np.real(xe[6]/(4*np.real(weierP(xe[2],xe[3],xe[4]-(Phip-xe[5])))-xe[7]/3)+xe[8]);
                            rtestm=np.real(xe[6]/(4*np.real(weierP(xe[2],xe[3],xe[4]-(Phim-xe[5])))-xe[7]/3)+xe[8]);
                            whit=0;
                            if alpha>=0:
                                if (rint_n<rtestp and rtestp<rext_n):
                                    rtesti=rtestp; DDr=rtesti**2-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                    Cf=From_spherical(Rs*rtestp/2,0,np.pi/2,0,Phip,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                    Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                    cobra=accretion_disk([Rs*rtestp/2,np.pi/2,Cf[4],Pr,Pth]);
                                    cobra=cobra[0].tolist()+cobra[1:];
                                    cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                    if Accretion_data[2]=="Custom":
                                        xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                    else:
                                        xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                                    
                                    whit=1;
                                
                                if (rint_n<rtestm and rtestm<rext_n):
                                    rtesti=rtestm; DDr=rtesti**2-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                    Cf=From_spherical(Rs*rtestm/2,0,np.pi/2,0,Phim,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                    Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                    cobra=accretion_disk([Rs*rtestm/2,np.pi/2,Cf[4],Pr,Pth]);
                                    cobra=cobra[0].tolist()+cobra[1:];
                                    cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                    if Accretion_data[2]=="Custom":
                                        xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                    else:
                                        xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                                    
                                    whit=1;
                                
                            else:
                                if (rint_n<rtestm and rtestm<rext_n):
                                    rtesti=rtestm; DDr=rtesti**2-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                    Cf=From_spherical(Rs*rtestm/2,0,np.pi/2,0,Phim,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                    Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                    cobra=accretion_disk([Rs*rtestm/2,np.pi/2,Cf[4],Pr,Pth]);
                                    cobra=cobra[0].tolist()+cobra[1:];
                                    cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                    if Accretion_data[2]=="Custom":
                                        xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                    else:
                                        xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                                    
                                    whit=1;
                                
                                if (rint_n<rtestp and rtestp<rext_n):
                                    rtesti=rtestp; DDr=rtesti**2-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                    Cf=From_spherical(Rs*rtestp/2,0,np.pi/2,0,Phip,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                    Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                    cobra=accretion_disk([Rs*rtestp/2,np.pi/2,Cf[4],Pr,Pth]);
                                    cobra=cobra[0].tolist()+cobra[1:];
                                    cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                    if Accretion_data[2]=="Custom":
                                        xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                    else:
                                        xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                                    
                                    whit=1;
                                
                            
                            if whit==0:
                                P=xe[1];
                                Z=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([np.cos(P),np.sin(P),0]));
                                Z=projtoplane_bis(rf*np.real(Z));
                                s1=(Z[1]+Umax)/(2*Umax); s2=(Z[2]+Vmax)/(2*Vmax);
                                s1=abs(1-abs(1-s1)); s2=abs(1-abs(1-s2));
                                ii=int(max(1,min(Npix,np.ceil(s1*Npix)))); jj=int(max(1,min(Npiy,np.ceil(s2*Npiy))));
                                xred[i,j,0]=IMG[ii-1,jj-1,0]; xred[i,j,1]=IMG[ii-1,jj-1,1]; xred[i,j,2]=IMG[ii-1,jj-1,2];
                            
                        
                    
                
                if Accretion_data[2]=="Custom":
                    dp_max=max((dop_max.flatten()).tolist())
                    for i in range(Npix):
                        for j in range(Npiy):
                            if xred[i,j,0]==-np.exp(1):
                                flo=np.floor(xred[i,j,2]*(T_int+(T_ext-T_int)*dop_max[i,j]/dp_max));
                                wef=blackbody[np.where(abs(flo-blackbody[:,0])==min(abs(flo-blackbody[:,0])))[0][0],range(1,4)];
                                xred[i,j,0]=xred[i,j,1]*wef[0]; xred[i,j,2]=xred[i,j,1]*wef[2]; xred[i,j,1]=xred[i,j,1]*wef[1];
                                
                    
                
                
                
                
            elif Accretion_data[0]>1:
                Xred=np.zeros((0,9));
                AR=[]; Yred=np.zeros((0,9))
                for xx in XX[int(np.floor(Npix/2)):Npix]:
                    for yy in YY[int(np.floor(Npiy/2)):Npiy]:
                        AR.append(np.sqrt(xx**2+yy**2));
                
                AR=np.sort(AR)
                
                for zz in [zu for zu in AR if zu!=0]:
                    X=init_conds_bis(zz,0,0);
                    r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
                    E=np.sqrt(rp**2+php**2*(r**2-2*r+rq)); L=r**2*php; pt=-E; pph=L;
                    
                    rpol=np.roots([(E**2)/L**2,0,-1,2,-rq]);
                    mi=min(abs(rpol-np.real(rpol))); frpol=[np.real(rpol[rr]) for rr in np.where(abs(rpol-np.real(rpol))==mi)][0];
                    rbar=frpol[np.where(abs(frpol)==min(abs(frpol)))[0][0]];
                    delta=(E**2)/L**2; gamma=2*(2*delta*rbar); beta=-1+3*rbar*(gamma-2*delta*rbar); alp=2+rbar*(2*beta-rbar*(3*gamma-4*delta*rbar));
                    g2=(beta**2/3-alp*gamma)/4; g3=(alp*beta*gamma/6-alp**2*delta/2-beta**3/27)/8;
                    rp2=np.roots([4,0,-g2,-g3]); z0=alp/(4*(r-rbar))+beta/12;
                    if abs(rp)<1e-12:
                        Z0=carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2]);
                    else:
                        Z0=np.sign(-rp)*carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2]);
                    new=newton(g2,g3,Z0,0); P=ph+np.sign(php)*new;
                    if P>np.pi/2:
                        Xred=np.vstack([Xred,np.array([zz,2,g2,g3,Z0,ph,alp,beta,rbar])]);
                    else:
                        Yred=np.vstack([Yred,np.array([zz,0,g2,g3,Z0,ph,alp,beta,rbar])]);
                        
                    

                Umax=0; Vmax=0; KK=np.zeros((Npix,Npiy)); LL=np.zeros((Npix,Npiy))
                for i in range(int(np.floor(Npix/2)),Npix):
                    x=XX[i];
                    for j in range(int(np.floor(Npiy/2)),Npiy):
                        y=YY[j]; r=np.sqrt(x**2+y**2);
                        mi=min(abs(r-Xred[:,0]));
                        if mi<1e-10:
                            k=np.where(abs(r-Xred[:,0])==mi)[0][0];
                            KK[i,j]=k+1; KK[i,Npiy-j-1]=k+1;
                            KK[Npix-i-1,j]=k+1; KK[Npix-i-1,Npiy-j-1]=k+1;
                        else:
                            k=np.where(abs(r-Yred[:,0])==min(abs(r-Yred[:,0])))[0][0]
                            LL[i,j]=k; LL[i,Npiy-j-1]=k;
                            LL[Npix-i-1,j]=k; LL[Npix-i-1,Npiy-j-1]=k;
                            
                        
                    
                Umax=np.pi/2; Vmax=Umax*Npiy/Npix; xred=np.zeros((Npix,Npiy,3)); dop_max=np.zeros((Npix,Npiy));

                for i in range(Npix):
                    xx=XX[i];
                    for j in range(Npiy):
                        yy=YY[j];
                        ata=np.arctan2(txi,-np.sign(xi)*yy/np.sqrt(yy**2+xx**2));
                        X=init_conds_bis(xx,yy,xi); r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
                        E=np.sqrt(rp**2+(php**2*np.sin(th)**2+thp**2)*(r**2-2*r+rq)); L=r**2*php*np.sin(th)**2; pt=-E; pph=L;
                        C=r**4*thp**2+L**2/np.tan(th)**2; Pth=np.sqrt(C);
                        if KK[i,j]==0:
                            xe=Yred[int(LL[i,j]),:];
                            Phi=min(np.pi+ata,np.pi-ata);
                            rtest=np.real(xe[6]/(4*np.real(weierP(xe[2],xe[3],xe[4]+(Phi-xe[5])))-xe[7]/3)+xe[8]);
                            if (rint_n<rtest and rtest<rext_n and Phi<np.pi/2):
                                rtesti=rtest; DDr=rtesti**2-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                Cf=From_spherical(Rs*rtest/2,0,np.pi/2,0,Phi,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                cobra=accretion_disk([Rs*rtest/2,np.pi/2,Cf[4],Pr,Pth]);
                                cobra=cobra[0].tolist()+cobra[1:];
                                cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                if Accretion_data[2]=="Custom":
                                    xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                else:
                                    xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                                
                            
                        else:
                            xe=Xred[int(KK[i,j]-1),:]; 
                            Phim=np.pi+np.arctan2(txi,yy/np.sqrt(yy**2+xx**2)); Phip=np.pi-np.arctan2(txi,-yy/np.sqrt(yy**2+xx**2));
                            rtestp=np.real(xe[6]/(4*np.real(weierP(xe[2],xe[3],xe[4]-(Phip-xe[5])))-xe[7]/3)+xe[8]);
                            rtestm=np.real(xe[6]/(4*np.real(weierP(xe[2],xe[3],xe[4]-(Phim-xe[5])))-xe[7]/3)+xe[8]);
                            #whit=0;
                            if alpha>=0:
                                if (rint_n<rtestp and rtestp<rext_n):
                                    rtesti=rtestp; DDr=rtesti**2-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                    Cf=From_spherical(Rs*rtestp/2,0,np.pi/2,0,Phip,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                    Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                    cobra=accretion_disk([Rs*rtestp/2,np.pi/2,Cf[4],Pr,Pth]);
                                    cobra=cobra[0].tolist()+cobra[1:];
                                    cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                    if Accretion_data[2]=="Custom":
                                        xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                    else:
                                        xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                                    
                                
                                if (rint_n<rtestm and rtestm<rext_n):
                                    rtesti=rtestm; DDr=rtesti**2-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                    Cf=From_spherical(Rs*rtestm/2,0,np.pi/2,0,Phim,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                    Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                    cobra=accretion_disk([Rs*rtestm/2,np.pi/2,Cf[4],Pr,Pth]);
                                    cobra=cobra[0].tolist()+cobra[1:];
                                    cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                    if Accretion_data[2]=="Custom":
                                        xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                    else:
                                        xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                                
                                
                            else:
                                if (rint_n<rtestm and rtestm<rext_n):
                                    rtesti=rtestm; DDr=rtesti**2-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                    Cf=From_spherical(Rs*rtestm/2,0,np.pi/2,0,Phim,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                    Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                    cobra=accretion_disk([Rs*rtestm/2,np.pi/2,Cf[4],Pr,Pth]);
                                    cobra=cobra[0].tolist()+cobra[1:];
                                    cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                    if Accretion_data[2]=="Custom":
                                        xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                    else:
                                        xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                                    
                                
                                if (rint_n<rtestp and rtestp<rext_n):
                                    rtesti=rtestp; DDr=rtesti**2-2*rtesti+rq; Pr=np.sqrt(abs(E**2*rtesti**4-DDr*(L**2+C)))/DDr;
                                    Cf=From_spherical(Rs*rtestp/2,0,np.pi/2,0,Phip,0); Cf=rot(np.array([1,0,0]),np.arctan2(yy,xx),np.array([Cf[0],Cf[2],Cf[4]]));
                                    Cf=rot(np.array([0,1,0]),xi,Cf); Cf=np.real(Cf); Cf=To_spherical(Cf[0],0,Cf[1],0,Cf[2],0);
                                    cobra=accretion_disk([Rs*rtestp/2,np.pi/2,Cf[4],Pr,Pth]);
                                    cobra=cobra[0].tolist()+cobra[1:];
                                    cobra=np.real(cobra).tolist(); Temp.append(cobra[-1]);
                                    if Accretion_data[2]=="Custom":
                                        xred[i,j,0]=-np.exp(1); xred[i,j,1]=cobra[3]; xred[i,j,2]=cobra[5]; dop_max[i,j]=cobra[4];
                                    else:
                                        xred[i,j,0]=cobra[3]*cobra[0]; xred[i,j,1]=cobra[3]*cobra[1]; xred[i,j,2]=cobra[3]*cobra[2];
                        
                    
                
                if Accretion_data[2]=="Custom":
                    dp_max=max((dop_max.flatten()).tolist())
                    for i in range(Npix):
                        for j in range(Npiy):
                            if xred[i,j,0]==-np.exp(1):
                                flo=np.floor(xred[i,j,2]*(T_int+(T_ext-T_int)*dop_max[i,j]/dp_max));
                                wef=blackbody[np.where(abs(flo-blackbody[:,0])==min(abs(flo-blackbody[:,0])))[0][0],range(1,4)];
                                xred[i,j,0]=xred[i,j,1]*wef[0]; xred[i,j,2]=xred[i,j,1]*wef[2]; xred[i,j,1]=xred[i,j,1]*wef[1];
                            
                        
                    
    
    
    xredt=np.zeros((Npiy,Npix,3));
    for i in range(3):
        xredt[:,:,i]=np.transpose(xred[:,:,i]);
    



    plt.figure(figsize=(12,12))
    plt.imshow(xredt)

    if (Accretion_data[0]>0 and Accretion_data[7]==1 and Accretion_data[2]!=" "):
        if Accretion_data[2]=="Custom":
            Text=T_ext; Tint=T_int;
        else:
            Text=min(Temp); Tint=max(Temp)
        n=np.where(abs(Text-blackbody[:,0])==min(abs(Text-blackbody[:,0])))[0][0]
        ncolors=np.zeros((0,4))
        while blackbody[n,0]<Tint:
            ncolors=np.vstack([ncolors,np.array(blackbody[n,1:4].tolist()+[1])])
            n+=1
        newcmp = ListedColormap(ncolors)
        norm = mpl.colors.Normalize(vmin=Text, vmax=Tint)
        sm = plt.cm.ScalarMappable(cmap=newcmp, norm=norm) 
        sm.set_array([])
        plt.colorbar(sm, ticks=np.linspace(Text,Tint, 10+11),label="Temperature [K]",shrink=0.812) 

    plt.grid(False)
    plt.axis('off')
    plt.show()
    
