#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 15:01:19 2024

@author: arthur
"""



import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import cmath
from scipy.optimize import newton as nt
from scipy.integrate import odeint
import matplotlib as mpl
from matplotlib.colors import ListedColormap

import cv2 as cv

import os
path0=os.getcwd();
os.chdir(path0);
import warnings

import pickle
import imageio




def shadow4gif(Lambda,Mass,Kerr,Newman,Image_matrix,Angle):
    if Kerr!=0: #Carter's equations
        warnings.filterwarnings("ignore");#warnings.filterwarnings("error")
        ##We define the refining parameters for details near the horizon(s)
        ##cft=0.95; itermax=100; N=1500;//N=3000;
        cft=0.85; itermax=50; N=1500;#N=1200;
        N=1500;
        
        if Lambda!=0:
            c=1; G=1; M=1; GSI=6.67408e-11; cSI=299792458; e0=8.854187e-12; sb=5.67e-8;
            Rs=2*GSI*Mass/cSI**2; A=Kerr*Rs/2; alpha=-Angle;
            rq=Newman**2;
            x0=50000; rf=60000;
            rs=2; rg=1; a=Kerr; lam=0.8; chi=1+Lambda*a**2/3;
            

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
            
            
            Img=Image_matrix; Npix=np.shape(Img); Npiy=Npix[1]; Npix=Npix[0]
            IMG=np.zeros((Npiy,Npix,3))
            for i in range(3):
                IMG[:,:,i]=np.transpose(Img[:,:,i])
            Npix=np.shape(IMG); Npiy=Npix[1]; Npix=Npix[0]
            XX=np.linspace(-Xmax,Xmax,Npix); YY=np.linspace(-Xmax*Npiy/Npix,Xmax*Npiy/Npix,Npiy);
            h=x0*Xmax*np.sqrt(1+Npiy**2/Npix**2)/(rf-Xmax*np.sqrt(1+Npiy**2/Npix**2));
            

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
            
            
            Umax=np.pi/2; Vmax=Umax*Npiy/Npix; xred=np.zeros((Npix,Npiy,3));
            npr=np.roots([-Lambda/3,0,1-Lambda*a**2/3,-2,rq+a**2])
            mi=min(abs(npr-np.real(npr)))
            tes=[np.real(npr[rr]) for rr in np.where(abs(npr-np.real(npr))==mi)[0].tolist() if np.real(npr[rr])<2]
            if (mi<1e-8 and len(tes)>0):
                ir=max(tes)
            else:
                ir=-np.Inf


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
                    if (dwef<2.5e2 and (len(np.where([abs(phii)>=np.Inf*np.pi for phii in phi])[0])==0 and len(np.where([Vec[vv,0]<=1.01*ir for vv in range(N)])[0])==0)):
                        l=np.where(dWEF==dwef)[0][0];
                        wef=BoyerLindquist_bis(R[l],theta[l],phi[l]);
                        wef=np.array([[np.cos(alpha),0,np.sin(alpha)],[0,1,0],[-np.sin(alpha),0,np.cos(alpha)]]).dot(np.array(wef));
                        wef=wef.tolist()
                        wef=projtoplane_bis(wef);
                        s1=np.real(wef[1]+Umax)/(2*Umax); s2=np.real(wef[2]+Vmax)/(2*Vmax);
                        s1=abs(1-abs(1-s1)); s2=abs(1-abs(1-s2));
                        ii=int(max(1,min(Npix,np.ceil(s1*Npix)))); jj=int(max(1,min(Npiy,np.ceil(s2*Npiy))));
                        xred[i,j,0]=IMG[ii-1,jj-1,0]; xred[i,j,1]=IMG[ii-1,jj-1,1]; xred[i,j,2]=IMG[ii-1,jj-1,2];
                        
                                  
            

        else:
            c=1; G=1; M=1; GSI=6.67408e-11; cSI=299792458; e0=8.854187e-12; sb=5.67e-8;
            Rs=2*GSI*Mass/cSI**2; A=Kerr*Rs/2; alpha=-Angle;
            rq=Newman**2; x0=50000; rf=60000; rs=2; rg=1; a=Kerr; lam=0.8;


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
            
            Img=Image_matrix; Npix=np.shape(Img); Npiy=Npix[1]; Npix=Npix[0]
            IMG=np.zeros((Npiy,Npix,3))
            for i in range(3):
                IMG[:,:,i]=np.transpose(Img[:,:,i])
            Npix=np.shape(IMG); Npiy=Npix[1]; Npix=Npix[0]
            XX=np.linspace(-Xmax,Xmax,Npix); YY=np.linspace(-Xmax*Npiy/Npix,Xmax*Npiy/Npix,Npiy);
            h=x0*Xmax*np.sqrt(1+Npiy**2/Npix**2)/(rf-Xmax*np.sqrt(1+Npiy**2/Npix**2));
            

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



            Umax=np.pi/2; Vmax=Umax*Npiy/Npix; xred=np.zeros((Npix,Npiy,3));
            npr=np.roots([1,-2,rq+a**2])
            mi=min(abs(npr-np.real(npr)))
            tes=[np.real(npr[rr]) for rr in np.where(abs(npr-np.real(npr))==mi)[0].tolist() if np.real(npr[rr])<2]
            if (mi<1e-8 and len(tes)>0):
                ir=max(tes)
            else:
                ir=-np.Inf


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
                    if (dwef<2.5e2 and (len(np.where([abs(phii)>=np.Inf*np.pi for phii in phi])[0])==0 and len(np.where([Vec[vv,0]<=1.01*ir for vv in range(N)])[0])==0)):
                        l=np.where(dWEF==dwef)[0][0];
                        wef=BoyerLindquist_bis(R[l],theta[l],phi[l]);
                        wef=np.array([[np.cos(alpha),0,np.sin(alpha)],[0,1,0],[-np.sin(alpha),0,np.cos(alpha)]]).dot(np.array(wef));
                        wef=wef.tolist()
                        wef=projtoplane_bis(wef);
                        s1=np.real(wef[1]+Umax)/(2*Umax); s2=np.real(wef[2]+Vmax)/(2*Vmax);
                        s1=abs(1-abs(1-s1)); s2=abs(1-abs(1-s2));
                        ii=int(max(1,min(Npix,np.ceil(s1*Npix)))); jj=int(max(1,min(Npiy,np.ceil(s2*Npiy))));
                        xred[i,j,0]=IMG[ii-1,jj-1,0]; xred[i,j,1]=IMG[ii-1,jj-1,1]; xred[i,j,2]=IMG[ii-1,jj-1,2];
                        
                    
    
        warnings.filterwarnings("ignore")
        
        
        
        
        
        
    else:
        ##Use Weierstrass' analytic method if a=0:
        if Lambda!=0:
            c=1; G=1; M=1; GSI=6.67408e-11; cSI=299792458; e0=8.854187e-12; sb=5.67e-8;
            Rs=2*GSI*Mass/cSI**2; A=Kerr*Rs/2; alpha=-Angle;
            x0=50000; rf=60000; rq=Newman**2; rs=2; rg=1; a=0; lam=0.8;
            
            
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
            
            Img=Image_matrix; Npix=np.shape(Img); Npiy=Npix[1]; Npix=Npix[0]
            IMG=np.zeros((Npiy,Npix,3))
            for i in range(3):
                IMG[:,:,i]=np.transpose(Img[:,:,i])
            Npix=np.shape(IMG); Npiy=Npix[1]; Npix=Npix[0]
            XX=np.linspace(-Xmax,Xmax,Npix); YY=np.linspace(-Xmax*Npiy/Npix,Xmax*Npiy/Npix,Npiy);
            h=x0*Xmax*np.sqrt(1+Npiy**2/Npix**2)/(rf-Xmax*np.sqrt(1+Npiy**2/Npix**2));
            
            
            
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
                            
                
                
                    

        else:
            c=1; G=1; M=1; GSI=6.67408e-11; cSI=299792458; e0=8.854187e-12; sb=5.67e-8;
            Rs=2*GSI*Mass/cSI**2; A=Kerr*Rs/2; alpha=-Angle;# xi=Angle; txi=np.tan(xi);
            x0=50000; rf=60000;
            rq=Newman**2;##Q=Newman*2*Mass*sqrt(%pi*e0*GSI); rq2=Q^2*GSI/(4*%pi*e0*cSI^4); rq=4*rq2/Rs^2;
            rs=2; rg=1; a=0; lam=0.8;
            
            
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
            
            Img=Image_matrix; Npix=np.shape(Img); Npiy=Npix[1]; Npix=Npix[0]
            IMG=np.zeros((Npiy,Npix,3))
            for i in range(3):
                IMG[:,:,i]=np.transpose(Img[:,:,i])
            Npix=np.shape(IMG); Npiy=Npix[1]; Npix=Npix[0]
            XX=np.linspace(-Xmax,Xmax,Npix); YY=np.linspace(-Xmax*Npiy/Npix,Xmax*Npiy/Npix,Npiy);
            h=x0*Xmax*np.sqrt(1+Npiy**2/Npix**2)/(rf-Xmax*np.sqrt(1+Npiy**2/Npix**2));
            
            
            
            
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
                            
                
                
    
    
    xredt=np.zeros((Npiy,Npix,3));
    for i in range(3):
        xredt[:,:,i]=np.transpose(xred[:,:,i]);
        
    return xredt



#------------------------------------------------------------------------------



def DatFile4gif(Resol,Lambda,Mass,Kerr,Newman,Angle):
    N1=Resol[0]; N2=Resol[1];
    Ns=[N1,N2];
    os.chdir(path0);
    img1=np.zeros((N1,N2,3))
    for k in range(3):
        img1[:,:,k]=np.array([k*N1*N2+np.linspace(i*N2,(i+1)*N2-1,N2).astype(int) for i in range(N1)])
    img2=shadow4gif(Lambda,Mass,Kerr,Newman,img1,Angle);
    lis1=np.zeros((0,3)); lis2=np.zeros((0,3))
    for i in range(N1):
        for j in range(N2):
            lis1=np.vstack([lis1,np.array([img1[i,j,0],img1[i,j,1],img1[i,j,2]])])
            lis2=np.vstack([lis2,np.array([img2[i,j,0],img2[i,j,1],img2[i,j,2]])])
    L=[];
    for k in range(N1*N2):
        pix=lis2[k,:]
        l=0;
        while (l<N1*N2 and (lis1[l,0]!=pix[0] or lis1[l,1]!=pix[1] or lis1[l,2]!=pix[2])):
            l+=1
        if l>=N1*N2:
            L.append(-1)
        else:
            L.append(l)
    Ls=list([N1,N2,L])

    path=path0+"/dat_files"
    try:
        os.mkdir(path)
        #print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    os.chdir(path)
    path=path+"/file"+"_"+str(Resol)+"_"+str(Lambda)+"_"+str(Mass)+"_"+str(Kerr)+"_"+str(Newman)+"_"+str(Angle)+".dat";

    with open(path, 'wb') as f:
        pickle.dump([Ns,Ls,Lambda,Mass,Kerr,Newman,img1,img2], f, protocol=-1)
        f.close()









def make_gif_with_DatFile(Nimages,Name,Image,Resol,Shifts,Direction,FPS,Lambda,Mass,Kerr,Newman,Angle):
    os.chdir(path0)
    N1=Resol[0]; N2=Resol[1]; K1=Shifts[0]; K2=Shifts[1]; coe=Shifts[2]; Ns=[N1,N2];
    img=cv.imread(Image); img=cv.cvtColor(img,cv.COLOR_BGR2RGB); img=img/256;
    N0=int(np.floor((np.shape(img)[0]-N1)/2)); names=[]; name0=Name; ext=".png"
    path=path0+'/'+Name+'_gif';
    try:
        os.mkdir(path)
        #print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    for j in range(Nimages):
        if j<10:
            names.append(name0+"000"+str(j)+ext)
        elif j<100:
            names.append(name0+"00"+str(j)+ext)
        elif j<1000:
            names.append(name0+"0"+str(j)+ext)
        else:
            names.append(name0+str(j)+ext)
    os.chdir(path)
    path=path+"/temp"
    try:
        os.mkdir(path)
        #print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    imgs=list([]);
    os.chdir(path)
    for i0 in range(len(names)):
        img0=np.zeros((N1,N2,3))
        floe1=int(np.floor(coe*i0)+K1); floe2=int(np.floor(coe*i0)+K2);
        for k in range(3):
            if Direction=="d2-":
                img0[:,:,k]=img[floe1:(floe1+N1),floe2:(floe2+N2),k]
            elif Direction=="d2+":
                img0[:,:,k]=img[(floe1+N1):floe1:-1,(floe2+N2):floe2:-1,k]
            elif Direction=="d1-":
                img0[:,:,k]=img[(floe1+N1):floe1:-1,floe2:(floe2+N2),k]
            elif Direction=="d1+":
                img0[:,:,k]=img[floe1:(floe1+N1),(floe2+N2):floe2:-1,k]
            elif Direction=="h-":
                img0[:,:,k]=img[N0+K1:N0+K1+N1,floe2:(floe2+N2),k]
            elif Direction=="h+":
                img0[:,:,k]=img[N0+K1+N1:N0+K1:-1,(floe2+N2):floe2:-1,k]
            elif Direction=="v+":
                img0[:,:,k]=img[floe1:(floe1+N1),N0+K2:N0+K2+N2,k]
            elif Direction=="v-":
                img0[:,:,k]=img[(floe1+N1):floe1:-1,N0+K2+N2:N0+K2:-1,k]
        imgs.append(img0)
        img0*=256
        img0.astype(int)
        cv.imwrite(names[i0],img0)
    
    file=path0+"/dat_files/file"+"_"+str(Resol)+"_"+str(Lambda)+"_"+str(Mass)+"_"+str(Kerr)+"_"+str(Newman)+"_"+str(Angle)+".dat";

    try:
        with open(file,'rb') as f:
            [Ns,Ls,Lambda,Mass,Kerr,Newman,img1,img2] = pickle.load(f)
            f.close()
        
        os.chdir(path)
        Imgs0=os.listdir(); Imgs0.sort(); Imgs=list([]); L=0;
        img=cv.imread(Imgs0[0]); N1=np.shape(img)[0]; N2=np.shape(img)[1];
        L=Ls[2] 
    
        for img in range(len(Imgs0)):
            img=Imgs0[img];
            imgp=cv.cvtColor(cv.imread(img),cv.COLOR_BGR2RGB); imgp=imgp/256;
            lisp=np.zeros((0,3))
            for i in range(N1):
                for j in range(N2):
                    lisp=np.vstack([lisp,np.array([imgp[i,j,0],imgp[i,j,1],imgp[i,j,2]])])
            img0=np.zeros((0,3))
            for m in L:
                if m<0:
                    img0=np.vstack([img0,np.array([0,0,0])])
                else:
                    img0=np.vstack([img0,lisp[m]])
            Img0=np.zeros((N1,N2,3)); cou=-1
            for i in range(N1):
                for j in range(N2):
                    cou+=1;
                    Img0[i,j,0]=img0[cou,0]; Img0[i,j,1]=img0[cou,1]; Img0[i,j,2]=img0[cou,2]
            Imgs.append(Img0)
        
        for img in Imgs0:
            os.remove(img)
        
        path=path0+'/'+Name+'_gif';
        os.chdir(path)
        os.rmdir(path+"/temp")
        for img in range(len(Imgs)):
            name=names[img]; img=Imgs[img];
            img*=256
            img.astype(int)
            cv.imwrite(name,img)
            
        #imageio.mimsave(Name+'.gif', names, loop=0, duration = 0.1)
        #for name in names:
        #    os.remove(name)
        with imageio.get_writer(Name+'.gif', mode='I',fps=FPS,loop=0) as writer:
            for name in names:
                image = imageio.imread(name)
                writer.append_data(image)
                os.remove(name)
                
    except Exception as e:
        print(f"Error: exception {e}.")    








def make_gif(Nimages,Name,Image,Resol,Shifts,Direction,FPS,Lambda,Mass,Kerr,Newman,Angle):
    N1=Resol[0]; N2=Resol[1]; K1=Shifts[0]; K2=Shifts[1]; coe=Shifts[2];
    os.chdir(path0);
    img1=np.zeros((N1,N2,3))
    for k in range(3):
        img1[:,:,k]=np.array([k*N1*N2+np.linspace(i*N2,(i+1)*N2-1,N2).astype(int) for i in range(N1)])
    img2=shadow4gif(Lambda,Mass,Kerr,Newman,img1,Angle);
    lis1=np.zeros((0,3)); lis2=np.zeros((0,3))
    for i in range(N1):
        for j in range(N2):
            lis1=np.vstack([lis1,np.array([img1[i,j,0],img1[i,j,1],img1[i,j,2]])])
            lis2=np.vstack([lis2,np.array([img2[i,j,0],img2[i,j,1],img2[i,j,2]])])
    L=[];
    for k in range(N1*N2):
        pix=lis2[k,:]
        l=0;
        while (l<N1*N2 and (lis1[l,0]!=pix[0] or lis1[l,1]!=pix[1] or lis1[l,2]!=pix[2])):
            l+=1
        if l>=N1*N2:
            L.append(-1)
        else:
            L.append(l)
    Ls=list([N1,N2,L])

    
    img=cv.imread(Image); img=cv.cvtColor(img,cv.COLOR_BGR2RGB); img=img/256;
    N0=int(np.floor((np.shape(img)[0]-N1)/2)); names=[]; name0=Name; ext=".png"
    path=path0+'/'+Name+'_gif';
    try:
        os.mkdir(path)
        #print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    for j in range(Nimages):
        if j<10:
            names.append(name0+"000"+str(j)+ext)
        elif j<100:
            names.append(name0+"00"+str(j)+ext)
        elif j<1000:
            names.append(name0+"0"+str(j)+ext)
        else:
            names.append(name0+str(j)+ext)
    os.chdir(path)
    path=path+"/temp"
    try:
        os.mkdir(path)
        #print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
    imgs=list([]);
    os.chdir(path)
    for i0 in range(len(names)):
        img0=np.zeros((N1,N2,3))
        floe1=int(np.floor(coe*i0)+K1); floe2=int(np.floor(coe*i0)+K2);
        for k in range(3):
            if Direction=="d2-":
                img0[:,:,k]=img[floe1:(floe1+N1),floe2:(floe2+N2),k]
            elif Direction=="d2+":
                img0[:,:,k]=img[(floe1+N1):floe1:-1,(floe2+N2):floe2:-1,k]
            elif Direction=="d1-":
                img0[:,:,k]=img[(floe1+N1):floe1:-1,floe2:(floe2+N2),k]
            elif Direction=="d1+":
                img0[:,:,k]=img[floe1:(floe1+N1),(floe2+N2):floe2:-1,k]
            elif Direction=="h-":
                img0[:,:,k]=img[N0+K1:N0+K1+N1,floe2:(floe2+N2),k]
            elif Direction=="h+":
                img0[:,:,k]=img[N0+K1+N1:N0+K1:-1,(floe2+N2):floe2:-1,k]
            elif Direction=="v+":
                img0[:,:,k]=img[floe1:(floe1+N1),N0+K2:N0+K2+N2,k]
            elif Direction=="v-":
                img0[:,:,k]=img[(floe1+N1):floe1:-1,N0+K2+N2:N0+K2:-1,k]
        imgs.append(img0)
        img0*=256
        img0.astype(int)
        cv.imwrite(names[i0],img0)
       
    os.chdir(path)
    Imgs0=os.listdir(); Imgs0.sort(); Imgs=list([]); L=0;
    img=cv.imread(Imgs0[0]); N1=np.shape(img)[0]; N2=np.shape(img)[1];
    L=Ls[2] 

    for img in range(len(Imgs0)):
        img=Imgs0[img];
        imgp=cv.cvtColor(cv.imread(img),cv.COLOR_BGR2RGB); imgp=imgp/256;
        lisp=np.zeros((0,3))
        for i in range(N1):
            for j in range(N2):
                lisp=np.vstack([lisp,np.array([imgp[i,j,0],imgp[i,j,1],imgp[i,j,2]])])
        img0=np.zeros((0,3))
        for m in L:
            if m<0:
                img0=np.vstack([img0,np.array([0,0,0])])
            else:
                img0=np.vstack([img0,lisp[m]])
        Img0=np.zeros((N1,N2,3)); cou=-1
        for i in range(N1):
            for j in range(N2):
                cou+=1;
                Img0[i,j,0]=img0[cou,0]; Img0[i,j,1]=img0[cou,1]; Img0[i,j,2]=img0[cou,2]
        Imgs.append(Img0)
    
    for img in Imgs0:
        os.remove(img)
    
    path=path0+'/'+Name+'_gif';
    os.chdir(path)
    os.rmdir(path+"/temp")
    for img in range(len(Imgs)):
        name=names[img]; img=Imgs[img];
        img*=256
        img.astype(int)
        cv.imwrite(name,img)
    with imageio.get_writer(Name+'.gif', mode='I',fps=FPS,loop=0) as writer:
        for name in names:
            image = imageio.imread(name)
            writer.append_data(image)
            os.remove(name)