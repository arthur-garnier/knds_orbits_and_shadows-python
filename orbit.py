#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:56:30 2024

@author: arthur
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import cmath
from scipy.optimize import fsolve
#from scipy.optimize import newton as nt
#import matplotlib as mpl

#from scipy.misc import derivative
import cv2 as cv

from scipy.integrate import odeint

from auxi import *

##The function that computes the trajectory of a test particle. 
def orbit(Lambda,Mass,Kerr,Newman,IniConds,Form,Tau,N,Mu,Conserv,e):
    if Lambda!=0:
        ##Fundamental constants, normalized parameters and initial conditions
        GSI=6.67408e-11; cSI=299792458; e0=8.854187e-12; ##e=0;
        ##Mass=cSI**2*Rs/(2*GSI); J=Kerr*GSI*Mass**2/cSI; a=J/(Mass*cSI);
        Rs=2*GSI*Mass/cSI**2; J=Kerr*GSI*Mass**2/cSI; a=J/(Mass*cSI);
        Q=Newman*2*Mass*np.sqrt(np.pi*e0*GSI*(1-0*Kerr**2)); rq2=Q**2*GSI/(4*np.pi*e0*cSI**4); rq=4*rq2/Rs**2;##WARNING 0*Kerr**2
        G=1; c=1; Mass=1; rs=2; rg=1; a=Kerr; Q=np.sqrt(rq*4*np.pi*e0); tau=2*cSI/Rs*Tau; chi=1+Lambda*a**2/3;
        IC=[2/Rs*IniConds[0],IniConds[1],IniConds[2],IniConds[3]/cSI,IniConds[4]*Rs/(2*cSI),IniConds[5]*Rs/(2*cSI)];
        ##Compute \dot{t}_0 using Carter's formulas.
        dtau=tau/N; r0=IC[0]; th0=IC[1]; ph0=IC[2]; rp0=IC[3]; thp0=IC[4]; php0=IC[5];
        Dr0=(1-Lambda*r0**2/3)*(r0**2+a**2)-2*r0+rq; Dt0=1+Lambda*a**2*np.cos(th0)**2/3; S0=r0**2+a**2*np.cos(th0)**2;
        E0=-e*r0*np.sqrt(rq)/(chi*S0)+np.sqrt(Dr0*Dt0*php0**2*np.sin(th0)**2/chi**4+(Dr0-Dt0*a**2*np.sin(th0)**2)/(chi**2*S0)*(Mu+rp0**2*S0/Dr0+thp0**2*S0/Dt0));
        L0=np.sin(th0)**2/(chi**2*(Dt0*a**2*np.sin(th0)**2-Dr0))*(Dt0*(a*chi*(chi*E0*(r0**2+a**2)+e*r0*np.sqrt(rq))-php0*S0*Dr0)-a*chi**2*E0*Dr0);
        time_coeff=chi/S0*((r0**2+a**2)*(chi*(E0*(r0**2+a**2)-a*L0)+e*r0*np.sqrt(rq))/Dr0-a*np.sin(th0)*(a*E0*np.sin(th0)-L0/np.sin(th0))/Dt0);
        ##Select the formulation:
        if Form=="Polar":
            Vecc=[]; HAM=[]; CAR=[];
            ##From spherical coordinates (with velocities) to Cartesian coordinates (with velocities)
            def From_spherical(R,Rp,T,Tp,P,Pp):
                x=R*np.sin(T)*np.cos(P);
                y=R*np.sin(T)*np.sin(P);
                z=R*np.cos(T);
                xp=(Tp*np.cos(T)*np.cos(P)*R-np.sin(T)*(Pp*np.sin(P)*R-Rp*np.cos(P)));
                yp=(Tp*np.cos(T)*np.sin(P)*R+np.sin(T)*(Pp*np.cos(P)*R+Rp*np.sin(P)));
                zp=Rp*np.cos(T)-R*Tp*np.sin(T);
                BB=[x,xp,y,yp,z,zp];
                return BB
            ##Back to spherical coordinates (with velocities)
            def To_spherical(x,xp,y,yp,z,zp):
                P=np.arctan2(y,x)
                R=np.sqrt(x**2+y**2+z**2);
                T=np.arccos(z/R);
                Rp=(x*xp+y*yp+z*zp)/R;
                Tp=(z*Rp-zp*R)/(R*np.sqrt(R**2-z**2));
                Pp=(yp*x-xp*y)/(x**2+y**2);
                XX=[R,Rp,T,Tp,P,Pp];
                return XX
            ##Compute a rotated vector using Rodrigues' formula. The input is the directed axis, the angle and the vector to be rotated.
            def rot(axe,theta,u):
                KK=np.array([[0,-axe[2],axe[1]],[axe[2],0,-axe[0]],[-axe[1],axe[0],0]]); KK=KK/lin.norm(axe,2);
                RR=np.identity(3)+np.sin(theta)*KK+(1-np.cos(theta))*KK**2;
                v=RR.dot(u);
                return v
            ##Bring the initial point and velocity in the plane theta=pi/2
            X=IC; r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5]; mu=Mu;
            BB=From_spherical(r,rp,th,thp,ph,php); BBi=np.array([BB[0],BB[2],BB[4]]); BBv=np.array([BB[1],BB[3],BB[5]]);
            x=BBi[0]; y=BBi[1]; z=BBi[2]; vx=BBv[0]; vy=BBv[1]; vz=BBv[2];
            if (abs(z)<1e-10 and abs(vz)<1e-10):
                th0=0; O0=np.array([1,0,0]);
            elif (abs(z)<1e-10 and abs(vz)>=1e-10):
                P0=np.array([y*(vx*y-vy*x)/(x**2+y**2),x*(-vx*y+vy*x)/(x**2+y**2),vz]); Q0=np.array([y*(vx*y-vy*x)/(x**2+y**2),x*(-vx*y+vy*x)/(x**2+y**2),0]);
                th0=np.sign(vz)*np.arccos((np.inner(P0,Q0))/(lin.norm(P0)*lin.norm(Q0))); O0=BBi;
            elif (abs(z)>=1e-10 and abs(vz)<1e-10):
                th0=np.pi/2; O0=BBv;
            else:
                O0=np.array([x-z*vx/vz,y-z*vy/vz,0]); P0=np.array([vy*z/vz-y,-vx*z/vz+x,0]);
                Q0=np.array([-z*(-vy*z+vz*y)*(-vx*y+vy*x)/((vx**2+vy**2)*z**2-2*vz*(vx*x+vy*y)*z+vz**2*(x**2+y**2)),z*(-vx*z+vz*x)*(-vx*y+vy*x)/((vx**2+vy**2)*z**2-2*vz*(vx*x+vy*y)*z+vz**2*(x**2+y**2)),z]);
                th0=np.sign(z)*np.arccos((np.inner(P0,Q0))/(lin.norm(P0)*lin.norm(Q0)));
            
            BBi=rot(O0,-th0,BBi); BBv=rot(O0,-th0,BBv);
            CC=To_spherical(BBi[0],BBv[0],BBi[1],BBv[1],BBi[2],BBv[2]);
            R=CC[0]; Rp=CC[1]; T=CC[2]; Tp=CC[3]; P=CC[4]; Pp=CC[5];
            ##Angular momentum (constant) and initial polar datum
            L=R**2*Pp; X=[1/R,-Rp/L];
            ##Function for 'ode'
            def f(U):
                ##Y=[U[1];U[0]*(-2*rq*U[0]**2+3*U[0]-1-mu*rq/L**2)+mu/L**2-Lambda*mu/(3*L**2*U[0]**3)];
                Y=[U[1],U[0]*(-2*rq*U[0]**2+3*U[0]-1+(e**2-mu)*rq/L**2)+(E0*e*np.sqrt(rq)+mu)/L**2-Lambda*mu/(3*L**2*U[0]**3)];
                return Y
            def F(V,t):
                U=f(V)
                return U
            ##Solve using 'ode'
            intt=np.arange(0,tau,dtau)[:N]
            Vec=odeint(F,X,Pp*intt);
            Vec=np.array([1/Vec[:,0],P+Pp*intt]); Vecc=np.zeros((0,3)); HAM=[];
            test=0; ii=0;
            ##Check if the trajectoy doesn't go too far or too close
            while (test==0 and ii<np.shape(Vec)[1]):
                Rf=Vec[0,ii]; Pf=Vec[1,ii];
                if (Rf<0 or Rf>np.Inf):#200*rs):
                    test=1;
                
                ##back to the original plane and to SI units
                Cf=rot(O0,th0,np.array([Rf*np.cos(Pf),Rf*np.sin(Pf),0]));
                Vecc=np.vstack([Vecc,np.array([Rs/2*Rf,np.arccos(Cf[2]/Rf),np.arctan2(Cf[1],Cf[0])])]); ii+=1;
                
        elif Form=="Euler-Lagrange":
            ##initialize
                    X0=[0,time_coeff,IC[0],IC[3],IC[1],IC[4],IC[2],IC[5]]; Vec=[X0]; X=X0; CAR=[];
                    ##choose the right 'ode' function from parameters
                    if (J==0 and Q==0):
                        f=cosmo_Schwarzschild;
                    elif (Q==0 and J!=0):
                        f=cosmo_Kerr;
                    elif (Q!=0 and J==0):
                        f=cosmo_ReissnerNordstrom;
                    else:
                        f=cosmo_KerrNewman;
                    
                    def F(V,t):
                        #U=np.transpose(np.array([V]))
                        U=f(V,Lambda,a,rq);
                        #U=np.transpose(U)[0].tolist()
                        return U
                    Vec=odeint(F,X,np.arange(0,tau,dtau));
                    Vecc=np.zeros((0,3)); HAM=[];
                    for ve in Vec:
                        ##back to SI units
                        Vecc=np.vstack([Vecc,np.array([Rs/2*ve[2],ve[4],ve[6]])]);
                        if Conserv==1:
                            ##If Conserv==1 is chosen,: we compute the Hamiltonian and Carter constant at each node
                            g=cosmo_met_mat([0,ve[2],ve[4],ve[6]],Lambda,a,rq);
                            HAM.append(np.inner(np.array([1,ve[3],ve[5],ve[7]]),g.dot(np.array([1,ve[3],ve[5],ve[7]]))));
                            DT=1+Lambda/3*a**2*np.cos(ve[4])**2; S=ve[2]**2+a**2*np.cos(ve[4])**2;
                            Car=S**2*ve[5]**2/DT+chi**2*np.cos(ve[4])**2/DT*(L0**2/np.sin(ve[4])**2-a**2*(E0**2-Mu*DT/chi**2+Lambda**2/3*(a*E0-L0)**2));
                            CAR.append(Car);
                            
        elif Form=="Carter":
                    ##Initialize and fix the motion constants
                    X=IC; r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5]; mu=-Mu; CAR=[];
                    Dr=(1-Lambda*r**2/3)*(r**2+a**2)-2*r+rq; Dt=1+Lambda*a**2*np.cos(th)**2/3; S=r**2+a**2*np.cos(th)**2;
                    E=-e*r*np.sqrt(rq)/(chi*S)+np.sqrt(Dr*Dt*php**2*np.sin(th)**2/chi**4+(Dr-Dt*a**2*np.sin(th)**2)/(chi**2*S)*(-mu+rp**2*S/Dr+thp**2*S/Dt));
                    Lz=np.sin(th)**2/(chi**2*(Dt*a**2*np.sin(th)**2-Dr))*(Dt*(a*chi*(chi*E*(r**2+a**2)+e*r*np.sqrt(rq))-php*S*Dr)-a*chi**2*E*Dr);
                    pt=-E; pr=S*rp/Dr; pth=S*thp/Dt; pph=Lz;
                    Q=Dt*pth**2+chi**2*np.cos(th)**2/Dt*(Lz**2/np.sin(th)**2-a**2*(E**2+mu*Dt/chi**2+Lambda**2/3*(a*E-Lz)**2));
                    k=Q+chi**2*(a*E-Lz)**2;
                    X=[r,th,ph,pr,pth]; Vec=[X];
                    ##'ode' solve
                    def F(V,t):
                        #U=np.transpose(np.array([V]))
                        U=cosmo_Carter_Newman(V,Lambda,a,rq,E,Lz,e,k,mu);
                        #U=np.transpose(U)[0].tolist()
                        return U
                    Vec=odeint(F,X,np.arange(0,tau,dtau));
                    Vecc=np.zeros((0,3)); HAM=[];
                    for ve in Vec:
                        Vecc=np.vstack([Vecc,np.array([Rs/2*ve[0],ve[1],ve[2]])]);
                        if Conserv==1:
                            gi=cosmo_inv_met_mat([0,ve[0],ve[1],ve[2]],Lambda,a,rq);
                            HAM.append(np.inner(np.array([pt,ve[3],ve[4],pph]),gi.dot(np.array([pt,ve[3],ve[4],pph]))));
                            DT=1+Lambda/3*a**2*np.cos(ve[1])**2; S=ve[0]**2+a**2*np.cos(ve[1])**2;
                            Car=DT*ve[4]**2+chi**2*np.cos(ve[1])**2/DT*(L0**2/np.sin(ve[1])**2-a**2*(E0**2-Mu*DT/chi**2+Lambda**2/3*(a*E0-L0)**2));
                            CAR.append(Car);
                            
        elif Form=="Hamilton":
                    X=cosmo_init_conds_hamiltonian([0,IC[0],IC[1],IC[2],time_coeff,IC[3],IC[4],IC[5]],Lambda,a,rq); Vec=[X]; CAR=[];
                    f=cosmo_Hamilton_equations;
                    def F(V,t):
                        U=f(V,Lambda,a,rq);
                        return U
                    Vec=odeint(F,X,np.arange(0,tau,dtau));
                    Vecc=np.zeros((0,3)); HAM=[];
                    for ve in Vec:
                        Vecc=np.vstack([Vecc,np.array([Rs/2*ve[1],ve[2],ve[3]])]);
                        if Conserv==1:
                            gi=cosmo_inv_met_mat([0,ve[1],ve[2],ve[3]],Lambda,a,rq);
                            HAM.append(np.inner(np.array([ve[4],ve[5],ve[6],ve[7]]),gi.dot(np.array([ve[4],ve[5],ve[6],ve[7]]))));
                            DT=1+Lambda/3*a**2*np.cos(ve[2])**2; S=ve[1]**2+a**2*np.cos(ve[2])**2;
                            Car=DT*ve[6]**2+chi**2*np.cos(ve[2])**2/DT*(L0**2/np.sin(ve[2])**2-a**2*(E0**2-Mu*DT/chi**2+Lambda**2/3*(a*E0-L0)**2));
                            CAR.append(Car);
                            
        elif Form=="Symplectic Euler p":
                    P=np.array([time_coeff,IC[3],IC[4],IC[5]]); X=[0,IC[0],IC[1],IC[2]]; Vec=np.zeros((0,4)); PVec=np.zeros((0,4)); Ham=[]; k=Kerr; rq=4*rq2/Rs**2; rs=2; CAR=[];
                    P=cosmo_met_mat(X,Lambda,a,rq).dot(P);
                    #X=np.transpose(np.array([[0,IC[0],IC[1],IC[2]]]));
                    for n in range(N):#N+1
                        ##the implicit part is done using 'fsolve'
                        def IL(x):
                            x1=(np.array(x)-np.array(X)-dtau*cosmo_inv_met_mat(x,Lambda,a,rq).dot(P)).tolist();
                            return x1
                        X=fsolve(IL,X,full_output=False); [g,gi,drpg,dthpg]=cosmo_inverse_metric_matrix(X,Lambda,a,rq);
                        P=P-dtau/2*np.array([0,np.inner(P,drpg.dot(P)),np.inner(P,dthpg.dot(P)),0]);
                        Vec=np.vstack([Vec,X]); PVec=np.vstack([PVec,P]); Ham.append(np.inner(P,g.dot(P)));
                    
                    Vecc=np.zeros((0,3)); HAM=[];
                    if Conserv==1:
                        HAM=Ham;
                    
                    for vee in range(np.shape(Vec)[0]):
                        ve=Vec[vee,:];
                        Vecc=np.vstack([Vecc,np.array([Rs/2*ve[1],ve[2],ve[3]])]);
                        if Conserv==1:
                            Pv=PVec[vee,:];
                            DT=1+Lambda/3*a**2*np.cos(ve[2])**2; S=ve[1]**2+a**2*np.cos(ve[2])**2;
                            Car=DT*Pv[2]**2+chi**2*np.cos(ve[2])**2/DT*(L0**2/np.sin(ve[2])**2-a**2*(E0**2-Mu*DT/chi**2+Lambda**2/3*(a*E0-L0)**2));
                            CAR.append(Car);
                            
                            
        elif Form=="Symplectic Euler q":
                    P=np.array([time_coeff,IC[3],IC[4],IC[5]]); X=[0,IC[0],IC[1],IC[2]]; Vec=np.zeros((0,4)); PVec=np.zeros((0,4)); Ham=[]; k=Kerr; rq=4*rq2/Rs**2; rs=2; CAR=[];
                    [g,gi,drpg,dthpg]=cosmo_inverse_metric_matrix(X,Lambda,a,rq); P=g.dot(P);
                    P=P.tolist()
                    for n in range(N):#N+1
                        def IR(p):
                            p1=(np.array(p)-np.array(P)+dtau/2*np.array([0,np.inner(np.array(p),drpg.dot(np.array(p))),np.inner(np.array(p),dthpg.dot(np.array(p))),0])).tolist()
                            return p1
                        P=fsolve(IR,P,full_output=False);
                        X=(np.array(X)+dtau*cosmo_inv_met_mat(X,Lambda,a,rq).dot(P)).tolist();
                        [g,gi,drpg,dthpg]=cosmo_inverse_metric_matrix(X,Lambda,a,rq);
                        Vec=np.vstack([Vec,np.array(X)]); PVec=np.vstack([PVec,P]); Ham.append(np.inner(P,g.dot(P)));
                    
                    Vecc=np.zeros((0,3)); HAM=[];
                    if Conserv==1:
                        HAM=Ham;
                    
                    for vee in range(np.shape(Vec)[0]):
                        ve=Vec[vee,:];
                        Vecc=np.vstack([Vecc,np.array([Rs/2*ve[1],ve[2],ve[3]])]);
                        if Conserv==1:
                            Pv=PVec[vee,:];
                            DT=1+Lambda/3*a**2*np.cos(ve[2])**2; S=ve[1]**2+a**2*np.cos(ve[2])**2;
                            Car=DT*Pv[2]**2+chi**2*np.cos(ve[2])**2/DT*(L0**2/np.sin(ve[2])**2-a**2*(E0**2-Mu*DT/chi**2+Lambda**2/3*(a*E0-L0)**2));
                            CAR.append(Car);
                            
        elif Form=="Verlet":
                    P=np.array([time_coeff,IC[3],IC[4],IC[5]]); X=[0,IC[0],IC[1],IC[2]]; Vec=np.zeros((0,4)); PVec=np.zeros((0,4)); Ham=[]; k=Kerr; rq=4*rq2/Rs**2; rs=2; CAR=[];
                    P=cosmo_met_mat(X,Lambda,a,rq).dot(P);
                    Ham=[]; [g,gi,drpg,dthpg]=cosmo_inverse_metric_matrix(X,Lambda,a,rq);
                    for n in range(N):#N+1
                        Pp=P-dtau/4*np.array([0,np.inner(P,drpg.dot(P)),np.inner(P,dthpg.dot(P)),0]);
                        X=(np.array(X)+dtau*gi.dot(Pp)).tolist();
                        [g,gi,drpg,dthpg]=cosmo_inverse_metric_matrix(X,Lambda,a,rq);
                        P=Pp-dtau/4*np.array([0,np.inner(P,drpg.dot(P)),np.inner(P,dthpg.dot(P)),0]);
                        Vec=np.vstack([Vec,np.array(X)]); PVec=np.vstack([PVec,np.transpose(P)]); Ham.append(np.inner(P,g.dot(P)));
                    
                    Vecc=np.zeros((0,3)); HAM=[];
                    if Conserv==1:
                        HAM=Ham;
                    
                    for vee in range(np.shape(Vec)[0]):
                        ve=Vec[vee,:];
                        Vecc=np.vstack([Vecc,np.array([Rs/2*ve[1],ve[2],ve[3]])]);
                        if Conserv==1:
                            Pv=PVec[vee,:];
                            DT=1+Lambda/3*a**2*np.cos(ve[2])**2; S=ve[1]**2+a**2*np.cos(ve[2])**2;
                            Car=DT*Pv[2]**2+chi**2*np.cos(ve[2])**2/DT*(L0**2/np.sin(ve[2])**2-a**2*(E0**2-Mu*DT/chi**2+Lambda**2/3*(a*E0-L0)**2));
                            CAR.append(Car);
        
        elif Form=="Stormer-Verlet":
                    P=[time_coeff,IC[3],IC[4],IC[5]]; X=[0,IC[0],IC[1],IC[2]]; Vec=np.zeros((0,4)); PVec=np.zeros((0,4)); rq=4*rq2/Rs**2; rs=2; CAR=[];
                    [g,Gam]=cosmo_metric_with_christoffel(X,Lambda,rs,a,rq); P=g.dot(np.array(P)); HAM=[];
                    gi=cosmo_inv_met_mat(X,Lambda,a,rq); Ham=[1/2*np.inner(P,gi.dot(P))];
                    Vec=np.vstack([Vec,X]); PVec=np.vstack([PVec,P]);
                    X=np.array(X);
                    for n in range(N-1):#N
                        def NI(x):
                            x1=(np.array(x)-np.array(X)-dtau/2*cosmo_inv_met_mat(x,Lambda,a,rq).dot(P)).tolist();
                            return x1
                        XP=fsolve(NI,X); [g,gi,drpg,dthpg]=cosmo_inverse_metric_matrix(XP.tolist(),Lambda,a,rq);
                        def NJ(p):
                            p1=(np.array(p)-np.array(P)+dtau/4*(np.array([0,np.inner(P,drpg.dot(P)),np.inner(P,dthpg.dot(P)),0])+np.array([0,np.inner(np.array(p),drpg.dot(np.array(p))),np.inner(np.array(p),dthpg.dot(np.array(p))),0]))).tolist();
                            return p1
                        #def NJp(p):
                        #    Ap=drpg.dot(P); Bp=dthpg.dot(P);
                        #    p1=np.array([[1,0,0,0],[dtau/2*Ap[0],1+dtau/2*Ap[1],dtau/2*Ap[2],dtau/2*Ap[3]],[dtau/2*Bp[0],dtau/2*Bp[1],1+dtau/2*Bp[2],dtau/2*Bp[3]],[0,0,0,1]])
                        #    return(p1)
                        P=fsolve(NJ,P); X=XP+dtau/2*gi.dot(np.transpose(P)); Vec=np.vstack([Vec,X]); Ham.append(np.inner(P,gi.dot(P))); PVec=np.vstack([PVec,P]);
                    
                    Vecc=np.zeros((0,3)); HAM=[];
                    if Conserv==1:
                        HAM=Ham;
                    
                    for vee in range(np.shape(Vec)[0]):
                        ve=Vec[vee,:];
                        Vecc=np.vstack([Vecc,np.array([Rs/2*ve[1],ve[2],ve[3]])]);
                        if Conserv==1:
                            Pv=PVec[vee,:];
                            DT=1+Lambda/3*a**2*np.cos(ve[2])**2; S=ve[1]**2+a**2*np.cos(ve[2])**2;
                            Car=DT*Pv[2]**2+chi**2*np.cos(ve[2])**2/DT*(L0**2/np.sin(ve[2])**2-a**2*(E0**2-Mu*DT/chi**2+Lambda**2/3*(a*E0-L0)**2));
                            CAR.append(Car);
                            
                            
    else:
        GSI=6.67408e-11; cSI=299792458; e0=8.854187e-12; ##e=0;
        ##Mass=cSI**2*Rs/(2*GSI); J=Kerr*GSI*Mass**2/cSI; a=J/(Mass*cSI);
        Rs=2*GSI*Mass/cSI**2; J=Kerr*GSI*Mass**2/cSI; a=J/(Mass*cSI);
        Q=Newman*2*Mass*np.sqrt(np.pi*e0*GSI*(1-0*Kerr**2)); rq2=Q**2*GSI/(4*np.pi*e0*cSI**4); rq=4*rq2/Rs**2;##WARNING 0*Kerr**2
        G=1; c=1; Mass=1; rs=2; rg=1; a=Kerr; Q=np.sqrt(rq*4*np.pi*e0); tau=2*cSI/Rs*Tau;
        IC=[2/Rs*IniConds[0],IniConds[1],IniConds[2],IniConds[3]/cSI,IniConds[4]*Rs/(2*cSI),IniConds[5]*Rs/(2*cSI)];
        ##Compute \dot{t}_0 using Carter's formulas.
        dtau=tau/N; r0=IC[0]; th0=IC[1]; ph0=IC[2]; rp0=IC[3]; thp0=IC[4]; php0=IC[5];
        Dr0=(r0**2+a**2)-2*r0+rq; S0=r0**2+a**2*np.cos(th0)**2;
        E0=-e*r0*np.sqrt(rq)/S0+np.sqrt(Dr0*php0**2*np.sin(th0)**2+(Dr0-a**2*np.sin(th0)**2)/S0*(Mu+rp0**2*S0/Dr0+thp0**2*S0));
        L0=np.sin(th0)**2/(a**2*np.sin(th0)**2-Dr0)*((a*(E0*(r0**2+a**2)+e*r0*np.sqrt(rq))-php0*S0*Dr0)-a*E0*Dr0);
        time_coeff=1/S0*((r0**2+a**2)*((E0*(r0**2+a**2)-a*L0)+e*r0*np.sqrt(rq))/Dr0-a*np.sin(th0)*(a*E0*np.sin(th0)-L0/np.sin(th0)));
        
        
        if Form=="Polar":
            Vecc=[]; HAM=[]; CAR=[];
            ##From spherical coordinates (with velocities) to Cartesian coordinates (with velocities)
            def From_spherical(R,Rp,T,Tp,P,Pp):
                x=R*np.sin(T)*np.cos(P);
                y=R*np.sin(T)*np.sin(P);
                z=R*np.cos(T);
                xp=(Tp*np.cos(T)*np.cos(P)*R-np.sin(T)*(Pp*np.sin(P)*R-Rp*np.cos(P)));
                yp=(Tp*np.cos(T)*np.sin(P)*R+np.sin(T)*(Pp*np.cos(P)*R+Rp*np.sin(P)));
                zp=Rp*np.cos(T)-R*Tp*np.sin(T);
                BB=[x,xp,y,yp,z,zp];
                return BB
            ##Back to spherical coordinates (with velocities)
            def To_spherical(x,xp,y,yp,z,zp):
                P=np.arctan2(y,x)
                R=np.sqrt(x**2+y**2+z**2);
                T=np.arccos(z/R);
                Rp=(x*xp+y*yp+z*zp)/R;
                Tp=(z*Rp-zp*R)/(R*np.sqrt(R**2-z**2));
                Pp=(yp*x-xp*y)/(x**2+y**2);
                XX=[R,Rp,T,Tp,P,Pp];
                return XX
            ##Compute a rotated vector using Rodrigues' formula. The input is the directed axis, the angle and the vector to be rotated.
            def rot(axe,theta,u):
                KK=np.array([[0,-axe[2],axe[1]],[axe[2],0,-axe[0]],[-axe[1],axe[0],0]]); KK=KK/lin.norm(axe,2);
                RR=np.identity(3)+np.sin(theta)*KK+(1-np.cos(theta))*KK**2;
                v=RR.dot(u);
                return v
            ##Bring the initial point and velocity in the plane theta=pi/2
            X=IC; r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5]; mu=Mu;
            BB=From_spherical(r,rp,th,thp,ph,php); BBi=np.array([BB[0],BB[2],BB[4]]); BBv=np.array([BB[1],BB[3],BB[5]]);
            x=BBi[0]; y=BBi[1]; z=BBi[2]; vx=BBv[0]; vy=BBv[1]; vz=BBv[2];
            if (abs(z)<1e-10 and abs(vz)<1e-10):
                th0=0; O0=np.array([1,0,0]);
            elif (abs(z)<1e-10 and abs(vz)>=1e-10):
                P0=np.array([y*(vx*y-vy*x)/(x**2+y**2),x*(-vx*y+vy*x)/(x**2+y**2),vz]); Q0=np.array([y*(vx*y-vy*x)/(x**2+y**2),x*(-vx*y+vy*x)/(x**2+y**2),0]);
                th0=np.sign(vz)*np.arccos((np.inner(P0,Q0))/(lin.norm(P0)*lin.norm(Q0))); O0=BBi;
            elif (abs(z)>=1e-10 and abs(vz)<1e-10):
                th0=np.pi/2; O0=BBv;
            else:
                O0=np.array([x-z*vx/vz,y-z*vy/vz,0]); P0=np.array([vy*z/vz-y,-vx*z/vz+x,0]);
                Q0=np.array([-z*(-vy*z+vz*y)*(-vx*y+vy*x)/((vx**2+vy**2)*z**2-2*vz*(vx*x+vy*y)*z+vz**2*(x**2+y**2)),z*(-vx*z+vz*x)*(-vx*y+vy*x)/((vx**2+vy**2)*z**2-2*vz*(vx*x+vy*y)*z+vz**2*(x**2+y**2)),z]);
                th0=np.sign(z)*np.arccos((np.inner(P0,Q0))/(lin.norm(P0)*lin.norm(Q0)));
            
            BBi=rot(O0,-th0,BBi); BBv=rot(O0,-th0,BBv);
            CC=To_spherical(BBi[0],BBv[0],BBi[1],BBv[1],BBi[2],BBv[2]);
            R=CC[0]; Rp=CC[1]; T=CC[2]; Tp=CC[3]; P=CC[4]; Pp=CC[5];
            ##Angular momentum (constant) and initial polar datum
            L=R**2*Pp; X=[1/R,-Rp/L];
            ##Function for 'ode'
            def f(U):
                ##Y=[U[1];U[0]*(-2*rq*U[0]**2+3*U[0]-1-mu*rq/L**2)+mu/L**2-Lambda*mu/(3*L**2*U[0]**3)];
                Y=[U[1],U[0]*(-2*rq*U[0]**2+3*U[0]-1+(e**2-mu)*rq/L**2)+(E0*e*np.sqrt(rq)+mu)/L**2];
                return Y
            def F(V,t):
                U=f(V)
                return U
            ##Solve using 'ode'
            intt=np.arange(0,tau,dtau)#[:N]
            Vec=odeint(F,X,Pp*intt);
            Vec=np.array([1/Vec[:,0],P+Pp*intt]); Vecc=np.zeros((0,3)); HAM=[];
            test=0; ii=0;
            ##Check if the trajectoy doesn't go too far or too close
            while (test==0 and ii<np.shape(Vec)[1]):
                Rf=Vec[0,ii]; Pf=Vec[1,ii];
                if (Rf<0 or Rf>np.Inf):#200*rs):
                    test=1;
                
                ##back to the original plane and to SI units
                Cf=rot(O0,th0,np.array([Rf*np.cos(Pf),Rf*np.sin(Pf),0]));
                Vecc=np.vstack([Vecc,np.array([Rs/2*Rf,np.arccos(Cf[2]/Rf),np.arctan2(Cf[1],Cf[0])])]); ii+=1;
        
        elif Form=="Weierstrass":
                    Vecc=np.zeros((0,3)); mu=Mu; HAM=[]; CAR=[];
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
                    
                    X=IC; r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5];
                    BB=From_spherical(r,rp,th,thp,ph,php); BBi=np.array([BB[0],BB[2],BB[4]]); BBv=np.array([BB[1],BB[3],BB[5]]);
                    x=BBi[0]; y=BBi[1]; z=BBi[2]; vx=BBv[0]; vy=BBv[1]; vz=BBv[2];
                    if (abs(z)<1e-10 and abs(vz)<1e-10):
                        th0=0; O0=np.array([1,0,0]);
                    elif (abs(z)<1e-10 and abs(vz)>=1e-10):
                        P0=np.array([y*(vx*y-vy*x)/(x**2+y**2),x*(-vx*y+vy*x)/(x**2+y**2),vz]); Q0=np.array([y*(vx*y-vy*x)/(x**2+y**2),x*(-vx*y+vy*x)/(x**2+y**2),0]);
                        th0=np.sign(vz)*np.arccos((np.inner(P0,Q0))/(lin.norm(P0)*lin.norm(Q0))); O0=BBi;
                    elif (abs(z)>=1e-10 and abs(vz)<1e-10):
                        th0=np.pi/2; O0=BBv;
                    else:
                        O0=np.array([x-z*vx/vz,y-z*vy/vz,0]); P0=np.array([vy*z/vz-y,-vx*z/vz+x,0]);
                        Q0=np.array([-z*(-vy*z+vz*y)*(-vx*y+vy*x)/((vx**2+vy**2)*z**2-2*vz*(vx*x+vy*y)*z+vz**2*(x**2+y**2)),z*(-vx*z+vz*x)*(-vx*y+vy*x)/((vx**2+vy**2)*z**2-2*vz*(vx*x+vy*y)*z+vz**2*(x**2+y**2)),z]);
                        th0=np.sign(z)*np.arccos((np.inner(P0,Q0))/(lin.norm(P0)*lin.norm(Q0)));
            
                    BBi=rot(O0,-th0,BBi); BBv=rot(O0,-th0,BBv);
                    CC=To_spherical(BBi[0],BBv[0],BBi[1],BBv[1],BBi[2],BBv[2]);
                    R=CC[0]; Rp=CC[1]; P=CC[4]; Pp=CC[5];# T=CC[2]; Tp=CC[3];
                    
                    E=-e*np.sqrt(rq)/R+np.sqrt(Rp**2+(1-2/R+rq/R**2)*mu+Pp**2*(R**2-2*R+rq)); L=R**2*Pp;
                    rpol=np.roots([(E**2-mu)/L**2,2*(E*e*np.sqrt(rq)+mu)/L**2,-1+(e**2-mu)*rq/L**2,2,-rq]);
                    mi=min(abs(rpol-np.real(rpol))); frpol=[np.real(rpol[rr]) for rr in np.where(abs(rpol-np.real(rpol))==mi)][0];
                    rbar=frpol[np.where(abs(frpol)==min(abs(frpol)))[0][0]];
                    delta=(E**2-mu)/L**2; gamma=2*(2*delta*rbar+(E*e*np.sqrt(rq)+mu)/L**2); beta=-1+(e**2-mu)*rq/L**2+3*rbar*(gamma-2*delta*rbar); alpha=2+rbar*(2*beta-rbar*(3*gamma-4*delta*rbar));
                    g2=(beta**2/3-alpha*gamma)/4; g3=(alpha*beta*gamma/6-alpha**2*delta/2-beta**3/27)/8;
                    
                    def weierP(z):
                        N0=12;
                        zz0=z/(2**N0); zz=1/zz0**2+g2/20*zz0**2+g3/28*zz0**4;
                        for j in range(1,N0+1):
                            zz=-2*zz+(6*zz**2-g2/2)**2/(4*(4*zz**3-g2*zz-g3));
                        return(zz)
                    
                    rp2=np.roots([4,0,-g2,-g3]); z0=alpha/(4*(R-rbar))+beta/12;
                    if abs(Rp)<1e-12:
                        Z0=carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2])
                    else:
                        Z0=np.sign(-Rp)*carlson(z0-rp2[0],z0-rp2[1],z0-rp2[2])
                    
                    def wrs(t):
                        return((2-rbar)*(4*np.real(weierP(Z0+t))-beta/3)-alpha);
                    
                    xxx=fsolve(wrs,0,full_output=False,maxfev=5000)[0]; vvv=wrs(xxx);
                    if (abs(vvv)<1e-8 and abs(np.sign(Pp)*xxx+P)<2*np.pi):
                        Z0=-Z0
                        tau=min(tau,abs(xxx)); dtau=tau/N;
                        
                    Vec=4*np.real(weierP(Z0+np.arange(0,tau,dtau)))-beta/3; si=len(Vec);
                    Vec=np.array([alpha/Vec+rbar,P+np.sign(Pp)*np.arange(0,tau,dtau)]); Vecc=np.zeros((0,3)); test=0; ii=0;
                    while (test==0 and ii<si):
                        Rf=Vec[0,ii]; Pf=Vec[1,ii];
                        if (Rf<0 or Rf>200*Rs):
                            test=1;
                        Cf=rot(O0,th0,np.array([Rf*np.cos(Pf),Rf*np.sin(Pf),0]));
                        Vecc=np.vstack([Vecc,np.array([Rs/2*Rf,np.arccos(Cf[2]/Rf),np.arctan2(Cf[1],Cf[0])])]); ii+=1;
        
        elif Form=="Euler-Lagrange":
            ##initialize
                    X0=[0,time_coeff,IC[0],IC[3],IC[1],IC[4],IC[2],IC[5]]; Vec=[X0]; X=X0; CAR=[];
                    ##choose the right 'ode' function from parameters
                    if (J==0 and Q==0):
                        f=Schwarzschild;
                    elif (Q==0 and J!=0):
                        f=Kerr;
                    elif (Q!=0 and J==0):
                        f=ReissnerNordstrom;
                    else:
                        f=KerrNewman;
                    
                    def F(V,t):
                        U=f(V,a,rq);
                        return U
                    Vec=odeint(F,X,np.arange(0,tau,dtau));
                    Vecc=np.zeros((0,3)); HAM=[];
                    for ve in Vec:
                        ##back to SI units
                        Vecc=np.vstack([Vecc,np.array([Rs/2*ve[2],ve[4],ve[6]])]);
                        if Conserv==1:
                            ##If Conserv==1 is chosen,: we compute the Hamiltonian and Carter constant at each node
                            g=met_mat([0,ve[2],ve[4],ve[6]],a,rq);
                            HAM.append(np.inner(np.array([1,ve[3],ve[5],ve[7]]),g.dot(np.array([1,ve[3],ve[5],ve[7]]))));
                            S=ve[2]**2+a**2*np.cos(ve[4])**2;
                            Car=S**2*ve[5]**2+np.cos(ve[4])**2*(L0**2/np.sin(ve[4])**2-a**2*(E0**2-Mu));
                            CAR.append(Car);
                            
        elif Form=="Carter":
                    ##Initialize and fix the motion constants
                    X=IC; r=X[0]; th=X[1]; ph=X[2]; rp=X[3]; thp=X[4]; php=X[5]; mu=-Mu; CAR=[];
                    Dr=(r**2+a**2)-2*r+rq; S=r**2+a**2*np.cos(th)**2;
                    E=-e*r*np.sqrt(rq)/S+np.sqrt(Dr*php**2*np.sin(th)**2+(Dr-a**2*np.sin(th)**2)/S*(-mu+rp**2*S/Dr+thp**2*S));
                    Lz=np.sin(th)**2/(a**2*np.sin(th)**2-Dr)*((a*(E*(r**2+a**2)+e*r*np.sqrt(rq))-php*S*Dr)-a*E*Dr);
                    pt=-E; pr=S*rp/Dr; pth=S*thp; pph=Lz;
                    Q=pth**2+np.cos(th)**2*(Lz**2/np.sin(th)**2-a**2*(E**2+mu));
                    k=Q+(a*E-Lz)**2;
                    X=[r,th,ph,pr,pth]; Vec=[X];
                    ##'ode' solve
                    def F(V,t):
                        #U=np.transpose(np.array([V]))
                        U=Carter_Newman(V,a,rq,E,Lz,e,k,mu);
                        #U=np.transpose(U)[0].tolist()
                        return U
                    Vec=odeint(F,X,np.arange(0,tau,dtau));
                    Vecc=np.zeros((0,3)); HAM=[];
                    for ve in Vec:
                        Vecc=np.vstack([Vecc,np.array([Rs/2*ve[0],ve[1],ve[2]])]);
                        if Conserv==1:
                            gi=inv_met_mat([0,ve[0],ve[1],ve[2]],a,rq);
                            HAM.append(np.inner(np.array([pt,ve[3],ve[4],pph]),gi.dot(np.array([pt,ve[3],ve[4],pph]))));
                            S=ve[0]**2+a**2*np.cos(ve[1])**2;
                            Car=ve[4]**2+np.cos(ve[1])**2*(L0**2/np.sin(ve[1])**2-a**2*(E0**2-Mu));
                            CAR.append(Car);
                            
        elif Form=="Hamilton":
                    X=init_conds_hamiltonian([0,IC[0],IC[1],IC[2],time_coeff,IC[3],IC[4],IC[5]],a,rq); Vec=[X]; CAR=[];
                    f=Hamilton_equations;
                    def F(V,t):
                        U=f(V,a,rq);
                        return U
                    Vec=odeint(F,X,np.arange(0,tau,dtau));
                    Vecc=np.zeros((0,3)); HAM=[];
                    for ve in Vec:
                        Vecc=np.vstack([Vecc,np.array([Rs/2*ve[1],ve[2],ve[3]])]);
                        if Conserv==1:
                            gi=inv_met_mat([0,ve[1],ve[2],ve[3]],a,rq);
                            HAM.append(np.inner(np.array([ve[4],ve[5],ve[6],ve[7]]),gi.dot(np.array([ve[4],ve[5],ve[6],ve[7]]))));
                            S=ve[1]**2+a**2*np.cos(ve[2])**2;
                            Car=ve[6]**2+np.cos(ve[2])**2*(L0**2/np.sin(ve[2])**2-a**2*(E0**2-Mu));
                            CAR.append(Car);
                            
        elif Form=="Symplectic Euler p":
                    P=np.array([time_coeff,IC[3],IC[4],IC[5]]); X=[0,IC[0],IC[1],IC[2]]; Vec=np.zeros((0,4)); PVec=np.zeros((0,4)); Ham=[]; k=Kerr; rq=4*rq2/Rs**2; rs=2; CAR=[];
                    P=met_mat(X,a,rq).dot(P);
                    #X=np.transpose(np.array([[0,IC[0],IC[1],IC[2]]]));
                    for n in range(N):#N+1
                        ##the implicit part is done using 'fsolve'
                        def IL(x):
                            x1=(np.array(x)-np.array(X)-dtau*inv_met_mat(x,a,rq).dot(P)).tolist();
                            return x1
                        X=fsolve(IL,X,full_output=False); [g,gi,drpg,dthpg]=inverse_metric_matrix(X,a,rq);
                        P=P-dtau/2*np.array([0,np.inner(P,drpg.dot(P)),np.inner(P,dthpg.dot(P)),0]);
                        Vec=np.vstack([Vec,X]); PVec=np.vstack([PVec,P]); Ham.append(np.inner(P,g.dot(P)));
                    
                    Vecc=np.zeros((0,3)); HAM=[];
                    if Conserv==1:
                        HAM=Ham;
                    
                    for vee in range(np.shape(Vec)[0]):
                        ve=Vec[vee,:];
                        Vecc=np.vstack([Vecc,np.array([Rs/2*ve[1],ve[2],ve[3]])]);
                        if Conserv==1:
                            Pv=PVec[vee,:];
                            S=ve[1]**2+a**2*np.cos(ve[2])**2;
                            Car=Pv[2]**2+np.cos(ve[2])**2*(L0**2/np.sin(ve[2])**2-a**2*(E0**2-Mu));
                            CAR.append(Car);
                            
                            
        elif Form=="Symplectic Euler q":
                    P=np.array([time_coeff,IC[3],IC[4],IC[5]]); X=[0,IC[0],IC[1],IC[2]]; Vec=np.zeros((0,4)); PVec=np.zeros((0,4)); Ham=[]; k=Kerr; rq=4*rq2/Rs**2; rs=2; CAR=[];
                    [g,gi,drpg,dthpg]=inverse_metric_matrix(X,a,rq); P=g.dot(P);
                    P=P.tolist()
                    for n in range(N):#N+1
                        def IR(p):
                            p1=(np.array(p)-np.array(P)+dtau/2*np.array([0,np.inner(np.array(p),drpg.dot(np.array(p))),np.inner(np.array(p),dthpg.dot(np.array(p))),0])).tolist()
                            return p1
                        P=fsolve(IR,P,full_output=False);
                        X=(np.array(X)+dtau*inv_met_mat(X,a,rq).dot(P)).tolist();
                        [g,gi,drpg,dthpg]=inverse_metric_matrix(X,a,rq);
                        Vec=np.vstack([Vec,np.array(X)]); PVec=np.vstack([PVec,P]); Ham.append(np.inner(P,g.dot(P)));
                    
                    Vecc=np.zeros((0,3)); HAM=[];
                    if Conserv==1:
                        HAM=Ham;
                    
                    for vee in range(np.shape(Vec)[0]):
                        ve=Vec[vee,:];
                        Vecc=np.vstack([Vecc,np.array([Rs/2*ve[1],ve[2],ve[3]])]);
                        if Conserv==1:
                            Pv=PVec[vee,:];
                            S=ve[1]**2+a**2*np.cos(ve[2])**2;
                            Car=Pv[2]**2+np.cos(ve[2])**2*(L0**2/np.sin(ve[2])**2-a**2*(E0**2-Mu));
                            CAR.append(Car);
                            
        elif Form=="Verlet":
                    P=np.array([time_coeff,IC[3],IC[4],IC[5]]); X=[0,IC[0],IC[1],IC[2]]; Vec=np.zeros((0,4)); PVec=np.zeros((0,4)); Ham=[]; k=Kerr; rq=4*rq2/Rs**2; rs=2; CAR=[];
                    P=met_mat(X,a,rq).dot(P);
                    Ham=[]; [g,gi,drpg,dthpg]=inverse_metric_matrix(X,a,rq);
                    for n in range(N):#N+1
                        Pp=P-dtau/4*np.array([0,np.inner(P,drpg.dot(P)),np.inner(P,dthpg.dot(P)),0]);
                        X=(np.array(X)+dtau*gi.dot(Pp)).tolist();
                        [g,gi,drpg,dthpg]=inverse_metric_matrix(X,a,rq);
                        P=Pp-dtau/4*np.array([0,np.inner(P,drpg.dot(P)),np.inner(P,dthpg.dot(P)),0]);
                        Vec=np.vstack([Vec,np.array(X)]); PVec=np.vstack([PVec,np.transpose(P)]); Ham.append(np.inner(P,g.dot(P)));
                    
                    Vecc=np.zeros((0,3)); HAM=[];
                    if Conserv==1:
                        HAM=Ham;
                    
                    for vee in range(np.shape(Vec)[0]):
                        ve=Vec[vee,:];
                        Vecc=np.vstack([Vecc,np.array([Rs/2*ve[1],ve[2],ve[3]])]);
                        if Conserv==1:
                            Pv=PVec[vee,:];
                            S=ve[1]**2+a**2*np.cos(ve[2])**2;
                            Car=Pv[2]**2+np.cos(ve[2])**2*(L0**2/np.sin(ve[2])**2-a**2*(E0**2-Mu));
                            CAR.append(Car);

        elif Form=="Stormer-Verlet":
                    P=[time_coeff,IC[3],IC[4],IC[5]]; X=[0,IC[0],IC[1],IC[2]]; Vec=np.zeros((0,4)); PVec=np.zeros((0,4)); rq=4*rq2/Rs**2; rs=2; CAR=[];
                    [g,Gam]=metric_with_christoffel(X,rs,a,rq); P=g.dot(np.array(P)); HAM=[];
                    gi=inv_met_mat(X,a,rq); Ham=[1/2*np.inner(P,gi.dot(P))];
                    X=np.array(X);
                    Vec=np.vstack([Vec,X]); PVec=np.vstack([PVec,P]);
                    for n in range(N-1):#N
                        def NI(x):
                            x1=(np.array(x)-np.array(X)-dtau/2*inv_met_mat(x,a,rq).dot(P)).tolist();
                            return x1
                        XP=fsolve(NI,X); [g,gi,drpg,dthpg]=inverse_metric_matrix(XP.tolist(),a,rq);
                        def NJ(p):
                            p1=(np.array(p)-np.array(P)+dtau/4*(np.array([0,np.inner(P,drpg.dot(P)),np.inner(P,dthpg.dot(P)),0])+np.array([0,np.inner(np.array(p),drpg.dot(np.array(p))),np.inner(np.array(p),dthpg.dot(np.array(p))),0]))).tolist();
                            return p1
                        #def NJp(p):
                        #    Ap=drpg.dot(P); Bp=dthpg.dot(P);
                        #    p1=np.array([[1,0,0,0],[dtau/2*Ap[0],1+dtau/2*Ap[1],dtau/2*Ap[2],dtau/2*Ap[3]],[dtau/2*Bp[0],dtau/2*Bp[1],1+dtau/2*Bp[2],dtau/2*Bp[3]],[0,0,0,1]])
                        #    return(p1)
                        P=fsolve(NJ,P); X=XP+dtau/2*gi.dot(P); Vec=np.vstack([Vec,X]); Ham.append(np.inner(P,gi.dot(P))); PVec=np.vstack([PVec,P]);
                    
                    Vecc=np.zeros((0,3)); HAM=[];
                    if Conserv==1:
                        HAM=Ham;
                    
                    for vee in range(np.shape(Vec)[0]):
                        ve=Vec[vee,:];
                        Vecc=np.vstack([Vecc,np.array([Rs/2*ve[1],ve[2],ve[3]])]);
                        if Conserv==1:
                            Pv=PVec[vee,:];
                            S=ve[1]**2+a**2*np.cos(ve[2])**2;
                            Car=Pv[2]**2+np.cos(ve[2])**2*(L0**2/np.sin(ve[2])**2-a**2*(E0**2-Mu));
                            CAR.append(Car);
    return [Vecc,HAM,CAR]