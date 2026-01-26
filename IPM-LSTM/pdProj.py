from __future__ import print_function

import aesara

import aesara.tensor as at
import numpy as np
import scipy
import time
from aesara.tensor.nlinalg import pinv
from aesara.tensor.basic import diag
from aesara.tensor.nlinalg import eigh
from aesara.tensor.slinalg import eigvalsh

try:
    FunctionType = aesara.compile.function_module.Function
except AttributeError:
    FunctionType = aesara.compile.function.types.Function

class pdProj:
    def __init__(self,prob=None,x0=None, y0=None, bl=None, bu=None, cl=None, cu=None, x_dev=None, s_dev=None, y_dev=None,
                 f=None, df=None, d2f=None, c=None, dc=None, d2c=None, muB0=1.0E-4, muL0=1.0, muP0=1.0E-4,
                 armijoTol=1.0E-3, muLfac=2, muPfac=10, muAfac=10, muBfac=10, gammaC=0.5, boundaryTol=0.8, jMmax=40, maxItns=500, 
                 tolStat=1.0E-4, tolIsp=1.0E-5, tolPrimFeas=1.0E-5, tolDualFeas=1.0E-5, infinity=1.0E15, eMax = 1.0E6, tolM=1.0E-1,
                 yMax=1.0E6, minPosEig=1.0E-8, unboundedf=-1.0E9, printLevel=1, float_dtype=np.float64, tol_res = 1.0E-1):
        
        self.prob = prob
        self.x0 = x0
        self.y0 = y0
        self.x_dev = x_dev
        self.y_dev = y_dev
        self.printLevel = printLevel

        self.n = 0
        self.m = 0
        if x0 is not None:
            self.n = x0.shape[0]
        if cl is not None:
            self.m = cl.shape[0]

        if bl is None:
            bl = -infinity * np.ones(self.n,dtype=float_dtype)
        if bu is None:
            bu = infinity * np.ones(self.n,dtype=float_dtype)
        self.bl = bl
        self.bu = bu
        self.cl = cl
        self.cu = cu
        if self.m > 0:
            self.bL = np.concatenate((bl,cl))
            self.bU = np.concatenate((bu,cu))
        else:
            self.bL = bl
            self.bU = bu

        self.f = f
        self.df = df
        self.d2f = d2f
        self.c = c
        self.dc = dc
        self.d2c = d2c
        self.infinity=infinity
        self.float_dtype = float_dtype

        # Find indices of the free/fixed variables and slacks.
        # A fixed slack indicates an equality constraint.
        self.jFreex = np.argwhere(bl!=bu).ravel()
        self.nFreex = self.jFreex.size
        self.jFixedx = np.argwhere(bl==bu).ravel()
        self.iFreec = np.argwhere(cl!=cu).ravel()
        self.nfreec = self.iFreec.size
        self.iFixedc = np.argwhere(cl==cu).ravel()
        self.jFixedc = self.iFixedc + self.n

        # Find jLo & jUp, indices of the free variables & slacks with lower/upper bound
        self.jxLo = np.argwhere((bl!=bu) & (bl>-infinity)).ravel()
        self.jxUp = np.argwhere((bl!=bu) & (bu<infinity)).ravel()
        self.jxLorU = np.argwhere((bl!=bu) & ((bl>-infinity)|(bu<infinity))).ravel()
        if self.m > 0:
            self.jcLo = np.argwhere((cl!=cu) & (cl>-infinity)).ravel() + self.n
            self.jcUp = np.argwhere((cl!=cu) & (cu<infinity)).ravel() +self.n
            self.icLorU = np.argwhere((cl!=cu) & ((cl>-infinity)|(cu<infinity))).ravel()
            self.jcLorU = self.icLorU + self.n
        else:
            self.jcLo = np.array([], dtype=int)
            self.jcUp = np.array([], dtype=int)
            self.icLorU = np.array([], dtype=int)
            self.jcLorU = np.array([], dtype=int)
        self.jLo = np.concatenate([self.jxLo, self.jcLo], axis=0)
        self.jUp = np.concatenate([self.jxUp, self.jcUp], axis=0)
        self.jLoA = np.array([], dtype=int)
        self.jUpA = np.array([], dtype=int)

        # Constants
        self.eps = np.finfo(float_dtype).eps
        self.maxItns = maxItns
        self.jMmax = jMmax
        self.eMax = eMax
        self.muPfac = muPfac
        self.muLfac = muLfac
        self.muAfac = muAfac
        self.muBfac = muBfac
        self.minPosEig = minPosEig
        self.boundaryTol = boundaryTol
        self.armijoTol = armijoTol
        self.gammaC = gammaC
        self.tolM = tolM
        self.yMax = yMax
        self.tolPrimFeas = tolPrimFeas
        self.tolDualFeas = tolDualFeas
        self.tol_res = tol_res
        self.unboundedf = unboundedf
        self.tolIsp = tolIsp

        # Constant vectors
        self.mOnes = np.ones(self.m,dtype=float_dtype)
        self.nmOnes = np.ones((self.n+self.m,),dtype=float_dtype)
        self.nmZeros = np.zeros(self.n+self.m,dtype=float_dtype)
        if self.m > 0:
            self.mZeros = np.zeros(self.m,float_dtype)
            self.mOnes =  np.ones(self.m,float_dtype)
        else:
            self.mZeros = np.array([], dtype=float_dtype)
            self.mOnes = np.array([], dtype=float_dtype)
            
        self.muB = muB0 * self.nmOnes
        self.muL = muL0 * self.mOnes
        self.muP = muP0 * self.mOnes
        self.muA = muB0 * self.nmOnes

        self.zeL = np.zeros(self.n+self.m,dtype=float_dtype)
        self.zeU = np.zeros(self.n+self.m,dtype=float_dtype)
        self.xeL = np.zeros(self.n+self.m,dtype=float_dtype)
        self.xeU = np.zeros(self.n+self.m,dtype=float_dtype)
        self.zL = np.zeros(self.n+self.m,dtype=float_dtype)
        self.zU = np.zeros(self.n+self.m,dtype=float_dtype)
        self.xL = np.zeros(self.n+self.m,dtype=float_dtype)
        self.xU = np.zeros(self.n+self.m,dtype=float_dtype)
        self.v = np.zeros(self.n,dtype=float_dtype)
        self.vL = np.zeros(self.n,dtype=float_dtype)
        self.vU = np.zeros(self.n,dtype=float_dtype)
        self.veL = np.zeros(self.n,dtype=float_dtype)
        self.veU = np.zeros(self.n,dtype=float_dtype)
        self.xLinv = np.zeros(self.n+self.m,dtype=float_dtype)
        self.xUinv = np.zeros(self.n+self.m,dtype=float_dtype)

        self.compiled = False

    def compile(self, nvar=None, nc=None):
        """ Compile the objective function if the formula is given,
           as well as the gradient, and the Hessian with constraints.
        """
        # Number of variables and constraints
        if nvar is not None:
            self.n = nvar
        if nc is not None:
            self.m = nc

        # Check if any functions are precompiled
        f_precompile = isinstance(self.f, FunctionType)
        df_precompile = isinstance(self.df, FunctionType)
        d2f_precompile = isinstance(self.d2f, FunctionType)
        c_precompile = isinstance(self.c, FunctionType)
        dc_precompile = isinstance(self.dc, FunctionType)
        d2c_precompile = isinstance(self.d2c, FunctionType)
        if any([f_precompile, df_precompile, d2f_precompile, c_precompile, dc_precompile, d2c_precompile]):
            precompile = True
        else:
            precompile = False

        # Declare device vectors
        if self.y_dev is None:
            self.y_dev = at.vector('y_dev')

        # Use automatic differentiation if expressions for gradient and/or Hessian of f are not provided
        if self.df is None:
            df = at.grad(self.f, self.x_dev)
        else:
            df = self.df
        if self.d2f is None:
            d2f = aesara.gradient.hessian(cost=self.f, wrt=self.x_dev)
        else:
            d2f = self.d2f

        # Construct expression for the constraint Jacobians and Hessians
        if self.m > 0:
            if self.dc is None:
                dc = aesara.gradient.jacobian(self.c, wrt=self.x_dev).reshape((self.m, self.n))
            else:
                dc = self.dc
            if self.d2c is None:
                d2c = aesara.gradient.hessian(cost=at.sum(self.c * self.y_dev), wrt=self.x_dev)
            else:
                d2c = self.d2c

        # If some expressions have been precompiled into functions, compile any remaining expressions
        if precompile:
            if not f_precompile:
                f_func = aesara.function(inputs=[self.x_dev], outputs=self.f)
            else:
                f_func = self.f
            if not df_precompile:
                df_func = aesara.function(inputs=[self.x_dev], outputs=df)
            else:
                df_func = self.df
            if not d2f_precompile:
                d2f_func = aesara.function(inputs=[self.x_dev], outputs=d2f)
            else:
                d2f_func = self.d2f
            if self.m > 0:
                if not c_precompile:
                    c_func = aesara.function(inputs=[self.x_dev], outputs=self.c.reshape((self.m,)))
                if not dc_compile:
                    dc_func = aesara.function(inputs=[self.x_dev], outputs=dc.reshape((self.m, self.n)))
                else:
                    def dc_func(x): return dc(x).reshape((self.m, self.n))
                if not d2c_precompile:
                    d2c_func = aesara.function(inputs=[self.x_dev, self.y_dev], outputs=d2c)
                else:
                    d2c_func = d2c

        # construct expression for initializing the Lagrange multipliers 
        if self.m > 0:
            if precompile:
                def init_yz(x):
                    if self.y0 is None:
                        ye = np.dot(np.linalg.pinv(dc_func(x)).T,df_func(x).reshape((self.n,1))).reshape((self.m,))
                    else:
                        ye = self.y0
                    ze = np.concatenate([df_func(x)-np.dot(dc_func(x).T,ye).reshape((self.n,)),ye], axis=0)
                    return ze
            else:
                init_yz = at.zeros((self.m+self.n,))
                if self.y0 is None:
                    ye = at.dot(pinv(dc).T, df.reshape(self.n,1)).reshape((self.m,))
                    init_yz = at.concatenate([df-at.dot(dc.T,ye).reshape((self.n,)),ye], axis=0)
                else:
                    init_yz = at.concatenate([df-at.dot(dc.T,self.y0).reshape((self.n,)),self.y0], axis=0)
        else:
            if precompile:
                def init_yz(x):
                    return df_func(x)
            else:
                init_yz = df


        # Construct expression for the Hessian of the Lagrangian
        if precompile:
            if self.m > 0:
                def d2L_func(x,y):
                    return d2f_func(x)-d2c_func(x,y)
            else:
                d2L_func = d2f_func
        else:
            if self.m > 0:
                d2L = d2f-d2c
            else:
                d2L = d2f
        
        # Compile expressions into device functions
        if precompile:
            self.obj = f_func
            self.gradx = df_func
            self.hess = d2L_func
        else:
            self.obj = aesara.function(inputs=[self.x_dev], outputs=self.f)
            self.gradx = aesara.function(inputs=[self.x_dev], outputs=df)
            if self.m > 0:
                self.hess = aesara.function(inputs=[self.x_dev,self.y_dev], outputs=d2L, on_unused_input='ignore')
            else:
                self.hess = aesara.function(inputs=[self.x_dev], outputs=d2L, on_unused_input='ignore')

        if self.m > 0:
            if precompile:
                self.init_yz = init_yz
                self.con = c_func
                self.jaco = dc_func
            else:
                self.init_yz = aesara.function(inputs=[self.x_dev], outputs=init_yz)
                self.con =  aesara.function(inputs=[self.x_dev], outputs=self.c.reshape((self.m,)))
                self.jaco = aesara.function(inputs=[self.x_dev], outputs=dc)
        elif precompile:
            self.init_yz = init_yz 
        else:
            self.init_yz = aesara.function(inputs=[self.x_dev], outputs=init_yz)

    def optCheck(self,x,y,g,J,zScale,cScale):
        """ Compute the optimality measures 
        """
        rxL = np.divide(x-self.bL, np.absolute(self.bL)+1)
        rxU = np.divide(self.bU-x, np.absolute(self.bU)+1)
        violL = np.minimum(0,rxL)
        violU = np.minimum(0,rxU)
        if self.m > 0:
            viol = np.array([np.linalg.norm(self.cs,ord=np.inf)/np.max([np.linalg.norm(x[self.n:],ord=np.inf),1])])
        else:
            viol = np.array([],dtype=self.float_dtype)
            
        compL = np.linalg.norm(np.multiply(self.zL,np.minimum(np.absolute(rxL),1)),ord=np.inf)
        compU = np.linalg.norm(np.multiply(self.zU,np.minimum(np.absolute(rxU),1)),ord=np.inf)

        z =  (g-np.dot(J.T,y))/zScale
        zxL = np.maximum(np.multiply(z,np.minimum(rxL,1)),0)
        zxU = np.maximum(np.multiply(-z,np.minimum(rxU,1)),0)

        maxViolBnds = np.linalg.norm(np.concatenate((violL,violU),axis=0),ord=np.inf)
        maxViolAll = np.linalg.norm(np.concatenate((violL,violU,viol),axis=0),ord=np.inf)
        maxComp = np.linalg.norm(np.array([compL,compU]),ord=np.inf)/zScale

        primInf =  np.linalg.norm(np.concatenate((violL,violU,viol),axis=0),ord=np.inf)
        dualInf = np.linalg.norm(np.concatenate([zxL,zxU],axis=0),ord=np.inf)

        zIsp = np.dot(J.T,self.cs)/cScale
        zxLIsp = np.maximum(np.multiply(zIsp,np.minimum(rxL,1)),0)
        zxUIsp = np.maximum(np.multiply(-zIsp,np.minimum(rxU,1)),0)
        ispDualInf = np.linalg.norm(np.concatenate([zxLIsp,zxUIsp],axis=0),ord=np.inf)
        
        return maxViolAll, maxComp, maxViolBnds, primInf, dualInf,ispDualInf        

    def merit(self,x,y,muB,muP,muA,muL):
        """ Calculate the merit function and its gradient
        """
        if self.m > 0:
            self.J = np.concatenate([self.Jx, -np.eye(self.m)], axis=1)
            self.cs = self.cx-x[self.n:]
        
        self.xL = np.zeros(self.n+self.m,dtype=self.float_dtype)
        self.xU = np.zeros(self.n+self.m,dtype=self.float_dtype)
        self.xLinv = np.zeros(self.n+self.m,dtype=self.float_dtype)
        self.xUinv = np.zeros(self.n+self.m,dtype=self.float_dtype)
        zLinv = np.zeros(self.n+self.m,dtype=self.float_dtype)
        zUinv = np.zeros(self.n+self.m,dtype=self.float_dtype)
        self.piL = np.zeros(self.n+self.m,dtype=self.float_dtype)
        self.piU = np.zeros(self.n+self.m,dtype=self.float_dtype)
        self.piZ = np.zeros(self.n+self.m,dtype=self.float_dtype)
        self.piVL = np.zeros(self.n,dtype=self.float_dtype)
        self.piVU = np.zeros(self.n,dtype=self.float_dtype)
        self.DBL = np.zeros(self.n+self.m,dtype=self.float_dtype)
        self.DBU = np.zeros(self.n+self.m,dtype=self.float_dtype)
        self.xL[self.jLo] = x[self.jLo]-self.bL[self.jLo]+muB[self.jLo]
        self.xU[self.jUp] = self.bU[self.jUp]-x[self.jUp]+muB[self.jUp]
        self.xLinv[self.jLo] = np.reciprocal(self.xL[self.jLo])
        self.xUinv[self.jUp] = np.reciprocal(self.xU[self.jUp])

        zLinv[self.jLo] = np.reciprocal(self.zL[self.jLo]+muB[self.jLo])
        zUinv[self.jUp] = np.reciprocal(self.zU[self.jUp]+muB[self.jUp])
        self.DBL[self.jLo] = np.multiply(zLinv[self.jLo],self.xL[self.jLo])
        self.DBU[self.jUp] = np.multiply(zUinv[self.jUp],self.xU[self.jUp])

        self.piY = self.ye-np.divide(self.cs,muP)
        
        self.piL[self.jLo] = np.prod(np.vstack([muB[self.jLo],self.xLinv[self.jLo],\
                                               self.zeL[self.jLo]+self.xeL[self.jLo]]),axis=0)-muB[self.jLo]
        self.piU[self.jUp] = np.prod(np.vstack([muB[self.jUp],self.xUinv[self.jUp],\
                                               self.zeU[self.jUp]+self.xeU[self.jUp]]),axis=0)-muB[self.jUp]

        self.piZ[self.jLo] = np.copy(self.piL[self.jLo])
        self.piZ[self.jUp] = self.piL[self.jUp]-self.piU[self.jUp]
        self.piZ[self.jFixedc] = np.copy(self.piY[self.iFixedc])
        
        self.piVL[self.jLoA] = self.veL[self.jLoA]\
            -np.divide(x[self.jLoA]-self.bL[self.jLoA],muA[self.jLoA])
        self.piVU[self.jUpA] = self.veU[self.jUpA]\
            -np.divide(self.bU[self.jUpA]-x[self.jUpA],muA[self.jUpA])
        self.piV = self.piVL-self.piVU

        self.vML =  np.zeros(self.n,dtype=self.float_dtype)
        self.vMU =  np.zeros(self.n,dtype=self.float_dtype)
        self.vML[self.jLoA] = self.piVL[self.jLoA]*2-self.vL[self.jLoA]
        self.vMU[self.jUpA] = self.piVU[self.jUpA]*2-self.vU[self.jUpA]
        self.vM = self.vML-self.vMU
        
        self.yM = self.piY*2-y

        self.zML = np.zeros(self.n+self.m,dtype=self.float_dtype)
        self.zMU = np.zeros(self.n+self.m,dtype=self.float_dtype)
        self.zML[self.jLo] = self.piL[self.jLo]*2-self.zL[self.jLo]
        self.zMU[self.jUp] = self.piU[self.jUp]*2-self.zU[self.jUp]
        
        self.zM = np.zeros(self.n+self.m,dtype=self.float_dtype)
        self.zM[self.jLo] = np.copy(self.zML[self.jLo])
        self.zM[self.jUp] = self.zML[self.jUp]-self.zMU[self.jUp]
        self.zM[self.jFixedc] = np.copy(self.yM[self.iFixedc])
        
        muPTerm = np.sum(np.divide(np.square(self.cs),muP))\
            + np.sum(np.divide(np.square(self.cs+np.multiply((y-self.ye),muP)),muP))\
            + np.sum(np.divide(np.square(x[self.jLoA]-self.bL[self.jLoA]),muA[self.jLoA]))\
            + np.sum(np.divide(np.square(self.bU[self.jUpA]-x[self.jUpA]),muA[self.jUpA]))\
            + np.sum(np.multiply(np.square(np.divide(x[self.jLoA]-self.bL[self.jLoA],muA[self.jLoA])+
                                           self.vL[self.jLoA]-self.veL[self.jLoA]),muA[self.jLoA]))\
            + np.sum(np.multiply(np.square(np.divide(self.bU[self.jUpA]-x[self.jUpA],muA[self.jUpA])+
                                           self.vU[self.jUpA]-self.veU[self.jUpA]),muA[self.jUpA]))
        
        muLTerm = np.sum(np.divide(np.square(self.cs),muL))\
            + np.sum(np.divide(np.square(self.cs+np.multiply((y-self.ye),muL)),muL))\
            + np.sum(np.divide(np.square(x[self.jLoA]-self.bL[self.jLoA]),muA[self.jLoA]))\
            + np.sum(np.divide(np.square(self.bU[self.jUpA]-x[self.jUpA]),muA[self.jUpA]))\
            + np.sum(np.multiply(np.square(np.divide(x[self.jLoA]-self.bL[self.jLoA],muA[self.jLoA])+
                                           self.vL[self.jLoA]-self.veL[self.jLoA]),muA[self.jLoA]))\
            + np.sum(np.multiply(np.square(np.divide(self.bU[self.jUpA]-x[self.jUpA],muA[self.jUpA])+
                                           self.vU[self.jUpA]-self.veU[self.jUpA]),muA[self.jUpA]))  

        muBTerm = -np.sum(np.prod(np.vstack([muB[self.jLo],self.zeL[self.jLo]+self.xeL[self.jLo],\
                                             np.log(self.zL[self.jLo]+muB[self.jLo])+2*np.log(self.xL[self.jLo])]),axis=0))\
                  -np.sum(np.prod(np.vstack([muB[self.jUp],self.zeU[self.jUp]+self.xeU[self.jUp],\
                                             np.log(self.zU[self.jUp]+muB[self.jUp])+2*np.log(self.xU[self.jUp])]),axis=0))\
                  +np.sum(np.multiply(self.zL[self.jLo],self.xL[self.jLo]))\
                  +2*np.sum(np.multiply(muB[self.jLo],x[self.jLo]-self.bL[self.jLo]))\
                  +np.sum(np.multiply(self.zU[self.jUp],self.xU[self.jUp]))\
                  +2*np.sum(np.multiply(muB[self.jUp],self.bU[self.jUp]-x[self.jUp]))
        
        fLag = self.fx-np.inner(self.cs,self.ye)\
            -np.sum(np.multiply(x[self.jLoA]-self.bL[self.jLoA],self.veL[self.jLoA]))\
            -np.sum(np.multiply(self.bU[self.jUpA]-x[self.jUpA],self.veU[self.jUpA]))
        
        fMmuL = fLag+muLTerm/2+muBTerm
        fMmuP = fLag+muPTerm/2+muBTerm
        gMmuP_x = self.gx-np.dot(self.Jx.T,self.yM)-self.vM-self.zM[:self.n]
        gMmuP_s = self.yM-self.zM[self.n:]
        gMmuP_y = np.multiply(muP,y-self.piY)
        gMmuP_vL = np.multiply(muA[self.jLoA],self.vL[self.jLoA]-self.piVL[self.jLoA])
        gMmuP_vU = np.multiply(muA[self.jUpA],self.vU[self.jUpA]-self.piVU[self.jUpA])
        gMmuP_zL = np.multiply(self.DBL[self.jLo],self.zL[self.jLo]-self.piL[self.jLo])
        gMmuP_zU = np.multiply(self.DBU[self.jUp],self.zU[self.jUp]-self.piU[self.jUp])
        gMmuP = np.concatenate([gMmuP_x,gMmuP_s,gMmuP_y,gMmuP_vL,gMmuP_vU,gMmuP_zL,gMmuP_zU],axis=0)

        return fMmuL, fMmuP, gMmuP


    def scaleSym(self,A,sclTol=1.0E-3):
        """ Scale an ill-conditioned system
        """
        Done = False
        itns = 0
        sclItns = 20
        n = np.size(A,1)
        S = np.ones(n,dtype=self.float_dtype)

        while itns <= sclItns:
            itns += 1
            D = np.amax(np.absolute(A),axis=0)
            D[np.argwhere(D<=0)]=1
            tol = np.amax(np.absolute(1-D))
            if tol <= sclTol or itns >= sclItns:
                Done = True

            D = np.sqrt(np.reciprocal(D))
            A = np.diagflat(D) @ A @ np.diagflat(D)
            S = np.multiply(S,D)

            if Done:
                break
        return S,A,itns       
        
    def kktSolver(self,K,rhs,n,m,minPosEig,Hmod):
        """ Regularize and solve the KKT system
        """
        KL = 1/3
        KU = 8
        Hmin = 1.0E-20
        Hmax = 1.0E40
        Hmod0 = 1.0E-1

        nOnes = np.ones(n,dtype=self.float_dtype)

        Hmods = 0
        lu, D, perm = scipy.linalg.ldl(K)

        eigs, v = scipy.linalg.eig(D)
        maxEigK = np.max(np.absolute(eigs))
        minEigK = np.min(np.absolute(eigs))
        condD = maxEigK/np.max([minEigK,minPosEig])

        numPos = np.argwhere(eigs>minPosEig*maxEigK).size
        numNeg = np.argwhere(eigs<-minPosEig*maxEigK).size
        numSing = n+m-numPos-numNeg

        K0 = np.copy(K[:n,:n])
        
        if numPos == n and numNeg == m:
            exitInfo = 0
        else:
            if numSing > 0:
                exitInfo = 1
            else:
                exitInfo = 2

            if Hmod == 0:
                Hmod = Hmod0
            else:
                Hmod = np.max([Hmin,KL*Hmod])
            
            convexified = False
            while not convexified:
                Hmods += 1
                
                K[:n,:n] = K0+np.diagflat(Hmod*nOnes)

                lu, D, perm = scipy.linalg.ldl(K)
                eigs, v = scipy.linalg.eig(D)
                maxEigK = np.max(np.absolute(eigs))
                minEigK = np.min(np.absolute(eigs))
                condD = maxEigK/np.max([minEigK,minPosEig])

                numPos = np.argwhere(eigs>0).size
                numNeg = np.argwhere(eigs<0).size
                numSing = n+m-numPos-numNeg
                if numPos == n and numNeg == m:
                    convexified = True
                else:
                    if KU*Hmod > Hmax:
                        break
                    Hmod = KU*Hmod
            if not convexified:
                exitInfo = -1
                
        if condD > self.eps**(-2/3):
            try:
                S,K,itns = self.scaleSym(K)
                d = np.multiply(np.linalg.solve(K,np.multiply(S,rhs)),S)
            except:
                if Hmod == 0:
                    Hmod = Hmod0
                else:
                    Hmod = np.min([KU*Hmod,Hmax])
                Hmods += 1
                K[:n,:n] = K0+np.diagflat(Hmod*nOnes)
                S,K,itns = self.scaleSym(K)
                d = np.multiply(np.linalg.solve(K,np.multiply(S,rhs)),S)                
        else:
            try:
                d = np.linalg.solve(K,rhs)
            except:
                if Hmod == 0:
                    Hmod = Hmod0
                else:
                    Hmod = np.min([KU*Hmod,Hmax])
                Hmods += 1
                K[:n,:n] = K0+np.diagflat(Hmod*nOnes)
                d = np.linalg.solve(K,rhs)
        
        return d,Hmod,Hmods,exitInfo
        
        

    def fqArmijoProj(self, step, vec, dvec, vbL, vbU, F, maxNormF, dzX, fMmuP, fMmuL, gMmuP,\
                     stepMax=1.0E15, maxMerit=1.0E12):
        """Flexible quasi-Armijo search to determine a step length.
           iExit     Result
           -----    -----------------------------
             1      The search is successful.
             2      A better point was found but no sufficient decrease.
             3      Too many function calls.
             4      No descent direction.
        """
        armijoTol = self.armijoTol
        gammaC = self.gammaC
        jMmax = self.jMmax
        step = np.min([step,stepMax])
        fMmuP0 = fMmuP
        fMmuL0 = fMmuL
        gMmuP0 = gMmuP
        F0 = F
        z0 = np.copy(self.z)
        vec0 = np.copy(vec)
        muPused = False
        normF0 = np.linalg.norm(F0,ord=np.inf)
        gMv = np.dot(gMmuP,dvec)
        
        if gMv >= 0:
            jfM = 0
            iExit = 4
        else:
            jfM = 0
            while jfM < jMmax:
                vec = vec0+step*dvec
                for j in range(vec.size):
                    vec[j] = np.min([np.max([vec[j],vbL[j]]),vbU[j]])

                if self.prob is not None:
                    self.fx, self.gx = self.prob.obj(vec[:self.n], gradient=True)
                else:
                    self.fx = self.obj(vec[:self.n])
                    self.gx = self.gradx(vec[:self.n])
                if self.m > 0:
                    if self.prob is not None:
                        self.cx, self.Jx = self.prob.cons(vec[:self.n], gradient=True)
                    else:
                        self.cx = self.con(vec[:self.n])
                        self.Jx = self.jaco(vec[:self.n])
                jfM += 1
                
                if np.sum(np.isnan(self.fx)+np.isinf(self.fx))+np.sum(np.isnan(self.gx)+np.isinf(self.gx)) == 0:
                    self.vL[self.jLoA] = np.copy(vec[self.lvL:self.lvU])
                    self.vU[self.jUpA] = np.copy(vec[self.lvU:self.lzL])
                    self.zL[self.jLo] = np.copy(vec[self.lzL:self.lzU])
                    self.zU[self.jUp] = np.copy(vec[self.lzU:])

                    self.v = self.vL-self.vU
                    self.z[self.jFixedx] = z0[self.jFixedx]+step*dzX[self.jFixedx]
                    self.z[self.jFixedc] = z0[self.jFixedc]+step*dzX[self.jFixedc]
                    self.z[self.jUp] = 0
                    self.z[self.jLo] = np.copy(self.zL[self.jLo])
                    self.z[self.jUp] = self.z[self.jUp]-self.zU[self.jUp]
                    
                    fMmuL, fMmuP, gMmuP = self.merit(vec[:self.ly],vec[self.ly:self.lvL],self.muB,self.muP,self.muA,self.muL)
 
                    F = np.concatenate([self.gx-np.dot(self.Jx.T,vec[self.ly:self.lvL])-self.v-self.z[:self.n],\
                           vec[self.ly:self.lvL]-self.z[self.n:],\
                           np.multiply(self.muP,vec[self.ly:self.lvL]-self.piY),\
                           np.multiply(self.muA[self.jLoA],self.vL[self.jLoA]-self.piVL[self.jLoA]),\
                           np.multiply(self.muA[self.jUpA],self.vU[self.jUpA]-self.piVU[self.jUpA]),\
                           np.multiply(self.xL[self.jLo],self.zL[self.jLo]-self.piL[self.jLo]),\
                           np.multiply(self.xU[self.jUp],self.zU[self.jUp]-self.piU[self.jUp])], axis=0)
                    
                    normF = np.linalg.norm(F,ord=np.inf)
                    if normF <= 0.9*np.min([normF0,maxNormF]) and fMmuP < np.max([fMmuP0,maxMerit])\
                       and fMmuL < np.max([fMmuL0,maxMerit]):
                        maxNormF = 0.9*maxNormF
                        break
                    elif fMmuL <= fMmuL0+armijoTol*step*gMv:
                        break
                    if fMmuP <= fMmuP0+armijoTol*step*gMv:
                        muPused = True
                        break
                if jfM < jMmax:
                    step = step* gammaC

            if  jfM <= jMmax:
                iExit = 1
            elif fMmuP < fMmuP0:
                iExit = 2
            else:
                iExit = 3

        return step,vec,fMmuP,fMmuL,gMmuP,jfM,F,iExit,muPused,maxNormF
        

    def solve(self, x0=None, force_recompile=False):
        """Main solver function that initiates and controls the iteraions
        """
        _time = time.time()
        
        if x0 is not None:
            self.x0 = x0

        # x variables must be initialized and have length greater than zero
        assert (self.x0 is not None) and (self.x0.size > 0)
        # x0 should be a one-dimensional array
        assert self.x0.size == self.x0.shape[0]
        # set the variable counter equal to the number of x variables
        self.n = self.x0.size
        # cast x0 to float_dtype
        self.x0 = self.float_dtype(self.x0)
        # validate class members
        #self.validate()
        
        # if expressions are not compiled or force_recompile=True, compile expressions into functions
        if (not self.compiled or force_recompile) and self.prob is None:
            self.compile()

        # Initialize the run statistics
        nf = 0
        itn = 0
        CvEitns = 0
        Eitns = 0
        Oitns = 0
        Mitns = 0
        Fitns = 0

        Hmod = 0
        Hmods = 0
        outcome = ''
        status = 0

        step = 0
        normdx = 0

        convexity = ''
        
        # Initialize x variables
        x = self.nmZeros
        x[:self.n] = np.copy(self.x0)
        x[self.jFixedx] = np.copy(self.bL[self.jFixedx])
        pert = 1
        for i in self.jFreex:
             if self.bL[i] > -self.infinity:
                pertL = np.max([self.muB[i]/(abs(self.bL[i])+1),pert])
                if self.bU[i] < self.infinity:
                    pertU = np.max([self.muB[i]/(abs(self.bU[i])+1),pert])
                    xMid = (self.bL[i]+self.bU[i])/2
                    if xMid < self.bL[i]+pertL or xMid > self.bU[i]-pertU:
                        x[i] = xMid
                    elif x[i] < xMid:
                        x[i] = np.max([self.bL[i]+pertL,x[i]])
                    else:
                        x[i] = np.min([self.bU[i]-pertU,x[i]])
                else:
                    x[i] = np.max([self.bL[i]+pertL,x[i]])
             elif self.bU[i] < self.infinity:
                pertU = np.max([self.muB[i]/(abs(self.bU[i])+1),pert])
                x[i] = np.min([self.bU[i]-pertU,x[i]])

                
        # Compute the problem functions and gradients
        if self.prob is not None:
            self.fx, self.gx = self.prob.obj(x[:self.n],gradient=True)
        else:
            self.fx = self.obj(x[:self.n])
            self.gx = self.gradx(x[:self.n])

        # Initialize slacks and multipliers  
        if self.m > 0:
            if self.prob is not None:
                self.cx, self.Jx = self.prob.cons(x[:self.n], gradient=True)
            else:
                self.cx = self.con(x[:self.n])
                self.Jx = self.jaco(x[:self.n])
            x[self.n:] = np.copy(self.cx)
            x[self.jFixedc] = np.copy(self.cl[self.iFixedc])
            for i in self.iFreec:
                j = self.n + i
                if self.cl[i] > -self.infinity:
                    pertL = np.max([self.muB[j]/(abs(self.cl[i])+1),pert])
                    if self.cu[i] < self.infinity:
                        pertU = np.max([self.muB[j]/(abs(self.cu[i])+1),pert])
                        sMid = (self.cl[i]+self.cu[i])/2
                        if sMid < self.cl[i]+pertL or sMid > self.cu[i]-pertU:
                            x[j] = sMid
                        elif x[j] < sMid:
                            x[j] = np.max([self.cl[i]+pertL,x[j]])
                        else:
                            x[j] = np.min([self.cu[i]-pertU,x[j]])
                    else:
                        x[j] = np.max([self.cl[i]+pertL,x[j]])
                elif self.cu[i] < self.infinity:
                    pertU = np.max([self.muB[j]/(abs(self.cu[i])+1),pert])
                    x[j] = np.min([self.cu[i]-pertU,x[j]])
            if self.prob is not None:
                self.ze = np.concatenate([self.gx-np.dot(self.Jx.T,self.y0),self.y0])
            else:
                self.ze = self.init_yz(x[:self.n])
            self.ye = np.copy(self.ze[self.m:])
        else:
            self.cx = np.array([], dtype=self.float_dtype)
            self.Jx = np.array([], dtype=self.float_dtype)
            self.ye = np.array([], dtype=self.float_dtype)
            if self.prob is not None:
                self.ze =self.gx
            else:
                self.ze = self.init_yz(x[:self.n])

        self.zeL[self.jLo] = pert*np.maximum(self.ze[self.jLo], self.nmOnes[self.jLo])
        self.zeU[self.jUp] = -pert*np.minimum(self.ze[self.jUp], -self.nmOnes[self.jUp])
        self.xe = np.maximum(np.minimum(x, self.bU), self.bL)
        self.xeL[self.jLo] = np.maximum(x[self.jLo]-self.bL[self.jLo]+self.muB[self.jLo],self.muB[self.jLo])
        self.xeU[self.jUp] = np.maximum(self.bU[self.jUp]-x[self.jUp]+self.muB[self.jUp],self.muB[self.jUp])

        self.zL[self.jLo] = np.copy(self.zeL[self.jLo])
        self.zU[self.jUp] = np.copy(self.zeU[self.jUp])
        self.z = self.zL-self.zU
        self.z[self.jFixedx] = np.copy(self.ze[self.jFixedx])
        y = np.copy(self.z[self.n:])
        self.ye = np.copy(self.z[self.n:])
        
        if self.m > 0:
            self.cs = self.cx-x[self.n:]
            self.J = np.concatenate([self.Jx, -np.eye(self.m)], axis=1)
            g = np.concatenate([self.gx,self.mZeros],axis=0)
        else:
            self.cs = np.array([], dtype=self.float_dtype)
            self.J = np.array([], dtype=self.float_dtype)
            g = np.copy(self.gx)
        nf += 1

        normx0 = np.linalg.norm(x,ord=np.inf)
        if self.m > 0:
            normJx = np.linalg.norm(self.Jx,ord=np.inf)
            normcx = np.linalg.norm(self.cx,ord=2)
        else:
            normJx = 0
            normcx = 0
        normgx = np.linalg.norm(self.gx,ord=2)
        
        zScale = np.max([1,normgx,normJx*np.max([1,np.linalg.norm(y,ord=2)])])
        cScale = np.max([1,normcx*normJx])
        maxViolAll,maxComp,maxViolBnds,primInf,dualInf, ispDualInf= self.optCheck(x,y,g,self.J,zScale,cScale)
        phiMax = np.max([maxViolAll+maxComp+10,1000])
        
        maxNormF = 1.0E8

        tjfixedcL = np.array([], dtype=int)
        tjfixedcU = np.array([], dtype=int)
        jLo0 = np.copy(self.jLo)
        jUp0 = np.copy(self.jUp)

        while True:
            # Reinitialize multipliers if infeasible
            if np.any(self.zL[self.jLo]+self.muB[self.jLo]<=self.eps) \
               or np.any(self.zU[self.jUp]+self.muB[self.jUp]<=self.eps):
                v0 = np.concatenate([self.v, self.mZeros],axis=0)
                indicesLo = self.jLo[np.argwhere(self.zL[self.jLo]+self.muB[self.jLo]<=self.eps).ravel()]
                if indicesLo.size > 0:
                    self.zL[indicesLo] = self.zL[indicesLo]/self.muBfac
                    try:
                        self.zL[indicesLo] = np.maximum(self.zL[indicesLo],g[indicesLo]-np.dot(self.J[:,indicesLo].T,y)\
                                             -v0[indicesLo]+self.zU[indicesLo])
                    except:
                        self.zL[indicesLo] = np.maximum(self.zL[indicesLo],g[indicesLo]\
                                             -v0[indicesLo]+self.zU[indicesLo])
                indicesUp = self.jUp[np.argwhere(self.zU[self.jUp]+self.muB[self.jUp]<=self.eps).ravel()]
                if indicesUp.size > 0:
                    self.zU[indicesUp] = self.zU[indicesUp]/self.muBfac
                    try:
                        self.zU[indicesUp] = np.maximum(self.zU[indicesUp],self.zL[indicesUp],-g[indicesUp]\
                                             +np.dot(self.J[:,indicesUp].T,y)+v0[indicesUp])
                    except:
                        self.zU[indicesUp] = np.maximum(self.zU[indicesUp],self.zL[indicesUp],-g[indicesUp]+v0[indicesUp])
            fMmuL, fMmuP, gMmuP = self.merit(x,y,self.muB,self.muP,self.muA,self.muL)

            F = np.concatenate([self.gx-np.dot(self.Jx.T,y)-self.v-self.z[:self.n],\
                           y-self.z[self.n:],np.multiply(self.muP,y-self.piY),\
                           np.multiply(self.muA[self.jLoA],self.vL[self.jLoA]-self.piVL[self.jLoA]),\
                           np.multiply(self.muA[self.jUpA],self.vU[self.jUpA]-self.piVU[self.jUpA]),\
                           np.multiply(self.xL[self.jLo],self.zL[self.jLo]-self.piL[self.jLo]),\
                           np.multiply(self.xU[self.jUp],self.zU[self.jUp]-self.piU[self.jUp])], axis=0)
            if itn == 0:
                fM0 = fMmuP

            if self.m > 0:
                if self.prob is not None:
                    self.Hx = self.prob.hess(x[:self.n],-y)
                else:
                    self.Hx = self.hess(x[:self.n],-y)
            else:
                if self.prob is not None:
                    self.Hx = self.prob.hess(x[:self.n])
                else:
                    self.Hx = self.hess(x[:self.n])
  
            # Test for optimality
            zScale = np.max([1,normgx,normJx*np.max([1,np.linalg.norm(y,ord=2)])])
            cScale = np.max([1,normcx*normJx])
            maxViolAll,maxComp,maxViolBnds,primInf,dualInf,ispDualInf = self.optCheck(x,y,g,self.J,zScale,cScale)
            raw_stationarity = np.linalg.norm(F, ord=2)

            if primInf <= self.tolPrimFeas and dualInf <= self.tolDualFeas:
                status = 1
            elif raw_stationarity < self.tol_res:
                status = 1
            elif primInf <= self.tolPrimFeas and self.fx <= self.unboundedf:
                status = 3
            elif maxViolAll >= self.tolPrimFeas and maxViolBnds <= self.tolPrimFeas\
                 and ispDualInf <= primInf*self.tolIsp\
                 and step*normdx <= (1+normx0)*1.0E-12 and itn > 0:
                status = 2
                
            # Print iteration details
            if self.printLevel>0:
                msg = []
                msg.append('Itn={}'.format(itn))
                msg.append('Nf={}'.format(nf))
                msg.append('Objective={}'.format(self.fx))
                msg.append('primInf={}'.format(primInf))
                msg.append('dualInf={}'.format(dualInf))
                msg.append('Res_norm={}'.format(raw_stationarity))
                print(', '.join(msg))


            # Test for termination
            Terminate = (status > 0 or itn > self.maxItns-1)
            if Terminate:
                if status == 1:
                    outcome = 'Converged'
                elif status == 2:
                    outcome = 'Infeasible stationary point'
                elif status == 3:
                    outcome = 'Unbounded problem'
                elif itn > self.maxItns-1:
                    status = 4
                    outcome = 'Too many iterations'
                else:
                    status =5
                break
            itn += 1

            #====================================================================
            #Compute the scaled  KKT matrix and the primal-dual search direction
            #====================================================================
            
            ADA = np.zeros(self.n,dtype=self.float_dtype)
            DZinv = np.zeros(self.n,dtype=self.float_dtype)
            DW = np.zeros(self.m, dtype=self.float_dtype)
            DWinv = np.zeros(self.m, dtype=self.float_dtype)
            
            ADA[self.jLoA] = np.reciprocal(self.muA[self.jLoA])
            ADA[self.jUpA] = np.reciprocal(self.muA[self.jUpA])
            DY = np.multiply(self.muP, self.mOnes)
            DWinv[self.icLorU] = np.multiply(self.xLinv[self.jcLorU],self.zL[self.jcLorU]+self.muB[self.jcLorU])\
                +np.multiply(self.xUinv[self.jcLorU],self.zU[self.jcLorU]+self.muB[self.jcLorU])
            DW[self.icLorU] = np.reciprocal(DWinv[self.icLorU])
            DZinv[self.jxLo] = np.multiply(self.xLinv[self.jxLo],self.zL[self.jxLo]+self.muB[self.jxLo])
            DZinv[self.jxUp] = DZinv[self.jxUp]+np.multiply(self.xUinv[self.jxUp],self.zU[self.jxUp]+self.muB[self.jxUp])

            yScale2 = np.ones(self.m,dtype=self.float_dtype)
            yScale = np.ones(self.m,dtype=self.float_dtype)
            yScale2[self.icLorU] = np.copy(DWinv[self.icLorU])
            yScale[self.icLorU] = np.sqrt(DWinv[self.icLorU])
            Iy = np.zeros(self.m,dtype=self.float_dtype)
            Iy[self.icLorU] = np.copy(self.mOnes[self.icLorU])

            Hk = np.copy(self.Hx[np.ix_(self.jFreex,self.jFreex)])
            Dk1= np.diagflat(DZinv[self.jFreex]+ADA[self.jFreex])
            try:
                Jk = np.matmul(np.diagflat(yScale),self.Jx[:,self.jFreex])
                Dk2 = np.diagflat(np.multiply(yScale2,DY)+Iy)
                K = np.block([[Hk+Dk1,Jk.T],[Jk,-Dk2]])
                rhs1 = self.gx[self.jFreex]-np.dot(self.Jx[:,self.jFreex].T,y)-self.piZ[self.jFreex]-self.piV[self.jFreex]
            except:
                K = Hk+Dk1
                rhs1 = self.gx[self.jFreex]-self.piZ[self.jFreex]-self.piV[self.jFreex]
            rhs2 = np.multiply(yScale,np.multiply(DY,y-self.piY)+np.multiply(DW,y-self.piZ[self.n:]))
            rhs = -np.concatenate([rhs1,rhs2],axis=0)
            dfree,Hmod,Hmods,exitInfo = self.kktSolver(K,rhs,self.n,self.m,self.minPosEig,Hmod)
            if exitInfo>1:
                convexity = 'Nonconvex'
            if self.m > 0:
                dfree[self.nFreex:(self.nFreex+self.m)] = -np.multiply(yScale,dfree[self.nFreex:(self.nFreex+self.m)])
            d = np.zeros(self.n+self.m,dtype=self.float_dtype)
            d[self.jFreex] = np.copy(dfree[:self.nFreex])
            d[self.n:(self.n+self.m)] = np.copy(dfree[self.nFreex:(self.nFreex+self.m)])
            dy = np.copy(d[self.n:(self.n+self.m)])
            yStep = y+dy
            dx = np.zeros(self.n+self.m,dtype=self.float_dtype)
            dx[:self.n] = np.copy(d[:self.n])
            dx[self.n:] = -np.multiply(DW,yStep-self.piZ[self.n:])
            xStep = x+dx
            normdx = np.linalg.norm(dx[:self.n],ord=np.inf)
            dzL =  np.zeros(self.n+self.m,dtype=self.float_dtype)
            dzU =  np.zeros(self.n+self.m,dtype=self.float_dtype)
            dzL[self.jLo] = -np.multiply(self.xLinv[self.jLo],np.multiply(self.zL[self.jLo],xStep[self.jLo]-self.bL[self.jLo])\
                                         +np.multiply(self.muB[self.jLo],self.zL[self.jLo]-self.zeL[self.jLo]+xStep[self.jLo]-self.xe[self.jLo]))
            dzU[self.jUp] = -np.multiply(self.xUinv[self.jUp],np.multiply(self.zU[self.jUp],self.bU[self.jUp]-xStep[self.jUp])\
                                         +np.multiply(self.muB[self.jUp],self.zU[self.jUp]-self.zeU[self.jUp]-xStep[self.jUp]+self.xe[self.jUp]))
            dzX = np.zeros(self.n+self.m,dtype=self.float_dtype)
            dzX[self.jFixedc] = yStep[self.iFixedc]-self.z[self.jFixedc]
            try:
                dzX[self.jFixedx] = self.gx[self.jFixedx]+np.dot(self.Hx[self.jFixedx,:],dx[:self.n])-np.dot(self.Jx[:,self.jFixedx].T,yStep)-self.z[self.jFixedx]
            except:
                dzX[self.jFixedx] = self.gx[self.jFixedx]+np.dot(self.Hx[self.jFixedx,:],dx[:self.n])-self.z[self.jFixedx]
            dvL = np.zeros(self.n,dtype=self.float_dtype)
            dvU = np.zeros(self.n,dtype=self.float_dtype)
            dvL[self.jLoA] = self.veL[self.jLoA]-np.divide(xStep[self.jLoA]-self.bL[self.jLoA],self.muA[self.jLoA])-self.vL[self.jLoA]
            dvU[self.jUpA] = self.veU[self.jUpA]-np.divide(self.bU[self.jUpA]-xStep[self.jUpA],self.muA[self.jUpA])-self.vU[self.jUpA]

            if exitInfo != 0:
                CvEitns += 1
                if exitInfo == -1:
                    outcome = 'No modification'
                    break
                else:
                    if Eitns == 0:
                        stepSum = 1
                    Eitns += 1
            else:
                Eitns = 0
                stepSum = 0

            #=================================================================
            # Compute the flexible quasi-Armijo step
            #=================================================================
            
            if Eitns > 0:
                step0 = 10*stepSum/Eitns
            else:
                step0 = 1

            step = np.min([1,step0])
            #stepMax = self.infinity

            vec = np.concatenate([x,y,self.vL[self.jLoA],self.vU[self.jUpA],self.zL[self.jLo],self.zU[self.jUp]],axis=0)
            dvec = np.concatenate([dx,dy,dvL[self.jLoA],dvU[self.jUpA],dzL[self.jLo],dzU[self.jUp]],axis=0)
 
            muPreduced = False
            gMmuP0 = np.inner(gMmuP,dvec)

            # Construct pointers to the array of primal-dual variables vec
            self.ly = self.n+self.m
            self.lvL = self.ly+self.m
            self.lvU = self.lvL+self.jLoA.size
            self.lzL = self.lvU+self.jUpA.size
            self.lzU = self.lzL+self.jLo.size

            # Lower and upper bounds
            vbL = np.concatenate([self.bL,-self.infinity*np.ones(self.m+self.jLoA.size+self.jUpA.size,dtype=self.float_dtype),\
                                  -self.muB[self.jLo],-self.muB[self.jUp]],axis=0)
            vbU =  np.concatenate([self.bU,self.infinity*np.ones(self.m+self.jLoA.size+self.jUpA.size,dtype=self.float_dtype),\
                                   self.infinity*np.ones(self.jLo.size,dtype=self.float_dtype),\
                                   self.infinity*np.ones(self.jUp.size,dtype=self.float_dtype)],axis=0)
            vbL[self.jLo] = self.bL[self.jLo]-self.muB[self.jLo]
            vbU[self.jUp] = self.bU[self.jUp]+self.muB[self.jUp]
            vbL[self.jLo] = np.minimum(vec[self.jLo]-self.boundaryTol*(vec[self.jLo]-vbL[self.jLo]),self.bL[self.jLo])
            vbU[self.jUp] = np.maximum(vec[self.jUp]+self.boundaryTol*(vbU[self.jUp]-vec[self.jUp]),self.bU[self.jUp])
            vbL[self.lzL:] = np.minimum(vec[self.lzL:]-self.boundaryTol*(vec[self.lzL:]-vbL[self.lzL:]),0)

            if gMmuP0 >= 0:
                outcome = 'No descent direction'
                status = 5
                break
            
            step,vec,fMmuP,fMmuL,gMmuP,jfM,F,iExit,muPused,maxNormF = self.fqArmijoProj(step,vec,dvec,\
                                                            vbL,vbU,F,maxNormF,dzX,fMmuP,fMmuL,gMmuP)

            x = np.copy(vec[:self.ly])
            y = np.copy(vec[self.ly:self.lvL])
            self.vL[self.jLoA] = np.copy(vec[self.lvL:self.lvU])
            self.vU[self.jUpA] = np.copy(vec[self.lvU:self.lzL])
            self.zL[self.jLo] = np.copy(vec[self.lzL:self.lzU])
            self.zU[self.jUp] = np.copy(vec[self.lzU:])
            
            if self.m > 0:
                self.J[:,:self.n] = self.Jx
                normcx = np.linalg.norm(self.cx,ord=2)
                normJx = np.linalg.norm(self.Jx,ord=np.inf)
                g = np.concatenate([self.gx,self.mZeros],axis=0)
            else:
               g = np.copy(self.gx) 
            normgx = np.linalg.norm(self.gx,ord=2)

            nf += jfM

            if iExit > 3:
                outcome = 'Linesearch failure'
                status = 5
                break
        
            # Update sum of step lengths asscociated with a sequence of modified KKTs
            if Eitns == 1:
                stepSum = step
            elif Eitns > 1:
                stepSum += step

            # Reset slack variables
            if self.jcLorU.size > 0:
                if muPused:
                    shat = self.cx-np.multiply(self.muP,self.ye+0.5*(self.z[self.n:]-y)+self.muB[self.n:])
                else:
                    shat = self.cx-np.multiply(self.muL,self.ye+0.5*(self.z[self.n:]-y)+self.muB[self.n:])
                for j in self.jcLorU:
                    i = j-self.n
                    if self.bL[j] > -self.infinity and self.bU[j] >= self.infinity:
                        x[j] = np.max([x[j],shat[i]])
                    elif self.bL[j] <= -self.infinity and self.bU[j] < self.infinity:
                        x[j] = np.min([x[j],shat[i]])
            self.cs = self.cx-x[self.n:]

            #=======================================================================
            # Update parameters if necessary
            #=======================================================================
            zScale = np.max([1,normgx,normJx*np.max([1,np.linalg.norm(y,ord=2)])])
            cScale = np.max([1,normcx*normJx])
            maxViolAll,maxComp,maxViolBnds,primInf,dualInf,ispDualInf= self.optCheck(x,y,g,self.J,zScale,cScale)

            Mtestx = np.linalg.norm(self.gx-np.dot(self.Jx.T,self.yM)-self.vM-self.zM[:self.n],ord=np.inf)
            if self.m > 0:
                Mtesty = np.linalg.norm(np.concatenate((y-self.piY,self.yM-self.zM[self.n:]),axis=0),ord=np.inf)
            else:
                Mtesty = 0
            try:
                MtestFL = np.linalg.norm(self.vL[self.jLoA]-self.piVL[self.jLoA],ord=np.inf)
            except:
                MtestFL = 0
            try:
                MtestFU = np.linalg.norm(self.vU[self.jUpA]-self.piVU[self.jUpA],ord=np.inf)
            except:
                MtestFU = 0
            try:
                MtestzL = np.linalg.norm(self.zL[self.jLo]-self.piL[self.jLo],ord=np.inf)
            except:
                MtestzL = 0
            try:
                MtestzU = np.linalg.norm(self.zU[self.jUp]-self.piU[self.jUp],ord=np.inf)
            except:
                MtestzU = 0

            Mtest = np.linalg.norm(np.array([Mtestx,Mtesty,MtestFL,MtestFU,MtestzL,MtestzU]),ord=np.inf)/np.max([1,fM0])

            phi0 = primInf+dualInf

            if phi0 <= phiMax:
                Oitns += 1
                phiMax = phiMax/2
                self.ye = np.copy(y)
                self.zeL[self.jLo] = np.copy(self.zL[self.jLo])
                self.zeU[self.jUp] = np.copy(self.zU[self.jUp])
                self.xe = np.maximum(np.minimum(x,self.bU),self.bL)
                self.xeL[self.jLo] = np.maximum(self.xL[self.jLo],self.muB[self.jLo])
                self.xeU[self.jUp] = np.maximum(self.xU[self.jUp],self.muB[self.jUp])
                self.ze[self.jxLorU] = self.zeL[self.jxLorU]-self.zeU[self.jxLorU]
                self.ze[self.n:] = self.zeL[self.n:]-self.zeU[self.n:]
                self.veL[self.jLoA] = np.copy(self.vL[self.jLoA])
                self.veU[self.jUpA] = np.copy(self.vU[self.jUpA])
                self.ve = self.veL-self.veU
            elif Mtest <=self.tolM:
                Mitns += 1
                self.ye = np.maximum(-self.yMax,np.minimum(y,self.yMax))
                self.zeL[self.n:] = np.maximum(-self.yMax,np.minimum(self.zeL[self.n:],self.yMax))
                self.zeU[self.n:] = np.maximum(-self.yMax,np.minimum(self.zeU[self.n:],self.yMax))
                self.ze[self.n:] = self.zeL[self.n:]-self.zeU[self.n:]
                self.xe = np.maximum(np.minimum(x,self.bU),self.bL)
                self.xeL[self.jLo] = np.maximum(np.minimum(self.xe[self.jLo]-self.bL[self.jLo]+self.muB[self.jLo],self.yMax),self.muB[self.jLo])
                self.xeU[self.jUp] = np.maximum(np.minimum(self.bU[self.jUp]-self.xe[self.jUp]+self.muB[self.jUp],self.yMax),self.muB[self.jUp])

                if np.linalg.norm(self.cs,ord=2) > self.tolM:
                    self.muP = self.muP/self.muPfac
                    muPreduced = True

                if maxComp > self.tolM or np.any(x[self.jLo]-self.bL[self.jLo]+self.tolM<0)\
                   or np.any(self.bU[self.jUp]-x[self.jUp]+self.tolM<0)\
                   or np.any(self.zL[self.jLo]+self.tolM<0)\
                   or np.any(self.zU[self.jUp]+self.tolM<0):
                    self.muB = self.muB/self.muBfac
                    self.muA = self.muA/self.muAfac
                self.tolM = self.tolM/2
            else:
                Fitns += 1

            #=======================================================================
            # Check if any temporaily fixed slacks can be freed.
            #=======================================================================
            jOkayL = tjfixedcL[np.argwhere(self.cx[tjfixedcL-self.n]-self.bL[tjfixedcL]+self.muB[tjfixedcL]>0).ravel()]
            jOkayU = tjfixedcL[np.argwhere(self.bU[tjfixedcL]-self.cx[tjfixedcL-self.n]+self.muB[tjfixedcL]>0).ravel()]
            jOkay = np.union1d(jOkayL,jOkayU)

            if jOkay.size > 0:
                x[jOkay] = self.cx[jOkay-self.n]
                self.xe[jOkay] = np.maximum(np.minimum(x[jOkay],self.bU[jOkay]),self.bL[jOkay])
                self.jFixedc = np.setdiff1d(self.jFixedc,jOkay)
                self.iFixedc = self.jFixedc - self.n

                if jOkayL.size > 0:
                    self.jLo = np.union1d(self.jLo,jOkayL)
                    self.zL[jOkayL] = np.maximum(np.absolute(self.piY[jOkayL-self.n]),self.eps)
                    self.zeL[jOkayL] = np.maximum(np.absolute(self.ye[jOkayL-self.n]),self.eps)
                    self.xeL[jOkayL] = np.maximum(x[jOkayL]-self.bL[jOkayL]+self.muB[jOkayL],self.muB[jOkayL])

                    jcLoUOkay = np.intersect1d(jOkayL, jUp0)
                    self.jUp = np.union1d(self.jUp, jcLoUOkay)
                    self.icLorU = np.union1d(self.icLorU,jOkayL-self.n)
                    self.jcLorU = self.icLorU+self.n
                    tjfixedcL = np.setdiff1d(tjfixedcL,jOkayL)
                    self.zU[jcLoUOkay] = np.copy(self.mZeros[jcLoUOkay-self.n])
                    self.zeU[jcLoUOkay] = np.copy(self.mZeros[jcLoUOkay-self.n])
                    self.xeU[jcLoUOkay] = np.maximum(self.bU[jcLoUOkay]-x[jcLoUOkay]+self.muB[jcLoUOkay],self.muB[jcLoUOkay])
                    
                if jOkayU.size > 0:
                    self.jUp = np.union1d(self.jUp,jOkayU)
                    self.zU[jOkayU] = np.maximum(np.absolute(self.piY[jOkayU-self.n]),self.eps)
                    self.zeU[jOkayU] = np.maximum(np.absolute(self.ye[jOkayU-self.n]),self.eps)
                    self.xeU[jOkayU] = np.maximum(self.bU[jOkayU]-x[jOkayU]+self.muB[jOkayU],self.muB[jOkayU])

                    jcLoUOkay = np.intersect1d(jOkayU, jLo0)
                    self.jLo = np.union1d(self.jLo, jcLoUOkay)
                    self.icLorU = np.union1d(self.icLorU,jOkayU-self.n)
                    self.jcLorU = self.icLorU+self.n
                    tjfixedcU = np.setdiff1d(tjfixedcU,jOkayU)
                    self.zL[jcLoUOkay] = self.mZeros[jcLoUOkay-self.n]
                    self.zeL[jcLoUOkay] = self.mZeros[jcLoUOkay-self.n]
                    self.xeL[jcLoUOkay] = np.maximum(x[jcLoUOkay]-self.bL[jcLoUOkay]+self.muB[jcLoUOkay],self.muB[jcLoUOkay])

                tjfixedc = np.union1d(tjfixedcL,tjfixedcU)
                self.cs = self.cx - x[self.n:]

                self.z[self.jLo] = np.copy(self.zL[self.jLo])
                self.z[self.jUp] = 0
                self.z[self.jUp] = np.copy(self.z[self.jUp] -self.zU[self.jUp])                    

            #===========================================================================
            # Check if any penalized infeasible variables can be returned to normality.
            #===========================================================================
            jOkayL = np.copy(self.jLoA[np.argwhere(x[self.jLoA]-self.bL[self.jLoA]+self.muB[self.jLoA]>0).ravel()])
            jOkayU = np.copy(self.jUpA[np.argwhere(self.bU[self.jUpA]-x[self.jUpA]+self.muB[self.jUpA]>0).ravel()])
            jOkay = np.union1d(jOkayL,jOkayU)

            if jOkay.size > 0:
                self.xe[jOkay] = np.maximum(np.minimum(x[jOkay],self.bU[jOkay]),self.bL[jOkay])

                if jOkayL.size > 0:
                    self.jLo = np.union1d(self.jLo,jOkayL)
                    self.piL[jOkayL] = np.copy(self.piVL[jOkayL])
                    self.zL[jOkayL] = np.maximum(self.vL[jOkayL],self.eps)
                    self.zeL[jOkayL] = np.maximum(self.veL[jOkayL],self.eps)
                    self.vL[jOkayL] = 0

                    jxLoUOkay = np.intersect1d(jOkayL,jUp0)
                    self.jUp = np.union1d(self.jUp,jxLoUOkay)
                    self.jLoA = np.setdiff1d(self.jLoA,jOkayL)
                    self.jxLorU = np.union1d(self.jxLorU,jOkayL)
                    self.xeU[jxLoUOkay] = np.maximum(self.bU[jxLoUOkay]-x[jxLoUOkay]+self.muB[jxLoUOkay],self.muB[jxLoUOkay])

                if jOkayU.size > 0:
                    self.jUp = np.union1d(self.jUp,jOkayU)
                    self.piU[jOkayU] = np.copy(self.piVU[jOkayU])
                    self.zU[jOkayU] = np.maximum(self.vU[jOkayU],self.eps)
                    self.zeU[jOkayU] = np.maximum(self.veU[jOkayU],self.eps)
                    self.xeU[jOkayU] = np.maximum(self.bU[jOkayU]-x[jOkayU]+self.muB[jOkayU],self.muB[jOkayU])
                    self.vU[jOkayU] = 0

                    jxLoUOkay = np.intersect1d(jOkayU,jLo0)
                    self.jLo = np.union1d(self.jLo,jxLoUOkay)
                    self.jUpA = np.setdiff1d(self.jUpA,jOkayU)
                    self.jxLorU = np.union1d(self.jxLorU,jOkayU)
                    self.xeL[jxLoUOkay] = np.maximum(x[jxLoUOkay]-self.bL[jxLoUOkay]+self.muB[jxLoUOkay],self.muB[jxLoUOkay])

                    self.v = self.vL-self.vU
                    
                    self.z[self.jLo] = np.copy(self.zL[self.jLo])
                    self.z[self.jUp] = 0
                    self.z[self.jUp] = np.copy(self.z[self.jUp] -self.zU[self.jUp])

            #===========================================================================
            # Check for variables and slacks outside their shifted bounds
            #===========================================================================
            jViolL = np.copy(self.jLo[np.argwhere(x[self.jLo]-self.bL[self.jLo]+self.muB[self.jLo]<=0).ravel()])
            jViolU = np.copy(self.jUp[np.argwhere(self.bU[self.jUp]-x[self.jUp]+self.muB[self.jUp]<=0).ravel()])

            if jViolL.size > 0:
                jxViolL = np.copy(jViolL[np.argwhere(jViolL<self.n).ravel()])
                jcViolL = np.copy(jViolL[np.argwhere(jViolL>=self.n).ravel()])
                
                if jcViolL.size > 0:
                    icViolL = jcViolL-self.n
                    self.muP[icViolL] = self.muB[jcViolL]/self.muAfac
                    x[jcViolL] = np.copy(self.bL[jcViolL])
                    self.ye[icViolL] = np.maximum(-self.yMax,np.minimum(self.zeL[jcViolL],self.yMax))
                    self.xL[jcViolL] = 0
                    self.zL[jcViolL] = 0
                    self.xe[jcViolL] = np.maximum(np.minimum(x[jcViolL],self.bU[jcViolL]),self.bL[jcViolL])

                    jcViolU = np.intersect1d(jcViolL,self.jUp)
                    self.jLo = np.setdiff1d(self.jLo,jcViolU)
                    self.jUp = np.setdiff1d(self.jUp,jcViolU)
                    self.jcLorU = np.setdiff1d(self.jcLorU,jcViolL)
                    self.icLorU = np.setdiff1d(self.icLorU,jcViolL-self.n)
                    self.jcLorU = np.setdiff1d(self.jcLorU,jcViolU)
                    self.icLorU = np.setdiff1d(self.icLorU,jcViolU-self.n)

                    tjfixedcL = np.union1d(tjfixedcL,jcViolL)
                    self.jFixedc = np.union1d(self.jFixedc,jcViolL)
                    self.iFixedc = self.jFixedc-self.n

                    self.z[jcViolL] = np.copy(self.zL[jcViolL])
                            

                if jxViolL.size > 0:
                    self.muA[jxViolL] = self.muB[jxViolL]/self.muAfac

                    self.veL[jxViolL] = np.copy(self.zeL[jxViolL])
                    self.ve[jxViolL] = np.copy(self.veL[jxViolL])
                    self.piVL[jxViolL] = np.copy(self.piL[jxViolL])
                    self.vL[jxViolL] = np.copy(self.zL[jxViolL])
                    self.v[jxViolL] = np.copy(self.vL[jxViolL])

                    self.jLoA = np.union1d(self.jLoA,jxViolL)
                    self.jLo = np.setdiff1d(self.jLo,jxViolL)
                    self.jUp = np.setdiff1d(self.jUp,jxViolL)
                    self.jxLorU = np.setdiff1d(self.jxLorU,jxViolL)

                    self.z[jxViolL] = 0
  
            if jViolU.size > 0:
                jxViolU = np.copy(jViolU[np.argwhere(jViolU<self.n).ravel()])
                jcViolU = np.copy(jViolU[np.argwhere(jViolU>=self.n).ravel()])

                if jcViolU.size > 0:
                    icViolU = jcViolU-self.n

                    self.muP[icViolU] = self.muB[jcViolU]/self.muAfac

                    x[jcViolU] = np.copy(self.bU[jcViolU])
                    self.xU[jcViolU] = 0
                    self.xe[jcViolU] = np.maximum(np.minimum(x[jcViolU],self.bU[jcViolU]),self.bL[jcViolU])
                    self.ye[icViolU] = -np.minimum(self.zeU[jcViolU],self.yMax)
                    self.xeU[jcViolU] = np.copy(self.muB[jcViolU])
                    self.zU[jcViolU] = 0

                    jcViolL = np.intersect1d(jcViolU,self.jLo)
                    self.jUp = np.setdiff1d(self.jUp,jcViolU)
                    self.jLo = np.setdiff1d(self.jLo,jcViolL)
                    self.jcLorU = np.setdiff1d(self.jcLorU,jcViolU)
                    self.icLorU = np.setdiff1d(self.icLorU,jcViolU-self.n)
                    self.jcLorU = np.setdiff1d(self.jcLorU,jcViolL)
                    self.icLorU = np.setdiff1d(self.icLorU,jcViolL-self.n)

                    tjfixedcU = np.union1d(tjfixedcU,jcViolU)
                    self.jFixedc = np.union1d(self.jFixedc,jcViolU)
                    self.iFixedc = self.jFixedc-self.n

                    self.z[jcViolU] = -self.zU[jcViolU]
                    

                if jxViolU.size > 0:
                    self.muA[jxViolU] = self.muB[jxViolU]/self.muAfac

                    self.veU[jxViolU] = np.copy(self.zeU[jxViolU])
                    self.ve[jxViolU] = -self.veU[jxViolU]
                    self.piVU[jxViolU] = np.copy(self.piU[jxViolU])
                    self.vU[jxViolU] = np.copy(self.zU[jxViolU])
                    self.v[jxViolU] = -self.vU[jxViolU]

                    self.jUpA = np.union1d(self.jUpA,jxViolU)
                    self.jLo = np.setdiff1d(self.jLo,jxViolU)
                    self.jUp = np.setdiff1d(self.jUp,jxViolU)
                    self.jxLorU = np.setdiff1d(self.jxLorU,jxViolU)

                    self.z[jxViolU] = 0             

            if jViolL.size > 0 or jViolU.size > 0:
                self.cs = self.cx - x[self.n:]
                
                self.z[self.jLo] = np.copy(self.zL[self.jLo])
                self.z[self.jUp] = 0
                self.z[self.jUp] = self.z[self.jUp] -self.zU[self.jUp]
                
            if muPused or muPreduced:
                self.muL = np.maximum(self.muL/2,self.muP)
            
        elapsed = time.time() - _time

        self.g       = g
        self.zScale  = zScale
        self.cScale  = cScale
        
        return x[:self.n],y,self.n,self.m,outcome,itn,nf,elapsed,self.fx
        
def main():
    import sys
    import os

    def list2str(lst):
        return [str(l) for l in lst]

    printLevel = 1
    
    # x_dev is a device vector that must be predefined by the user and is used to build theano
    # expressions.
    x_dev = at.vector('x_dev')

    # get the problem number/name from the command line argument list.
    if sys.argv[1].isdigit():
        prob = int(sys.argv[1])
    else:
        prob = str(sys.argv[1])

    float_dtype = np.float64

    infinity = float_dtype(1.0E15)

    np.random.seed(22)
    if prob == 1:
        print('minimize f(x, y) = x**2 - 4*x + y**2 - y - x*y')
        print('')
        x0 = np.random.randn(2).astype(float_dtype)

        f = x_dev[0] ** 2 - 4 * x_dev[0] + \
            x_dev[1] ** 2 - x_dev[1] - x_dev[0] * x_dev[1]

        p = pdProj(x0=x0, x_dev=x_dev, f=f, infinity=infinity,
                  float_dtype=float_dtype, printLevel=printLevel)
        x,y,n,m,outcome,itn,nf,elapsed,fval = p.solve()

        gt = [float_dtype(3.0), float_dtype(2.0)]
        print('')
        print('Ground truth: [x, y] = [{}, {}]'.format(*gt))
        print('Solver solution: [x, y] = [{}, {}]'.format(*x))    
     
    elif prob == 2:
        print('Find the global minimum of the 2D Rosenbrock function.')
        print('minimize f(x, y) = 100*(y - x**2)**2 + (1 - x)**2')
        print('')
        x0 = np.random.randn(2).astype(float_dtype)

        f = 100 * (x_dev[1] - x_dev[0] ** 2) ** 2 + (1 - x_dev[0]) ** 2

        p = pdProj(x0=x0, x_dev=x_dev, f=f, infinity=infinity,
                  float_dtype=float_dtype, printLevel=printLevel)
        x,y,n,m,outcome,itn,nf,elapsed,fval = p.solve()

        gt = [float_dtype(1.0), float_dtype(1.0)]
        print('')
        print('Ground truth: [x, y] = [{}, {}]'.format(*gt))
        print('Solver solution: [{}, {}]'.format(*x))
        
    elif prob == 3:
        print('maximize f(x, y) = x + y subject to x**2 + y**2 = 1')
        print('')
        x0 = np.random.randn(2).astype(float_dtype)
        y0 = np.array([0], dtype=float_dtype)

        f = -at.sum(x_dev)
        c = at.sum(x_dev ** 2) - 1.0

        bl = np.array([-infinity, -infinity], dtype=float_dtype)
        bu = np.array([infinity, infinity], dtype=float_dtype)
        cl = np.array([0], dtype=float_dtype)
        cu = np.array([0], dtype=float_dtype)

        p = pdProj(x0=x0,y0=y0, bl=bl, bu=bu, cl=cl, cu=cu, x_dev=x_dev, f=f, c=c, infinity=infinity,
                  float_dtype=float_dtype, printLevel=printLevel)
        x,y,n,m,outcome,itn,nf,elapsed,fval = p.solve()

        gt = [float_dtype(np.sqrt(2.0) / 2.0), float_dtype(np.sqrt(2.0) / 2.0)]
        print('')
        print('Ground truth: [x, y] = [{}, {}]'.format(*gt))
        print('Solver solution: [x, y] = [{}, {}]'.format(*x))
        
    elif prob == 4:
        print('maximize f(x, y) = (x**2)*y subject to x**2 + y**2 = 3')
        print('')
        x0 = np.random.randn(2).astype(float_dtype)

        f = -(x_dev[0] ** 2) * x_dev[1]
        c = at.sum(x_dev ** 2) - 3.0

        bl = np.array([-infinity, -infinity])
        bu = np.array([infinity, infinity])
        cl = np.array([0], dtype=float_dtype)
        cu = np.array([0], dtype=float_dtype)

        p = pdProj(x0=x0, bl=bl, bu=bu, cl=cl, cu=cu, x_dev=x_dev, f=f, c=c, infinity=infinity,
                  float_dtype=float_dtype, printLevel=printLevel)
        x,y,n,m,outcome,itn,nf,elapsed,fval = p.solve()

        gt = [
            float_dtype(np.sqrt(2.0)), float_dtype(1.0),
            -float_dtype(np.sqrt(2.0)), float_dtype(1.0),
            float_dtype(0.0), -float_dtype(-np.sqrt(3.0)),
        ]
        print('')
        print(
            'Ground truth: global max. @ [x, y] = [{}, {}] or [{}, {}], local max. @ [{}, {}]'.format(*gt))
        print('Solver solution: [x, y] = [{}, {}]'.format(*x))
        
    elif prob == 5:
        print('minimize f(x, y) = x**2 + 2*y**2 + 2*x + 8*y subject to -x - 2*y + 10 <= 0, x >= 0, y >= 0')
        print('')
        x0 = np.random.randn(2).astype(float_dtype)

        f = x_dev[0] ** 2 + 2.0 * x_dev[1] ** 2 + \
            2.0 * x_dev[0] + 8.0 * x_dev[1]
        c = at.zeros((1,))
        c = at.set_subtensor(c[0], x_dev[0] + 2.0 * x_dev[1] - 10.0)

        bl = np.array([0, 0],dtype=float_dtype)
        bu = np.array([infinity, infinity])
        cl = np.array([0], dtype=float_dtype)
        cu = np.array([0], dtype=float_dtype)

        p = pdProj(x0=x0, bl=bl, bu=bu, cl=cl, cu=cu, x_dev=x_dev, f=f, c=c, infinity=infinity,
                  float_dtype=float_dtype, printLevel=printLevel)
        x,y,n,m,outcome,itn,nf,elapsed,fval = p.solve()

        gt = [float_dtype(4.0), float_dtype(3.0)]
        print('')
        print('Ground truth: [x, y] = [{}, {}]'.format(*gt))
        print('Solver solution: [x, y] = [{}, {}]'.format(*x))
        
    elif prob == 6:
        print('Find the maximum entropy distribution of a six-sided die:')
        print(
            'maximize f(x) = -sum(x*log(x)) subject to sum(x) = 1 and x >= 0 (x.size == 6)')
        print('')
        x0 = np.random.rand(6).astype(float_dtype)
        x0 = x0 / np.sum(x0)

        f = at.sum(x_dev * at.log(x_dev + np.finfo(float_dtype).eps))
        c = at.zeros((1,))
        c = at.set_subtensor(c[0],at.sum(x_dev)-1.0)

        bl = np.array([0, 0, 0, 0, 0, 0], dtype=float_dtype)
        bu = np.array([infinity, infinity, infinity, infinity, infinity, infinity])
        cl = np.array([0], dtype=float_dtype)
        cu = np.array([0], dtype=float_dtype)

        p = pdProj(x0=x0, bl=bl, bu=bu, cl=cl, cu=cu, x_dev=x_dev, f=f, c=c, infinity=infinity,
                  float_dtype=float_dtype, printLevel=printLevel)
        x,y,n,m,outcome,itn,nf,elapsed,fval = p.solve()

        gt = str(float_dtype(1.0 / 6.0))
        print('')
        print('Ground truth: [{}]'.format(', '.join([gt] * 6)))
        print('Solver solution: x = [{}]'.format(', '.join(list2str(x))))
        
        
    elif prob == 7:
        print(
            'maximize f(x, y, z) = x*y*z subject to x + y + z = 1, x >= 0, y >= 0, z >= 0')
        print('')
        x0 = np.random.randn(3).astype(float_dtype)

        f = -x_dev[0] * x_dev[1] * x_dev[2]
        c = at.zeros((1,))
        c = at.set_subtensor(c[0],x_dev[0] + x_dev[1] + x_dev[2]-1.0)

        bl = np.array([0, 0, 0], dtype=float_dtype)
        bu = np.array([infinity, infinity, infinity])
        cl = np.array([0], dtype=float_dtype)
        cu = np.array([0], dtype=float_dtype)

        p = pdProj(x0=x0, bl=bl, bu=bu, cl=cl, cu=cu, x_dev=x_dev, f=f, c=c, infinity=infinity,
                  float_dtype=float_dtype, printLevel=printLevel)
        x,n,m,outcome,itn,nf,elapsed,fval = p.solve()

        gt = [float_dtype(1.0 / 3.0), float_dtype(1.0 / 3.0),
              float_dtype(1.0 / 3.0)]
        print('')
        print('Ground truth: [x, y, z] = [{}, {}, {}]'.format(*gt))
        print('Solver solution: [x, y, z] = [{}, {}, {}]'.format(*x))
        
    elif prob == 8:
        print('minimize f(x,y,z) = 4*x - 2*z subject to 2*x - y - z = 2, x**2 + y**2 = 1')
        print('')
        x0 = np.random.randn(3).astype(float_dtype)

        f = 4.0 * x_dev[1] - 2.0 * x_dev[2]
        c = at.zeros((2,))
        c = at.set_subtensor(c[0], 2.0 * x_dev[0] - x_dev[1] - x_dev[2] - 2.0)
        c = at.set_subtensor(c[1], x_dev[0] ** 2 + x_dev[1] ** 2 - 1.0)

        bl = np.array([-infinity, -infinity, -infinity])
        bu = np.array([infinity, infinity, infinity])
        cl = np.array([0, 0], dtype=float_dtype)
        cu = np.array([0, 0], dtype=float_dtype)

        p = pdProj(x0=x0, bl=bl, bu=bu, cl=cl, cu=cu, x_dev=x_dev, f=f, c=c, infinity=infinity,
                  float_dtype=float_dtype, printLevel=printLevel)
        x,y,n,m,outcome,itn,nf,elapsed,fval = p.solve()

        gt = [
            float_dtype(2.0 / np.sqrt(13.0)),
            float_dtype(-3.0 / np.sqrt(13.0)),
            float_dtype(-2.0 + 7.0 / np.sqrt(13.0))
        ]
        print('')
        print('Ground truth: [x, y, z] = [{}, {}, {}]'.format(*gt))
        print('Solver solution: [x, y, z] = [{}, {}, {}]'.format(*x))
        
    elif prob == 9:
        print(
            'minimize f(x, y) = (x - 2)**2 + 2*(y - 1)**2 subject to x + 4*y <= 3, x >= y')
        print('')
        x0 = np.random.randn(2).astype(float_dtype)
        y0 = np.array([0,0], dtype=float_dtype)

        f = (x_dev[0] - 2.0) ** 2 + 2.0 * (x_dev[1] - 1.0) ** 2
        c = at.zeros(2)
        c = at.set_subtensor(c[0], -x_dev[0] - 4.0 * x_dev[1] + 3.0)
        c = at.set_subtensor(c[1], x_dev[0] - x_dev[1])

        bl = np.array([-infinity, -infinity])
        bu = np.array([infinity, infinity])
        cl = np.array([0,0],dtype=float_dtype)
        cu = np.array([infinity,infinity])

        p = pdProj(x0=x0, y0=y0,bl=bl, bu=bu, cl=cl, cu=cu, x_dev=x_dev, f=f, c=c, infinity=infinity,
                  float_dtype=float_dtype, printLevel=printLevel)
        x,y,n,m,outcome,itn,nf,elapsed,fval = p.solve()

        gt = [float_dtype(5.0 / 3.0), float_dtype(1.0 / 3.0)]
        print('')
        print('Ground truth: [x, y] = [{}, {}]'.format(*gt))
        print('Solver solution: [x, y] = [{}, {}]'.format(*x))
        
    elif prob == 10:
       print('minimize f(x, y, z) = (x - 1)**2 + 2*(y + 2)**2 + 3*(z + 3)**2 subject to z - y - x = 1, z - x**2 >= 0')
       print('')
       x0 = np.random.randn(3).astype(float_dtype)

       f = (x_dev[0] - 1.0) ** 2 + 2.0 * \
           (x_dev[1] + 2.0) ** 2 + 3.0 * (x_dev[2] + 3.0) ** 2
       c = at.zeros(2)
       c = at.set_subtensor(c[0], x_dev[2] - x_dev[1] - x_dev[0] - 1.0)
       c = at.set_subtensor(c[1], x_dev[2] - x_dev[0] ** 2)

       bl = np.array([-infinity, -infinity, -infinity])
       bu = np.array([infinity, infinity, infinity])
       cl = np.array([0, 0], dtype=float_dtype)
       cu = np.array([0,infinity], dtype=float_dtype)

       p = pdProj(x0=x0, bl=bl, bu=bu, cl=cl, cu=cu, x_dev=x_dev, f=f, c=c, infinity=infinity,
                  float_dtype=float_dtype, printLevel=printLevel)
       x,y,n,m,outcome,itn,nf,elapsed,fval = p.solve()

       gt = [0.12288, -1.1078, 0.015100]
       print('')
       print('Ground truth: [x, y, z] = [{}, {}, {}]'.format(*gt))
       print('Solver solution: [x, y, z] = [{}, {}, {}]'.format(*x))
    else:
        # Solve CUTEst problem
        #import pycutest
        #prob = pycutest.import_problem(prob)
        #x0 = prob.x0
        #y0 = prob.v0
        #bl = prob.bl
        #bu = prob.bu
        #cl = prob.cl
        #cu = prob.cu
        #c,J =prob.cons(x0, gradient=True)
        
        #p = pdProj(prob=prob,x0=x0, y0=y0, bl=bl, bu=bu, cl=cl, cu=cu)
        #x,y,n,m,outcome,itn,nf,elapsed,fval = p.solve()
        #print(outcome)

        # Solve user-defined problem
        import problem        
        bl     = np.array([-infinity,-infinity], dtype=float_dtype)
        bu     = np.array([infinity,infinity], dtype=float_dtype)
        cl     = np.array([0], dtype=float_dtype)
        cu     = np.array([0], dtype=float_dtype)
        x0     = np.random.randn(2).astype(float_dtype)
        y0     = np.array([0], dtype=float_dtype)

        prob = problem.prob(x0=x0,y0=y0,bl=bl,bu=bu,cl=cl,cu=cu)
        p = pdProj(prob=prob,x0=x0, y0=y0, bl=bl, bu=bu, cl=cl, cu=cu)
        x,y,n,m,outcome,itn,nf,elapsed,fval= p.solve()
        print(x,outcome)
        
        

if __name__ == '__main__':
    main()
    
