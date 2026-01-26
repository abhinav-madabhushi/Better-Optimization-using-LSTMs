      SUBROUTINE ELFUN ( FUVALS, XVALUE, EPVALU, NCALCF, ITYPEE, 
     *                   ISTAEV, IELVAR, INTVAR, ISTADH, ISTEPA, 
     *                   ICALCF, LTYPEE, LSTAEV, LELVAR, LNTVAR, 
     *                   LSTADH, LSTEPA, LCALCF, LFVALU, LXVALU, 
     *                   LEPVLU, IFFLAG, IFSTAT )
      INTEGER NCALCF, IFFLAG, LTYPEE, LSTAEV, LELVAR, LNTVAR
      INTEGER LSTADH, LSTEPA, LCALCF, LFVALU, LXVALU, LEPVLU
      INTEGER IFSTAT
      INTEGER ITYPEE(LTYPEE), ISTAEV(LSTAEV), IELVAR(LELVAR)
      INTEGER INTVAR(LNTVAR), ISTADH(LSTADH), ISTEPA(LSTEPA)
      INTEGER ICALCF(LCALCF)
      DOUBLE PRECISION FUVALS(LFVALU), XVALUE(LXVALU), EPVALU(LEPVLU)
C
C  Problem name : SARO      
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION X1    , X2    , X3    , X4    , X5    
      DOUBLE PRECISION X6    , X7    , X8    , X9    , U1    
      DOUBLE PRECISION U2    , RK    , RIDX  , SAROFN, SAROGN
      DOUBLE PRECISION F     , GX1   , GX2   , GX3   , GX4   
      DOUBLE PRECISION GX5   , GX6   , GX7   , GX8   , GX9   
      DOUBLE PRECISION GU1   , GU2   
      EXTERNAL SAROFN, SAROGN
      IFSTAT = 0
      DO     3 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
       IELTYP = ITYPEE(IELEMN)
       GO TO (    1,    2
     *                                                        ), IELTYP
C
C  Element type : FTYPE     
C
    1  CONTINUE
       X1     = XVALUE(IELVAR(ILSTRT+     1))
       X2     = XVALUE(IELVAR(ILSTRT+     2))
       X3     = XVALUE(IELVAR(ILSTRT+     3))
       X4     = XVALUE(IELVAR(ILSTRT+     4))
       X5     = XVALUE(IELVAR(ILSTRT+     5))
       X6     = XVALUE(IELVAR(ILSTRT+     6))
       X7     = XVALUE(IELVAR(ILSTRT+     7))
       X8     = XVALUE(IELVAR(ILSTRT+     8))
       X9     = XVALUE(IELVAR(ILSTRT+     9))
       U1     = XVALUE(IELVAR(ILSTRT+    10))
       U2     = XVALUE(IELVAR(ILSTRT+    11))
       RK     = EPVALU(IPSTRT+     1)
       RIDX   = EPVALU(IPSTRT+     2)
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= SAROFN(RK, RIDX, .TRUE.,                 
     *                      X1, X2, X3, X4, X5, X6, X7, X8, X9,  
     *                      U1, U2)                              
       ELSE
        F      = SAROGN(RK, RIDX, .TRUE.,                 
     *               X1, X2, X3, X4, X5, X6, X7, X8, X9,  
     *               U1, U2,                              
     *               GX1, GX2, GX3, GX4, GX5, GX6, GX7,   
     *               GX8, GX9, GU1, GU2)                  
        FUVALS(IGSTRT+     1)= GX1                                      
        FUVALS(IGSTRT+     2)= GX2                                      
        FUVALS(IGSTRT+     3)= GX3                                      
        FUVALS(IGSTRT+     4)= GX4                                      
        FUVALS(IGSTRT+     5)= GX5                                      
        FUVALS(IGSTRT+     6)= GX6                                      
        FUVALS(IGSTRT+     7)= GX7                                      
        FUVALS(IGSTRT+     8)= GX8                                      
        FUVALS(IGSTRT+     9)= GX9                                      
        FUVALS(IGSTRT+    10)= GU1                                      
        FUVALS(IGSTRT+    11)= GU2                                      
       END IF
       GO TO     3
C
C  Element type : GTYPE     
C
    2  CONTINUE
       X1     = XVALUE(IELVAR(ILSTRT+     1))
       X2     = XVALUE(IELVAR(ILSTRT+     2))
       X3     = XVALUE(IELVAR(ILSTRT+     3))
       X4     = XVALUE(IELVAR(ILSTRT+     4))
       X5     = XVALUE(IELVAR(ILSTRT+     5))
       X6     = XVALUE(IELVAR(ILSTRT+     6))
       X7     = XVALUE(IELVAR(ILSTRT+     7))
       X8     = XVALUE(IELVAR(ILSTRT+     8))
       X9     = XVALUE(IELVAR(ILSTRT+     9))
       U1     = XVALUE(IELVAR(ILSTRT+    10))
       U2     = XVALUE(IELVAR(ILSTRT+    11))
       RK     = EPVALU(IPSTRT+     1)
       RIDX   = EPVALU(IPSTRT+     2)
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= SAROFN(RK, RIDX, .FALSE.,                
     *                      X1, X2, X3, X4, X5, X6, X7, X8, X9,  
     *                      U1, U2)                              
       ELSE
        F      = SAROGN(RK, RIDX, .FALSE.,                
     *               X1, X2, X3, X4, X5, X6, X7, X8, X9,  
     *               U1, U2,                              
     *               GX1, GX2, GX3, GX4, GX5, GX6, GX7,   
     *               GX8, GX9, GU1, GU2)                  
        FUVALS(IGSTRT+     1)= GX1                                      
        FUVALS(IGSTRT+     2)= GX2                                      
        FUVALS(IGSTRT+     3)= GX3                                      
        FUVALS(IGSTRT+     4)= GX4                                      
        FUVALS(IGSTRT+     5)= GX5                                      
        FUVALS(IGSTRT+     6)= GX6                                      
        FUVALS(IGSTRT+     7)= GX7                                      
        FUVALS(IGSTRT+     8)= GX8                                      
        FUVALS(IGSTRT+     9)= GX9                                      
        FUVALS(IGSTRT+    10)= GU1                                      
        FUVALS(IGSTRT+    11)= GU2                                      
       END IF
    3 CONTINUE
      RETURN
      END
