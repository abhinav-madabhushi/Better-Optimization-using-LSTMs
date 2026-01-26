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
C  Problem name : STEENBRA  
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION FLOW  , X1    , X2    , X3    , X4    
      DOUBLE PRECISION X5    , X6    , X7    , X8    , X9    
      DOUBLE PRECISION X10   , X11   , X12   , ALPHA , BETA  
      DOUBLE PRECISION TWOBET
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : QUAD      
C
       X1     = XVALUE(IELVAR(ILSTRT+     1))
       X2     = XVALUE(IELVAR(ILSTRT+     2))
       X3     = XVALUE(IELVAR(ILSTRT+     3))
       X4     = XVALUE(IELVAR(ILSTRT+     4))
       X5     = XVALUE(IELVAR(ILSTRT+     5))
       X6     = XVALUE(IELVAR(ILSTRT+     6))
       X7     = XVALUE(IELVAR(ILSTRT+     7))
       X8     = XVALUE(IELVAR(ILSTRT+     8))
       X9     = XVALUE(IELVAR(ILSTRT+     9))
       X10    = XVALUE(IELVAR(ILSTRT+    10))
       X11    = XVALUE(IELVAR(ILSTRT+    11))
       X12    = XVALUE(IELVAR(ILSTRT+    12))
       ALPHA  = EPVALU(IPSTRT+     1)
       BETA   = EPVALU(IPSTRT+     2)
       FLOW   =   X1    
     *          + X2    
     *          + X3    
     *          + X4    
     *          + X5    
     *          + X6    
     *          + X7    
     *          + X8    
     *          + X9    
     *          + X10   
     *          + X11   
     *          + X12   
       TWOBET = BETA + BETA                              
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= ALPHA * FLOW + BETA * FLOW * FLOW        
       ELSE
        FUVALS(IGSTRT+     1)= ALPHA + TWOBET * FLOW                    
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=TWOBET                                   
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
