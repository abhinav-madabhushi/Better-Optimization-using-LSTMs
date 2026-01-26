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
C  Problem name : EG3       
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION X     , Y     , X1    , XN    , X2    
      DOUBLE PRECISION Z     , SINX  , COSX  
      INTRINSIC SIN   , COS   
      IFSTAT = 0
      DO     5 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
       IELTYP = ITYPEE(IELEMN)
       GO TO (    1,    2,    3,    4
     *                                                        ), IELTYP
C
C  Element type : PROD      
C
    2  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y      = XVALUE(IELVAR(ILSTRT+     2))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= X * Y                                    
       ELSE
        FUVALS(IGSTRT+     1)= Y                                        
        FUVALS(IGSTRT+     2)= X                                        
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=1.0                                      
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
       GO TO     5
C
C  Element type : RPROD     
C
    1  CONTINUE
       X1     = XVALUE(IELVAR(ILSTRT+     1))
       XN     = XVALUE(IELVAR(ILSTRT+     2))
       X2     = XVALUE(IELVAR(ILSTRT+     3))
       X      =   X1    
     *          - XN    
       Y      =   X2    
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= X * Y                                    
       ELSE
        FUVALS(IGSTRT+     1)= Y                                        
        FUVALS(IGSTRT+     2)= X                                        
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=1.0                                      
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
       GO TO     5
C
C  Element type : SQR2      
C
    4  CONTINUE
       Y      = XVALUE(IELVAR(ILSTRT+     1))
       Z      = XVALUE(IELVAR(ILSTRT+     2))
       X      =   Y     
     *          + Z     
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= X * X                                    
       ELSE
        FUVALS(IGSTRT+     1)= X + X                                    
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=2.0                                      
        END IF
       END IF
       GO TO     5
C
C  Element type : SINSQR    
C
    3  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       SINX   = SIN( X )                                 
       COSX   = COS( X )                                 
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= SINX * SINX                              
       ELSE
        FUVALS(IGSTRT+     1)= 2.0 * SINX * COSX                        
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=2.0 * ( COSX * COSX - SINX * SINX )      
        END IF
       END IF
    5 CONTINUE
      RETURN
      END
