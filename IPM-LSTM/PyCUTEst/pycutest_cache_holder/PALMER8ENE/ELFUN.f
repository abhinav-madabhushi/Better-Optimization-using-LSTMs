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
C  Problem name : PALMER8ENE
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION K     , L     , XSQR  , EXPON 
      INTRINSIC EXP   
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : PROD      
C
       K      = XVALUE(IELVAR(ILSTRT+     1))
       L      = XVALUE(IELVAR(ILSTRT+     2))
       XSQR   = EPVALU(IPSTRT+     1)
       EXPON  = EXP( - K * XSQR )                        
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= L * EXPON                                
       ELSE
        FUVALS(IGSTRT+     1)= - XSQR * L * EXPON                       
        FUVALS(IGSTRT+     2)= EXPON                                    
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=XSQR * XSQR * L * EXPON                  
         FUVALS(IHSTRT+     2)=- XSQR * EXPON                           
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
