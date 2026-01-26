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
C  Problem name : DMN15332  
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION WEIGHT, WIDTH , POSIT , X     , DENOM 
      DOUBLE PRECISION PIINV , RATIO , WOPI  
      INTRINSIC ATAN  
      IFSTAT = 0
      PIINV  = 0.25D0 / ATAN( 1.0D0 )                   
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : LORENTZ   
C
       WEIGHT = XVALUE(IELVAR(ILSTRT+     1))
       WIDTH  = XVALUE(IELVAR(ILSTRT+     2))
       POSIT  = EPVALU(IPSTRT+     1)
       X      = EPVALU(IPSTRT+     2)
       DENOM  = ( X - POSIT ) ** 2 + WIDTH ** 2          
       RATIO  = WIDTH / DENOM                            
       WOPI   = PIINV * WEIGHT                           
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= WOPI * RATIO                             
       ELSE
        FUVALS(IGSTRT+     1)= PIINV * RATIO                            
        FUVALS(IGSTRT+     2)= WOPI / DENOM -                           
     *                         2.0D0 * WOPI * RATIO ** 2                
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=PIINV / DENOM                            
     *                         - 2.0D+0 * PIINV * RATIO ** 2            
         FUVALS(IHSTRT+     3)=- 6.0D+0 * WOPI * WIDTH / DENOM ** 2     
     *                         + 8.0D+0 * WOPI * RATIO ** 3             
         FUVALS(IHSTRT+     1)=0.0D+0
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
