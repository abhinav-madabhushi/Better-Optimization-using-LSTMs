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
C  Problem name : FBRAINNE  
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION ALPHA , C0    , COEFF , LAMBDA, BETA  
      DOUBLE PRECISION LOGLAM, LAMBET
      INTRINSIC EXP   , LOG   
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : FBRAIN    
C
       ALPHA  = XVALUE(IELVAR(ILSTRT+     1))
       C0     = XVALUE(IELVAR(ILSTRT+     2))
       COEFF  = EPVALU(IPSTRT+     1)
       LAMBDA = EPVALU(IPSTRT+     2)
       BETA   = ALPHA + ALPHA - 1.0D0                    
       LOGLAM = LOG( LAMBDA )                            
       LAMBET = COEFF * ( LAMBDA ** BETA )               
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= C0 * LAMBET                              
       ELSE
        FUVALS(IGSTRT+     1)= 2.0D0 * C0 * LAMBET * LOGLAM             
        FUVALS(IGSTRT+     2)= LAMBET                                   
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=4.0D0 * C0 * LAMBET * LOGLAM * LOGLAM    
         FUVALS(IHSTRT+     2)=2.0D0 * LAMBET * LOGLAM                  
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
