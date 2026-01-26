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
C  Problem name : KOEBHELB  
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION VN    , VA    , VB    , XX    , T     
      DOUBLE PRECISION M1OX  , XOB2  , M2XOB3
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : KHE       
C
       VN     = XVALUE(IELVAR(ILSTRT+     1))
       VA     = XVALUE(IELVAR(ILSTRT+     2))
       VB     = XVALUE(IELVAR(ILSTRT+     3))
       XX     = EPVALU(IPSTRT+     1)
       T      = EXP( - VA / XX - XX / VB )               
       M1OX   = - 1.0 / XX                               
       XOB2   = XX / ( VB * VB )                         
       M2XOB3 = - 2.0 * XX / VB**3                       
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= VN * T                                   
       ELSE
        FUVALS(IGSTRT+     1)= T                                        
        FUVALS(IGSTRT+     2)= VN * T * M1OX                            
        FUVALS(IGSTRT+     3)= VN * T * XOB2                            
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=T * M1OX                                 
         FUVALS(IHSTRT+     4)=T * XOB2                                 
         FUVALS(IHSTRT+     3)=VN * T * M1OX * M1OX                     
         FUVALS(IHSTRT+     5)=VN * T * M1OX * XOB2                     
         FUVALS(IHSTRT+     6)=VN * T * ( XOB2 * XOB2 + M2XOB3 )        
         FUVALS(IHSTRT+     1)=0.0D+0
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
