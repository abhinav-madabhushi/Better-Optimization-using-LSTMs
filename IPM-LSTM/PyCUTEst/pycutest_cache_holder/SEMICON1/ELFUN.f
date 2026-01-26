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
C  Problem name : SEMICON1  
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION X     , LAC   , LAB   , LU    , FVAL  
      INTRINSIC EXP   
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : WE1       
C
       X      = XVALUE(IELVAR(ILSTRT+     1))
       LAC    = EPVALU(IPSTRT+     1)
       LAB    = EPVALU(IPSTRT+     2)
       LU     = EPVALU(IPSTRT+     3)
       FVAL   = LAC * EXP( LAB * ( X - LU ) )            
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= FVAL                                     
       ELSE
        FUVALS(IGSTRT+     1)= LAB * FVAL                               
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=LAB * LAB * FVAL                         
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
