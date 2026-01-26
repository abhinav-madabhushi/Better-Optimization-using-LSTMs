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
C  Problem name : SCW1      
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION T     , S     , C     
      INTRINSIC SIN   , COS   
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
C  Element type : SINT      
C
    1  CONTINUE
       T      = XVALUE(IELVAR(ILSTRT+     1))
       S      = SIN( T )                                 
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= S                                        
       ELSE
        FUVALS(IGSTRT+     1)= COS( T )                                 
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=-S                                       
        END IF
       END IF
       GO TO     3
C
C  Element type : COST      
C
    2  CONTINUE
       T      = XVALUE(IELVAR(ILSTRT+     1))
       C      = COS( T )                                 
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= C                                        
       ELSE
        FUVALS(IGSTRT+     1)= - SIN( T )                               
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=- C                                      
        END IF
       END IF
    3 CONTINUE
      RETURN
      END
