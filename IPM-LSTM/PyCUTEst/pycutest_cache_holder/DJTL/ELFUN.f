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
C  Problem name : DJTL      
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION V1    , DIF   
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
C  Element type : CB-10     
C
    1  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       DIF    = V1 - 10.0                                
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= DIF**3                                   
       ELSE
        FUVALS(IGSTRT+     1)= 3.0 * DIF * DIF                          
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=6.0 * DIF                                
        END IF
       END IF
       GO TO     5
C
C  Element type : CB-20     
C
    2  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       DIF    = V1 - 20.0                                
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= DIF**3                                   
       ELSE
        FUVALS(IGSTRT+     1)= 3.0 * DIF * DIF                          
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=6.0 * DIF                                
        END IF
       END IF
       GO TO     5
C
C  Element type : SQ-5      
C
    3  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       DIF    = V1 - 5.0                                 
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= DIF**2                                   
       ELSE
        FUVALS(IGSTRT+     1)= 2.0 * DIF                                
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=2.0                                      
        END IF
       END IF
       GO TO     5
C
C  Element type : SQ-6      
C
    4  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       DIF    = V1 - 6.0                                 
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= DIF**2                                   
       ELSE
        FUVALS(IGSTRT+     1)= 2.0 * DIF                                
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=2.0                                      
        END IF
       END IF
    5 CONTINUE
      RETURN
      END
