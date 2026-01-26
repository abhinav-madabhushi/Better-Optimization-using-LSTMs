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
C  Problem name : HS56      
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION V1    , V2    , V3    , P     , SUM   
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
C  Element type : 3PROD     
C
    1  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       V3     = XVALUE(IELVAR(ILSTRT+     3))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= V1 * V2 * V3                             
       ELSE
        FUVALS(IGSTRT+     1)= V2 * V3                                  
        FUVALS(IGSTRT+     2)= V1 * V3                                  
        FUVALS(IGSTRT+     3)= V1 * V2                                  
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=V3                                       
         FUVALS(IHSTRT+     4)=V2                                       
         FUVALS(IHSTRT+     5)=V1                                       
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
         FUVALS(IHSTRT+     6)=0.0D+0
        END IF
       END IF
       GO TO     3
C
C  Element type : PSNSQ     
C
    2  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       P      = EPVALU(IPSTRT+     1)
       SUM    = V1 + V1                                  
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= -P * SIN(V1)**2                          
       ELSE
        FUVALS(IGSTRT+     1)= -P * SIN(SUM)                            
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=-2.0 * P * COS(SUM)                      
        END IF
       END IF
    3 CONTINUE
      RETURN
      END
