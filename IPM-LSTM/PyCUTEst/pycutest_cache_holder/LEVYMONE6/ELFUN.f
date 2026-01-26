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
C  Problem name : LEVYMONE6 
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION X     , Z     , L     , C     , A     
      DOUBLE PRECISION PI    , PIL   , U     , V     , SINV  
      DOUBLE PRECISION COSV  
      INTRINSIC SIN   , COS   , ATAN  
      IFSTAT = 0
      PI     = 4.0 * ATAN( 1.0D0 )                      
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
C  Element type : S2        
C
    1  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       L      = EPVALU(IPSTRT+     1)
       C      = EPVALU(IPSTRT+     2)
       PIL    = PI * L                                   
       V      = PIL * X + PI * C                         
       SINV   = SIN( V )                                 
       COSV   = COS( V )                                 
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= SINV                                     
       ELSE
        FUVALS(IGSTRT+     1)= PIL * COSV                               
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=- PIL * PIL * SINV                       
        END IF
       END IF
       GO TO     3
C
C  Element type : PS2       
C
    2  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Z      = XVALUE(IELVAR(ILSTRT+     2))
       L      = EPVALU(IPSTRT+     1)
       C      = EPVALU(IPSTRT+     2)
       A      = EPVALU(IPSTRT+     3)
       PIL    = PI * L                                   
       U      = L * Z + C - A                            
       V      = PIL * X + PI * C                         
       SINV   = SIN( V )                                 
       COSV   = COS( V )                                 
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= U * SINV                                 
       ELSE
        FUVALS(IGSTRT+     1)= PIL * U * COSV                           
        FUVALS(IGSTRT+     2)= L * SINV                                 
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=- PIL * PIL * U * SINV                   
         FUVALS(IHSTRT+     2)=L * PIL * COSV                           
         FUVALS(IHSTRT+     3)=0.0                                      
        END IF
       END IF
    3 CONTINUE
      RETURN
      END
