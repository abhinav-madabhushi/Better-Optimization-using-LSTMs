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
C  Problem name : ACOPR30   
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION X     , V1    , V2    , V3    
      IFSTAT = 0
      DO     8 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
       IELTYP = ITYPEE(IELEMN)
       GO TO (    1,    2,    3,    4,    5,    6,    7
     *                                                        ), IELTYP
C
C  Element type : 123456789S
C
    1  CONTINUE
       X = XVALUE(IELVAR(ILSTRT+     1))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= 5.0D-1 * X * X
       ELSE
        FUVALS(IGSTRT+     1)= X
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)= 1.0D+0
        END IF
       END IF
       GO TO     8
C
C  Element type : P2        
C
    2  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= V1 ** 2                                  
       ELSE
        FUVALS(IGSTRT+     1)= 2.0D+0 * V1                              
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=2.0D+0                                   
        END IF
       END IF
       GO TO     8
C
C  Element type : P4        
C
    3  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= V1 ** 4                                  
       ELSE
        FUVALS(IGSTRT+     1)= 4.0D+0 * V1 ** 3                         
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=12.0D+0 * V1 ** 2                        
        END IF
       END IF
       GO TO     8
C
C  Element type : P11       
C
    4  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= V1 * V2                                  
       ELSE
        FUVALS(IGSTRT+     1)= V2                                       
        FUVALS(IGSTRT+     2)= V1                                       
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=1.0D+0                                   
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
       GO TO     8
C
C  Element type : P31       
C
    5  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= V2 * ( V1 ** 3 )                         
       ELSE
        FUVALS(IGSTRT+     1)= 3.0D+0 * V2 * V1 ** 2                    
        FUVALS(IGSTRT+     2)= V1 ** 3                                  
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=6.0D+0 * V2 * V1                         
         FUVALS(IHSTRT+     2)=3.0D+0 * V1 ** 2                         
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
       GO TO     8
C
C  Element type : P22       
C
    6  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= ( V1 * V2 ) ** 2                         
       ELSE
        FUVALS(IGSTRT+     1)= 2.0D+0 * V1 * V2 ** 2                    
        FUVALS(IGSTRT+     2)= 2.0D+0 * V2 * V1 ** 2                    
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=2.0D+0 * V2 ** 2                         
         FUVALS(IHSTRT+     2)=4.0D+0 * V1 * V2                         
         FUVALS(IHSTRT+     3)=2.0D+0 * V1 ** 2                         
        END IF
       END IF
       GO TO     8
C
C  Element type : P211      
C
    7  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       V3     = XVALUE(IELVAR(ILSTRT+     3))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= ( V3 * V2 ) * ( V1 ** 2 )                
       ELSE
        FUVALS(IGSTRT+     1)= 2.0D+0 * V3 * V2 * V1                    
        FUVALS(IGSTRT+     2)= V3 * V1 ** 2                             
        FUVALS(IGSTRT+     3)= V2 * V1 ** 2                             
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=2.0D+0 * V3 * V2                         
         FUVALS(IHSTRT+     2)=2.0D+0 * V3 * V1                         
         FUVALS(IHSTRT+     4)=2.0D+0 * V2 * V1                         
         FUVALS(IHSTRT+     5)=V1 ** 2                                  
         FUVALS(IHSTRT+     6)=0.0D+0                                   
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
    8 CONTINUE
      RETURN
      END
