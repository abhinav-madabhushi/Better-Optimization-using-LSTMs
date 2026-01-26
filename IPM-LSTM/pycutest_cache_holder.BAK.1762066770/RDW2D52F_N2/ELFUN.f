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
C  Problem name : RDW2D52F  
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION U1    , U2    , U3    , U4    , F1    
      DOUBLE PRECISION F2    , F3    , F4    , V1    , V2    
      DOUBLE PRECISION V3    , V4    , UV1   , UV2   , UV3   
      DOUBLE PRECISION UV4   , C1    , C2    , C3    , C4    
      IFSTAT = 0
      DO    11 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
       IELTYP = ITYPEE(IELEMN)
       GO TO (    1,    2,    3,    4,    5,    6,    7,    8,
     *            9,   10
     *                                                        ), IELTYP
C
C  Element type : M         
C
    1  CONTINUE
       U1     = XVALUE(IELVAR(ILSTRT+     1))
       U2     = XVALUE(IELVAR(ILSTRT+     2))
       U3     = XVALUE(IELVAR(ILSTRT+     3))
       U4     = XVALUE(IELVAR(ILSTRT+     4))
       V1     = EPVALU(IPSTRT+     1)
       V2     = EPVALU(IPSTRT+     2)
       V3     = EPVALU(IPSTRT+     3)
       V4     = EPVALU(IPSTRT+     4)
       UV1    = U1 - V1                                  
       UV2    = U2 - V2                                  
       UV3    = U3 - V3                                  
       UV4    = U4 - V4                                  
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= 2.0 * UV1 ** 2 + 2.0 * UV2 ** 2 +        
     *                  2.0 * UV3 ** 2 + 2.0 * UV4 ** 2 +        
     *                  2.0 * UV1*UV2 +  2.0 * UV1*UV3 +         
     *                        UV1*UV4 +       UV2*UV3 +          
     *                  2.0 * UV2*UV4 + 2.0 * UV3*UV4            
       ELSE
        FUVALS(IGSTRT+     1)= 4.0 * UV1 + 2.0 * UV2 + 2.0 * UV3 + UV4  
        FUVALS(IGSTRT+     2)= 2.0 * UV1 + 4.0 * UV2 + UV3 + 2.0 * UV4  
        FUVALS(IGSTRT+     3)= 2.0 * UV1 + UV2 + 4.0 * UV3 + 2.0 * UV4  
        FUVALS(IGSTRT+     4)= UV1 + 2.0 * UV2 + 2.0 * UV3 + 4.0 * UV4  
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=4.0                                      
         FUVALS(IHSTRT+     2)=2.0                                      
         FUVALS(IHSTRT+     4)=2.0                                      
         FUVALS(IHSTRT+     7)=1.0                                      
         FUVALS(IHSTRT+     3)=4.0                                      
         FUVALS(IHSTRT+     5)=1.0                                      
         FUVALS(IHSTRT+     8)=2.0                                      
         FUVALS(IHSTRT+     6)=4.0                                      
         FUVALS(IHSTRT+     9)=2.0                                      
         FUVALS(IHSTRT+    10)=4.0                                      
        END IF
       END IF
       GO TO    11
C
C  Element type : M0        
C
    2  CONTINUE
       F1     = XVALUE(IELVAR(ILSTRT+     1))
       F2     = XVALUE(IELVAR(ILSTRT+     2))
       F3     = XVALUE(IELVAR(ILSTRT+     3))
       F4     = XVALUE(IELVAR(ILSTRT+     4))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= 2.0 * F1 ** 2 + 2.0 * F2 ** 2 +          
     *                  2.0 * F3 ** 2 + 2.0 * F4 ** 2 +          
     *                  2.0 * F1*F2 +  2.0 * F1*F3 +             
     *                        F1*F4 +       F2*F3 +              
     *                  2.0 * F2*F4 + 2.0 * F3*F4                
       ELSE
        FUVALS(IGSTRT+     1)= 4.0 * F1 + 2.0 * F2 + 2.0 * F3 + F4      
        FUVALS(IGSTRT+     2)= 2.0 * F1 + 4.0 * F2 + F3 + 2.0 * F4      
        FUVALS(IGSTRT+     3)= 2.0 * F1 + F2 + 4.0 * F3 + 2.0 * F4      
        FUVALS(IGSTRT+     4)= F1 + 2.0 * F2 + 2.0 * F3 + 4.0 * F4      
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=4.0                                      
         FUVALS(IHSTRT+     2)=2.0                                      
         FUVALS(IHSTRT+     4)=2.0                                      
         FUVALS(IHSTRT+     7)=1.0                                      
         FUVALS(IHSTRT+     3)=4.0                                      
         FUVALS(IHSTRT+     5)=1.0                                      
         FUVALS(IHSTRT+     8)=2.0                                      
         FUVALS(IHSTRT+     6)=4.0                                      
         FUVALS(IHSTRT+     9)=2.0                                      
         FUVALS(IHSTRT+    10)=4.0                                      
        END IF
       END IF
       GO TO    11
C
C  Element type : A         
C
    3  CONTINUE
       U1     = XVALUE(IELVAR(ILSTRT+     1))
       U2     = XVALUE(IELVAR(ILSTRT+     2))
       U3     = XVALUE(IELVAR(ILSTRT+     3))
       U4     = XVALUE(IELVAR(ILSTRT+     4))
       C1     = 4.0                                      
       C2     = -1.0                                     
       C3     = -1.0                                     
       C4     = -2.0                                     
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= C1 * U1 + C2 * U2 + C3 * U3 + C4 * U4    
       ELSE
        FUVALS(IGSTRT+     1)= C1                                       
        FUVALS(IGSTRT+     2)= C2                                       
        FUVALS(IGSTRT+     3)= C3                                       
        FUVALS(IGSTRT+     4)= C4                                       
       END IF
       GO TO    11
C
C  Element type : B         
C
    4  CONTINUE
       U1     = XVALUE(IELVAR(ILSTRT+     1))
       U2     = XVALUE(IELVAR(ILSTRT+     2))
       U3     = XVALUE(IELVAR(ILSTRT+     3))
       U4     = XVALUE(IELVAR(ILSTRT+     4))
       C1     = -1.0                                     
       C2     = 4.0                                      
       C3     = -2.0                                     
       C4     = -1.0                                     
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= C1 * U1 + C2 * U2 + C3 * U3 + C4 * U4    
       ELSE
        FUVALS(IGSTRT+     1)= C1                                       
        FUVALS(IGSTRT+     2)= C2                                       
        FUVALS(IGSTRT+     3)= C3                                       
        FUVALS(IGSTRT+     4)= C4                                       
       END IF
       GO TO    11
C
C  Element type : C         
C
    5  CONTINUE
       U1     = XVALUE(IELVAR(ILSTRT+     1))
       U2     = XVALUE(IELVAR(ILSTRT+     2))
       U3     = XVALUE(IELVAR(ILSTRT+     3))
       U4     = XVALUE(IELVAR(ILSTRT+     4))
       C1     = -1.0                                     
       C2     = -2.0                                     
       C3     = 4.0                                      
       C4     = -1.0                                     
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= C1 * U1 + C2 * U2 + C3 * U3 + C4 * U4    
       ELSE
        FUVALS(IGSTRT+     1)= C1                                       
        FUVALS(IGSTRT+     2)= C2                                       
        FUVALS(IGSTRT+     3)= C3                                       
        FUVALS(IGSTRT+     4)= C4                                       
       END IF
       GO TO    11
C
C  Element type : D         
C
    6  CONTINUE
       U1     = XVALUE(IELVAR(ILSTRT+     1))
       U2     = XVALUE(IELVAR(ILSTRT+     2))
       U3     = XVALUE(IELVAR(ILSTRT+     3))
       U4     = XVALUE(IELVAR(ILSTRT+     4))
       C1     = -2.0                                     
       C2     = -1.0                                     
       C3     = -1.0                                     
       C4     = 4.0                                      
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= C1 * U1 + C2 * U2 + C3 * U3 + C4 * U4    
       ELSE
        FUVALS(IGSTRT+     1)= C1                                       
        FUVALS(IGSTRT+     2)= C2                                       
        FUVALS(IGSTRT+     3)= C3                                       
        FUVALS(IGSTRT+     4)= C4                                       
       END IF
       GO TO    11
C
C  Element type : P         
C
    7  CONTINUE
       F1     = XVALUE(IELVAR(ILSTRT+     1))
       F2     = XVALUE(IELVAR(ILSTRT+     2))
       F3     = XVALUE(IELVAR(ILSTRT+     3))
       F4     = XVALUE(IELVAR(ILSTRT+     4))
       C1     = 4.0                                      
       C2     = 2.0                                      
       C3     = 2.0                                      
       C4     = 1.0                                      
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= C1 * F1 + C2 * F2 + C3 * F3 + C4 * F4    
       ELSE
        FUVALS(IGSTRT+     1)= C1                                       
        FUVALS(IGSTRT+     2)= C2                                       
        FUVALS(IGSTRT+     3)= C3                                       
        FUVALS(IGSTRT+     4)= C4                                       
       END IF
       GO TO    11
C
C  Element type : Q         
C
    8  CONTINUE
       F1     = XVALUE(IELVAR(ILSTRT+     1))
       F2     = XVALUE(IELVAR(ILSTRT+     2))
       F3     = XVALUE(IELVAR(ILSTRT+     3))
       F4     = XVALUE(IELVAR(ILSTRT+     4))
       C1     = 2.0                                      
       C2     = 4.0                                      
       C3     = 1.0                                      
       C4     = 2.0                                      
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= C1 * F1 + C2 * F2 + C3 * F3 + C4 * F4    
       ELSE
        FUVALS(IGSTRT+     1)= C1                                       
        FUVALS(IGSTRT+     2)= C2                                       
        FUVALS(IGSTRT+     3)= C3                                       
        FUVALS(IGSTRT+     4)= C4                                       
       END IF
       GO TO    11
C
C  Element type : R         
C
    9  CONTINUE
       F1     = XVALUE(IELVAR(ILSTRT+     1))
       F2     = XVALUE(IELVAR(ILSTRT+     2))
       F3     = XVALUE(IELVAR(ILSTRT+     3))
       F4     = XVALUE(IELVAR(ILSTRT+     4))
       C1     = 2.0                                      
       C2     = 1.0                                      
       C3     = 4.0                                      
       C4     = 2.0                                      
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= C1 * F1 + C2 * F2 + C3 * F3 + C4 * F4    
       ELSE
        FUVALS(IGSTRT+     1)= C1                                       
        FUVALS(IGSTRT+     2)= C2                                       
        FUVALS(IGSTRT+     3)= C3                                       
        FUVALS(IGSTRT+     4)= C4                                       
       END IF
       GO TO    11
C
C  Element type : S         
C
   10  CONTINUE
       F1     = XVALUE(IELVAR(ILSTRT+     1))
       F2     = XVALUE(IELVAR(ILSTRT+     2))
       F3     = XVALUE(IELVAR(ILSTRT+     3))
       F4     = XVALUE(IELVAR(ILSTRT+     4))
       C1     = 1.0                                      
       C2     = 2.0                                      
       C3     = 2.0                                      
       C4     = 4.0                                      
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= C1 * F1 + C2 * F2 + C3 * F3 + C4 * F4    
       ELSE
        FUVALS(IGSTRT+     1)= C1                                       
        FUVALS(IGSTRT+     2)= C2                                       
        FUVALS(IGSTRT+     3)= C3                                       
        FUVALS(IGSTRT+     4)= C4                                       
       END IF
   11 CONTINUE
      RETURN
      END
