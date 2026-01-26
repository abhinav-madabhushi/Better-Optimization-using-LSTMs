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
C  Problem name : HS109     
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION X     , Y     , Z     , U1    , U2    
      DOUBLE PRECISION U3    , W     , P     , ARG   , S     
      DOUBLE PRECISION C     
      INTRINSIC SIN   , COS   
      IFSTAT = 0
      DO     7 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
       IELTYP = ITYPEE(IELEMN)
       GO TO (    1,    2,    3,    4,    5,    6
     *                                                        ), IELTYP
C
C  Element type : CB        
C
    2  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= X**3                                     
       ELSE
        FUVALS(IGSTRT+     1)= 3.0 * (X**2)                             
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=6.0 * X                                  
        END IF
       END IF
       GO TO     7
C
C  Element type : SQ        
C
    1  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= X**2                                     
       ELSE
        FUVALS(IGSTRT+     1)= 2.0 * X                                  
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=2.0                                      
        END IF
       END IF
       GO TO     7
C
C  Element type : SIN       
C
    3  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y      = XVALUE(IELVAR(ILSTRT+     2))
       Z      = XVALUE(IELVAR(ILSTRT+     3))
       P      = EPVALU(IPSTRT+     1)
       ARG    = P*Z - 0.25                               
       S      = SIN( ARG )                               
       C      = COS( ARG )                               
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= X*Y*S                                    
       ELSE
        FUVALS(IGSTRT+     1)= Y*S                                      
        FUVALS(IGSTRT+     2)= X*S                                      
        FUVALS(IGSTRT+     3)= P*X*Y*C                                  
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=S                                        
         FUVALS(IHSTRT+     4)=P*Y*C                                    
         FUVALS(IHSTRT+     5)=P*X*C                                    
         FUVALS(IHSTRT+     6)=-P*P*X*Y*S                               
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
       GO TO     7
C
C  Element type : SIN2      
C
    4  CONTINUE
       W      = XVALUE(IELVAR(ILSTRT+     1))
       X      = XVALUE(IELVAR(ILSTRT+     2))
       Y      = XVALUE(IELVAR(ILSTRT+     3))
       Z      = XVALUE(IELVAR(ILSTRT+     4))
       U1     =   W     
       U2     =   X     
       U3     =   Y     
     *          - Z     
       ARG    = U3 - 0.25                                
       C      = COS( ARG )                               
       S      = SIN( ARG )                               
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= U1*U2*S                                  
       ELSE
        FUVALS(IGSTRT+     1)= U2*S                                     
        FUVALS(IGSTRT+     2)= U1*S                                     
        FUVALS(IGSTRT+     3)= U1*U2*C                                  
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=S                                        
         FUVALS(IHSTRT+     4)=U2*C                                     
         FUVALS(IHSTRT+     5)=U1*C                                     
         FUVALS(IHSTRT+     6)=-U1*U2*S                                 
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
       GO TO     7
C
C  Element type : COS       
C
    5  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y      = XVALUE(IELVAR(ILSTRT+     2))
       Z      = XVALUE(IELVAR(ILSTRT+     3))
       P      = EPVALU(IPSTRT+     1)
       ARG    = P*Z - 0.25                               
       C      = COS( ARG )                               
       S      = SIN( ARG )                               
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= X*Y*C                                    
       ELSE
        FUVALS(IGSTRT+     1)= Y*C                                      
        FUVALS(IGSTRT+     2)= X*C                                      
        FUVALS(IGSTRT+     3)= -P*X*Y*S                                 
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=C                                        
         FUVALS(IHSTRT+     4)=-P*Y*S                                   
         FUVALS(IHSTRT+     5)=-P*X*S                                   
         FUVALS(IHSTRT+     6)=-P*P*X*Y*C                               
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
       GO TO     7
C
C  Element type : COS2      
C
    6  CONTINUE
       W      = XVALUE(IELVAR(ILSTRT+     1))
       X      = XVALUE(IELVAR(ILSTRT+     2))
       Y      = XVALUE(IELVAR(ILSTRT+     3))
       Z      = XVALUE(IELVAR(ILSTRT+     4))
       U1     =   W     
       U2     =   X     
       U3     =   Y     
     *          - Z     
       ARG    = U3 - 0.25                                
       C      = COS( ARG )                               
       S      = SIN( ARG )                               
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= U1*U2*C                                  
       ELSE
        FUVALS(IGSTRT+     1)= U2*C                                     
        FUVALS(IGSTRT+     2)= U1*C                                     
        FUVALS(IGSTRT+     3)= -U1*U2*S                                 
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=C                                        
         FUVALS(IHSTRT+     4)=-U2*S                                    
         FUVALS(IHSTRT+     5)=-U1*S                                    
         FUVALS(IHSTRT+     6)=-U1*U2*C                                 
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
    7 CONTINUE
      RETURN
      END
