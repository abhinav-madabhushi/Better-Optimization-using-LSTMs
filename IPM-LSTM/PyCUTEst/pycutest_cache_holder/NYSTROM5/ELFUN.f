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
C  Problem name : NYSTROM5  
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION X     , Y     , XX    , YY    , Z     
      DOUBLE PRECISION W     , ZZ    , Y1    , Y2    , Y3    
      DOUBLE PRECISION Z1    , Z2    , Z3    
      IFSTAT = 0
      DO    16 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
       IELTYP = ITYPEE(IELEMN)
       GO TO (    1,    2,    3,    4,    5,    6,    7,    8,
     *            9,   10,   11,   12,   13,   14,   15
     *                                                        ), IELTYP
C
C  Element type : 2PR       
C
    1  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y      = XVALUE(IELVAR(ILSTRT+     2))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= X * Y                                    
       ELSE
        FUVALS(IGSTRT+     1)= Y                                        
        FUVALS(IGSTRT+     2)= X                                        
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=1.0                                      
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
       GO TO    16
C
C  Element type : 2PRI2     
C
    2  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y1     = XVALUE(IELVAR(ILSTRT+     2))
       Y2     = XVALUE(IELVAR(ILSTRT+     3))
       XX     =   X     
       YY     =   Y1    
     *          + Y2    
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= XX * YY                                  
       ELSE
        FUVALS(IGSTRT+     1)= YY                                       
        FUVALS(IGSTRT+     2)= XX                                       
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=1.0                                      
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
       GO TO    16
C
C  Element type : 2PRI3     
C
    3  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y1     = XVALUE(IELVAR(ILSTRT+     2))
       Y2     = XVALUE(IELVAR(ILSTRT+     3))
       Y3     = XVALUE(IELVAR(ILSTRT+     4))
       XX     =   X     
       YY     =   Y1    
     *          + Y2    
     *          + Y3    
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= XX * YY                                  
       ELSE
        FUVALS(IGSTRT+     1)= YY                                       
        FUVALS(IGSTRT+     2)= XX                                       
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=1.0                                      
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
       GO TO    16
C
C  Element type : 3PR       
C
   10  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y      = XVALUE(IELVAR(ILSTRT+     2))
       Z      = XVALUE(IELVAR(ILSTRT+     3))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= X * Y * Z                                
       ELSE
        FUVALS(IGSTRT+     1)= Y * Z                                    
        FUVALS(IGSTRT+     2)= X * Z                                    
        FUVALS(IGSTRT+     3)= X * Y                                    
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=Z                                        
         FUVALS(IHSTRT+     4)=Y                                        
         FUVALS(IHSTRT+     5)=X                                        
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
         FUVALS(IHSTRT+     6)=0.0D+0
        END IF
       END IF
       GO TO    16
C
C  Element type : 3PRI2     
C
   11  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y      = XVALUE(IELVAR(ILSTRT+     2))
       Z1     = XVALUE(IELVAR(ILSTRT+     3))
       Z2     = XVALUE(IELVAR(ILSTRT+     4))
       XX     =   X     
       YY     =   Y     
       ZZ     =   Z1    
     *          + Z2    
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= XX * YY * ZZ                             
       ELSE
        FUVALS(IGSTRT+     1)= YY * ZZ                                  
        FUVALS(IGSTRT+     2)= XX * ZZ                                  
        FUVALS(IGSTRT+     3)= XX * YY                                  
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=ZZ                                       
         FUVALS(IHSTRT+     4)=YY                                       
         FUVALS(IHSTRT+     5)=XX                                       
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
         FUVALS(IHSTRT+     6)=0.0D+0
        END IF
       END IF
       GO TO    16
C
C  Element type : 3PRI3     
C
   12  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y      = XVALUE(IELVAR(ILSTRT+     2))
       Z1     = XVALUE(IELVAR(ILSTRT+     3))
       Z2     = XVALUE(IELVAR(ILSTRT+     4))
       Z3     = XVALUE(IELVAR(ILSTRT+     5))
       XX     =   X     
       YY     =   Y     
       ZZ     =   Z1    
     *          + Z2    
     *          + Z3    
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= XX * YY * ZZ                             
       ELSE
        FUVALS(IGSTRT+     1)= YY * ZZ                                  
        FUVALS(IGSTRT+     2)= XX * ZZ                                  
        FUVALS(IGSTRT+     3)= XX * YY                                  
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=ZZ                                       
         FUVALS(IHSTRT+     4)=YY                                       
         FUVALS(IHSTRT+     5)=XX                                       
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
         FUVALS(IHSTRT+     6)=0.0D+0
        END IF
       END IF
       GO TO    16
C
C  Element type : 4PR       
C
    4  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y      = XVALUE(IELVAR(ILSTRT+     2))
       Z      = XVALUE(IELVAR(ILSTRT+     3))
       W      = XVALUE(IELVAR(ILSTRT+     4))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= X * Y * Z * W                            
       ELSE
        FUVALS(IGSTRT+     1)= Y * Z * W                                
        FUVALS(IGSTRT+     2)= X * Z * W                                
        FUVALS(IGSTRT+     3)= X * Y * W                                
        FUVALS(IGSTRT+     4)= X * Y * Z                                
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=Z * W                                    
         FUVALS(IHSTRT+     4)=Y * W                                    
         FUVALS(IHSTRT+     7)=Y * Z                                    
         FUVALS(IHSTRT+     5)=X * W                                    
         FUVALS(IHSTRT+     8)=X * Z                                    
         FUVALS(IHSTRT+     9)=X * Y                                    
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
         FUVALS(IHSTRT+     6)=0.0D+0
         FUVALS(IHSTRT+    10)=0.0D+0
        END IF
       END IF
       GO TO    16
C
C  Element type : XY2       
C
    5  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y      = XVALUE(IELVAR(ILSTRT+     2))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= X * Y * Y                                
       ELSE
        FUVALS(IGSTRT+     1)= Y * Y                                    
        FUVALS(IGSTRT+     2)= 2.0 * X * Y                              
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=2.0 * Y                                  
         FUVALS(IHSTRT+     3)=2.0 * X                                  
         FUVALS(IHSTRT+     1)=0.0D+0
        END IF
       END IF
       GO TO    16
C
C  Element type : XY2I2     
C
    6  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y1     = XVALUE(IELVAR(ILSTRT+     2))
       Y2     = XVALUE(IELVAR(ILSTRT+     3))
       XX     =   X     
       YY     =   Y1    
     *          + Y2    
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= XX * YY * YY                             
       ELSE
        FUVALS(IGSTRT+     1)= YY * YY                                  
        FUVALS(IGSTRT+     2)= 2.0 * XX * YY                            
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=2.0 * YY                                 
         FUVALS(IHSTRT+     3)=2.0 * XX                                 
         FUVALS(IHSTRT+     1)=0.0D+0
        END IF
       END IF
       GO TO    16
C
C  Element type : XY2I3     
C
    7  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y1     = XVALUE(IELVAR(ILSTRT+     2))
       Y2     = XVALUE(IELVAR(ILSTRT+     3))
       Y3     = XVALUE(IELVAR(ILSTRT+     4))
       XX     =   X     
       YY     =   Y1    
     *          + Y2    
     *          + Y3    
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= XX * YY * YY                             
       ELSE
        FUVALS(IGSTRT+     1)= YY * YY                                  
        FUVALS(IGSTRT+     2)= 2.0 * XX * YY                            
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=2.0 * YY                                 
         FUVALS(IHSTRT+     3)=2.0 * XX                                 
         FUVALS(IHSTRT+     1)=0.0D+0
        END IF
       END IF
       GO TO    16
C
C  Element type : XY3       
C
    8  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y      = XVALUE(IELVAR(ILSTRT+     2))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= X * Y**3                                 
       ELSE
        FUVALS(IGSTRT+     1)= Y**3                                     
        FUVALS(IGSTRT+     2)= 3.0 * X * Y**2                           
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=3.0 * Y**2                               
         FUVALS(IHSTRT+     3)=6.0 * X * Y                              
         FUVALS(IHSTRT+     1)=0.0D+0
        END IF
       END IF
       GO TO    16
C
C  Element type : XY4       
C
    9  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y      = XVALUE(IELVAR(ILSTRT+     2))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= X * Y**4                                 
       ELSE
        FUVALS(IGSTRT+     1)= Y**4                                     
        FUVALS(IGSTRT+     2)= 4.0 * X * Y**3                           
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=4.0 * Y**3                               
         FUVALS(IHSTRT+     3)=12.0 * X * Y**2                          
         FUVALS(IHSTRT+     1)=0.0D+0
        END IF
       END IF
       GO TO    16
C
C  Element type : XY2Z      
C
   13  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y      = XVALUE(IELVAR(ILSTRT+     2))
       Z      = XVALUE(IELVAR(ILSTRT+     3))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= X * Y * Y * Z                            
       ELSE
        FUVALS(IGSTRT+     1)= Y * Y * Z                                
        FUVALS(IGSTRT+     2)= 2.0 * X * Y * Z                          
        FUVALS(IGSTRT+     3)= X * Y * Y                                
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=2.0 * Y * Z                              
         FUVALS(IHSTRT+     4)=Y * Y                                    
         FUVALS(IHSTRT+     3)=2.0 * X * Z                              
         FUVALS(IHSTRT+     5)=2.0 * X * Y                              
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     6)=0.0D+0
        END IF
       END IF
       GO TO    16
C
C  Element type : XY2ZI2    
C
   14  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y      = XVALUE(IELVAR(ILSTRT+     2))
       Z1     = XVALUE(IELVAR(ILSTRT+     3))
       Z2     = XVALUE(IELVAR(ILSTRT+     4))
       XX     =   X     
       YY     =   Y     
       ZZ     =   Z1    
     *          + Z2    
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= XX * YY * YY * ZZ                        
       ELSE
        FUVALS(IGSTRT+     1)= YY * YY * ZZ                             
        FUVALS(IGSTRT+     2)= 2.0 * XX * YY * ZZ                       
        FUVALS(IGSTRT+     3)= XX * YY * YY                             
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=2.0 * YY * ZZ                            
         FUVALS(IHSTRT+     4)=YY * YY                                  
         FUVALS(IHSTRT+     3)=2.0 * XX * ZZ                            
         FUVALS(IHSTRT+     5)=2.0 * XX * YY                            
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     6)=0.0D+0
        END IF
       END IF
       GO TO    16
C
C  Element type : XY2ZI3    
C
   15  CONTINUE
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y      = XVALUE(IELVAR(ILSTRT+     2))
       Z1     = XVALUE(IELVAR(ILSTRT+     3))
       Z2     = XVALUE(IELVAR(ILSTRT+     4))
       Z3     = XVALUE(IELVAR(ILSTRT+     5))
       XX     =   X     
       YY     =   Y     
       ZZ     =   Z1    
     *          + Z2    
     *          + Z3    
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= XX * YY * YY * ZZ                        
       ELSE
        FUVALS(IGSTRT+     1)= YY * YY * ZZ                             
        FUVALS(IGSTRT+     2)= 2.0 * XX * YY * ZZ                       
        FUVALS(IGSTRT+     3)= XX * YY * YY                             
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=2.0 * YY * ZZ                            
         FUVALS(IHSTRT+     4)=YY * YY                                  
         FUVALS(IHSTRT+     3)=2.0 * XX * ZZ                            
         FUVALS(IHSTRT+     5)=2.0 * XX * YY                            
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     6)=0.0D+0
        END IF
       END IF
   16 CONTINUE
      RETURN
      END
