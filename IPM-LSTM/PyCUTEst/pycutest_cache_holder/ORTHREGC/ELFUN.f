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
C  Problem name : ORTHREGC  
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION H     , X     , Y     , G     
      IFSTAT = 0
      DO     4 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
       IELTYP = ITYPEE(IELEMN)
       GO TO (    1,    2,    3
     *                                                        ), IELTYP
C
C  Element type : HXX       
C
    1  CONTINUE
       H      = XVALUE(IELVAR(ILSTRT+     1))
       X      = XVALUE(IELVAR(ILSTRT+     2))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= H * X * X                                
       ELSE
        FUVALS(IGSTRT+     1)= X * X                                    
        FUVALS(IGSTRT+     2)= 2.0 * H * X                              
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=X + X                                    
         FUVALS(IHSTRT+     3)=H + H                                    
         FUVALS(IHSTRT+     1)=0.0D+0
        END IF
       END IF
       GO TO     4
C
C  Element type : HXY       
C
    2  CONTINUE
       H      = XVALUE(IELVAR(ILSTRT+     1))
       X      = XVALUE(IELVAR(ILSTRT+     2))
       Y      = XVALUE(IELVAR(ILSTRT+     3))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= H * X * Y                                
       ELSE
        FUVALS(IGSTRT+     1)= X * Y                                    
        FUVALS(IGSTRT+     2)= H * Y                                    
        FUVALS(IGSTRT+     3)= H * X                                    
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=Y                                        
         FUVALS(IHSTRT+     4)=X                                        
         FUVALS(IHSTRT+     5)=H                                        
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
         FUVALS(IHSTRT+     6)=0.0D+0
        END IF
       END IF
       GO TO     4
C
C  Element type : GX        
C
    3  CONTINUE
       G      = XVALUE(IELVAR(ILSTRT+     1))
       X      = XVALUE(IELVAR(ILSTRT+     2))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= G * X                                    
       ELSE
        FUVALS(IGSTRT+     1)= X                                        
        FUVALS(IGSTRT+     2)= G                                        
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=1.0                                      
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
    4 CONTINUE
      RETURN
      END
