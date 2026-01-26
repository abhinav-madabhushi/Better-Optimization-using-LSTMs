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
C  Problem name : CHWIRUT2  
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION V1    , V2    , V3    , X     , E     
      DOUBLE PRECISION V2PV3X, V2PV32, V2PV33, EX    , EX2   
      INTRINSIC EXP   
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : E16       
C
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       V3     = XVALUE(IELVAR(ILSTRT+     3))
       X      = EPVALU(IPSTRT+     1)
       E      = EXP( - V1 * X )                          
       EX     = E * X                                    
       EX2    = EX * X                                   
       V2PV3X = V2 + V3 * X                              
       V2PV32 = V2PV3X * V2PV3X                          
       V2PV33 = V2PV3X * V2PV32                          
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= E / V2PV3X                               
       ELSE
        FUVALS(IGSTRT+     1)= - EX / V2PV3X                            
        FUVALS(IGSTRT+     2)= - E / V2PV32                             
        FUVALS(IGSTRT+     3)= - EX / V2PV32                            
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=EX2 / V2PV3X                             
         FUVALS(IHSTRT+     2)=EX / V2PV32                              
         FUVALS(IHSTRT+     4)=EX2 / V2PV32                             
         FUVALS(IHSTRT+     3)=2.0 * E / V2PV33                         
         FUVALS(IHSTRT+     5)=2.0 * EX / V2PV33                        
         FUVALS(IHSTRT+     6)=2.0 * EX2 / V2PV33                       
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
