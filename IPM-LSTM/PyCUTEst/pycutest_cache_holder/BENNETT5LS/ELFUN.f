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
C  Problem name : BENNETT5LS
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION V1    , V2    , V3    , X     , E     
      DOUBLE PRECISION V3INV , V2PX  , V2PXP , V2PXP1, V2PXP2
      DOUBLE PRECISION V2PXL 
      INTRINSIC LOG   
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : E15       
C
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       V3     = XVALUE(IELVAR(ILSTRT+     3))
       X      = EPVALU(IPSTRT+     1)
       V3INV  = 1.0 / V3                                 
       V2PX   = V2 + X                                   
       V2PXL  = LOG( V2PX )                              
       V2PXP  = V2PX ** V3INV                            
       V2PXP1 = V2PX ** ( V3INV + 1.0 )                  
       V2PXP2 = V2PX ** ( V3INV + 2.0 )                  
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= V1 / V2PXP                               
       ELSE
        FUVALS(IGSTRT+     1)= 1.0 / V2PXP                              
        FUVALS(IGSTRT+     2)= - V1 / ( V3 * V2PXP1 )                   
        FUVALS(IGSTRT+     3)= V1 * V2PXL / ( V2PXP * V3 ** 2 )         
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=- 1.0 / (V3 * V2PXP1)                    
         FUVALS(IHSTRT+     4)=V2PXL / ( V2PXP * V3 ** 2 )              
         FUVALS(IHSTRT+     3)=V1 * (1.0 / V3 + 1.0) / ( V3 * V2PXP2 )  
         FUVALS(IHSTRT+     5)=V1 / ( V2PX * V2PXP * V3 **2 )           
     *                         - V1 * V2PXL / ( V2PXP1 * V3 ** 3 )      
         FUVALS(IHSTRT+     6)=V1 * V2PXL**2 / ( V2PXP * V3 ** 4 )      
     *                         - 2.0 * V1 * V2PXL / ( V2PXP  * V3 ** 3 )
         FUVALS(IHSTRT+     1)=0.0D+0
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
