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
C  Problem name : HS57      
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION V1    , V2    , AA    , BB    , AM8   
      DOUBLE PRECISION CMV1  , E     , DED2  , R     , DRD1  
      DOUBLE PRECISION DRD2  , D2RD22
      INTRINSIC EXP   
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
C  Element type : OBSQ      
C
    1  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       AA     = EPVALU(IPSTRT+     1)
       BB     = EPVALU(IPSTRT+     2)
       AM8    = AA - 8.0                                 
       CMV1   = 0.49 - V1                                
       E      = EXP( - V2 * AM8 )                        
       DED2   = - AM8 * E                                
       R      = BB - V1 - CMV1 * E                       
       DRD1   = E - 1.0                                  
       DRD2   = - CMV1 * DED2                            
       D2RD22 = - CMV1 * AM8 * AM8 * E                   
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= R * R                                    
       ELSE
        FUVALS(IGSTRT+     1)= 2.0 * R * DRD1                           
        FUVALS(IGSTRT+     2)= 2.0 * R * DRD2                           
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=2.0 * DRD1 * DRD1                        
         FUVALS(IHSTRT+     2)=2.0 * ( DRD2 * DRD1 + R * DED2 )         
         FUVALS(IHSTRT+     3)=2.0 * ( DRD2 * DRD2 + R * D2RD22 )       
        END IF
       END IF
       GO TO     3
C
C  Element type : 2PR       
C
    2  CONTINUE
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= V1 * V2                                  
       ELSE
        FUVALS(IGSTRT+     1)= V2                                       
        FUVALS(IGSTRT+     2)= V1                                       
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=1.0                                      
         FUVALS(IHSTRT+     1)=0.0D+0
         FUVALS(IHSTRT+     3)=0.0D+0
        END IF
       END IF
    3 CONTINUE
      RETURN
      END
