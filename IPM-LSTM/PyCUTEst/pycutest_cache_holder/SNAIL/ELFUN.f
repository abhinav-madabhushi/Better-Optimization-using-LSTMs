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
C  Problem name : SNAIL     
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION X     , Y     , CL    , CU    , X2    
      DOUBLE PRECISION Y2    , A     , B     , R2    , R     
      DOUBLE PRECISION R3    , R4    , DRDX  , DRDY  , D2RDX2
      DOUBLE PRECISION D2RDY2, D2RDXY, D     , D2    , D3    
      DOUBLE PRECISION THETA , DTDX  , DTDY  , D2TDX2, D2TDY2
      DOUBLE PRECISION D2TDXY, ARG   , C     , DCDX  , DCDY  
      DOUBLE PRECISION D2CDX2, D2CDY2, D2CDXY, S     , U     
      DOUBLE PRECISION DUDX  , DUDY  , D2UDX2, D2UDY2, D2UDXY
      DOUBLE PRECISION V     , DVDX  , DVDY  , D2VDX2, D2VDY2
      DOUBLE PRECISION D2VDXY
      INTRINSIC SQRT  , COS   , SIN   , ATAN2 
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : SPIRAL    
C
       X      = XVALUE(IELVAR(ILSTRT+     1))
       Y      = XVALUE(IELVAR(ILSTRT+     2))
       CL     = EPVALU(IPSTRT+     1)
       CU     = EPVALU(IPSTRT+     2)
       A      = 0.5 * ( CU + CL )                        
       B      = 0.5 * ( CU - CL )                        
       X2     = X * X                                    
       Y2     = Y * Y                                    
       R2     = X2 + Y2                                  
       D      = 1.0 + R2                                 
       D2     = D * D                                    
       D3     = D2 * D                                   
       U      = R2 / D                                   
       DUDX   = ( X + X ) / D2                           
       DUDY   = ( Y + Y ) / D2                           
       D2UDX2 = 2.0 * ( D - 4.0 * X2 ) / D3              
       D2UDY2 = 2.0 * ( D - 4.0 * Y2 ) / D3              
       D2UDXY = -8.0 * X * Y / D3                        
       THETA  = ATAN2( Y , X )                           
       DTDX   = - Y / R2                                 
       DTDY   = X / R2                                   
       R4     = R2 * R2                                  
       D2TDX2 = 2.0 * X * Y / R4                         
       D2TDY2 = -2.0 * Y * X / R4                        
       D2TDXY = ( Y2 - X2 ) / R4                         
       R      = SQRT( R2 )                               
       R3     = R * R2                                   
       DRDX   = X / R                                    
       DRDY   = Y / R                                    
       D2RDX2 = Y2 / R3                                  
       D2RDY2 = X2 / R3                                  
       D2RDXY = - X * Y / R3                             
       ARG    = R - THETA                                
       S      = B * SIN( ARG )                           
       C      = B * COS( ARG )                           
       DCDX   = - S * ( DRDX - DTDX )                    
       DCDY   = - S * ( DRDY - DTDY )                    
       D2CDX2 = - C * ( DRDX - DTDX )**2                 
     *                - S * ( D2RDX2 - D2TDX2 )          
       D2CDY2 = - C * ( DRDY - DTDY )**2                 
     *                - S * ( D2RDY2 - D2TDY2 )          
       D2CDXY = - C * ( DRDX - DTDX ) * ( DRDY - DTDY )  
     *                - S * ( D2RDXY - D2TDXY )          
       V      = 1.0 + A * R - R * C                      
       DVDX   = A * DRDX - DRDX * C - R * DCDX           
       DVDY   = A * DRDY - DRDY * C - R * DCDY           
       D2VDX2 = A * D2RDX2 - D2RDX2 * C                  
     *                - 2.0 * DRDX * DCDX                
     *                - R * D2CDX2                       
       D2VDY2 = A * D2RDY2 - D2RDY2 * C                  
     *                - 2.0 * DRDY * DCDY                
     *                - R * D2CDY2                       
       D2VDXY = A * D2RDXY - D2RDXY * C                  
     *                - DRDX * DCDY                      
     *                - DRDY * DCDX                      
     *                - R * D2CDXY                       
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= U * V                                    
       ELSE
        FUVALS(IGSTRT+     1)= DUDX * V + U * DVDX                      
        FUVALS(IGSTRT+     2)= DUDY * V + U * DVDY                      
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=D2UDX2 * V + 2.0 * DUDX * DVDX           
     *                                + U * D2VDX2                      
         FUVALS(IHSTRT+     2)=D2UDXY * V + DUDX * DVDY                 
     *                                + DUDY * DVDX                     
     *                                + U * D2VDXY                      
         FUVALS(IHSTRT+     3)=D2UDY2 * V + 2.0 * DUDY * DVDY           
     *                                + U * D2VDY2                      
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
