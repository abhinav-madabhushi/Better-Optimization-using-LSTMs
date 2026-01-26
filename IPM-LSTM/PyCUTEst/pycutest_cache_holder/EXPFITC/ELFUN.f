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
C  Problem name : EXPFITC   
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION P0    , P1    , P2    , Q1    , Q2    
      DOUBLE PRECISION T     , T2    , ET    , QT    , ETQT  
      DOUBLE PRECISION ETQT2 , ETQT3 , PT    , TM5   , TM5SQ 
      DOUBLE PRECISION F     , TWOF  , DFDP0 , DFDP1 , DFDP2 
      DOUBLE PRECISION DFDQ1 , DFDQ2 , D2P0Q1, D2P0Q2, D2P1Q1
      DOUBLE PRECISION D2P1Q2, D2P2Q1, D2P2Q2, D2Q1Q1, D2Q1Q2
      DOUBLE PRECISION D2Q2Q2
      INTRINSIC EXP   
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : FIT       
C
       P0     = XVALUE(IELVAR(ILSTRT+     1))
       P1     = XVALUE(IELVAR(ILSTRT+     2))
       P2     = XVALUE(IELVAR(ILSTRT+     3))
       Q1     = XVALUE(IELVAR(ILSTRT+     4))
       Q2     = XVALUE(IELVAR(ILSTRT+     5))
       T      = EPVALU(IPSTRT+     1)
       TM5    = T - 5.0                                  
       TM5SQ  = TM5 * TM5                                
       T2     = T * T                                    
       ET     = EXP( T )                                 
       QT     = 1.0 + Q1 * TM5 + Q2 * TM5SQ              
       ETQT   = ET * QT                                  
       ETQT2  = ETQT * QT                                
       ETQT3  = ETQT2 * QT                               
       PT     = P0 + P1 * T + P2 * T2                    
       F      = PT / ETQT - 1.0                          
       TWOF   = F + F                                    
       DFDP0  = 1.0 / ETQT                               
       DFDP1  = T / ETQT                                 
       DFDP2  = T2 / ETQT                                
       DFDQ1  = - PT * TM5 / ETQT2                       
       DFDQ2  = - PT * TM5SQ / ETQT2                     
       D2P0Q1 = - TM5 / ETQT2                            
       D2P0Q2 = - TM5SQ / ETQT2                          
       D2P1Q1 = - T * TM5 / ETQT2                        
       D2P1Q2 = - T * TM5SQ / ETQT2                      
       D2P2Q1 = - T2 * TM5 / ETQT2                       
       D2P2Q2 = - T2 * TM5SQ / ETQT2                     
       D2Q1Q1 = 2.0 * PT * TM5SQ / ETQT3                 
       D2Q1Q2 = 2.0 * PT * TM5SQ * TM5 / ETQT3           
       D2Q2Q2 = 2.0 * PT * TM5SQ * TM5SQ / ETQT3         
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= F * F                                    
       ELSE
        FUVALS(IGSTRT+     1)= TWOF * DFDP0                             
        FUVALS(IGSTRT+     2)= TWOF * DFDP1                             
        FUVALS(IGSTRT+     3)= TWOF * DFDP2                             
        FUVALS(IGSTRT+     4)= TWOF * DFDQ1                             
        FUVALS(IGSTRT+     5)= TWOF * DFDQ2                             
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=2.0 * DFDP0 * DFDP0                      
         FUVALS(IHSTRT+     2)=2.0 * DFDP0 * DFDP1                      
         FUVALS(IHSTRT+     4)=2.0 * DFDP0 * DFDP2                      
         FUVALS(IHSTRT+     3)=2.0 * DFDP1 * DFDP1                      
         FUVALS(IHSTRT+     5)=2.0 * DFDP1 * DFDP2                      
         FUVALS(IHSTRT+     6)=2.0 * DFDP2 * DFDP2                      
         FUVALS(IHSTRT+     7)=TWOF * D2P0Q1 + 2.0 * DFDP0 * DFDQ1      
         FUVALS(IHSTRT+    11)=TWOF * D2P0Q2 + 2.0 * DFDP0 * DFDQ2      
         FUVALS(IHSTRT+     8)=TWOF * D2P1Q1 + 2.0 * DFDP1 * DFDQ1      
         FUVALS(IHSTRT+    12)=TWOF * D2P1Q2 + 2.0 * DFDP1 * DFDQ2      
         FUVALS(IHSTRT+     9)=TWOF * D2P2Q1 + 2.0 * DFDP2 * DFDQ1      
         FUVALS(IHSTRT+    13)=TWOF * D2P2Q2 + 2.0 * DFDP2 * DFDQ2      
         FUVALS(IHSTRT+    10)=TWOF * D2Q1Q1 + 2.0 * DFDQ1 * DFDQ1      
         FUVALS(IHSTRT+    14)=TWOF * D2Q1Q2 + 2.0 * DFDQ1 * DFDQ2      
         FUVALS(IHSTRT+    15)=TWOF * D2Q2Q2 + 2.0 * DFDQ2 * DFDQ2      
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
