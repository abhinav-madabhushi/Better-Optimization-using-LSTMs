C                                                                       
C FORTRAN ROUTINES                                                      
C                                                                       
C                                                                       
C CALCULATE FUNCTION VALUE OF DISCRETE TIME SYSTEM EQUATIONS AND CONSTRA
C THIS FUNCTION PROVIDES AN INTERFACE TO SRFG.                          
C                                                                       
      DOUBLE PRECISION FUNCTION SAROFN(                                 
     1   RK, RIDX, SYSEQ,                                               
     2   X1, X2, X3, X4, X5, X6, X7, X8, X9,                            
     3   U1, U2)                                                        
C	                                                                      
      INTEGER NX, NU, NG                                                
      PARAMETER(NX = 9, NU = 2, NG = 2)                                 
      DOUBLE PRECISION RK, RIDX                                         
      LOGICAL SYSEQ                                                     
      DOUBLE PRECISION X1, X2, X3, X4, X5, X6, X7, X8, X9               
      DOUBLE PRECISION U1, U2                                           
      INTEGER K, IDX                                                    
      DOUBLE PRECISION X(NX), U(NU), F(NX), G(NG)                       
      DOUBLE PRECISION RKZBAK                                           
C                                                                       
C     SAVE                                                              
C                                                                       
C     DATA RKZBAK /-1.0D0/                                              
C                                                                       
C     IF ((RK .NE. RKZBAK)                                              
C    *     .OR. (X1 .NE. X(1))                                          
C    *     .OR. (X2 .NE. X(2))                                          
C    *     .OR. (X3 .NE. X(3))                                          
C    *     .OR. (X4 .NE. X(4))                                          
C    *     .OR. (X5 .NE. X(5))                                          
C    *     .OR. (X6 .NE. X(6))                                          
C    *     .OR. (X7 .NE. X(7))                                          
C    *     .OR. (X8 .NE. X(8))                                          
C    *     .OR. (X9 .NE. X(9))                                          
C    *     .OR. (U1 .NE. U(1))                                          
C    *     .OR. (U2 .NE. U(2))) THEN                                    
C                                                                       
         X(1) = X1                                                      
         X(2) = X2                                                      
         X(3) = X3                                                      
         X(4) = X4                                                      
         X(5) = X5                                                      
         X(6) = X6                                                      
         X(7) = X7                                                      
         X(8) = X8                                                      
         X(9) = X9                                                      
C                                                                       
         U(1) = U1                                                      
         U(2) = U2                                                      
C                                                                       
         K = RK                                                         
C        RKZBAK = RK                                                    
C                                                                       
         CALL SRFG(K, X, U, F, G)                                       
C     END IF                                                            
C                                                                       
      IDX = RIDX                                                        
      IF (SYSEQ) THEN                                                   
         SAROFN = F(IDX)                                                
      ELSE                                                              
         SAROFN = G(IDX)                                                
      END IF                                                            
C                                                                       
      RETURN                                                            
      END                                                               
C                                                                       
C                                                                       
C CALCULATE JACOBIAN OF DISCRETE TIME SYSTEM EQUATIONS AND CONSTRAINTS. 
C THIS FUNCTION PROVIDES AN INTERFACE TO GZSRFG THAT HAS BEEN DERIVED   
C FROM SRFG USING ADIFOR.                                               
C                                                                       
      DOUBLE PRECISION FUNCTION SAROGN(                                 
     1   RK, RIDX, SYSEQ,                                               
     2   X1, X2, X3, X4, X5, X6, X7, X8, X9,                            
     3   U1, U2,                                                        
     4   GX1, GX2, GX3, GX4, GX5, GX6, GX7, GX8, GX9,                   
     5   GU1, GU2)                                                      
C                                                                       
      INTEGER NX, NU, NG, PMAX                                          
      PARAMETER(NX = 9, NU = 2, NG = 2, PMAX = 11)                      
      DOUBLE PRECISION RK, RIDX                                         
      LOGICAL SYSEQ                                                     
      DOUBLE PRECISION X1, X2, X3, X4, X5, X6, X7, X8, X9               
      DOUBLE PRECISION U1, U2                                           
      DOUBLE PRECISION GX1, GX2, GX3, GX4, GX5, GX6, GX7, GX8, GX9      
      DOUBLE PRECISION GU1, GU2                                         
      INTEGER I, J, K, IDX                                              
      DOUBLE PRECISION X(NX), U(NU), F(NX), G(NG)                       
      DOUBLE PRECISION GZF(PMAX, NX), GZG(PMAX, NG)                     
      DOUBLE PRECISION GZX(PMAX, NX), GZU(PMAX, NU)                     
      DOUBLE PRECISION RKZBAK                                           
C                                                                       
C     SAVE                                                              
C                                                                       
C     DATA RKZBAK /-1.0D0/                                              
C                                                                       
C     IF ((RK .NE. RKZBAK)                                              
C    *     .OR. (X1 .NE. X(1))                                          
C    *     .OR. (X2 .NE. X(2))                                          
C    *     .OR. (X3 .NE. X(3))                                          
C    *     .OR. (X4 .NE. X(4))                                          
C    *     .OR. (X5 .NE. X(5))                                          
C    *     .OR. (X6 .NE. X(6))                                          
C    *     .OR. (X7 .NE. X(7))                                          
C    *     .OR. (X8 .NE. X(8))                                          
C    *     .OR. (X9 .NE. X(9))                                          
C    *     .OR. (U1 .NE. U(1))                                          
C    *     .OR. (U2 .NE. U(2))) THEN                                    
C                                                                       
         X(1) = X1                                                      
         X(2) = X2                                                      
         X(3) = X3                                                      
         X(4) = X4                                                      
         X(5) = X5                                                      
         X(6) = X6                                                      
         X(7) = X7                                                      
         X(8) = X8                                                      
         X(9) = X9                                                      
C                                                                       
         U(1) = U1                                                      
         U(2) = U2                                                      
C                                                                       
         K = RK                                                         
C        RKZBAK = RK                                                    
C                                                                       
         DO 120 I = 1, PMAX                                             
            DO 100 J = 1, NX                                            
               GZX(I, J) = 0.0D0                                        
 100        CONTINUE                                                    
            DO 110 J = 1, NU                                            
               GZU(I, J) = 0.0D0                                        
 110        CONTINUE                                                    
            IF (I .LE. NX) THEN                                         
               GZX(I, I) = 1.0D0                                        
            ELSE                                                        
               GZU(I, I - NX) = 1.0D0                                   
            END IF                                                      
 120     CONTINUE                                                       
C                                                                       
         CALL GZSRFG(PMAX, K, X, GZX, PMAX, U, GZU, PMAX,               
     *        F, GZF, PMAX, G, GZG, PMAX)                               
C                                                                       
C     END IF                                                            
C                                                                       
      IDX = RIDX                                                        
C                                                                       
      IF (SYSEQ) THEN                                                   
         GX1 = GZF(1,IDX)                                               
         GX2 = GZF(2,IDX)                                               
         GX3 = GZF(3,IDX)                                               
         GX4 = GZF(4,IDX)                                               
         GX5 = GZF(5,IDX)                                               
         GX6 = GZF(6,IDX)                                               
         GX7 = GZF(7,IDX)                                               
         GX8 = GZF(8,IDX)                                               
         GX9 = GZF(9,IDX)                                               
C                                                                       
         GU1 = GZF(10,IDX)                                              
         GU2 = GZF(11,IDX)                                              
C                                                                       
         SAROGN = F(IDX)                                                
      ELSE                                                              
         GX1 = GZG(1,IDX)                                               
         GX2 = GZG(2,IDX)                                               
         GX3 = GZG(3,IDX)                                               
         GX4 = GZG(4,IDX)                                               
         GX5 = GZG(5,IDX)                                               
         GX6 = GZG(6,IDX)                                               
         GX7 = GZG(7,IDX)                                               
         GX8 = GZG(8,IDX)                                               
         GX9 = GZG(9,IDX)                                               
C                                                                       
         GU1 = GZG(10,IDX)                                              
         GU2 = GZG(11,IDX)                                              
C                                                                       
         SAROGN = G(IDX)                                                
      END IF                                                            
C                                                                       
      RETURN                                                            
      END                                                               
C                                                                       
C                                                                       
C CALCULATE FUNCTION VALUES OF DISCRETE-TIME SYSTEM EQUATIONS AND CONSTR
C THIS SUBROUTINE DIVIDES EACH TIME STEP K INTO TWO SAMPLE PERIODS AND  
C INTEGRATES CONTINUOUS-TIME SYSTEM EQUATIONS.                          
C                                                                       
      SUBROUTINE SRFG(K, X, U, F, G)                                    
C                                                                       
      INTEGER NX, NU, NG, NP, NW                                        
      PARAMETER (NX = 9, NU = 2, NG = 2, NP = 1, NW = 1)                
      INTEGER K                                                         
      DOUBLE PRECISION X(NX), U(NU), F(NX), G(NG)                       
      INTEGER KK                                                        
      DOUBLE PRECISION DT, FH(NX), P(NP), W(NW)                         
C                                                                       
      KK = 2 * (K - 1) + 1                                              
      DT = 12.0D0                                                       
C                                                                       
      P(1) = U(1)                                                       
      CALL SRK4(KK, DT, X, P, FH, W)                                    
      G(1) = W(1)                                                       
C                                                                       
      P(1) = U(2)                                                       
      CALL SRK4(KK + 1, DT, FH, P, F, W)                                
      G(2) = W(1)                                                       
C                                                                       
      RETURN                                                            
      END                                                               
C                                                                       
C                                                                       
C RUNGE KUTTA TYPE INTEGRATION PROCEDURE.                               
C                                                                       
      SUBROUTINE SRK4(KK, DT, X, P, F, W)                               
C                                                                       
      INTEGER NX, NP, NW                                                
      PARAMETER (NX = 9, NP = 1, NW = 1)                                
      INTEGER KK                                                        
      DOUBLE PRECISION DT, X(NX), P(NP), F(NX), W(NW)                   
      INTEGER I                                                         
      DOUBLE PRECISION DF(NX), XH(NX), W1(NW)                           
C                                                                       
      CALL SREQ(KK, X, P, DF, W)                                        
      DO 200 I = 1, NX                                                  
         F(I) = X(I) + DT/6.0D0 * DF(I)                                 
         XH(I) = X(I) + DT/2.0D0 * DF(I)                                
 200  CONTINUE                                                          
C                                                                       
      CALL SREQ(KK, XH, P, DF, W1)                                      
      DO 210 I = 1, NX                                                  
         F(I) = F(I) + DT/3.0D0 * DF(I)                                 
         XH(I) = X(I) + DT/2.0D0 * DF(I)                                
 210  CONTINUE                                                          
C                                                                       
      CALL SREQ(KK, XH, P, DF, W1)                                      
      DO 220 I = 1, NX                                                  
         F(I) = F(I) + DT/3.0D0 * DF(I)                                 
         XH(I) = X(I) + DT * DF(I)                                      
 220  CONTINUE                                                          
C                                                                       
      CALL SREQ(KK, XH, P, DF, W1)                                      
      DO 230 I = 1, NX                                                  
         F(I) = F(I) + DT/6.0D0 * DF(I)                                 
 230  CONTINUE                                                          
C                                                                       
      END                                                               
C                                                                       
C                                                                       
C CALCULATE RIGHT HAND SIDES OF THE ODES DEFINING THE SYSTEM.           
C                                                                       
      SUBROUTINE SREQ(KK, X, P, DF, W)                                  
C                                                                       
      INTEGER KMAX, NX, NP, NW                                          
      LOGICAL EXTU                                                      
      PARAMETER (KMAX = 366, NX = 9, NP = 1, NW = 1, EXTU = .TRUE.)     
      INTEGER KK                                                        
      DOUBLE PRECISION X(NX), P(NP), DF(NX), W(NW), V(199)              
      INTEGER K                                                         
      DOUBLE PRECISION TAS(KMAX), TRSHS(KMAX), PSHS(KMAX)               
      DOUBLE PRECISION H0S(KMAX), H1S(KMAX), H2S(KMAX), H3S(KMAX)       
      DOUBLE PRECISION OP0S(KMAX), OP1S(KMAX), OP2S(KMAX), OP3S(KMAX)   
      COMMON /SARDAT/                                                   
     1     TAS, TRSHS, PSHS, H0S, H1S, H2S, H3S, OP0S, OP1S, OP2S, OP3S 
C                                                                       
      K = ((KK - 1) / 2) + 1                                            
C                                                                       
      TA = TAS(K)                                                       
      TRSH = TRSHS(K)                                                   
      PSH = PSHS(K)                                                     
      IF (2*(K-1) .EQ. (KK-1)) THEN                                     
         H0 = 2.0D0 * H0S(K)                                            
         H1 = 2.0D0 * H1S(K)                                            
         H2 = 2.0D0 * H2S(K)                                            
         H3 = 2.0D0 * H3S(K)                                            
         OP0 = 2.0D0 * OP0S(K)                                          
         OP1 = 2.0D0 * OP1S(K)                                          
         OP2 = 2.0D0 * OP2S(K)                                          
         OP3 = 2.0D0 * OP3S(K)                                          
      ELSE                                                              
         H0 = 0.0D0                                                     
         H1 = 0.0D0                                                     
         H2 = 0.0D0                                                     
         H3 = 0.0D0                                                     
         OP0 = 0.0D0                                                    
         OP1 = 0.0D0                                                    
         OP2 = 0.0D0                                                    
         OP3 = 0.0D0                                                    
      END IF                                                            
C                                                                       
C SORTED AND SOLVED EQUATIONS, GENERATED BY DYMOLA                      
C (POSTPROCESSED IN ORDER TO PLACE ALL TEMPORARY VARIABLES IN V)        
C   M.                                                                  
      V(12) = 0.001D0*PSH                                               
      IF (EXTU) THEN                                                    
         V(13) = P(1)                                                   
      ELSE                                                              
         IF (X(1) .GT. 60.0D0) THEN                                     
            V(199) = (60.0D0 - TRSH)/(X(1) -                            
     1           TRSH)                                                  
         ELSE                                                           
            V(198) = EXP(15.0D0 - 20.0D0                                
     1           *(X(1) - TRSH))                                        
            V(199) = 1.0D0/(1.0D0 + V(198))                             
         END IF                                                         
         V(13) = V(199)                                                 
      END IF                                                            
        V(6) = 1000.0D0*V(13)*V(12)/(                                   
     1    69666.6666666667D0 - 1161.11111111111D0*TRSH)                 
        V(11) = 1.16111111111111D0*(X(1) - TRSH)*                       
     1    V(6)                                                          
        V(10) = V(12) - V(11)                                           
        DF(7) = 0.001D0*V(10)                                           
        V(19) = OP0 + (OP1 + (OP2 + OP3*X(6))*X(6))*                    
     1    X(6)                                                          
        V(14) = 740.0D0*V(19)                                           
        V(16) = 0.000277777777777778D0*V(14)                            
        V(22) = - V(16)                                                 
        V(26) = - V(16)                                                 
        V(35) = - V(16)                                                 
        V(43) = 3600.0D0*V(16)                                          
        V(18) = H0 + (H1 + (H2 + H3*X(6))*X(6))*                        
     1    X(6)                                                          
        V(46) = X(6) + V(18)/(1.16111111111111D-27                      
     1     + 1161.11111111111D0*V(19))                                  
        V(8) = 0.000277777777777778D0*V(6)                              
        V(24) = - V(8)                                                  
        V(28) = - V(8)                                                  
        V(123) = - V(8)                                                 
        V(106) = - V(8)                                                 
        V(89) = - V(8)                                                  
        V(72) = - V(8)                                                  
        V(55) = - V(8)                                                  
        V(37) = - V(8)                                                  
        V(48) = 3600.0D0*V(8)                                           
        V(166) = 47.5428571428572D0*(X(1) -                             
     1    X(2))                                                         
        V(157) = 10.1587301587302D0*(X(1) - TA)                         
        V(40) = - (V(166) + V(157))                                     
        V(138) = 3.76619850066144D0*(X(1) - TA)                         
        V(41) = - V(138)                                                
        DF(1) = 9.41985645933014D-06*(                                  
     1    1161.11111111111D0*V(43)*(V(46) -                             
     1    X(1)) + 1161.11111111111D0*V(48)*(                            
     1    X(2) - X(1)) + V(40) -                                        
     1    V(138))                                                       
        V(53) = - V(16)                                                 
        V(61) = 3600.0D0*V(16)                                          
        V(65) = 3600.0D0*V(8)                                           
        V(167) = - V(166)                                               
        V(170) = 39.6190476190476D0*(X(2) -                             
     1    X(3))                                                         
        V(58) = V(166) - V(170)                                         
        V(142) = 4.0D0 +                                                
     1    0.5D0*TA                                                      
        V(141) = 5.64929775099216D0*(X(2) -                             
     1    V(142))                                                       
        V(59) = - V(141)                                                
        DF(2) = 6.27990430622009D-06*(                                  
     1    1161.11111111111D0*V(61)*(X(1) -                              
     1    X(2)) + 1161.11111111111D0*V(65)*(                            
     1    X(3) - X(2)) + V(58) -                                        
     1    V(141))                                                       
        V(70) = - V(16)                                                 
        V(78) = 3600.0D0*V(16)                                          
        V(82) = 3600.0D0*V(8)                                           
        V(172) = - V(170)                                               
        V(175) = 39.6190476190476D0*(X(3) -                             
     1    X(4))                                                         
        V(75) = V(170) - V(175)                                         
        V(145) = 5.64929775099216D0*X(3) -                              
     1    45.1943820079373D0                                            
        V(76) = - V(145)                                                
        DF(3) = 6.27990430622009D-06*(                                  
     1    1161.11111111111D0*V(78)*(X(2) -                              
     1    X(3)) + 1161.11111111111D0*V(82)*(                            
     1    X(4) - X(3)) + V(75) -                                        
     1    V(145))                                                       
        V(87) = - V(16)                                                 
        V(95) = 3600.0D0*V(16)                                          
        V(99) = 3600.0D0*V(8)                                           
        V(177) = - V(175)                                               
        V(180) = 47.5428571428572D0*(X(4) -                             
     1    X(5))                                                         
        V(92) = V(175) - V(180)                                         
        V(148) = 5.64929775099216D0*X(4) -                              
     1    45.1943820079373D0                                            
        V(93) = - V(148)                                                
        DF(4) = 6.27990430622009D-06*(                                  
     1    1161.11111111111D0*V(95)*(X(3) -                              
     1    X(4)) + 1161.11111111111D0*V(99)*(                            
     1    X(5) - X(4)) + V(92) -                                        
     1    V(148))                                                       
        V(104) = - V(16)                                                
        V(112) = 3600.0D0*V(16)                                         
        V(116) = 3600.0D0*V(8)                                          
        V(182) = - V(180)                                               
        V(185) = 79.2380952380953D0*(X(5) -                             
     1    X(6))                                                         
        V(109) = V(180) - V(185)                                        
        V(151) = 3.76619850066144D0*X(5) -                              
     1    30.1295880052915D0                                            
        V(110) = - V(151)                                               
        DF(5) = 9.41985645933014D-06*(                                  
     1    1161.11111111111D0*V(112)*(X(4) -                             
     1    X(5)) + 1161.11111111111D0*V(116)*(                           
     1    X(6) - X(5)) + V(109) -                                       
     1    V(151))                                                       
        V(121) = - V(16)                                                
        V(129) = 3600.0D0*V(16)                                         
        V(133) = 3600.0D0*V(8)                                          
        V(187) = - V(185)                                               
        V(161) = 73.1428571428572D0*X(6) -                              
     1    585.142857142857D0                                            
        V(126) = V(185) - V(161)                                        
        V(154) = 1.88309925033072D0*X(6) -                              
     1    15.0647940026458D0                                            
        V(127) = - V(154)                                               
        DF(6) = 1.88397129186603D-05*(                                  
     1    1161.11111111111D0*V(129)*(X(5) -                             
     1    X(6)) + 1161.11111111111D0*V(133)*(TRSH -                     
     1    X(6)) + V(126) - V(154))                                      
        V(196) = 4180.0D0*(X(1) - TRSH)*V(8)                            
        DF(8) = 0.001D0*V(196)                                          
        V(197) = 4180.0D0*(V(46) - X(6))*                               
     1    V(16)                                                         
        DF(9) = 0.001D0*V(197)                                          
C END OF SORTED AND SOLVED EQUATIONS                                    
C                                                                       
      W(1) = V(10)                                                      
C                                                                       
      RETURN                                                            
      END                                                               
C                                                                       
C                                                                       
      BLOCK DATA SARCOM                                                 
C                                                                       
      INTEGER NK1, NK2, NK3, NK4                                        
      PARAMETER (NK1 = 108, NK2 = 108, NK3 = 108, NK4 = 42)             
      DOUBLE PRECISION TA1(NK1), TA2(NK2), TA3(NK3), TA4(NK4)           
      DOUBLE PRECISION TRSH1(NK1), TRSH2(NK2), TRSH3(NK3), TRSH4(NK4)   
      DOUBLE PRECISION PSH1(NK1), PSH2(NK2), PSH3(NK3), PSH4(NK4)       
      DOUBLE PRECISION H01(NK1), H02(NK2), H03(NK3), H04(NK4)           
      DOUBLE PRECISION H11(NK1), H12(NK2), H13(NK3), H14(NK4)           
      DOUBLE PRECISION H21(NK1), H22(NK2), H23(NK3), H24(NK4)           
      DOUBLE PRECISION H31(NK1), H32(NK2), H33(NK3), H34(NK4)           
      DOUBLE PRECISION OP01(NK1), OP02(NK2), OP03(NK3), OP04(NK4)       
      DOUBLE PRECISION OP11(NK1), OP12(NK2), OP13(NK3), OP14(NK4)       
      DOUBLE PRECISION OP21(NK1), OP22(NK2), OP23(NK3), OP24(NK4)       
      DOUBLE PRECISION OP31(NK1), OP32(NK2), OP33(NK3), OP34(NK4)       
C                                                                       
      COMMON /SARDAT/                                                   
     1     TA1, TA2, TA3, TA4, TRSH1, TRSH2, TRSH3, TRSH4,              
     2     PSH1, PSH2, PSH3, PSH4,                                      
     3     H01, H02, H03, H04, H11, H12, H13, H14,                      
     4     H21, H22, H23, H24, H31, H32, H33, H34,                      
     5     OP01, OP02, OP03, OP04, OP11, OP12, OP13, OP14,              
     6     OP21, OP22, OP23, OP24, OP31, OP32, OP33, OP34               
C                                                                       
      DATA TA1/                                                         
     1   1.52D+01, 1.60D+01, 1.67D+01, 1.77D+01, 1.54D+01, 1.14D+01,    
     2   9.10D+00, 1.23D+01, 1.21D+01, 1.46D+01, 1.28D+01, 9.44D+00,    
     3   1.25D+01, 1.53D+01, 1.51D+01, 1.37D+01, 1.18D+01, 1.26D+01,    
     4   1.31D+01, 1.36D+01, 1.25D+01, 1.25D+01, 1.14D+01, 1.33D+01,    
     5   1.35D+01, 1.18D+01, 9.37D+00, 7.50D+00, 8.75D+00, 1.25D+01,    
     6   1.03D+01, 9.90D+00, 7.45D+00, 1.10D+01, 1.17D+01, 1.14D+01,    
     7   7.96D+00, 1.34D+01, 1.25D+01, 8.81D+00, 1.14D+01, 1.12D+01,    
     8   1.04D+01, 1.12D+01, 1.06D+01, 1.07D+01, 9.89D+00, 7.89D+00,    
     9   5.32D+00, 2.33D-01,-7.92D-02, 2.34D+00, 1.65D+00, 4.45D+00,    
     1   4.46D+00, 6.70D+00, 6.65D+00, 6.72D+00, 6.22D+00, 5.08D+00,    
     2   4.08D+00, 4.65D+00, 7.82D+00, 8.24D+00, 6.96D+00, 5.36D+00,    
     3   4.56D+00, 7.21D+00, 6.74D+00, 4.94D+00, 1.10D+00, 5.45D+00,    
     4   6.33D+00, 5.92D+00, 5.50D+00, 3.67D+00, 2.60D+00, 1.38D+00,    
     5   1.59D+00,-2.08D-01,-4.81D+00,-1.77D+00, 6.08D+00, 7.07D+00,    
     6   6.96D+00, 6.12D+00, 4.92D+00, 5.89D+00, 6.32D+00, 6.60D+00,    
     7   5.90D+00, 4.59D+00, 4.70D+00, 5.61D+00, 5.00D+00,-5.21D-01,    
     8   1.70D+00,-1.49D+00,-4.02D+00, 1.21D+00, 1.83D+00, 1.73D+00,    
     9   1.83D+00, 1.73D+00, 1.83D+00, 1.73D+00, 3.66D+00, 4.05D+00/    
      DATA TA2/                                                         
     1   4.98D+00, 4.32D+00, 3.55D+00, 4.17D+00, 4.33D-01, 5.30D+00,    
     2  -5.08D-01,-2.08D+00, 4.50D+00,-8.17D-01, 3.60D+00, 5.70D+00,    
     3   3.45D+00,-6.00D-01, 3.83D+00, 3.46D+00, 3.83D+00, 3.46D+00,    
     4   4.36D+00, 3.72D+00, 3.65D+00, 3.16D+00, 3.56D+00, 4.45D+00,    
     5   3.87D+00, 1.90D+00,-3.90D+00,-2.36D+00, 4.08D-01, 3.92D-01,    
     6  -1.97D+00, 3.83D-01, 3.92D-01, 2.60D+00, 2.95D+00,-3.21D-01,    
     7   3.31D+00, 6.00D-01, 2.40D+00,-1.42D-01,-1.93D+00,-1.48D+00,    
     8  -2.29D+00,-3.65D+00,-4.23D+00,-5.12D+00,-3.84D+00,-3.91D+00,    
     9  -5.16D+00,-4.20D+00,-4.86D+00,-4.05D+00,-2.55D+00,-2.36D+00,    
     1  -2.08D+00,-2.38D+00,-2.43D+00,-5.47D+00,-6.94D+00,-7.25D+00,    
     2  -4.92D+00,-2.78D+00,-3.52D+00,-4.62D-01, 4.42D-01, 9.37D-01,    
     3   1.97D+00, 2.65D+00, 3.95D+00, 2.75D+00,-5.33D-01,-3.28D+00,    
     4  -2.62D+00, 1.12D+00, 1.26D+00, 6.00D-01, 1.45D+00, 8.13D-01,    
     5   2.48D+00, 2.80D+00, 3.57D+00, 3.33D+00, 2.89D+00, 4.14D+00,    
     6   3.61D+00, 1.89D+00, 4.30D+00, 4.69D+00, 3.46D+00, 4.04D+00,    
     7   4.72D+00, 4.74D+00, 4.92D+00, 4.01D+00, 4.04D+00, 3.16D+00,    
     8   2.78D+00, 3.49D+00, 6.55D+00, 6.41D+00, 2.83D+00, 3.03D+00,    
     9   4.19D+00, 2.69D+00, 5.51D+00, 6.39D+00, 6.08D+00, 4.89D+00/    
      DATA TA3/                                                         
     1   7.48D+00, 7.07D+00, 5.88D+00, 5.92D+00, 4.61D+00, 6.31D+00,    
     2   9.54D+00, 1.11D+01, 1.11D+01, 9.06D+00, 6.74D+00, 6.02D+00,    
     3   2.68D+00, 1.36D+00, 1.01D+00, 1.67D+00, 2.49D+00, 3.35D+00,    
     4   4.59D+00, 4.76D+00, 6.83D+00, 4.51D+00, 6.67D+00, 8.83D+00,    
     5   7.94D+00, 8.48D+00, 5.20D+00, 7.71D+00, 6.14D+00, 5.75D+00,    
     6   7.69D+00, 7.78D+00, 7.82D+00, 8.29D+00, 8.85D+00, 8.78D+00,    
     7   8.93D+00, 1.03D+01, 9.17D+00, 9.85D+00, 8.28D+00, 9.65D+00,    
     8   8.20D+00, 9.11D+00, 9.03D+00, 1.06D+01, 1.10D+01, 1.09D+01,    
     9   9.14D+00, 8.81D+00, 9.72D+00, 1.01D+01, 1.17D+01, 1.15D+01,    
     1   1.18D+01, 1.47D+01, 1.40D+01, 1.11D+01, 1.04D+01, 7.71D+00,    
     2   7.31D+00, 9.96D+00, 1.05D+01, 1.12D+01, 1.08D+01, 1.16D+01,    
     3   1.33D+01, 1.22D+01, 1.14D+01, 1.28D+01, 1.13D+01, 1.25D+01,    
     4   1.16D+01, 1.25D+01, 9.95D+00, 1.19D+01, 1.18D+01, 1.29D+01,    
     5   1.34D+01, 1.41D+01, 1.39D+01, 1.47D+01, 1.40D+01, 1.43D+01,    
     6   1.44D+01, 1.23D+01, 1.35D+01, 1.51D+01, 1.73D+01, 1.95D+01,    
     7   2.03D+01, 2.09D+01, 2.11D+01, 2.11D+01, 2.18D+01, 1.93D+01,    
     8   1.71D+01, 1.74D+01, 1.80D+01, 1.74D+01, 1.68D+01, 1.66D+01,    
     9   1.55D+01, 1.63D+01, 1.65D+01, 1.55D+01, 1.67D+01, 1.68D+01/    
      DATA TA4/                                                         
     1   1.83D+01, 1.55D+01, 1.66D+01, 1.73D+01, 1.72D+01, 1.69D+01,    
     2   1.69D+01, 1.72D+01, 2.07D+01, 2.22D+01, 2.24D+01, 2.17D+01,    
     3   2.00D+01, 1.89D+01, 1.86D+01, 1.75D+01, 1.94D+01, 1.88D+01,    
     4   1.67D+01, 1.79D+01, 1.68D+01, 1.55D+01, 1.65D+01, 1.66D+01,    
     5   1.70D+01, 1.66D+01, 1.47D+01, 1.46D+01, 1.59D+01, 1.62D+01,    
     6   1.65D+01, 1.74D+01, 1.86D+01, 1.58D+01, 1.61D+01, 1.49D+01,    
     7   1.45D+01, 1.46D+01, 1.45D+01, 1.69D+01, 1.53D+01, 1.52D+01/    
C                                                                       
      DATA TRSH1/                                                       
     1   3.03D+01, 3.02D+01, 3.10D+01, 3.11D+01, 3.11D+01, 3.01D+01,    
     2   3.19D+01, 3.11D+01, 3.16D+01, 3.08D+01, 3.07D+01, 3.21D+01,    
     3   2.97D+01, 3.09D+01, 3.01D+01, 3.02D+01, 3.08D+01, 3.07D+01,    
     4   3.07D+01, 2.93D+01, 3.10D+01, 3.02D+01, 3.04D+01, 3.08D+01,    
     5   3.06D+01, 3.09D+01, 3.09D+01, 3.28D+01, 3.15D+01, 3.02D+01,    
     6   3.11D+01, 3.15D+01, 3.30D+01, 3.01D+01, 3.08D+01, 2.98D+01,    
     7   3.24D+01, 3.15D+01, 3.12D+01, 3.14D+01, 3.09D+01, 3.12D+01,    
     8   3.04D+01, 3.04D+01, 3.15D+01, 3.08D+01, 3.14D+01, 3.18D+01,    
     9   3.32D+01, 3.58D+01, 3.64D+01, 3.56D+01, 3.55D+01, 3.43D+01,    
     1   3.36D+01, 3.27D+01, 3.21D+01, 3.21D+01, 3.32D+01, 3.35D+01,    
     2   3.37D+01, 3.35D+01, 3.23D+01, 3.10D+01, 3.18D+01, 3.35D+01,    
     3   3.35D+01, 3.20D+01, 3.24D+01, 3.34D+01, 3.50D+01, 3.25D+01,    
     4   3.28D+01, 3.27D+01, 3.31D+01, 3.38D+01, 3.45D+01, 3.46D+01,    
     5   3.48D+01, 3.65D+01, 3.97D+01, 3.83D+01, 3.28D+01, 3.21D+01,    
     6   3.14D+01, 3.23D+01, 3.34D+01, 3.29D+01, 3.27D+01, 3.25D+01,    
     7   3.27D+01, 3.30D+01, 3.30D+01, 3.34D+01, 3.37D+01, 3.67D+01,    
     8   3.53D+01, 3.72D+01, 3.86D+01, 3.49D+01, 3.52D+01, 3.51D+01,    
     9   3.49D+01, 3.50D+01, 3.50D+01, 3.45D+01, 3.38D+01, 3.42D+01/    
      DATA TRSH2/                                                       
     1   3.33D+01, 3.36D+01, 3.39D+01, 3.37D+01, 3.54D+01, 3.28D+01,    
     2   3.69D+01, 3.80D+01, 3.34D+01, 3.71D+01, 3.47D+01, 3.21D+01,    
     3   3.40D+01, 3.71D+01, 3.37D+01, 3.42D+01, 3.38D+01, 3.43D+01,    
     4   3.36D+01, 3.33D+01, 3.33D+01, 3.40D+01, 3.41D+01, 3.34D+01,    
     5   3.42D+01, 3.53D+01, 3.82D+01, 3.73D+01, 3.55D+01, 3.60D+01,    
     6   3.72D+01, 3.61D+01, 3.58D+01, 3.38D+01, 3.38D+01, 3.66D+01,    
     7   3.45D+01, 3.56D+01, 3.49D+01, 3.61D+01, 3.67D+01, 3.64D+01,    
     8   3.75D+01, 3.86D+01, 3.88D+01, 3.98D+01, 3.85D+01, 3.80D+01,    
     9   3.94D+01, 3.87D+01, 3.95D+01, 3.90D+01, 3.78D+01, 3.73D+01,    
     1   3.64D+01, 3.71D+01, 3.73D+01, 3.97D+01, 4.12D+01, 4.14D+01,    
     2   3.91D+01, 3.71D+01, 3.81D+01, 3.60D+01, 3.55D+01, 3.53D+01,    
     3   3.47D+01, 3.42D+01, 3.26D+01, 3.39D+01, 3.62D+01, 3.81D+01,    
     4   3.77D+01, 3.53D+01, 3.49D+01, 3.46D+01, 3.46D+01, 3.50D+01,    
     5   3.44D+01, 3.42D+01, 3.41D+01, 3.40D+01, 3.33D+01, 3.31D+01,    
     6   3.38D+01, 3.47D+01, 3.33D+01, 3.35D+01, 3.38D+01, 3.28D+01,    
     7   3.28D+01, 3.31D+01, 3.35D+01, 3.37D+01, 3.39D+01, 3.41D+01,    
     8   3.35D+01, 3.40D+01, 3.33D+01, 3.30D+01, 3.45D+01, 3.53D+01,    
     9   3.37D+01, 3.38D+01, 3.21D+01, 3.24D+01, 3.31D+01, 3.35D+01/    
      DATA TRSH3/                                                       
     1   3.26D+01, 3.26D+01, 3.20D+01, 3.20D+01, 3.36D+01, 3.36D+01,    
     2   3.17D+01, 3.24D+01, 3.17D+01, 3.09D+01, 3.20D+01, 3.26D+01,    
     3   3.50D+01, 3.55D+01, 3.60D+01, 3.57D+01, 3.43D+01, 3.36D+01,    
     4   3.38D+01, 3.44D+01, 3.30D+01, 3.40D+01, 3.30D+01, 3.13D+01,    
     5   3.15D+01, 3.20D+01, 3.34D+01, 3.20D+01, 3.33D+01, 3.27D+01,    
     6   3.10D+01, 3.16D+01, 3.31D+01, 3.29D+01, 3.26D+01, 3.30D+01,    
     7   3.14D+01, 3.02D+01, 3.04D+01, 3.15D+01, 3.21D+01, 3.11D+01,    
     8   3.23D+01, 3.20D+01, 3.14D+01, 3.02D+01, 3.11D+01, 3.14D+01,    
     9   3.15D+01, 3.23D+01, 3.17D+01, 3.10D+01, 3.01D+01, 3.19D+01,    
     1   3.14D+01, 3.06D+01, 3.15D+01, 3.08D+01, 3.06D+01, 3.18D+01,    
     2   3.26D+01, 3.16D+01, 3.13D+01, 3.13D+01, 3.09D+01, 2.99D+01,    
     3   2.97D+01, 3.11D+01, 3.08D+01, 3.03D+01, 3.08D+01, 3.12D+01,    
     4   3.03D+01, 2.98D+01, 3.20D+01, 3.06D+01, 3.09D+01, 3.00D+01,    
     5   3.08D+01, 2.95D+01, 3.04D+01, 3.05D+01, 2.98D+01, 3.16D+01,    
     6   3.18D+01, 3.05D+01, 2.96D+01, 2.76D+01, 3.07D+01, 3.23D+01,    
     7   3.05D+01, 3.07D+01, 3.09D+01, 2.98D+01, 2.98D+01, 3.09D+01,    
     8   3.37D+01, 3.12D+01, 3.22D+01, 3.14D+01, 2.97D+01, 2.86D+01,    
     9   3.12D+01, 3.11D+01, 3.23D+01, 3.25D+01, 3.16D+01, 2.98D+01/    
      DATA TRSH4/                                                       
     1   2.95D+01, 3.09D+01, 3.07D+01, 3.06D+01, 2.93D+01, 3.05D+01,    
     2   3.00D+01, 2.98D+01, 3.06D+01, 3.11D+01, 3.12D+01, 2.92D+01,    
     3   3.11D+01, 3.03D+01, 3.03D+01, 3.12D+01, 3.11D+01, 3.12D+01,    
     4   2.92D+01, 3.12D+01, 3.03D+01, 3.02D+01, 3.11D+01, 3.09D+01,    
     5   3.11D+01, 2.92D+01, 3.09D+01, 3.01D+01, 3.02D+01, 3.10D+01,    
     6   3.09D+01, 3.10D+01, 2.92D+01, 3.09D+01, 3.02D+01, 3.04D+01,    
     7   3.12D+01, 3.10D+01, 3.12D+01, 2.92D+01, 3.09D+01, 3.03D+01/    
C                                                                       
      DATA PSH1/                                                        
     1   2.46D+04, 1.96D+04, 2.12D+04, 1.55D+04, 1.74D+04, 2.89D+04,    
     2   3.63D+04, 2.99D+04, 2.68D+04, 2.12D+04, 2.41D+04, 3.33D+04,    
     3   2.50D+04, 1.91D+04, 2.09D+04, 2.11D+04, 2.94D+04, 2.47D+04,    
     4   2.20D+04, 2.21D+04, 2.61D+04, 2.82D+04, 2.80D+04, 2.48D+04,    
     5   2.22D+04, 2.62D+04, 3.52D+04, 4.10D+04, 3.92D+04, 2.46D+04,    
     6   3.39D+04, 3.21D+04, 4.14D+04, 2.98D+04, 2.85D+04, 3.30D+04,    
     7   3.92D+04, 2.48D+04, 2.44D+04, 3.72D+04, 2.84D+04, 3.11D+04,    
     8   3.70D+04, 3.22D+04, 3.20D+04, 3.20D+04, 3.29D+04, 3.97D+04,    
     9   4.92D+04, 6.54D+04, 6.58D+04, 5.78D+04, 5.99D+04, 4.86D+04,    
     1   5.10D+04, 4.53D+04, 4.71D+04, 4.41D+04, 4.61D+04, 4.81D+04,    
     2   5.12D+04, 4.99D+04, 4.06D+04, 4.24D+04, 4.58D+04, 4.99D+04,    
     3   5.12D+04, 4.16D+04, 4.30D+04, 5.25D+04, 6.48D+04, 5.02D+04,    
     4   4.66D+04, 4.84D+04, 4.77D+04, 5.46D+04, 5.96D+04, 6.40D+04,    
     5   6.14D+04, 6.67D+04, 8.03D+04, 6.84D+04, 4.54D+04, 4.71D+04,    
     6   4.76D+04, 4.80D+04, 5.09D+04, 4.80D+04, 4.38D+04, 4.41D+04,    
     7   5.07D+04, 5.40D+04, 5.23D+04, 4.89D+04, 5.12D+04, 6.48D+04,    
     8   5.97D+04, 7.27D+04, 7.96D+04, 6.28D+04, 6.04D+04, 6.06D+04,    
     9   5.98D+04, 6.01D+04, 6.12D+04, 6.28D+04, 5.48D+04, 5.34D+04/    
      DATA PSH2/                                                        
     1   5.09D+04, 5.34D+04, 5.46D+04, 5.46D+04, 6.66D+04, 5.02D+04,    
     2   6.65D+04, 7.16D+04, 5.43D+04, 6.62D+04, 5.51D+04, 5.13D+04,    
     3   5.83D+04, 6.71D+04, 5.63D+04, 5.50D+04, 5.60D+04, 5.48D+04,    
     4   5.29D+04, 5.79D+04, 5.73D+04, 5.83D+04, 5.48D+04, 5.42D+04,    
     5   5.33D+04, 6.11D+04, 8.12D+04, 7.50D+04, 6.66D+04, 6.38D+04,    
     6   7.31D+04, 6.38D+04, 6.66D+04, 6.14D+04, 6.00D+04, 6.81D+04,    
     7   5.53D+04, 6.54D+04, 5.78D+04, 6.85D+04, 7.54D+04, 7.28D+04,    
     8   7.47D+04, 7.60D+04, 8.03D+04, 8.04D+04, 7.96D+04, 8.29D+04,    
     9   8.39D+04, 8.00D+04, 8.04D+04, 7.93D+04, 7.33D+04, 7.54D+04,    
     1   7.78D+04, 7.59D+04, 7.52D+04, 8.23D+04, 8.83D+04, 8.78D+04,    
     2   8.28D+04, 7.96D+04, 7.95D+04, 7.00D+04, 6.58D+04, 6.37D+04,    
     3   6.15D+04, 6.07D+04, 6.00D+04, 6.21D+04, 7.14D+04, 7.72D+04,    
     4   7.57D+04, 6.29D+04, 6.53D+04, 7.02D+04, 6.58D+04, 6.72D+04,    
     5   5.95D+04, 5.83D+04, 5.67D+04, 5.87D+04, 6.28D+04, 5.65D+04,    
     6   5.82D+04, 6.11D+04, 5.40D+04, 5.21D+04, 5.81D+04, 5.94D+04,    
     7   5.46D+04, 5.35D+04, 5.12D+04, 5.46D+04, 5.37D+04, 5.89D+04,    
     8   6.32D+04, 5.80D+04, 4.77D+04, 4.63D+04, 5.94D+04, 5.59D+04,    
     9   5.55D+04, 6.28D+04, 5.20D+04, 4.83D+04, 4.71D+04, 5.26D+04/    
      DATA PSH3/                                                        
     1   4.25D+04, 4.67D+04, 5.17D+04, 5.08D+04, 5.38D+04, 4.57D+04,    
     2   3.86D+04, 3.20D+04, 3.48D+04, 4.23D+04, 4.80D+04, 5.00D+04,    
     3   5.71D+04, 6.36D+04, 6.18D+04, 6.15D+04, 6.14D+04, 5.81D+04,    
     4   5.42D+04, 5.11D+04, 4.70D+04, 5.17D+04, 4.58D+04, 4.24D+04,    
     5   4.34D+04, 4.21D+04, 5.00D+04, 4.46D+04, 4.62D+04, 4.81D+04,    
     6   4.60D+04, 4.42D+04, 4.15D+04, 4.09D+04, 4.09D+04, 3.66D+04,    
     7   4.09D+04, 3.76D+04, 3.97D+04, 3.52D+04, 4.06D+04, 3.90D+04,    
     8   3.92D+04, 4.04D+04, 4.10D+04, 3.72D+04, 3.19D+04, 3.29D+04,    
     9   4.04D+04, 3.73D+04, 3.88D+04, 3.79D+04, 3.32D+04, 3.08D+04,    
     1   3.00D+04, 2.66D+04, 2.34D+04, 3.11D+04, 3.70D+04, 4.28D+04,    
     2   4.41D+04, 3.57D+04, 3.16D+04, 3.04D+04, 3.60D+04, 3.35D+04,    
     3   2.75D+04, 2.91D+04, 3.07D+04, 2.53D+04, 3.03D+04, 2.84D+04,    
     4   3.16D+04, 2.82D+04, 3.38D+04, 2.70D+04, 2.82D+04, 2.38D+04,    
     5   2.39D+04, 2.53D+04, 2.05D+04, 2.21D+04, 2.15D+04, 2.13D+04,    
     6   2.30D+04, 2.56D+04, 2.23D+04, 1.88D+04, 1.82D+04, 1.37D+04,    
     7   1.49D+04, 1.52D+04, 1.53D+04, 1.78D+04, 1.73D+04, 1.57D+04,    
     8   1.43D+04, 1.70D+04, 1.66D+04, 1.53D+04, 1.83D+04, 1.73D+04,    
     9   1.65D+04, 1.78D+04, 1.56D+04, 1.93D+04, 1.86D+04, 1.97D+04/    
      DATA PSH4/                                                        
     1   1.59D+04, 1.83D+04, 1.88D+04, 1.79D+04, 1.75D+04, 1.89D+04,    
     2   1.99D+04, 1.85D+04, 1.93D+04, 1.55D+04, 1.44D+04, 1.60D+04,    
     3   1.68D+04, 1.92D+04, 1.69D+04, 1.77D+04, 1.55D+04, 1.46D+04,    
     4   1.71D+04, 1.65D+04, 1.90D+04, 1.74D+04, 1.79D+04, 1.65D+04,    
     5   1.49D+04, 1.61D+04, 1.98D+04, 2.33D+04, 1.76D+04, 1.92D+04,    
     6   1.69D+04, 1.70D+04, 1.72D+04, 1.84D+04, 2.11D+04, 2.08D+04,    
     7   2.41D+04, 2.19D+04, 2.03D+04, 1.60D+04, 2.10D+04, 2.46D+04/    
C                                                                       
      DATA H01/                                                         
     1   1.42D+02, 1.41D+02, 1.32D+02, 1.06D+02, 1.25D+02, 9.71D+01,    
     2   1.42D+02, 1.29D+02, 1.38D+02, 5.90D+00, 6.94D+01, 1.32D+02,    
     3   3.47D+01, 8.86D+01, 4.85D+01, 9.46D+01, 3.61D-02, 7.08D+00,    
     4   8.88D+01, 8.90D+01, 9.93D+01, 1.75D+00, 4.35D+01, 3.80D+01,    
     5   7.91D+01, 3.42D+01, 6.91D+01, 9.25D+01, 2.79D+01, 2.77D-01,    
     6   2.70D+01, 8.50D+01, 3.14D+01, 8.66D+01, 1.11D+01, 5.27D+01,    
     7   9.15D+01, 6.29D+01, 7.95D+01, 7.48D+01, 6.86D+01, 6.56D+01,    
     8   7.80D+01, 1.42D+01, 8.35D+00, 8.31D+00, 1.23D+01, 4.96D+00,    
     9   1.65D+00, 7.48D+01, 5.86D+01, 5.72D+01, 7.01D+01, 6.03D+01,    
     1   3.95D-01, 2.63D+01, 3.31D+01, 3.17D+01, 3.69D+01, 5.00D+01,    
     2   1.11D-03, 1.17D+00, 7.22D-02, 4.00D-04, 3.79D+01, 6.36D+00,    
     3   4.14D+01, 3.84D+01, 2.42D+01, 4.97D+01, 5.26D+01, 6.96D-01,    
     4   2.37D+01, 3.95D+01, 8.23D+00, 3.70D-03, 8.21D-03, 3.68D-02,    
     5   2.56D+01, 2.73D+01, 4.29D+01, 2.48D-02, 6.82D-04, 1.17D-04,    
     6   4.59D-05, 4.00D-04,-6.94D-05,-3.16D-06,-3.32D-05, 1.44D-03,    
     7   7.92D-05, 5.78D-05, 6.83D-04, 4.28D-04, 2.13D+01, 2.62D+01,    
     8   5.63D+00, 1.90D+01, 1.12D+01, 7.96D-04, 1.16D-03, 3.11D+01,    
     9   1.15D-03, 3.11D+01, 1.15D-03, 3.11D+01, 8.58D-03,-6.30D-05/    
      DATA H02/                                                         
     1   1.60D-04,-5.12D-05, 2.70D-03, 9.89D-01, 2.80D-02,-5.64D-05,    
     2   1.99D+01, 1.44D+01, 2.16D-04, 2.82D+01,-1.19D-05, 4.70D-02,    
     3   1.44D+00, 1.14D+01, 1.79D+01, 1.34D+01, 1.81D+01, 1.34D+01,    
     4   3.41D-02,-5.98D-05, 7.35D-02, 6.62D+00,-4.44D-05, 5.20D-02,    
     5   1.16D-03, 2.50D+01, 3.76D+01, 1.66D+01,-4.83D-05,-4.80D-05,    
     6   3.55D+01,-6.58D-05, 3.01D+01, 2.16D-03, 3.78D-01, 3.26D+01,    
     7   2.79D-03, 4.61D+01, 3.51D+01, 1.67D-02, 2.04D-01,-5.36D-05,    
     8   4.51D+01, 5.39D+00, 4.56D+01, 3.26D+01, 8.70D+00, 2.69D+01,    
     9   5.58D+01, 5.73D+01, 5.42D+01, 1.56D-01, 9.36D-01, 3.59D-02,    
     1   6.10D-05, 7.66D-02, 7.77D-02, 1.43D+00, 5.90D+01, 5.91D+01,    
     2   1.16D-01, 4.12D-02, 4.48D+01, 1.11D-03,-7.81D-05,-8.43D-05,    
     3   1.23D-02,-6.84D-05,-2.75D-05, 4.76D+01, 8.23D+01, 8.27D+01,    
     4   2.35D-01, 3.88D+00, 3.93D-03,-5.05D-05, 2.84D-01, 1.80D+00,    
     5   4.02D-02, 5.61D-03, 7.34D+01, 2.76D+01, 1.72D-03, 1.09D+00,    
     6   4.23D-02, 1.27D-02, 3.63D-04, 4.36D+00, 2.41D+00, 4.59D+01,    
     7   3.89D-02, 3.72D-04, 1.12D+00, 2.52D+01, 3.26D+01, 7.03D-02,    
     8   3.96D-02, 1.19D+02, 1.23D+02, 3.73D+01, 5.71D+01, 1.21D+02,    
     9   3.22D+01, 1.13D+02, 2.93D-02, 6.83D-02, 3.73D-03, 7.69D+01/    
      DATA H03/                                                         
     1   3.84D-01, 1.03D+02, 1.31D+01, 5.60D+01, 1.24D+02, 6.43D+01,    
     2   1.18D+02, 1.41D+02, 1.18D+02, 1.25D+02, 1.42D+02, 1.20D+02,    
     3   1.47D+02, 1.32D+02, 9.48D+01, 1.29D+02, 1.53D+02, 5.55D+01,    
     4   9.29D+01, 1.45D+02, 8.68D+01, 1.15D+02, 7.61D+01, 1.02D+02,    
     5   1.42D+02, 6.86D+01, 3.24D-01, 1.17D+02, 4.91D+01, 1.58D+00,    
     6   4.44D+01, 1.80D+02, 1.81D+02, 1.78D+02, 1.71D+02, 1.36D+02,    
     7   4.37D+01, 1.33D+02, 5.24D-01, 1.82D+02, 7.91D+01, 1.05D+02,    
     8   9.85D+01, 1.76D+02, 1.44D+02, 7.31D+01, 1.78D+02, 1.70D+02,    
     9   1.24D+02, 1.83D+02, 1.60D+02, 1.50D+02, 1.65D+02, 1.77D+02,    
     1   1.40D+02, 1.79D+02, 1.87D+02, 1.43D+02, 1.57D+02, 7.74D+01,    
     2   9.52D+00, 1.05D+02, 1.68D+02, 7.88D+01, 5.03D+01, 1.67D+02,    
     3   1.31D+02, 3.90D+01, 4.97D+01, 7.79D+01, 7.56D+01, 1.17D+02,    
     4   1.00D+02, 5.98D+01, 3.11D+01, 9.91D+01, 7.72D+01, 1.09D+02,    
     5   5.39D+01, 9.05D+01, 1.25D+02, 1.10D+02, 2.73D+01, 7.98D+01,    
     6   8.01D+01, 6.54D+01, 1.13D+02, 1.05D+02, 1.72D+02, 1.53D+02,    
     7   1.78D+02, 1.82D+02, 1.79D+02, 1.78D+02, 1.79D+02, 8.61D+01,    
     8   1.64D+02, 1.73D+02, 1.00D+02, 4.97D+01, 7.27D+01, 7.47D+01,    
     9   6.04D+01, 1.22D+02, 1.02D+02, 9.24D+01, 1.18D+02, 9.49D+01/    
      DATA H04/                                                         
     1   1.68D+02, 5.55D+01, 1.32D+02, 1.04D+02, 1.58D+02, 1.55D+02,    
     2   5.85D+01, 1.47D+02, 1.64D+02, 1.60D+02, 1.38D+02, 1.59D+02,    
     3   1.30D+02, 1.11D+02, 1.32D+02, 8.08D+00, 8.90D+01, 1.34D+02,    
     4   9.00D+01, 9.33D+01, 1.28D+02, 1.24D+02, 1.40D+02, 1.06D+02,    
     5   8.68D+01, 4.83D+01, 3.85D+01, 8.49D+01, 1.41D+02, 1.09D+02,    
     6   1.42D+02, 1.40D+02, 1.27D+02, 7.20D+01, 1.42D+02, 1.49D+02,    
     7   1.24D+02, 1.40D+02, 1.46D+02, 3.54D+01, 1.40D+02, 1.42D+02/    
C                                                                       
      DATA H11/                                                         
     1  -9.47D-02,-1.67D-01,-5.06D-01,-3.75D-01,-7.74D-02, 5.03D-01,    
     2  -5.85D-01,-2.90D-01,-4.46D-01,-3.45D-01,-1.89D-01,-7.77D-01,    
     3  -5.13D-01, 3.95D-01, 2.29D-01,-3.13D-01,-2.29D-03,-4.43D-01,    
     4   1.82D-01, 3.81D-01,-1.73D-01,-1.10D-01,-4.50D-01,-1.77D-01,    
     5  -4.37D-02,-1.45D+00,-1.17D-02, 2.83D-01,-5.89D-01,-1.75D-02,    
     6  -1.03D+00,-1.35D-01,-7.59D-01,-4.05D-02,-6.65D-01,-2.43D-01,    
     7   2.78D-01, 2.70D-01, 2.73D-01, 2.64D-01, 4.20D-01, 3.68D-01,    
     8   2.05D-01,-8.19D-01,-5.16D-01,-5.14D-01,-7.72D-01,-3.13D-01,    
     9  -1.05D-01,-1.60D-01,-1.05D+00,-1.79D+00,-3.64D-01, 1.89D-01,    
     1  -2.51D-02,-1.28D+00,-9.31D-01,-7.45D-01,-1.09D+00,-1.28D+00,    
     2  -7.25D-05,-7.42D-02,-4.58D-03,-2.70D-05,-1.50D+00,-3.88D-01,    
     3  -1.88D+00,-7.71D-01,-1.23D+00,-5.44D-01,-9.02D-01,-4.42D-02,    
     4  -1.23D+00,-1.14D+00,-4.98D-01,-2.38D-04,-5.26D-04,-2.35D-03,    
     5  -1.25D+00,-1.44D+00,-1.46D+00,-1.59D-03,-4.48D-05,-8.44D-06,    
     6  -3.75D-06,-2.65D-05, 4.09D-06,-4.57D-07, 1.51D-06,-9.37D-05,    
     7  -5.82D-06,-4.36D-06,-4.46D-05,-2.83D-05,-9.86D-01,-1.45D+00,    
     8  -3.50D-01,-1.10D+00,-7.09D-01,-5.15D-05,-7.49D-05,-1.36D+00,    
     9  -7.46D-05,-1.36D+00,-7.45D-05,-1.36D+00,-5.50D-04, 3.81D-06/    
      DATA H12/                                                         
     1  -1.10D-05, 2.98D-06,-1.74D-04,-6.29D-02,-1.80D-03, 3.19D-06,    
     2  -1.19D+00,-9.01D-01,-1.46D-05,-1.35D+00, 2.10D-07,-2.99D-03,    
     3  -9.15D-02,-7.12D-01,-8.91D-01,-7.78D-01,-9.01D-01,-7.80D-01,    
     4  -2.18D-03, 3.58D-06,-4.69D-03,-4.03D-01, 2.42D-06,-3.31D-03,    
     5  -7.53D-05,-1.10D+00,-1.58D+00,-9.36D-01, 2.86D-06, 2.85D-06,    
     6  -8.88D-01, 4.03D-06,-1.45D+00,-1.39D-04,-2.41D-02,-1.11D+00,    
     7  -1.80D-04,-1.37D+00,-1.29D+00,-1.07D-03,-1.31D-02, 3.21D-06,    
     8  -1.40D+00,-3.40D-01,-9.44D-01,-1.45D+00,-5.52D-01,-1.33D+00,    
     9  -9.62D-01,-9.83D-01,-1.64D+00,-1.00D-02,-5.98D-02,-2.30D-03,    
     1  -4.23D-06,-4.90D-03,-4.97D-03,-9.15D-02,-1.39D+00,-1.38D+00,    
     2  -7.45D-03,-2.64D-03,-1.10D+00,-7.19D-05, 4.79D-06, 5.11D-06,    
     3  -7.90D-04, 4.07D-06, 1.21D-06,-6.25D-01, 1.06D-01, 9.82D-03,    
     4  -1.50D-02,-2.47D-01,-2.53D-04, 2.90D-06,-1.81D-02,-1.15D-01,    
     5  -2.57D-03,-3.61D-04,-6.80D-02,-1.57D+00,-1.11D-04,-6.91D-02,    
     6  -2.70D-03,-8.13D-04,-2.43D-05,-2.77D-01,-1.53D-01,-6.22D-01,    
     7  -2.48D-03,-2.50D-05,-7.09D-02,-1.00D+00,-1.44D+00,-4.49D-03,    
     8  -2.53D-03,-3.51D-01,-3.64D-01,-6.71D-01,-2.98D+00,-1.57D-01,    
     9  -7.11D-01,-3.95D-01,-1.87D-03,-4.35D-03,-2.41D-04,-1.10D+00/    
      DATA H13/                                                         
     1  -2.44D-02,-2.94D-01,-7.63D-01,-5.62D-01,-1.42D+00,-1.58D+00,    
     2  -4.04D-01,-6.52D-01,-4.61D-01,-5.86D-01,-6.00D-01,-8.63D-01,    
     3  -8.73D-01,-3.73D-01,-1.40D+00,-6.28D-01,-8.71D-01,-1.54D+00,    
     4  -9.52D-01,-6.67D-01, 6.10D-01,-1.47D+00,-1.15D+00, 6.50D-01,    
     5  -1.45D-01,-8.10D-01,-2.07D-02,-6.24D-01,-2.36D+00,-1.00D-01,    
     6  -7.22D-01,-8.19D-01,-8.12D-01,-7.99D-01,-5.38D-01, 6.46D-02,    
     7  -2.20D-01,-1.32D+00,-3.31D-02,-8.96D-01,-1.56D+00, 3.78D-01,    
     8  -8.68D-01,-7.47D-01,-3.02D-01, 4.83D-01,-7.49D-01,-8.06D-01,    
     9   5.71D-01,-9.39D-01,-6.41D-01,-5.42D-01,-5.24D-01,-7.67D-01,    
     1  -7.66D-01,-5.50D-01,-6.58D-01,-6.54D-01,-5.86D-01,-2.51D+00,    
     2  -5.99D-01,-1.57D+00,-6.08D-01,-6.56D-01,-2.44D+00,-9.34D-01,    
     3  -2.70D-01,-1.94D+00,-1.76D+00, 4.72D-01, 3.31D-01,-6.00D-01,    
     4  -1.12D+00,-6.53D-01,-1.79D+00, 2.23D-01,-1.09D+00, 1.11D+00,    
     5  -2.19D-02,-3.12D-01,-4.71D-01,-3.68D-01,-2.34D-01,-1.26D-01,    
     6  -7.54D-01,-1.66D+00, 1.76D-01,-4.91D-01,-3.51D-01,-2.17D-01,    
     7  -6.91D-02,-7.97D-02,-4.64D-04,-1.03D-03, 6.80D-03, 4.02D-01,    
     8  -3.04D-01,-2.99D-01, 6.28D-01, 1.17D-01,-1.73D+00, 6.06D-01,    
     9  -2.13D+00,-8.56D-05,-7.94D-02, 5.39D-01, 1.05D-01,-5.81D-01/    
      DATA H14/                                                         
     1  -2.80D-01,-2.61D+00,-2.30D-01, 2.96D-01,-9.86D-02,-1.48D-01,    
     2  -9.43D-01,-3.56D-01, 1.98D-02,-1.16D-01, 2.79D-01,-7.48D-02,    
     3   2.40D-01,-1.00D-01,-4.68D-01,-4.74D-01,-6.87D-01, 5.31D-02,    
     4  -1.62D-01,-2.67D-01,-7.63D-01,-6.01D-01,-4.04D-01, 5.16D-01,    
     5   2.73D-01,-1.97D-02,-1.87D+00, 4.34D-01,-6.60D-01,-7.29D-02,    
     6  -3.31D-01,-4.48D-01, 1.30D-01,-1.57D+00,-3.20D-01,-1.42D-01,    
     7  -3.87D-01, 4.21D-02,-7.18D-02, 2.46D-01,-1.35D-01,-9.47D-02/    
C                                                                       
      DATA H21/                                                         
     1  -2.47D-02,-2.36D-02,-1.34D-02,-1.25D-02,-2.89D-02,-4.45D-02,    
     2  -1.84D-02,-1.84D-02,-1.77D-02, 6.14D-03,-2.93D-02,-1.10D-02,    
     3  -4.56D-03,-3.48D-02,-2.14D-02,-1.94D-02, 4.30D-05, 8.25D-03,    
     4  -3.04D-02,-4.02D-02,-2.36D-02, 2.04D-03,-1.12D-02,-1.25D-02,    
     5  -2.72D-02, 1.96D-02,-2.87D-02,-4.02D-02, 3.00D-04, 3.26D-04,    
     6   1.22D-02,-2.37D-02, 2.52D-03,-2.59D-02, 1.20D-02,-1.99D-02,    
     7  -3.65D-02,-3.18D-02,-3.37D-02,-3.70D-02,-3.80D-02,-3.78D-02,    
     8  -3.17D-02, 1.45D-02, 9.53D-03, 9.49D-03, 1.44D-02, 5.85D-03,    
     9   1.96D-03,-3.08D-02,-2.21D-03, 1.59D-02,-2.58D-02,-3.41D-02,    
     1   4.70D-04, 1.98D-02, 6.02D-03, 2.04D-03, 8.26D-03, 6.16D-03,    
     2   1.38D-06, 1.39D-03, 8.59D-05, 5.25D-07, 1.88D-02, 7.10D-03,    
     3   2.73D-02,-8.54D-04, 1.97D-02,-1.28D-02,-5.64D-03, 8.28D-04,    
     4   2.02D-02, 7.68D-03, 9.07D-03, 4.50D-06, 9.92D-06, 4.43D-05,    
     5   1.96D-02, 2.37D-02, 1.47D-02, 3.00D-05, 8.56D-07, 1.69D-07,    
     6   7.95D-08, 5.11D-07,-7.32D-08, 1.59D-08,-2.17D-08, 1.78D-06,    
     7   1.18D-07, 8.95D-08, 8.50D-07, 5.44D-07, 1.46D-02, 2.49D-02,    
     8   6.48D-03, 1.93D-02, 1.33D-02, 9.78D-07, 1.42D-06, 1.90D-02,    
     9   1.42D-06, 1.90D-02, 1.41D-06, 1.91D-02, 1.04D-05,-6.93D-08/    
      DATA H22/                                                         
     1   2.15D-07,-5.28D-08, 3.28D-06, 1.18D-03, 3.39D-05,-5.56D-08,    
     2   2.15D-02, 1.67D-02, 2.84D-07, 2.05D-02, 2.11D-09, 5.63D-05,    
     3   1.72D-03, 1.32D-02, 1.41D-02, 1.38D-02, 1.43D-02, 1.38D-02,    
     4   4.09D-05,-6.46D-08, 8.81D-05, 7.37D-03,-4.10D-08, 6.23D-05,    
     5   1.43D-06, 1.54D-02, 2.12D-02, 1.63D-02,-5.15D-08,-5.11D-08,    
     6   3.43D-03,-7.40D-08, 2.21D-02, 2.63D-06, 4.53D-04, 1.14D-02,    
     7   3.41D-06, 1.06D-02, 1.46D-02, 2.01D-05, 2.46D-04,-5.80D-08,    
     8   1.18D-02, 6.36D-03,-5.91D-04, 2.06D-02, 1.04D-02, 2.08D-02,    
     9  -5.39D-03,-5.83D-03, 1.29D-02, 1.89D-04, 1.13D-03, 4.34D-05,    
     1   8.34D-08, 9.24D-05, 9.37D-05, 1.72D-03, 3.44D-03, 3.13D-03,    
     2   1.40D-04, 4.97D-05, 4.09D-03, 1.37D-06,-8.80D-08,-9.32D-08,    
     3   1.49D-05,-7.32D-08,-1.66D-08,-8.40D-03,-3.77D-02,-3.75D-02,    
     4   2.82D-04, 4.63D-03, 4.79D-06,-5.11D-08, 3.41D-04, 2.16D-03,    
     5   4.84D-05, 6.82D-06,-3.07D-02, 2.73D-02, 2.12D-06, 1.30D-03,    
     6   5.08D-05, 1.53D-05, 4.69D-07, 5.18D-03, 2.87D-03,-6.65D-03,    
     7   4.66D-05, 4.84D-07, 1.33D-03, 1.28D-02, 2.04D-02, 8.45D-05,    
     8   4.76D-05,-2.42D-02,-2.06D-02,-1.44D-03, 4.88D-02,-3.12D-02,    
     9   5.33D-04,-2.94D-02, 3.53D-05, 8.17D-05, 4.55D-06,-3.05D-03/    
      DATA H23/                                                         
     1   4.57D-04,-2.25D-02, 1.35D-02,-8.82D-03, 3.03D-03, 6.24D-03,    
     2  -1.99D-02,-1.42D-02,-1.96D-02,-1.87D-02,-1.90D-02,-8.21D-03,    
     3  -1.70D-02,-3.02D-02,-1.68D-02,-3.00D-02,-1.51D-02, 9.67D-03,    
     4  -1.59D-02,-1.94D-02,-5.64D-02,-3.69D-03,-9.35D-03,-5.37D-02,    
     5  -3.03D-02,-5.68D-03, 3.89D-04,-1.39D-02, 3.62D-02, 1.88D-03,    
     6  -3.55D-03,-1.76D-02,-1.75D-02,-1.62D-02,-2.09D-02,-3.85D-02,    
     7  -1.71D-02,-2.97D-03, 6.20D-04,-1.54D-02, 3.88D-04,-6.07D-02,    
     8  -2.50D-02,-1.92D-02,-2.90D-02,-4.18D-02,-1.83D-02,-1.80D-02,    
     9  -5.48D-02,-1.51D-02,-1.83D-02,-2.45D-02,-2.09D-02,-1.58D-02,    
     1  -9.56D-03,-1.98D-02,-1.76D-02,-2.05D-02,-2.13D-02, 2.54D-02,    
     2   1.12D-02,-3.39D-03,-2.28D-02,-2.44D-02, 3.77D-02,-1.07D-02,    
     3  -1.97D-02, 3.06D-02, 1.91D-02,-4.91D-02,-4.75D-02,-1.31D-02,    
     4  -9.37D-03,-9.34D-03, 3.17D-02,-4.87D-02,-1.43D-02,-7.10D-02,    
     5  -2.05D-02,-2.92D-02,-2.16D-02,-2.47D-02,-8.31D-03,-2.45D-02,    
     6  -2.10D-02, 1.03D-02,-4.07D-02,-1.07D-02,-2.30D-02,-2.03D-02,    
     7  -2.62D-02,-2.60D-02,-2.71D-02,-2.71D-02,-2.66D-02,-4.23D-02,    
     8  -2.21D-02,-2.32D-02,-4.89D-02,-2.60D-02, 5.18D-03,-5.12D-02,    
     9   2.46D-02,-2.86D-02,-3.12D-02,-5.38D-02,-3.81D-02,-2.21D-02/    
      DATA H24/                                                         
     1  -2.21D-02, 3.92D-02,-2.29D-02,-4.58D-02,-2.89D-02,-2.69D-02,    
     2  -8.07D-03,-1.75D-02,-2.74D-02,-2.34D-02,-3.19D-02,-2.37D-02,    
     3  -3.20D-02,-2.42D-02,-1.65D-02, 8.47D-03,-5.92D-03,-3.05D-02,    
     4  -1.91D-02,-2.15D-02,-5.01D-03,-1.28D-02,-2.36D-02,-4.56D-02,    
     5  -3.70D-02,-2.27D-02, 2.91D-02,-4.12D-02,-1.60D-02,-3.19D-02,    
     6  -2.04D-02,-1.48D-02,-2.69D-02, 2.64D-03,-2.17D-02,-2.49D-02,    
     7  -2.04D-02,-3.19D-02,-2.69D-02,-2.35D-02,-2.44D-02,-2.47D-02/    
C                                                                       
      DATA H31/                                                         
     1   1.15D-04, 1.13D-04, 4.29D-05, 5.45D-05, 1.64D-04, 2.98D-04,    
     2   8.76D-05, 7.70D-05, 7.74D-05,-3.40D-05, 2.56D-04, 4.21D-05,    
     3   6.72D-05, 2.14D-04, 1.42D-04, 1.26D-04,-2.46D-07,-4.71D-05,    
     4   1.91D-04, 2.75D-04, 1.44D-04,-1.16D-05, 1.22D-04, 1.08D-04,    
     5   1.98D-04,-8.57D-05, 2.26D-04, 2.83D-04, 3.22D-05,-1.86D-06,    
     6  -4.52D-05, 1.60D-04, 2.41D-05, 1.71D-04,-6.72D-05, 1.80D-04,    
     7   2.43D-04, 2.32D-04, 2.30D-04, 2.76D-04, 2.75D-04, 2.84D-04,    
     8   2.18D-04,-7.97D-05,-5.41D-05,-5.38D-05,-8.22D-05,-3.35D-05,    
     9  -1.13D-05, 2.61D-04, 7.52D-05,-3.19D-05, 2.38D-04, 2.74D-04,    
     1  -2.69D-06,-9.86D-05, 3.99D-06, 2.70D-05,-7.12D-06, 2.34D-05,    
     2  -7.99D-09,-7.99D-06,-4.92D-07,-3.08D-09,-7.54D-05,-4.01D-05,    
     3  -1.28D-04, 5.37D-05,-1.01D-04, 1.44D-04, 1.05D-04,-4.75D-06,    
     4  -1.05D-04, 2.34D-06,-5.10D-05,-2.59D-08,-5.71D-08,-2.55D-07,    
     5  -9.79D-05,-1.24D-04,-4.04D-05,-1.72D-07,-4.97D-09,-1.01D-09,    
     6  -4.90D-10,-2.98D-09, 4.06D-10,-1.18D-10, 1.00D-10,-1.03D-08,    
     7  -7.08D-10,-5.42D-10,-4.92D-09,-3.17D-09,-6.94D-05,-1.34D-04,    
     8  -3.68D-05,-1.06D-04,-7.58D-05,-5.65D-09,-8.21D-09,-8.55D-05,    
     9  -8.18D-09,-8.57D-05,-8.17D-09,-8.58D-05,-5.96D-08, 3.90D-10/    
      DATA H32/                                                         
     1  -1.27D-09, 2.92D-10,-1.89D-08,-6.78D-06,-1.95D-07, 3.03D-10,    
     2  -1.20D-04,-9.51D-05,-1.66D-09,-1.00D-04,-3.45D-11,-3.23D-07,    
     3  -9.85D-06,-7.51D-05,-7.15D-05,-7.60D-05,-7.25D-05,-7.62D-05,    
     4  -2.35D-07, 3.62D-10,-5.06D-07,-4.15D-05, 2.19D-10,-3.58D-07,    
     5  -8.25D-09,-6.98D-05,-9.20D-05,-8.86D-05, 2.87D-10, 2.85D-10,    
     6   2.44D-05, 4.19D-10,-1.08D-04,-1.52D-08,-2.60D-06,-3.30D-05,    
     7  -1.97D-08,-1.00D-05,-5.07D-05,-1.16D-07,-1.41D-06, 3.24D-10,    
     8  -1.84D-05,-3.64D-05, 6.26D-05,-9.50D-05,-5.94D-05,-1.04D-04,    
     9   1.05D-04, 1.10D-04,-1.44D-05,-1.08D-06,-6.48D-06,-2.50D-07,    
     1  -4.93D-10,-5.32D-07,-5.39D-07,-9.92D-06, 5.47D-05, 5.68D-05,    
     2  -8.08D-07,-2.86D-07, 3.14D-05,-7.90D-09, 4.98D-10, 5.24D-10,    
     3  -8.56D-08, 4.09D-10, 7.33D-11, 1.07D-04, 2.92D-04, 3.03D-04,    
     4  -1.62D-06,-2.66D-05,-2.76D-08, 2.81D-10,-1.96D-06,-1.24D-05,    
     5  -2.78D-07,-3.93D-08, 2.49D-04,-1.49D-04,-1.22D-08,-7.45D-06,    
     6  -2.92D-07,-8.84D-08,-2.74D-09,-2.97D-05,-1.65D-05, 8.84D-05,    
     7  -2.68D-07,-2.83D-09,-7.64D-06,-5.27D-05,-9.35D-05,-4.86D-07,    
     8  -2.74D-07, 1.47D-04, 1.06D-04, 4.82D-05,-2.55D-04, 1.99D-04,    
     9   3.92D-05, 2.20D-04,-2.03D-07,-4.69D-07,-2.63D-08, 6.27D-05/    
      DATA H33/                                                         
     1  -2.62D-06, 1.45D-04,-7.48D-05, 8.98D-05,-2.71D-05, 4.04D-05,    
     2   1.08D-04, 5.53D-05, 1.12D-04, 1.08D-04, 9.14D-05, 3.54D-05,    
     3   9.94D-05, 1.97D-04, 2.34D-04, 2.36D-04, 7.06D-05, 8.02D-06,    
     4   1.66D-04, 1.01D-04, 4.33D-04, 6.25D-05, 1.44D-04, 3.75D-04,    
     5   1.66D-04, 6.77D-05,-2.23D-06, 7.39D-05,-1.79D-04,-1.08D-05,    
     6   6.82D-05, 7.91D-05, 7.63D-05, 6.40D-05, 9.41D-05, 2.29D-04,    
     7   1.59D-04, 1.61D-05,-3.55D-06, 6.18D-05, 8.20D-05, 4.86D-04,    
     8   2.52D-04, 8.70D-05, 1.78D-04, 3.03D-04, 8.41D-05, 9.68D-05,    
     9   3.63D-04, 6.05D-05, 7.78D-05, 1.36D-04, 1.01D-04, 5.75D-05,    
     1   3.07D-05, 8.28D-05, 6.26D-05, 1.20D-04, 1.12D-04,-7.86D-05,    
     2  -6.39D-05, 8.74D-05, 1.10D-04, 2.47D-04,-1.87D-04, 4.14D-05,    
     3   9.67D-05,-1.54D-04,-6.11D-05, 3.83D-04, 3.86D-04, 7.51D-05,    
     4   1.04D-04, 1.02D-04,-1.74D-04, 3.78D-04, 1.92D-04, 5.01D-04,    
     5   1.56D-04, 2.39D-04, 1.27D-04, 1.68D-04, 8.50D-05, 1.79D-04,    
     6   2.19D-04, 1.27D-06, 2.73D-04, 4.77D-05, 1.01D-04, 6.85D-05,    
     7   1.13D-04, 1.12D-04, 1.16D-04, 1.17D-04, 1.12D-04, 3.02D-04,    
     8   8.59D-05, 1.00D-04, 3.27D-04, 2.07D-04, 5.95D-05, 3.94D-04,    
     9  -9.32D-05, 1.59D-04, 2.19D-04, 4.06D-04, 2.62D-04, 1.90D-04/    
      DATA H34/                                                         
     1   9.40D-05,-1.90D-04, 9.96D-05, 3.36D-04, 1.46D-04, 1.34D-04,    
     2   1.29D-04, 7.16D-05, 1.23D-04, 1.01D-04, 1.55D-04, 9.24D-05,    
     3   1.54D-04, 1.25D-04, 7.28D-05,-4.70D-05, 3.56D-05, 1.69D-04,    
     4   1.10D-04, 1.43D-04,-2.58D-06, 4.88D-05, 1.38D-04, 2.94D-04,    
     5   2.56D-04, 1.89D-04,-1.45D-04, 2.85D-04, 8.93D-05, 2.11D-04,    
     6   8.04D-05, 5.11D-05, 1.25D-04, 6.77D-05, 1.08D-04, 1.14D-04,    
     7   1.20D-04, 1.83D-04, 1.30D-04, 1.83D-04, 1.14D-04, 1.15D-04/    
C                                                                       
      DATA OP01/                                                        
     1   4.46D-03, 4.63D-03, 4.78D-03, 3.65D-03, 4.39D-03, 3.37D-03,    
     2   4.95D-03, 4.16D-03, 4.77D-03, 2.05D-04, 3.27D-03, 4.64D-03,    
     3   1.58D-03, 3.07D-03, 1.71D-03, 3.58D-03, 1.60D-06, 1.41D-04,    
     4   3.28D-03, 3.62D-03, 4.19D-03, 3.73D-05, 2.04D-03, 2.10D-03,    
     5   3.46D-03, 2.48D-03, 2.63D-03, 3.44D-03, 1.25D-03, 8.23D-06,    
     6   2.12D-03, 3.18D-03, 1.97D-03, 3.07D-03, 4.05D-04, 2.30D-03,    
     7   3.07D-03, 2.44D-03, 3.06D-03, 2.84D-03, 2.37D-03, 2.27D-03,    
     8   2.59D-03, 6.98D-04, 2.47D-04, 2.45D-04, 2.77D-04, 1.18D-04,    
     9   4.14D-05, 2.55D-03, 2.33D-03, 4.03D-03, 2.85D-03, 2.12D-03,    
     1   1.17D-05, 1.91D-03, 2.11D-03, 1.77D-03, 2.49D-03, 3.13D-03,    
     2   1.46D-07, 3.02D-05, 2.52D-06, 9.92D-08, 2.73D-03, 3.00D-04,    
     3   3.34D-03, 2.26D-03, 1.78D-03, 2.60D-03, 2.84D-03, 1.71D-05,    
     4   1.81D-03, 2.76D-03, 4.01D-04, 2.96D-07, 4.75D-07, 1.59D-06,    
     5   1.88D-03, 1.97D-03, 2.74D-03, 1.26D-06, 1.07D-07, 5.96D-08,    
     6   4.96D-08, 8.19D-08, 1.97D-08, 4.09D-08, 3.59D-08, 1.59D-07,    
     7   5.20D-08, 4.65D-08, 9.49D-08, 8.16D-08, 1.72D-03, 2.09D-03,    
     8   2.23D-04, 1.34D-03, 4.80D-04, 9.13D-08, 1.29D-07, 2.54D-03,    
     9   1.29D-07, 2.54D-03, 1.29D-07, 2.54D-03, 4.97D-07, 1.34D-08/    
      DATA OP02/                                                        
     1   5.71D-08, 1.88D-08, 2.13D-07, 2.39D-05, 1.30D-06, 2.42D-08,    
     2   1.26D-03, 7.28D-04, 6.26D-08, 2.23D-03, 3.46D-08, 1.84D-06,    
     3   3.49D-05, 4.91D-04, 1.18D-03, 8.66D-04, 1.18D-03, 8.71D-04,    
     4   1.39D-06, 1.53D-08, 2.61D-06, 3.56D-04, 2.75D-08, 1.95D-06,    
     5   1.19D-07, 1.78D-03, 2.48D-03, 1.31D-03, 1.82D-08, 1.83D-08,    
     6   1.90D-03, 1.19D-08, 2.49D-03, 1.80D-07, 1.09D-05, 2.02D-03,    
     7   2.59D-07, 2.63D-03, 2.34D-03, 9.19D-07, 6.89D-06, 1.96D-08,    
     8   2.61D-03, 2.28D-04, 2.34D-03, 1.90D-03, 3.16D-04, 1.92D-03,    
     9   2.57D-03, 2.64D-03, 3.19D-03, 5.57D-06, 2.75D-05, 1.71D-06,    
     1   3.80D-08, 3.09D-06, 3.13D-06, 4.16D-05, 3.03D-03, 3.01D-03,    
     2   4.68D-06, 1.93D-06, 2.55D-03, 1.43D-07, 1.57D-08, 2.11D-08,    
     3   6.21D-07, 2.06D-08, 3.79D-08, 2.11D-03, 2.33D-03, 2.37D-03,    
     4   7.57D-06, 9.41D-05, 3.15D-07, 2.83D-08, 9.32D-06, 4.58D-05,    
     5   1.83D-06, 4.05D-07, 2.58D-03, 1.83D-03, 1.98D-07, 2.85D-05,    
     6   1.91D-06, 7.75D-07, 9.21D-08, 1.05D-04, 5.89D-05, 2.16D-03,    
     7   1.63D-06, 1.00D-07, 2.71D-05, 1.71D-03, 2.44D-03, 3.03D-06,    
     8   1.80D-06, 4.23D-03, 4.23D-03, 1.71D-03, 4.30D-03, 3.80D-03,    
     9   1.89D-03, 3.86D-03, 1.44D-06, 2.63D-06, 3.30D-07, 4.06D-03/    
      DATA OP03/                                                        
     1   1.16D-05, 3.47D-03, 7.24D-04, 2.91D-03, 4.42D-03, 4.13D-03,    
     2   4.15D-03, 4.89D-03, 4.58D-03, 4.71D-03, 4.80D-03, 3.91D-03,    
     3   5.04D-03, 4.56D-03, 4.78D-03, 4.48D-03, 4.87D-03, 3.33D-03,    
     4   4.25D-03, 4.81D-03, 3.03D-03, 5.42D-03, 4.38D-03, 3.66D-03,    
     5   4.36D-03, 2.99D-03, 1.08D-05, 3.95D-03, 3.84D-03, 3.89D-05,    
     6   2.28D-03, 5.20D-03, 5.24D-03, 5.18D-03, 5.05D-03, 4.55D-03,    
     7   1.86D-03, 5.70D-03, 1.45D-05, 5.42D-03, 3.96D-03, 4.07D-03,    
     8   4.39D-03, 5.76D-03, 4.64D-03, 2.81D-03, 5.10D-03, 5.02D-03,    
     9   3.99D-03, 5.74D-03, 5.52D-03, 5.36D-03, 5.30D-03, 5.78D-03,    
     1   4.46D-03, 5.94D-03, 5.86D-03, 5.40D-03, 5.10D-03, 4.82D-03,    
     2   2.54D-04, 5.62D-03, 5.81D-03, 4.32D-03, 4.08D-03, 5.26D-03,    
     3   4.17D-03, 2.89D-03, 3.57D-03, 3.08D-03, 3.04D-03, 4.15D-03,    
     4   4.90D-03, 2.80D-03, 1.51D-03, 3.68D-03, 4.10D-03, 3.51D-03,    
     5   2.27D-03, 3.95D-03, 4.86D-03, 4.88D-03, 1.57D-03, 3.41D-03,    
     6   4.18D-03, 3.52D-03, 4.59D-03, 3.98D-03, 5.31D-03, 5.48D-03,    
     7   5.50D-03, 5.52D-03, 5.47D-03, 5.48D-03, 5.46D-03, 4.17D-03,    
     8   5.28D-03, 5.14D-03, 4.31D-03, 2.22D-03, 5.43D-03, 3.07D-03,    
     9   3.20D-03, 4.53D-03, 4.54D-03, 4.34D-03, 4.65D-03, 5.07D-03/    
      DATA OP04/                                                        
     1   5.25D-03, 4.19D-03, 5.53D-03, 4.58D-03, 4.95D-03, 5.10D-03,    
     2   4.11D-03, 4.82D-03, 5.01D-03, 5.09D-03, 4.88D-03, 5.11D-03,    
     3   5.01D-03, 4.77D-03, 5.50D-03, 2.35D-04, 3.52D-03, 4.90D-03,    
     4   3.58D-03, 3.78D-03, 4.73D-03, 4.87D-03, 5.33D-03, 4.14D-03,    
     5   3.46D-03, 2.07D-03, 2.50D-03, 2.89D-03, 5.12D-03, 4.92D-03,    
     6   4.80D-03, 4.57D-03, 4.58D-03, 5.02D-03, 4.69D-03, 4.40D-03,    
     7   4.77D-03, 4.18D-03, 4.33D-03, 1.71D-03, 4.55D-03, 4.46D-03/    
C                                                                       
      DATA OP11/                                                        
     1   7.17D-06,-1.23D-05,-4.97D-05,-3.09D-05, 3.18D-06, 7.93D-05,    
     2  -5.17D-05,-1.49D-05,-3.77D-05,-1.24D-05, 6.06D-05,-5.72D-05,    
     3   1.46D-05, 5.30D-05, 3.69D-05, 5.50D-06,-9.90D-08,-8.74D-06,    
     4   3.04D-05, 4.75D-05,-3.58D-05,-2.30D-06, 2.50D-05,-1.13D-05,    
     5   1.56D-05,-1.08D-04, 5.87D-05, 5.33D-05, 5.18D-06,-5.08D-07,    
     6  -8.57D-05, 7.25D-07,-3.56D-05, 2.14D-05,-2.49D-05, 4.35D-05,    
     7   5.09D-05, 6.19D-05, 4.58D-05, 6.72D-05, 9.39D-05, 1.04D-04,    
     8   6.55D-05,-4.25D-05,-1.53D-05,-1.52D-05,-1.73D-05,-7.38D-06,    
     9  -2.60D-06, 7.47D-05, 8.12D-06,-1.28D-04, 5.57D-05, 9.71D-05,    
     1  -7.31D-07,-9.73D-05,-4.51D-05,-2.16D-05,-7.11D-05,-6.65D-05,    
     2  -9.23D-09,-1.90D-06,-1.58D-07,-6.20D-09,-1.08D-04,-1.87D-05,    
     3  -1.60D-04,-2.92D-05,-9.49D-05, 5.43D-06,-1.29D-05,-1.07D-06,    
     4  -9.94D-05,-7.41D-05,-2.49D-05,-1.86D-08,-3.00D-08,-1.01D-07,    
     5  -9.49D-05,-1.07D-04,-8.00D-05,-7.95D-08,-6.72D-09,-3.74D-09,    
     6  -3.11D-09,-5.15D-09,-1.24D-09,-2.57D-09,-2.25D-09,-9.97D-09,    
     7  -3.27D-09,-2.92D-09,-5.97D-09,-5.13D-09,-8.56D-05,-1.25D-04,    
     8  -1.40D-05,-8.11D-05,-3.04D-05,-5.78D-09,-8.15D-09,-1.18D-04,    
     9  -8.13D-09,-1.18D-04,-8.13D-09,-1.18D-04,-3.13D-08,-8.47D-10/    
      DATA OP12/                                                        
     1  -3.60D-09,-1.19D-09,-1.34D-08,-1.51D-06,-8.26D-08,-1.52D-09,    
     2  -7.80D-05,-4.59D-05,-3.94D-09,-1.11D-04,-2.18D-09,-1.16D-07,    
     3  -2.20D-06,-3.09D-05,-5.84D-05,-5.27D-05,-5.85D-05,-5.30D-05,    
     4  -8.74D-08,-9.64D-10,-1.65D-07,-2.23D-05,-1.73D-09,-1.23D-07,    
     5  -7.52D-09,-7.62D-05,-9.51D-05,-7.84D-05,-1.15D-09,-1.16D-09,    
     6  -2.11D-05,-7.54D-10,-1.27D-04,-1.14D-08,-6.91D-07,-6.01D-05,    
     7  -1.63D-08,-5.44D-05,-7.83D-05,-5.82D-08,-4.37D-07,-1.25D-09,    
     8  -5.85D-05,-1.45D-05,-1.54D-05,-7.10D-05,-2.01D-05,-9.45D-05,    
     9   4.17D-06, 6.32D-06,-7.54D-05,-3.54D-07,-1.75D-06,-1.09D-07,    
     1  -2.42D-09,-1.96D-07,-1.99D-07,-2.65D-06,-3.25D-05,-3.05D-05,    
     2  -2.98D-07,-1.22D-07,-3.98D-05,-9.05D-09,-9.95D-10,-1.34D-09,    
     3  -3.92D-08,-1.30D-09,-2.39D-09, 1.29D-05, 1.02D-04, 1.02D-04,    
     4  -4.81D-07,-5.95D-06,-1.99D-08,-1.79D-09,-5.90D-07,-2.90D-06,    
     5  -1.16D-07,-2.56D-08, 6.83D-05,-1.08D-04,-1.25D-08,-1.80D-06,    
     6  -1.20D-07,-4.90D-08,-5.80D-09,-6.63D-06,-3.72D-06,-4.05D-06,    
     7  -1.02D-07,-6.30D-09,-1.71D-06,-6.75D-05,-1.11D-04,-1.91D-07,    
     8  -1.14D-07,-2.89D-05,-2.36D-05,-4.24D-06,-2.37D-04, 2.87D-05,    
     9  -2.17D-05, 3.04D-05,-9.09D-08,-1.65D-07,-2.07D-08,-8.72D-05/    
      DATA OP13/                                                        
     1  -7.26D-07, 1.40D-05,-4.38D-05,-4.46D-05,-8.81D-05,-8.82D-05,    
     2  -1.58D-05,-5.22D-05,-3.61D-05,-4.47D-05,-4.00D-05,-3.99D-05,    
     3  -6.25D-05,-1.38D-05, 1.98D-05, 3.48D-05,-5.39D-05,-6.79D-05,    
     4  -2.39D-05,-3.58D-05, 1.57D-04,-1.13D-04,-4.80D-05, 9.27D-05,    
     5   1.61D-05,-2.95D-05,-6.84D-07,-2.93D-05,-1.95D-04,-2.45D-06,    
     6  -1.64D-05,-3.41D-05,-3.59D-05,-3.38D-05,-2.51D-05, 2.51D-05,    
     7   4.77D-05,-1.23D-04,-9.00D-07,-4.61D-05,-2.99D-05, 1.53D-04,    
     8   3.11D-05,-6.19D-05, 9.86D-06, 7.87D-05,-1.87D-05,-1.83D-05,    
     9   9.19D-05,-6.51D-05,-6.14D-05,-3.22D-05,-4.13D-05,-6.67D-05,    
     1  -4.85D-05,-5.79D-05,-6.04D-05,-4.78D-05,-2.15D-05,-1.52D-04,    
     2  -1.59D-05,-1.02D-04,-6.52D-05, 1.47D-05,-2.20D-04,-5.97D-05,    
     3  -1.32D-05,-1.55D-04,-1.27D-04, 1.32D-04, 1.38D-04,-3.43D-05,    
     4  -6.43D-05,-8.14D-06,-9.00D-05, 1.23D-04, 1.51D-05, 1.92D-04,    
     5   3.35D-05, 3.68D-05,-2.55D-05,-2.79D-05, 1.67D-06, 2.33D-05,    
     6   1.34D-05,-6.84D-05, 1.79D-05,-5.40D-05,-1.14D-05,-5.26D-05,    
     7  -1.59D-05,-1.48D-05,-8.53D-06,-1.23D-05,-7.75D-06, 4.65D-05,    
     8  -2.88D-05,-4.11D-06, 7.42D-05, 6.12D-05,-1.42D-04, 1.42D-04,    
     9  -9.81D-05,-6.44D-06, 9.27D-06, 9.62D-05, 4.12D-05,-3.28D-05/    
      DATA OP14/                                                        
     1  -2.36D-05,-2.19D-04,-6.33D-05, 6.92D-05, 1.58D-05,-5.37D-06,    
     2  -5.93D-05,-3.14D-05, 9.46D-06,-3.67D-06, 1.56D-05,-1.06D-05,    
     3  -8.79D-06,-4.37D-05,-8.11D-05,-1.41D-05,-3.80D-05,-9.07D-06,    
     4  -1.02D-05, 1.57D-07,-9.71D-05,-7.63D-05,-3.69D-05, 4.96D-05,    
     5   6.22D-05, 5.43D-05,-1.27D-04, 9.48D-05,-4.49D-05,-2.11D-05,    
     6  -2.50D-05,-2.78D-05,-7.13D-06,-1.19D-04,-1.49D-05, 1.41D-05,    
     7  -3.33D-05, 4.69D-05, 2.05D-05, 4.88D-05,-3.96D-06, 7.17D-06/    
C                                                                       
      DATA OP21/                                                        
     1  -3.37D-07, 7.95D-08, 1.11D-06, 5.63D-07,-3.75D-07,-2.49D-06,    
     2   1.03D-06, 2.82D-07, 6.41D-07, 2.26D-07,-3.18D-06, 1.18D-06,    
     3  -1.24D-06,-1.45D-06,-1.14D-06,-7.25D-07, 1.83D-09, 1.61D-07,    
     4  -1.10D-06,-1.93D-06, 4.22D-07, 4.23D-08,-1.77D-06,-5.55D-07,    
     5  -1.30D-06, 1.52D-06,-2.36D-06,-1.98D-06,-8.71D-07, 9.37D-09,    
     6   1.09D-06,-4.19D-07,-1.95D-07,-8.67D-07, 4.56D-07,-2.23D-06,    
     7  -1.57D-06,-2.23D-06,-1.65D-06,-2.52D-06,-2.99D-06,-3.39D-06,    
     8  -2.00D-06, 7.78D-07, 2.83D-07, 2.81D-07, 3.20D-07, 1.37D-07,    
     9   4.86D-08,-2.91D-06,-1.35D-06, 1.16D-06,-2.77D-06,-3.42D-06,    
     1   1.36D-08, 1.57D-06,-3.65D-08,-4.50D-07, 4.87D-07,-7.13D-10,    
     2   1.72D-10, 3.55D-08, 2.93D-09, 1.15D-10, 1.34D-06, 3.48D-07,    
     3   2.43D-06,-4.85D-07, 1.58D-06,-1.57D-06,-1.30D-06, 2.00D-08,    
     4   1.69D-06, 3.81D-07, 4.62D-07, 3.48D-10, 5.60D-10, 1.89D-09,    
     5   1.52D-06, 1.82D-06, 5.53D-07, 1.49D-09, 1.25D-10, 6.96D-11,    
     6   5.79D-11, 9.59D-11, 2.32D-11, 4.79D-11, 4.20D-11, 1.86D-10,    
     7   6.10D-11, 5.46D-11, 1.11D-10, 9.56D-11, 1.35D-06, 2.26D-06,    
     8   2.61D-07, 1.48D-06, 5.70D-07, 1.08D-10, 1.53D-10, 1.73D-06,    
     9   1.52D-10, 1.73D-06, 1.52D-10, 1.74D-06, 5.86D-10, 1.58D-11/    
      DATA OP22/                                                        
     1   6.71D-11, 2.22D-11, 2.51D-10, 2.82D-08, 1.55D-09, 2.85D-11,    
     2   1.44D-06, 8.58D-07, 7.34D-11, 1.75D-06, 4.06D-11, 2.16D-09,    
     3   4.11D-08, 5.76D-07, 9.17D-07, 9.63D-07, 9.19D-07, 9.68D-07,    
     4   1.63D-09, 1.80D-11, 3.08D-09, 4.14D-07, 3.23D-11, 2.29D-09,    
     5   1.41D-10, 1.04D-06, 1.13D-06, 1.41D-06, 2.16D-11, 2.17D-11,    
     6  -5.76D-07, 1.41D-11, 2.06D-06, 2.13D-10, 1.29D-08, 4.45D-07,    
     7   3.05D-10,-7.51D-08, 7.48D-07, 1.09D-09, 8.20D-09, 2.34D-11,    
     8   4.38D-08, 2.71D-07,-9.53D-07, 8.15D-07, 3.77D-07, 1.48D-06,    
     9  -1.56D-06,-1.68D-06, 1.73D-07, 6.65D-09, 3.28D-08, 2.04D-09,    
     1   4.53D-11, 3.69D-09, 3.73D-09, 4.98D-08,-9.12D-07,-9.60D-07,    
     2   5.59D-09, 2.29D-09,-4.18D-07, 1.70D-10, 1.86D-11, 2.50D-11,    
     3   7.33D-10, 2.43D-11, 4.47D-11,-1.43D-06,-3.21D-06,-3.41D-06,    
     4   9.01D-09, 1.11D-07, 3.73D-10, 3.36D-11, 1.10D-08, 5.43D-08,    
     5   2.16D-09, 4.79D-10,-2.62D-06, 1.95D-06, 2.34D-10, 3.36D-08,    
     6   2.25D-09, 9.16D-10, 1.08D-10, 1.24D-07, 6.95D-08,-9.16D-07,    
     7   1.91D-09, 1.18D-10, 3.19D-08, 8.58D-07, 1.61D-06, 3.57D-09,    
     8   2.13D-09, 4.32D-07, 4.31D-07,-7.40D-07, 4.05D-06,-9.68D-07,    
     9  -5.46D-07,-1.57D-06, 1.70D-09, 3.08D-09, 3.86D-10, 8.28D-07/    
      DATA OP23/                                                        
     1   1.35D-08,-7.39D-07, 7.96D-07, 1.19D-07, 1.73D-06, 1.30D-08,    
     2   1.60D-07, 1.09D-06, 4.73D-07, 6.58D-07, 8.72D-07, 7.70D-07,    
     3   1.05D-06,-1.04D-07,-3.34D-06,-1.99D-06, 1.15D-06,-1.10D-07,    
     4  -9.69D-07, 6.80D-07,-5.25D-06, 1.33D-06,-9.72D-07,-3.05D-06,    
     5  -5.06D-07,-1.22D-07, 1.28D-08, 4.76D-07, 3.14D-06, 4.57D-08,    
     6  -6.98D-07, 5.55D-07, 5.97D-07, 6.00D-07, 3.48D-07,-6.73D-07,    
     7  -2.22D-06, 2.16D-06, 1.67D-08, 7.93D-07,-1.28D-06,-5.71D-06,    
     8  -2.76D-06, 1.17D-06,-8.05D-07,-2.79D-06, 7.71D-08,-8.75D-08,    
     9  -2.58D-06, 1.19D-06, 1.19D-06, 4.30D-07, 5.23D-07, 1.28D-06,    
     1   8.71D-07, 8.74D-07, 1.07D-06, 4.76D-07, 7.41D-08, 1.49D-06,    
     2   2.96D-07, 4.44D-07, 1.33D-06,-2.66D-06, 3.71D-06, 8.07D-07,    
     3  -1.51D-08, 2.59D-06, 1.36D-06,-4.74D-06,-5.15D-06, 1.96D-07,    
     4   6.90D-08,-8.74D-07, 1.62D-06,-4.47D-06,-2.80D-06,-5.71D-06,    
     5  -1.54D-06,-2.27D-06, 9.27D-08,-2.71D-07,-8.89D-07,-1.52D-06,    
     6  -2.42D-06, 3.07D-08,-1.15D-06, 9.39D-07,-6.72D-08, 1.18D-06,    
     7  -6.80D-09,-7.73D-08,-1.74D-07,-7.60D-08,-1.87D-07,-2.40D-06,    
     8   5.76D-07,-1.81D-07,-2.75D-06,-2.53D-06, 7.19D-07,-5.05D-06,    
     9   9.38D-07,-2.44D-07,-1.37D-06,-4.13D-06,-2.23D-06,-1.08D-06/    
      DATA OP24/                                                        
     1   1.38D-07, 3.60D-06, 1.40D-06,-3.26D-06,-7.12D-07,-2.95D-07,    
     2  -7.87D-07, 2.96D-07,-4.76D-07,-2.86D-07,-6.70D-07, 4.96D-08,    
     3   1.85D-07, 8.56D-07, 1.46D-06, 2.54D-07, 3.75D-07,-2.38D-07,    
     4  -1.12D-07,-5.71D-07, 1.95D-06, 1.65D-06, 5.45D-08,-1.64D-06,    
     5  -2.28D-06,-2.32D-06, 2.06D-06,-2.83D-06, 2.88D-07,-2.96D-07,    
     6   6.86D-07, 5.70D-07, 6.81D-08, 4.69D-07, 2.93D-08,-4.10D-07,    
     7   5.29D-08,-1.60D-06,-5.82D-07,-2.04D-06,-4.22D-08,-3.37D-07/    
C                                                                       
      DATA OP31/                                                        
     1  -1.20D-09,-3.91D-09,-1.18D-08,-6.32D-09,-1.74D-09, 1.31D-08,    
     2  -1.15D-08,-5.74D-09,-7.66D-09,-1.27D-09, 2.36D-08,-1.19D-08,    
     3   9.87D-09, 5.04D-09, 5.64D-09, 2.61D-09,-1.04D-11,-9.15D-10,    
     4   3.82D-09, 1.03D-08,-6.39D-09,-2.40D-10, 1.39D-08, 4.62D-09,    
     5   7.70D-09,-6.89D-09, 1.53D-08, 1.04D-08, 7.44D-09,-5.31D-11,    
     6  -4.44D-09,-1.34D-10, 3.97D-09, 2.48D-09,-2.58D-09, 1.63D-08,    
     7   6.69D-09, 1.36D-08, 8.14D-09, 1.56D-08, 1.82D-08, 2.16D-08,    
     8   1.04D-08,-4.39D-09,-1.61D-09,-1.60D-09,-1.82D-09,-7.83D-10,    
     9  -2.78D-10, 1.96D-08, 1.08D-08,-2.51D-09, 2.01D-08, 2.32D-08,    
     1  -7.76D-11,-8.07D-09, 3.21D-09, 5.35D-09, 3.64D-11, 4.07D-09,    
     2  -9.85D-13,-2.03D-10,-1.67D-11,-6.57D-13,-5.28D-09,-1.98D-09,    
     3  -1.19D-08, 6.01D-09,-8.30D-09, 1.34D-08, 1.23D-08,-1.14D-10,    
     4  -9.08D-09, 1.27D-09,-2.63D-09,-1.99D-12,-3.21D-12,-1.08D-11,    
     5  -7.77D-09,-9.72D-09, 8.38D-11,-8.53D-12,-7.14D-13,-3.97D-13,    
     6  -3.30D-13,-5.48D-13,-1.32D-13,-2.74D-13,-2.40D-13,-1.06D-12,    
     7  -3.48D-13,-3.12D-13,-6.36D-13,-5.46D-13,-6.78D-09,-1.26D-08,    
     8  -1.49D-09,-8.33D-09,-3.27D-09,-6.20D-13,-8.73D-13,-8.18D-09,    
     9  -8.71D-13,-8.22D-09,-8.70D-13,-8.24D-09,-3.35D-12,-9.05D-14/    
      DATA OP32/                                                        
     1  -3.84D-13,-1.27D-13,-1.44D-12,-1.61D-10,-8.89D-12,-1.63D-13,    
     2  -8.14D-09,-4.91D-09,-4.20D-13,-8.84D-09,-2.32D-13,-1.23D-11,    
     3  -2.35D-10,-3.29D-09,-4.61D-09,-5.43D-09,-4.62D-09,-5.46D-09,    
     4  -9.34D-12,-1.03D-13,-1.76D-11,-2.36D-09,-1.85D-13,-1.31D-11,    
     5  -8.04D-13,-4.56D-09,-4.13D-09,-7.90D-09,-1.24D-13,-1.24D-13,    
     6   6.57D-09,-8.08D-14,-1.06D-08,-1.22D-12,-7.40D-11,-2.16D-10,    
     7  -1.74D-12, 4.10D-09,-1.77D-09,-6.25D-12,-4.70D-11,-1.34D-13,    
     8   3.31D-09,-1.55D-09, 9.47D-09,-2.83D-09,-2.16D-09,-7.46D-09,    
     9   1.35D-08, 1.45D-08, 3.17D-09,-3.81D-11,-1.88D-10,-1.17D-11,    
     1  -2.60D-13,-2.11D-11,-2.14D-11,-2.86D-10, 1.02D-08, 1.05D-08,    
     2  -3.21D-11,-1.32D-11, 6.22D-09,-9.71D-13,-1.07D-13,-1.43D-13,    
     3  -4.20D-12,-1.39D-13,-2.56D-13, 1.15D-08, 1.97D-08, 2.19D-08,    
     4  -5.17D-11,-6.38D-10,-2.14D-12,-1.92D-13,-6.32D-11,-3.11D-10,    
     5  -1.24D-11,-2.74D-12, 1.70D-08,-1.09D-08,-1.34D-12,-1.92D-10,    
     6  -1.29D-11,-5.25D-12,-6.20D-13,-7.07D-10,-3.98D-10, 7.72D-09,    
     7  -1.09D-11,-6.72D-13,-1.82D-10,-3.53D-09,-7.59D-09,-2.04D-11,    
     8  -1.22D-11,-7.23D-09,-7.59D-09, 6.45D-09,-2.18D-08, 1.89D-09,    
     9   6.31D-09, 8.31D-09,-9.69D-12,-1.76D-11,-2.20D-12,-4.17D-09/    
      DATA OP33/                                                        
     1  -7.71D-11, 1.76D-09,-4.47D-09, 6.02D-11,-1.43D-08, 5.26D-09,    
     2  -5.55D-09,-1.15D-08,-7.14D-09,-8.26D-09,-1.13D-08,-8.74D-09,    
     3  -1.05D-08,-3.79D-09, 2.86D-08, 1.19D-08,-1.23D-08, 5.21D-09,    
     4   7.73D-09,-9.58D-09, 3.47D-08,-8.52D-09, 1.08D-08, 1.70D-08,    
     5  -1.75D-09, 7.33D-10,-7.32D-11,-6.84D-09,-1.61D-08,-2.61D-10,    
     6   6.69D-09,-6.80D-09,-6.99D-09,-7.31D-09,-5.26D-09,-2.38D-09,    
     7   1.64D-08,-1.64D-08,-9.50D-11,-8.21D-09, 1.27D-08, 3.91D-08,    
     8   2.08D-08,-1.15D-08, 2.65D-09, 1.72D-08,-2.96D-09,-1.36D-09,    
     9   1.12D-08,-1.10D-08,-1.19D-08,-7.97D-09,-5.70D-09,-1.17D-08,    
     1  -8.07D-09,-7.93D-09,-9.56D-09,-6.17D-09,-3.54D-09,-4.44D-09,    
     2  -1.69D-09,-6.92D-11,-1.37D-08, 2.21D-08,-1.98D-08,-6.14D-09,    
     3  -2.23D-09,-1.37D-08,-4.27D-09, 3.22D-08, 3.63D-08,-2.33D-09,    
     4   3.21D-10, 6.79D-09,-9.04D-09, 2.94D-08, 2.41D-08, 3.47D-08,    
     5   9.75D-09, 1.52D-08,-4.53D-09,-4.05D-10, 7.61D-09, 9.51D-09,    
     6   1.97D-08, 3.31D-09, 3.75D-09,-8.28D-09,-2.28D-09,-1.22D-08,    
     7  -1.82D-09,-1.04D-09,-6.65D-10,-1.39D-09,-5.56D-10, 1.51D-08,    
     8  -7.87D-09,-1.49D-09, 1.53D-08, 1.75D-08, 2.25D-09, 3.44D-08,    
     9  -2.70D-09,-2.00D-09, 7.92D-09, 2.81D-08, 1.42D-08, 9.34D-09/    
      DATA OP34/                                                        
     1  -2.81D-09,-1.88D-08,-1.54D-08, 2.18D-08, 1.37D-09,-7.76D-10,    
     2   1.07D-08,-3.36D-09, 2.18D-10,-3.92D-10, 8.51D-10,-3.36D-09,    
     3  -7.38D-09,-1.11D-08,-1.28D-08,-1.42D-09,-3.66D-09,-1.45D-09,    
     4  -2.39D-09, 9.60D-10,-1.43D-08,-1.55D-08,-1.88D-09, 5.78D-09,    
     5   1.27D-08, 1.62D-08,-1.06D-08, 1.55D-08,-2.89D-09,-1.25D-09,    
     6  -1.04D-08,-7.20D-09,-4.86D-09, 2.70D-09,-3.16D-09,-1.17D-09,    
     7  -1.71D-09, 8.31D-09,-5.64D-11, 1.43D-08,-3.72D-09,-1.20D-09/    
C                                                                       
      END                                                               
C                                                                       
C                           DISCLAIMER                                  
C                                                                       
C   THIS FILE WAS GENERATED ON 01/27/98 BY THE VERSION OF               
C   ADIFOR COMPILED ON AUG 21 1995.                                     
C                                                                       
C   POSTPROCESSED IN ORDER TO OBTAIN STRICT F77 COMPATIBILITY           
C                                                                       
C   ADIFOR WAS PREPARED AS AN ACCOUNT OF WORK SPONSORED BY AN           
C   AGENCY OF THE UNITED STATES GOVERNMENT, RICE UNIVERSITY, AND        
C   THE UNIVERSITY OF CHICAGO.  NEITHER THE AUTHOR(S), THE UNITED       
C   STATES GOVERNMENT NOR ANY AGENCY THEREOF, NOR RICE UNIVERSITY,      
C   NOR THE UNIVERSITY OF CHICAGO, INCLUDING ANY OF THEIR EMPLOYEES     
C   OR OFFICERS, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES     
C   ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETE-   
C   NESS, OR USEFULNESS OF ANY INFORMATION OR PROCESS DISCLOSED, OR     
C   REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.  
C                                                                       
C                                                                       
C CALCULATE FUNCTION VALUES OF DISCRETE-TIME SYSTEM EQUATIONS AND CONSTR
C THIS SUBROUTINE DIVIDES EACH TIME STEP K INTO TWO SAMPLE PERIODS AND  
C INTEGRATES CONTINUOUS-TIME SYSTEM EQUATIONS.                          
C                                                                       
      SUBROUTINE GZSRFG(GZPZ, K, X, GZX, LDGZX, U, GZU, LDGZU, F, GZF, L
     *DGZF, G, GZG, LDGZG)                                              
C                                                                       
        INTEGER NX, NU, NG, NP, NW                                      
        PARAMETER (NX = 9, NU = 2, NG = 2, NP = 1, NW = 1)              
        INTEGER K                                                       
        DOUBLE PRECISION X(NX), U(NU), F(NX), G(NG)                     
        INTEGER KK                                                      
        DOUBLE PRECISION DT, FH(NX), P(NP), W(NW)                       
C                                                                       
        INTEGER GZPMXZ                                                  
        PARAMETER (GZPMXZ = 11)                                         
        INTEGER GZIZ, GZPZ, LDGZU, LDGZG, LDGZX, LDGZF                  
        DOUBLE PRECISION GZP(GZPMXZ, NP), GZU(LDGZU, NU), GZG(LDGZG, NG)
     *, GZW(GZPMXZ, NW), GZX(LDGZX, NX), GZF(LDGZF, NX), GZFH(GZPMXZ, NX
     *)                                                                 
        SAVE GZP, GZW, GZFH                                             
        EXTERNAL GZSRK4                                                 
C                                                                       
C                                                                       
C                                                                       
        IF (GZPZ .GT. GZPMXZ) THEN                                      
          PRINT *, 'PARAMETER GZPZ IS GREATER THAN GZPMXZ'              
          STOP                                                          
        ENDIF                                                           
        KK = 2 * (K - 1) + 1                                            
        DT = 12.0D0                                                     
C                                                                       
        DO 500 GZIZ = 1, GZPZ                                           
          GZP(GZIZ, 1) = GZU(GZIZ, 1)                                   
 500    CONTINUE                                                        
        P(1) = U(1)                                                     
C--------                                                               
        CALL GZSRK4(GZPZ, KK, DT, X, GZX, LDGZX, P, GZP, GZPMXZ, FH, GZF
     *H, GZPMXZ, W, GZW, GZPMXZ)                                        
        DO 501 GZIZ = 1, GZPZ                                           
          GZG(GZIZ, 1) = GZW(GZIZ, 1)                                   
 501    CONTINUE                                                        
        G(1) = W(1)                                                     
C--------                                                               
C                                                                       
        DO 502 GZIZ = 1, GZPZ                                           
          GZP(GZIZ, 1) = GZU(GZIZ, 2)                                   
 502    CONTINUE                                                        
        P(1) = U(2)                                                     
C--------                                                               
        CALL GZSRK4(GZPZ, KK + 1, DT, FH, GZFH, GZPMXZ, P, GZP, GZPMXZ, 
     *F, GZF, LDGZF, W, GZW, GZPMXZ)                                    
        DO 503 GZIZ = 1, GZPZ                                           
          GZG(GZIZ, 2) = GZW(GZIZ, 1)                                   
 503    CONTINUE                                                        
        G(2) = W(1)                                                     
C--------                                                               
C                                                                       
        RETURN                                                          
      END                                                               
C                                                                       
C                                                                       
C RUNGE KUTTA TYPE INTEGRATION PROCEDURE.                               
C                                                                       
      SUBROUTINE GZSRK4(GZPZ, KK, DT, X, GZX, LDGZX, P, GZP, LDGZP, F, G
     *ZF, LDGZF, W, GZW, LDGZW)                                         
C                                                                       
        INTEGER NX, NP, NW                                              
        PARAMETER (NX = 9, NP = 1, NW = 1)                              
        INTEGER KK                                                      
        DOUBLE PRECISION DT, X(NX), P(NP), F(NX), W(NW)                 
        INTEGER I                                                       
        DOUBLE PRECISION DF(NX), XH(NX), W1(NW)                         
C                                                                       
        INTEGER GZPMXZ                                                  
        PARAMETER (GZPMXZ = 11)                                         
        INTEGER GZIZ, GZPZ, LDGZF, LDGZX, LDGZP, LDGZW                  
        DOUBLE PRECISION D4ZB, GZF(LDGZF, NX), GZX(LDGZX, NX), GZDF(GZPM
     *XZ, NX), GZXH(GZPMXZ, NX), GZP(LDGZP, NP), GZW(LDGZW, NW), GZW1(  
     *GZPMXZ, NW)                                                       
        SAVE GZDF, GZXH, GZW1                                           
        EXTERNAL GZSREQ                                                 
C                                                                       
C                                                                       
C                                                                       
        IF (GZPZ .GT. GZPMXZ) THEN                                      
          PRINT *, 'PARAMETER GZPZ IS GREATER THAN GZPMXZ'              
          STOP                                                          
        ENDIF                                                           
        CALL GZSREQ(GZPZ, KK, X, GZX, LDGZX, P, GZP, LDGZP, DF, GZDF, GZ
     *PMXZ, W, GZW, LDGZW)                                              
        DO 506 I = 1, NX                                                
          D4ZB = DT / 6.0D0                                             
          DO 504 GZIZ = 1, GZPZ                                         
            GZF(GZIZ, I) = D4ZB * GZDF(GZIZ, I) + GZX(GZIZ, I)          
 504      CONTINUE                                                      
          F(I) = X(I) + DT / 6.0D0 * DF(I)                              
C--------                                                               
          D4ZB = DT / 2.0D0                                             
          DO 505 GZIZ = 1, GZPZ                                         
            GZXH(GZIZ, I) = D4ZB * GZDF(GZIZ, I) + GZX(GZIZ, I)         
 505      CONTINUE                                                      
          XH(I) = X(I) + DT / 2.0D0 * DF(I)                             
C--------                                                               
 506    CONTINUE                                                        
C                                                                       
        CALL GZSREQ(GZPZ, KK, XH, GZXH, GZPMXZ, P, GZP, LDGZP, DF, GZDF 
     *, GZPMXZ, W1, GZW1, GZPMXZ)                                       
        DO 509 I = 1, NX                                                
          D4ZB = DT / 3.0D0                                             
          DO 507 GZIZ = 1, GZPZ                                         
            GZF(GZIZ, I) = D4ZB * GZDF(GZIZ, I) + GZF(GZIZ, I)          
 507      CONTINUE                                                      
          F(I) = F(I) + DT / 3.0D0 * DF(I)                              
C--------                                                               
          D4ZB = DT / 2.0D0                                             
          DO 508 GZIZ = 1, GZPZ                                         
            GZXH(GZIZ, I) = D4ZB * GZDF(GZIZ, I) + GZX(GZIZ, I)         
 508      CONTINUE                                                      
          XH(I) = X(I) + DT / 2.0D0 * DF(I)                             
C--------                                                               
 509    CONTINUE                                                        
C                                                                       
        CALL GZSREQ(GZPZ, KK, XH, GZXH, GZPMXZ, P, GZP, LDGZP, DF, GZDF 
     *, GZPMXZ, W1, GZW1, GZPMXZ)                                       
        DO 512 I = 1, NX                                                
          D4ZB = DT / 3.0D0                                             
          DO 510 GZIZ = 1, GZPZ                                         
            GZF(GZIZ, I) = D4ZB * GZDF(GZIZ, I) + GZF(GZIZ, I)          
 510      CONTINUE                                                      
          F(I) = F(I) + DT / 3.0D0 * DF(I)                              
C--------                                                               
          DO 511 GZIZ = 1, GZPZ                                         
            GZXH(GZIZ, I) = DT * GZDF(GZIZ, I) + GZX(GZIZ, I)           
 511      CONTINUE                                                      
          XH(I) = X(I) + DT * DF(I)                                     
C--------                                                               
 512    CONTINUE                                                        
C                                                                       
        CALL GZSREQ(GZPZ, KK, XH, GZXH, GZPMXZ, P, GZP, LDGZP, DF, GZDF 
     *, GZPMXZ, W1, GZW1, GZPMXZ)                                       
        DO 514 I = 1, NX                                                
          D4ZB = DT / 6.0D0                                             
          DO 513 GZIZ = 1, GZPZ                                         
            GZF(GZIZ, I) = D4ZB * GZDF(GZIZ, I) + GZF(GZIZ, I)          
 513      CONTINUE                                                      
          F(I) = F(I) + DT / 6.0D0 * DF(I)                              
C--------                                                               
 514    CONTINUE                                                        
C                                                                       
      END                                                               
C                                                                       
C                                                                       
C CALCULATE RIGHT HAND SIDES OF THE ODES DEFINING THE SYSTEM.           
C                                                                       
      SUBROUTINE GZSREQ(GZPZ, KK, X, GZX, LDGZX, P, GZP, LDGZP, DF, GZDF
     *, LDGZDF, W, GZW, LDGZW)                                          
C                                                                       
        INTEGER KMAX, NX, NP, NW                                        
        LOGICAL EXTU                                                    
        PARAMETER (KMAX = 366, NX = 9, NP = 1, NW = 1, EXTU = .TRUE.)   
        INTEGER KK                                                      
        DOUBLE PRECISION X(NX), P(NP), DF(NX), W(NW), V(199)            
        INTEGER K                                                       
        DOUBLE PRECISION TAS(KMAX), TRSHS(KMAX), PSHS(KMAX)             
        DOUBLE PRECISION H0S(KMAX), H1S(KMAX), H2S(KMAX), H3S(KMAX)     
        DOUBLE PRECISION OP0S(KMAX), OP1S(KMAX), OP2S(KMAX), OP3S(KMAX) 
        COMMON /SARDAT/ TAS, TRSHS, PSHS, H0S, H1S, H2S, H3S, OP0S, OP1S
     *, OP2S, OP3S                                                      
C                                                                       
        INTEGER GZPMXZ                                                  
        PARAMETER (GZPMXZ = 11)                                         
        INTEGER GZIZ, GZPZ, LDGZP, LDGZX, LDGZDF, LDGZW                 
        DOUBLE PRECISION D11ZB, D5ZB, D10ZB, D6ZV, D7ZB, D6ZB, D9ZV, D8Z
     *V, D2ZV, D3ZV                                                     
        DOUBLE PRECISION D2ZW, D2ZB, D3ZB, D1ZW, D4ZV, D4ZB, D1ZP, D5ZV,
     * GZV(GZPMXZ, 199), GZP(LDGZP, NP)                                 
        DOUBLE PRECISION GZX(LDGZX, NX), GZD1ZW(GZPMXZ), GZDF(LDGZDF, N 
     *X), GZD2ZW(GZPMXZ), GZW(LDGZW, NW)                                
        SAVE GZV, GZD1ZW, GZD2ZW                                        
        INTRINSIC DBLE                                                  
C                                                                       
C                                                                       
C                                                                       
        IF (GZPZ .GT. GZPMXZ) THEN                                      
          PRINT *, 'PARAMETER GZPZ IS GREATER THAN GZPMXZ'              
          STOP                                                          
        ENDIF                                                           
        K = ((KK - 1) / 2) + 1                                          
C                                                                       
        TA = TAS(K)                                                     
        TRSH = TRSHS(K)                                                 
        PSH = PSHS(K)                                                   
        IF (2*(K-1) .EQ. (KK-1)) THEN                                   
           H0 = 2.0D0 * H0S(K)                                          
           H1 = 2.0D0 * H1S(K)                                          
           H2 = 2.0D0 * H2S(K)                                          
           H3 = 2.0D0 * H3S(K)                                          
           OP0 = 2.0D0 * OP0S(K)                                        
           OP1 = 2.0D0 * OP1S(K)                                        
           OP2 = 2.0D0 * OP2S(K)                                        
           OP3 = 2.0D0 * OP3S(K)                                        
        ELSE                                                            
           H0 = 0.0D0                                                   
           H1 = 0.0D0                                                   
           H2 = 0.0D0                                                   
           H3 = 0.0D0                                                   
           OP0 = 0.0D0                                                  
           OP1 = 0.0D0                                                  
           OP2 = 0.0D0                                                  
           OP3 = 0.0D0                                                  
        END IF                                                          
C                                                                       
C SORTED AND SOLVED EQUATIONS, GENERATED BY DYMOLA                      
C (POSTPROCESSED IN ORDER TO PLACE ALL TEMPORARY VARIABLES IN V)        
C   M.                                                                  
        DO 515 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 12) = 0.0D0                                         
 515    CONTINUE                                                        
        V(12) = 0.001D0 * DBLE(PSH)                                     
C--------                                                               
        IF (EXTU) THEN                                                  
          DO 516 GZIZ = 1, GZPZ                                         
            GZV(GZIZ, 13) = GZP(GZIZ, 1)                                
 516      CONTINUE                                                      
          V(13) = P(1)                                                  
C--------                                                               
        ELSE                                                            
          IF (X(1) .GT. 60.0D0) THEN                                    
            D2ZV = X(1) - DBLE(TRSH)                                    
            D3ZV = (60.0D0 - DBLE(TRSH)) / D2ZV                         
            D3ZB = -D3ZV / D2ZV                                         
            DO 517 GZIZ = 1, GZPZ                                       
              GZV(GZIZ, 199) = D3ZB * GZX(GZIZ, 1)                      
 517        CONTINUE                                                    
            V(199) = D3ZV                                               
C--------                                                               
          ELSE                                                          
            DO 518 GZIZ = 1, GZPZ                                       
              GZD1ZW(GZIZ) = -20.0D0 * GZX(GZIZ, 1)                     
 518        CONTINUE                                                    
            D1ZW = 15.0D0 - 20.0D0 * (X(1) - DBLE(TRSH))                
            D2ZV = EXP(D1ZW)                                            
            D1ZP =  D2ZV                                                
            DO 519 GZIZ = 1, GZPZ                                       
              GZV(GZIZ, 198) = D1ZP * GZD1ZW(GZIZ)                      
 519        CONTINUE                                                    
            V(198) = D2ZV                                               
C--------                                                               
            D2ZV = 1.0D0 + V(198)                                       
            D3ZV = 1.0D0 / D2ZV                                         
            D3ZB = -D3ZV / D2ZV                                         
            DO 520 GZIZ = 1, GZPZ                                       
              GZV(GZIZ, 199) = D3ZB * GZV(GZIZ, 198)                    
 520        CONTINUE                                                    
            V(199) = D3ZV                                               
C--------                                                               
          ENDIF                                                         
          DO 521 GZIZ = 1, GZPZ                                         
            GZV(GZIZ, 13) = GZV(GZIZ, 199)                              
 521      CONTINUE                                                      
          V(13) = V(199)                                                
C--------                                                               
        ENDIF                                                           
        D2ZV = 1000.0D0 * V(13)                                         
        D2ZB = 1.0D0 / (69666.6666666667D0 - 1161.11111111111D0 * DBLE(T
     *RSH))                                                             
        D4ZB = D2ZB * D2ZV                                              
        D5ZB = D2ZB * V(12) * 1000.0D0                                  
        DO 522 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 6) = D4ZB * GZV(GZIZ, 12) + D5ZB * GZV(GZIZ, 13)    
 522    CONTINUE                                                        
        V(6) = D2ZV * V(12) / (69666.6666666667D0 - 1161.11111111111D0 *
     * DBLE(TRSH))                                                      
C--------                                                               
        D3ZV = 1.16111111111111D0 * (X(1) - DBLE(TRSH))                 
        D5ZB = V(6) * 1.16111111111111D0                                
        DO 523 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 11) = D3ZV * GZV(GZIZ, 6) + D5ZB * GZX(GZIZ, 1)     
 523    CONTINUE                                                        
        V(11) = D3ZV * V(6)                                             
C--------                                                               
        DO 524 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 10) = -GZV(GZIZ, 11) + GZV(GZIZ, 12)                
 524    CONTINUE                                                        
        V(10) = V(12) - V(11)                                           
C--------                                                               
        DO 525 GZIZ = 1, GZPZ                                           
          GZDF(GZIZ, 7) = 0.001D0 * GZV(GZIZ, 10)                       
 525    CONTINUE                                                        
        DF(7) = 0.001D0 * V(10)                                         
C--------                                                               
        D3ZV = DBLE(OP2) + DBLE(OP3) * X(6)                             
        D5ZV = DBLE(OP1) + D3ZV * X(6)                                  
        D4ZB = D5ZV + X(6) * D3ZV + X(6) * X(6) * DBLE(OP3)             
        DO 526 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 19) = D4ZB * GZX(GZIZ, 6)                           
 526    CONTINUE                                                        
        V(19) = DBLE(OP0) + D5ZV * X(6)                                 
C--------                                                               
        DO 527 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 14) = 740.0D0 * GZV(GZIZ, 19)                       
 527    CONTINUE                                                        
        V(14) = 740.0D0 * V(19)                                         
C--------                                                               
        DO 528 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 16) = 0.000277777777777778D0 * GZV(GZIZ, 14)        
 528    CONTINUE                                                        
        V(16) = 0.000277777777777778D0 * V(14)                          
C--------                                                               
        DO 529 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 22) = -GZV(GZIZ, 16)                                
 529    CONTINUE                                                        
        V(22) = -V(16)                                                  
C--------                                                               
        DO 530 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 26) = -GZV(GZIZ, 16)                                
 530    CONTINUE                                                        
        V(26) = -V(16)                                                  
C--------                                                               
        DO 531 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 35) = -GZV(GZIZ, 16)                                
 531    CONTINUE                                                        
        V(35) = -V(16)                                                  
C--------                                                               
        DO 532 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 43) = 3600.0D0 * GZV(GZIZ, 16)                      
 532    CONTINUE                                                        
        V(43) = 3600.0D0 * V(16)                                        
C--------                                                               
        D3ZV = DBLE(H2) + DBLE(H3) * X(6)                               
        D5ZV = DBLE(H1) + D3ZV * X(6)                                   
        D4ZB = D5ZV + X(6) * D3ZV + X(6) * X(6) * DBLE(H3)              
        DO 533 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 18) = D4ZB * GZX(GZIZ, 6)                           
 533    CONTINUE                                                        
        V(18) = DBLE(H0) + D5ZV * X(6)                                  
C--------                                                               
        D5ZV = 1.16111111111111D-27 + 1161.11111111111D0 * V(19)        
        D6ZV = V(18) / D5ZV                                             
        D4ZB = 1.0D0 / D5ZV                                             
        D7ZB = -D6ZV / D5ZV * 1161.11111111111D0                        
        DO 534 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 46) = D7ZB * GZV(GZIZ, 19) + D4ZB * GZV(GZIZ, 18) + 
     *GZX(GZIZ, 6)                                                      
 534    CONTINUE                                                        
        V(46) = X(6) + D6ZV                                             
C--------                                                               
        DO 535 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 8) = 0.000277777777777778D0 * GZV(GZIZ, 6)          
 535    CONTINUE                                                        
        V(8) = 0.000277777777777778D0 * V(6)                            
C--------                                                               
        DO 536 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 24) = -GZV(GZIZ, 8)                                 
 536    CONTINUE                                                        
        V(24) = -V(8)                                                   
C--------                                                               
        DO 537 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 28) = -GZV(GZIZ, 8)                                 
 537    CONTINUE                                                        
        V(28) = -V(8)                                                   
C--------                                                               
        DO 538 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 123) = -GZV(GZIZ, 8)                                
 538    CONTINUE                                                        
        V(123) = -V(8)                                                  
C--------                                                               
        DO 539 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 106) = -GZV(GZIZ, 8)                                
 539    CONTINUE                                                        
        V(106) = -V(8)                                                  
C--------                                                               
        DO 540 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 89) = -GZV(GZIZ, 8)                                 
 540    CONTINUE                                                        
        V(89) = -V(8)                                                   
C--------                                                               
        DO 541 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 72) = -GZV(GZIZ, 8)                                 
 541    CONTINUE                                                        
        V(72) = -V(8)                                                   
C--------                                                               
        DO 542 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 55) = -GZV(GZIZ, 8)                                 
 542    CONTINUE                                                        
        V(55) = -V(8)                                                   
C--------                                                               
        DO 543 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 37) = -GZV(GZIZ, 8)                                 
 543    CONTINUE                                                        
        V(37) = -V(8)                                                   
C--------                                                               
        DO 544 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 48) = 3600.0D0 * GZV(GZIZ, 8)                       
 544    CONTINUE                                                        
        V(48) = 3600.0D0 * V(8)                                         
C--------                                                               
        DO 545 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 166) = -47.5428571428572D0 * GZX(GZIZ, 2) + 47.54285
     *71428572D0 * GZX(GZIZ, 1)                                         
 545    CONTINUE                                                        
        V(166) = 47.5428571428572D0 * (X(1) - X(2))                     
C--------                                                               
        DO 546 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 157) = 10.1587301587302D0 * GZX(GZIZ, 1)            
 546    CONTINUE                                                        
        V(157) = 10.1587301587302D0 * (X(1) - DBLE(TA))                 
C--------                                                               
        DO 547 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 40) = -GZV(GZIZ, 157) + (-GZV(GZIZ, 166))           
 547    CONTINUE                                                        
        V(40) = -(V(166) + V(157))                                      
C--------                                                               
        DO 548 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 138) = 3.76619850066144D0 * GZX(GZIZ, 1)            
 548    CONTINUE                                                        
        V(138) = 3.76619850066144D0 * (X(1) - DBLE(TA))                 
C--------                                                               
        DO 549 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 41) = -GZV(GZIZ, 138)                               
 549    CONTINUE                                                        
        V(41) = -V(138)                                                 
C--------                                                               
        D2ZV = 1161.11111111111D0 * V(43)                               
        D5ZV = V(46) - X(1)                                             
        D6ZB = D5ZV * 1161.11111111111D0                                
        DO 550 GZIZ = 1, GZPZ                                           
          GZD1ZW(GZIZ) = -D2ZV * GZX(GZIZ, 1) + D2ZV * GZV(GZIZ, 46) + D
     *6ZB * GZV(GZIZ, 43)                                               
 550    CONTINUE                                                        
        D1ZW = D2ZV * D5ZV                                              
        D3ZV = 1161.11111111111D0 * V(48)                               
        D6ZV = X(2) - X(1)                                              
        D10ZB = D6ZV * 1161.11111111111D0                               
        DO 551 GZIZ = 1, GZPZ                                           
          GZD2ZW(GZIZ) = GZV(GZIZ, 40) + (-D3ZV) * GZX(GZIZ, 1) + D3ZV *
     * GZX(GZIZ, 2) + D10ZB * GZV(GZIZ, 48) + GZD1ZW(GZIZ)              
 551    CONTINUE                                                        
        D2ZW = D1ZW + D3ZV * D6ZV + V(40)                               
        DO 552 GZIZ = 1, GZPZ                                           
          GZDF(GZIZ, 1) = -9.41985645933014D-06 * GZV(GZIZ, 138) + 9.419
     *85645933014D-06 * GZD2ZW(GZIZ)                                    
 552    CONTINUE                                                        
        DF(1) = 9.41985645933014D-06 * (D2ZW - V(138))                  
C--------                                                               
        DO 553 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 53) = -GZV(GZIZ, 16)                                
 553    CONTINUE                                                        
        V(53) = -V(16)                                                  
C--------                                                               
        DO 554 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 61) = 3600.0D0 * GZV(GZIZ, 16)                      
 554    CONTINUE                                                        
        V(61) = 3600.0D0 * V(16)                                        
C--------                                                               
        DO 555 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 65) = 3600.0D0 * GZV(GZIZ, 8)                       
 555    CONTINUE                                                        
        V(65) = 3600.0D0 * V(8)                                         
C--------                                                               
        DO 556 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 167) = -GZV(GZIZ, 166)                              
 556    CONTINUE                                                        
        V(167) = -V(166)                                                
C--------                                                               
        DO 557 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 170) = -39.6190476190476D0 * GZX(GZIZ, 3) + 39.61904
     *76190476D0 * GZX(GZIZ, 2)                                         
 557    CONTINUE                                                        
        V(170) = 39.6190476190476D0 * (X(2) - X(3))                     
C--------                                                               
        DO 558 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 58) = -GZV(GZIZ, 170) + GZV(GZIZ, 166)              
 558    CONTINUE                                                        
        V(58) = V(166) - V(170)                                         
C--------                                                               
        DO 559 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 142) = 0.0D0                                        
 559    CONTINUE                                                        
        V(142) = 4.0D0 + 0.5D0 * DBLE(TA)                               
C--------                                                               
        DO 560 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 141) = -5.64929775099216D0 * GZV(GZIZ, 142) + 5.6492
     *9775099216D0 * GZX(GZIZ, 2)                                       
 560    CONTINUE                                                        
        V(141) = 5.64929775099216D0 * (X(2) - V(142))                   
C--------                                                               
        DO 561 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 59) = -GZV(GZIZ, 141)                               
 561    CONTINUE                                                        
        V(59) = -V(141)                                                 
C--------                                                               
        D2ZV = 1161.11111111111D0 * V(61)                               
        D5ZV = X(1) - X(2)                                              
        D6ZB = D5ZV * 1161.11111111111D0                                
        DO 562 GZIZ = 1, GZPZ                                           
          GZD1ZW(GZIZ) = -D2ZV * GZX(GZIZ, 2) + D2ZV * GZX(GZIZ, 1) + D6
     *ZB * GZV(GZIZ, 61)                                                
 562    CONTINUE                                                        
        D1ZW = D2ZV * D5ZV                                              
        D3ZV = 1161.11111111111D0 * V(65)                               
        D6ZV = X(3) - X(2)                                              
        D10ZB = D6ZV * 1161.11111111111D0                               
        DO 563 GZIZ = 1, GZPZ                                           
          GZD2ZW(GZIZ) = GZV(GZIZ, 58) + (-D3ZV) * GZX(GZIZ, 2) + D3ZV *
     * GZX(GZIZ, 3) + D10ZB * GZV(GZIZ, 65) + GZD1ZW(GZIZ)              
 563    CONTINUE                                                        
        D2ZW = D1ZW + D3ZV * D6ZV + V(58)                               
        DO 564 GZIZ = 1, GZPZ                                           
          GZDF(GZIZ, 2) = -6.27990430622009D-06 * GZV(GZIZ, 141) + 6.279
     *90430622009D-06 * GZD2ZW(GZIZ)                                    
 564    CONTINUE                                                        
        DF(2) = 6.27990430622009D-06 * (D2ZW - V(141))                  
C--------                                                               
        DO 565 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 70) = -GZV(GZIZ, 16)                                
 565    CONTINUE                                                        
        V(70) = -V(16)                                                  
C--------                                                               
        DO 566 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 78) = 3600.0D0 * GZV(GZIZ, 16)                      
 566    CONTINUE                                                        
        V(78) = 3600.0D0 * V(16)                                        
C--------                                                               
        DO 567 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 82) = 3600.0D0 * GZV(GZIZ, 8)                       
 567    CONTINUE                                                        
        V(82) = 3600.0D0 * V(8)                                         
C--------                                                               
        DO 568 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 172) = -GZV(GZIZ, 170)                              
 568    CONTINUE                                                        
        V(172) = -V(170)                                                
C--------                                                               
        DO 569 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 175) = -39.6190476190476D0 * GZX(GZIZ, 4) + 39.61904
     *76190476D0 * GZX(GZIZ, 3)                                         
 569    CONTINUE                                                        
        V(175) = 39.6190476190476D0 * (X(3) - X(4))                     
C--------                                                               
        DO 570 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 75) = -GZV(GZIZ, 175) + GZV(GZIZ, 170)              
 570    CONTINUE                                                        
        V(75) = V(170) - V(175)                                         
C--------                                                               
        DO 571 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 145) = 5.64929775099216D0 * GZX(GZIZ, 3)            
 571    CONTINUE                                                        
        V(145) = 5.64929775099216D0 * X(3) - 45.1943820079373D0         
C--------                                                               
        DO 572 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 76) = -GZV(GZIZ, 145)                               
 572    CONTINUE                                                        
        V(76) = -V(145)                                                 
C--------                                                               
        D2ZV = 1161.11111111111D0 * V(78)                               
        D5ZV = X(2) - X(3)                                              
        D6ZB = D5ZV * 1161.11111111111D0                                
        DO 573 GZIZ = 1, GZPZ                                           
          GZD1ZW(GZIZ) = -D2ZV * GZX(GZIZ, 3) + D2ZV * GZX(GZIZ, 2) + D6
     *ZB * GZV(GZIZ, 78)                                                
 573    CONTINUE                                                        
        D1ZW = D2ZV * D5ZV                                              
        D3ZV = 1161.11111111111D0 * V(82)                               
        D6ZV = X(4) - X(3)                                              
        D10ZB = D6ZV * 1161.11111111111D0                               
        DO 574 GZIZ = 1, GZPZ                                           
          GZD2ZW(GZIZ) = GZV(GZIZ, 75) + (-D3ZV) * GZX(GZIZ, 3) + D3ZV *
     * GZX(GZIZ, 4) + D10ZB * GZV(GZIZ, 82) + GZD1ZW(GZIZ)              
 574    CONTINUE                                                        
        D2ZW = D1ZW + D3ZV * D6ZV + V(75)                               
        DO 575 GZIZ = 1, GZPZ                                           
          GZDF(GZIZ, 3) = -6.27990430622009D-06 * GZV(GZIZ, 145) + 6.279
     *90430622009D-06 * GZD2ZW(GZIZ)                                    
 575    CONTINUE                                                        
        DF(3) = 6.27990430622009D-06 * (D2ZW - V(145))                  
C--------                                                               
        DO 576 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 87) = -GZV(GZIZ, 16)                                
 576    CONTINUE                                                        
        V(87) = -V(16)                                                  
C--------                                                               
        DO 577 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 95) = 3600.0D0 * GZV(GZIZ, 16)                      
 577    CONTINUE                                                        
        V(95) = 3600.0D0 * V(16)                                        
C--------                                                               
        DO 578 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 99) = 3600.0D0 * GZV(GZIZ, 8)                       
 578    CONTINUE                                                        
        V(99) = 3600.0D0 * V(8)                                         
C--------                                                               
        DO 579 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 177) = -GZV(GZIZ, 175)                              
 579    CONTINUE                                                        
        V(177) = -V(175)                                                
C--------                                                               
        DO 580 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 180) = -47.5428571428572D0 * GZX(GZIZ, 5) + 47.54285
     *71428572D0 * GZX(GZIZ, 4)                                         
 580    CONTINUE                                                        
        V(180) = 47.5428571428572D0 * (X(4) - X(5))                     
C--------                                                               
        DO 581 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 92) = -GZV(GZIZ, 180) + GZV(GZIZ, 175)              
 581    CONTINUE                                                        
        V(92) = V(175) - V(180)                                         
C--------                                                               
        DO 582 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 148) = 5.64929775099216D0 * GZX(GZIZ, 4)            
 582    CONTINUE                                                        
        V(148) = 5.64929775099216D0 * X(4) - 45.1943820079373D0         
C--------                                                               
        DO 583 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 93) = -GZV(GZIZ, 148)                               
 583    CONTINUE                                                        
        V(93) = -V(148)                                                 
C--------                                                               
        D2ZV = 1161.11111111111D0 * V(95)                               
        D5ZV = X(3) - X(4)                                              
        D6ZB = D5ZV * 1161.11111111111D0                                
        DO 584 GZIZ = 1, GZPZ                                           
          GZD1ZW(GZIZ) = -D2ZV * GZX(GZIZ, 4) + D2ZV * GZX(GZIZ, 3) + D6
     *ZB * GZV(GZIZ, 95)                                                
 584    CONTINUE                                                        
        D1ZW = D2ZV * D5ZV                                              
        D3ZV = 1161.11111111111D0 * V(99)                               
        D6ZV = X(5) - X(4)                                              
        D10ZB = D6ZV * 1161.11111111111D0                               
        DO 585 GZIZ = 1, GZPZ                                           
          GZD2ZW(GZIZ) = GZV(GZIZ, 92) + (-D3ZV) * GZX(GZIZ, 4) + D3ZV *
     * GZX(GZIZ, 5) + D10ZB * GZV(GZIZ, 99) + GZD1ZW(GZIZ)              
 585    CONTINUE                                                        
        D2ZW = D1ZW + D3ZV * D6ZV + V(92)                               
        DO 586 GZIZ = 1, GZPZ                                           
          GZDF(GZIZ, 4) = -6.27990430622009D-06 * GZV(GZIZ, 148) + 6.279
     *90430622009D-06 * GZD2ZW(GZIZ)                                    
 586    CONTINUE                                                        
        DF(4) = 6.27990430622009D-06 * (D2ZW - V(148))                  
C--------                                                               
        DO 587 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 104) = -GZV(GZIZ, 16)                               
 587    CONTINUE                                                        
        V(104) = -V(16)                                                 
C--------                                                               
        DO 588 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 112) = 3600.0D0 * GZV(GZIZ, 16)                     
 588    CONTINUE                                                        
        V(112) = 3600.0D0 * V(16)                                       
C--------                                                               
        DO 589 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 116) = 3600.0D0 * GZV(GZIZ, 8)                      
 589    CONTINUE                                                        
        V(116) = 3600.0D0 * V(8)                                        
C--------                                                               
        DO 590 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 182) = -GZV(GZIZ, 180)                              
 590    CONTINUE                                                        
        V(182) = -V(180)                                                
C--------                                                               
        DO 591 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 185) = -79.2380952380953D0 * GZX(GZIZ, 6) + 79.23809
     *52380953D0 * GZX(GZIZ, 5)                                         
 591    CONTINUE                                                        
        V(185) = 79.2380952380953D0 * (X(5) - X(6))                     
C--------                                                               
        DO 592 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 109) = -GZV(GZIZ, 185) + GZV(GZIZ, 180)             
 592    CONTINUE                                                        
        V(109) = V(180) - V(185)                                        
C--------                                                               
        DO 593 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 151) = 3.76619850066144D0 * GZX(GZIZ, 5)            
 593    CONTINUE                                                        
        V(151) = 3.76619850066144D0 * X(5) - 30.1295880052915D0         
C--------                                                               
        DO 594 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 110) = -GZV(GZIZ, 151)                              
 594    CONTINUE                                                        
        V(110) = -V(151)                                                
C--------                                                               
        D2ZV = 1161.11111111111D0 * V(112)                              
        D5ZV = X(4) - X(5)                                              
        D6ZB = D5ZV * 1161.11111111111D0                                
        DO 595 GZIZ = 1, GZPZ                                           
          GZD1ZW(GZIZ) = -D2ZV * GZX(GZIZ, 5) + D2ZV * GZX(GZIZ, 4) + D6
     *ZB * GZV(GZIZ, 112)                                               
 595    CONTINUE                                                        
        D1ZW = D2ZV * D5ZV                                              
        D3ZV = 1161.11111111111D0 * V(116)                              
        D6ZV = X(6) - X(5)                                              
        D10ZB = D6ZV * 1161.11111111111D0                               
        DO 596 GZIZ = 1, GZPZ                                           
          GZD2ZW(GZIZ) = GZV(GZIZ, 109) + (-D3ZV) * GZX(GZIZ, 5) + D3ZV 
     ** GZX(GZIZ, 6) + D10ZB * GZV(GZIZ, 116) + GZD1ZW(GZIZ)            
 596    CONTINUE                                                        
        D2ZW = D1ZW + D3ZV * D6ZV + V(109)                              
        DO 597 GZIZ = 1, GZPZ                                           
          GZDF(GZIZ, 5) = -9.41985645933014D-06 * GZV(GZIZ, 151) + 9.419
     *85645933014D-06 * GZD2ZW(GZIZ)                                    
 597    CONTINUE                                                        
        DF(5) = 9.41985645933014D-06 * (D2ZW - V(151))                  
C--------                                                               
        DO 598 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 121) = -GZV(GZIZ, 16)                               
 598    CONTINUE                                                        
        V(121) = -V(16)                                                 
C--------                                                               
        DO 599 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 129) = 3600.0D0 * GZV(GZIZ, 16)                     
 599    CONTINUE                                                        
        V(129) = 3600.0D0 * V(16)                                       
C--------                                                               
        DO 600 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 133) = 3600.0D0 * GZV(GZIZ, 8)                      
 600    CONTINUE                                                        
        V(133) = 3600.0D0 * V(8)                                        
C--------                                                               
        DO 601 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 187) = -GZV(GZIZ, 185)                              
 601    CONTINUE                                                        
        V(187) = -V(185)                                                
C--------                                                               
        DO 602 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 161) = 73.1428571428572D0 * GZX(GZIZ, 6)            
 602    CONTINUE                                                        
        V(161) = 73.1428571428572D0 * X(6) - 585.142857142857D0         
C--------                                                               
        DO 603 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 126) = -GZV(GZIZ, 161) + GZV(GZIZ, 185)             
 603    CONTINUE                                                        
        V(126) = V(185) - V(161)                                        
C--------                                                               
        DO 604 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 154) = 1.88309925033072D0 * GZX(GZIZ, 6)            
 604    CONTINUE                                                        
        V(154) = 1.88309925033072D0 * X(6) - 15.0647940026458D0         
C--------                                                               
        DO 605 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 127) = -GZV(GZIZ, 154)                              
 605    CONTINUE                                                        
        V(127) = -V(154)                                                
C--------                                                               
        D2ZV = 1161.11111111111D0 * V(129)                              
        D5ZV = X(5) - X(6)                                              
        D8ZV = 1161.11111111111D0 * V(133)                              
        D9ZV = DBLE(TRSH) - X(6)                                        
        D7ZB = D9ZV * 1161.11111111111D0                                
        D6ZB = -D8ZV + (-D2ZV)                                          
        D11ZB = D5ZV * 1161.11111111111D0                               
        DO 606 GZIZ = 1, GZPZ                                           
          GZD1ZW(GZIZ) = D7ZB * GZV(GZIZ, 133) + D6ZB * GZX(GZIZ, 6) + D
     *2ZV * GZX(GZIZ, 5) + D11ZB * GZV(GZIZ, 129)                       
 606    CONTINUE                                                        
        D1ZW = D2ZV * D5ZV + D8ZV * D9ZV                                
        DO 607 GZIZ = 1, GZPZ                                           
          GZDF(GZIZ, 6) = -1.88397129186603D-05 * GZV(GZIZ, 154) + 1.883
     *97129186603D-05 * GZV(GZIZ, 126) + 1.88397129186603D-05 * GZD1ZW(G
     *ZIZ)                                                              
 607    CONTINUE                                                        
        DF(6) = 1.88397129186603D-05 * (D1ZW + V(126) - V(154))         
C--------                                                               
        D3ZV = 4180.0D0 * (X(1) - DBLE(TRSH))                           
        D5ZB = V(8) * 4180.0D0                                          
        DO 608 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 196) = D3ZV * GZV(GZIZ, 8) + D5ZB * GZX(GZIZ, 1)    
 608    CONTINUE                                                        
        V(196) = D3ZV * V(8)                                            
C--------                                                               
        DO 609 GZIZ = 1, GZPZ                                           
          GZDF(GZIZ, 8) = 0.001D0 * GZV(GZIZ, 196)                      
 609    CONTINUE                                                        
        DF(8) = 0.001D0 * V(196)                                        
C--------                                                               
        D4ZV = 4180.0D0 * (V(46) - X(6))                                
        D4ZB = V(16) * 4180.0D0                                         
        DO 610 GZIZ = 1, GZPZ                                           
          GZV(GZIZ, 197) = D4ZV * GZV(GZIZ, 16) + (-D4ZB) * GZX(GZIZ, 6)
     * + D4ZB * GZV(GZIZ, 46)                                           
 610    CONTINUE                                                        
        V(197) = D4ZV * V(16)                                           
C--------                                                               
        DO 611 GZIZ = 1, GZPZ                                           
          GZDF(GZIZ, 9) = 0.001D0 * GZV(GZIZ, 197)                      
 611    CONTINUE                                                        
        DF(9) = 0.001D0 * V(197)                                        
C--------                                                               
C END OF SORTED AND SOLVED EQUATIONS                                    
C                                                                       
        DO 612 GZIZ = 1, GZPZ                                           
          GZW(GZIZ, 1) = GZV(GZIZ, 10)                                  
 612    CONTINUE                                                        
        W(1) = V(10)                                                    
C--------                                                               
C                                                                       
        RETURN                                                          
      END                                                               
