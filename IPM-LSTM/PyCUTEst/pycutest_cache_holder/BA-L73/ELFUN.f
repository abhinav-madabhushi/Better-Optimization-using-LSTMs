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
C  Problem name : BA-L73    
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      INTEGER BA    , IBA   
      DOUBLE PRECISION RX    , RY    , RZ    , X     , Y     
      DOUBLE PRECISION Z     , TX    , TY    , TZ    , KA    
      DOUBLE PRECISION KB    , F     , YRES  , O     , G(12) 
      DOUBLE PRECISION H(78) , R     , S(12) , T(78) 
      EXTERNAL BA    
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : E         
C
       RX     = XVALUE(IELVAR(ILSTRT+     1))
       RY     = XVALUE(IELVAR(ILSTRT+     2))
       RZ     = XVALUE(IELVAR(ILSTRT+     3))
       X      = XVALUE(IELVAR(ILSTRT+     4))
       Y      = XVALUE(IELVAR(ILSTRT+     5))
       Z      = XVALUE(IELVAR(ILSTRT+     6))
       TX     = XVALUE(IELVAR(ILSTRT+     7))
       TY     = XVALUE(IELVAR(ILSTRT+     8))
       TZ     = XVALUE(IELVAR(ILSTRT+     9))
       KA     = XVALUE(IELVAR(ILSTRT+    10))
       KB     = XVALUE(IELVAR(ILSTRT+    11))
       F      = XVALUE(IELVAR(ILSTRT+    12))
       YRES   = EPVALU(IPSTRT+     1)
       IBA    = BA( RX, RY, RZ, X, Y, Z, TX,             
     *              TY, TZ, KA, KB, F, O, G, H,          
     *              R, S, T, YRES > 0.0 )                
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= O                                        
       ELSE
        FUVALS(IGSTRT+     1)= G(1)                                     
        FUVALS(IGSTRT+     2)= G(2)                                     
        FUVALS(IGSTRT+     3)= G(3)                                     
        FUVALS(IGSTRT+     4)= G(4)                                     
        FUVALS(IGSTRT+     5)= G(5)                                     
        FUVALS(IGSTRT+     6)= G(6)                                     
        FUVALS(IGSTRT+     7)= G(7)                                     
        FUVALS(IGSTRT+     8)= G(8)                                     
        FUVALS(IGSTRT+     9)= G(9)                                     
        FUVALS(IGSTRT+    10)= G(10)                                    
        FUVALS(IGSTRT+    11)= G(11)                                    
        FUVALS(IGSTRT+    12)= G(12)                                    
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=H(1)                                     
         FUVALS(IHSTRT+     2)=H(2)                                     
         FUVALS(IHSTRT+     3)=H(3)                                     
         FUVALS(IHSTRT+     4)=H(4)                                     
         FUVALS(IHSTRT+     5)=H(5)                                     
         FUVALS(IHSTRT+     6)=H(6)                                     
         FUVALS(IHSTRT+     7)=H(7)                                     
         FUVALS(IHSTRT+     8)=H(8)                                     
         FUVALS(IHSTRT+     9)=H(9)                                     
         FUVALS(IHSTRT+    10)=H(10)                                    
         FUVALS(IHSTRT+    11)=H(11)                                    
         FUVALS(IHSTRT+    12)=H(12)                                    
         FUVALS(IHSTRT+    13)=H(13)                                    
         FUVALS(IHSTRT+    14)=H(14)                                    
         FUVALS(IHSTRT+    15)=H(15)                                    
         FUVALS(IHSTRT+    16)=H(16)                                    
         FUVALS(IHSTRT+    17)=H(17)                                    
         FUVALS(IHSTRT+    18)=H(18)                                    
         FUVALS(IHSTRT+    19)=H(19)                                    
         FUVALS(IHSTRT+    20)=H(20)                                    
         FUVALS(IHSTRT+    21)=H(21)                                    
         FUVALS(IHSTRT+    22)=H(22)                                    
         FUVALS(IHSTRT+    23)=H(23)                                    
         FUVALS(IHSTRT+    24)=H(24)                                    
         FUVALS(IHSTRT+    25)=H(25)                                    
         FUVALS(IHSTRT+    26)=H(26)                                    
         FUVALS(IHSTRT+    27)=H(27)                                    
         FUVALS(IHSTRT+    28)=H(28)                                    
         FUVALS(IHSTRT+    29)=H(29)                                    
         FUVALS(IHSTRT+    30)=H(30)                                    
         FUVALS(IHSTRT+    31)=H(31)                                    
         FUVALS(IHSTRT+    32)=H(32)                                    
         FUVALS(IHSTRT+    33)=H(33)                                    
         FUVALS(IHSTRT+    34)=H(34)                                    
         FUVALS(IHSTRT+    35)=H(35)                                    
         FUVALS(IHSTRT+    36)=H(36)                                    
         FUVALS(IHSTRT+    37)=H(37)                                    
         FUVALS(IHSTRT+    38)=H(38)                                    
         FUVALS(IHSTRT+    39)=H(39)                                    
         FUVALS(IHSTRT+    40)=H(40)                                    
         FUVALS(IHSTRT+    41)=H(41)                                    
         FUVALS(IHSTRT+    42)=H(42)                                    
         FUVALS(IHSTRT+    43)=H(43)                                    
         FUVALS(IHSTRT+    44)=H(44)                                    
         FUVALS(IHSTRT+    45)=H(45)                                    
         FUVALS(IHSTRT+    46)=H(46)                                    
         FUVALS(IHSTRT+    47)=H(47)                                    
         FUVALS(IHSTRT+    48)=H(48)                                    
         FUVALS(IHSTRT+    49)=H(49)                                    
         FUVALS(IHSTRT+    50)=H(50)                                    
         FUVALS(IHSTRT+    51)=H(51)                                    
         FUVALS(IHSTRT+    52)=H(52)                                    
         FUVALS(IHSTRT+    53)=H(53)                                    
         FUVALS(IHSTRT+    54)=H(54)                                    
         FUVALS(IHSTRT+    55)=H(55)                                    
         FUVALS(IHSTRT+    56)=H(56)                                    
         FUVALS(IHSTRT+    57)=H(57)                                    
         FUVALS(IHSTRT+    58)=H(58)                                    
         FUVALS(IHSTRT+    59)=H(59)                                    
         FUVALS(IHSTRT+    60)=H(60)                                    
         FUVALS(IHSTRT+    61)=H(61)                                    
         FUVALS(IHSTRT+    62)=H(62)                                    
         FUVALS(IHSTRT+    63)=H(63)                                    
         FUVALS(IHSTRT+    64)=H(64)                                    
         FUVALS(IHSTRT+    65)=H(65)                                    
         FUVALS(IHSTRT+    66)=H(66)                                    
         FUVALS(IHSTRT+    67)=H(67)                                    
         FUVALS(IHSTRT+    68)=H(68)                                    
         FUVALS(IHSTRT+    69)=H(69)                                    
         FUVALS(IHSTRT+    70)=H(70)                                    
         FUVALS(IHSTRT+    71)=H(71)                                    
         FUVALS(IHSTRT+    72)=H(72)                                    
         FUVALS(IHSTRT+    73)=H(73)                                    
         FUVALS(IHSTRT+    74)=H(74)                                    
         FUVALS(IHSTRT+    75)=H(75)                                    
         FUVALS(IHSTRT+    76)=H(76)                                    
         FUVALS(IHSTRT+    77)=H(77)                                    
         FUVALS(IHSTRT+    78)=H(78)                                    
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
