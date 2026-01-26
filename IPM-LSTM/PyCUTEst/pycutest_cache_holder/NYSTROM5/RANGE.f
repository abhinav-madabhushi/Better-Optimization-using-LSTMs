      SUBROUTINE RANGE( IELEMN, TRANSP, W1, W2, nelvar, ninvar,
     *                  itype, LW1, LW2 )
      INTEGER IELEMN, nelvar, ninvar, itype, LW1, LW2
      LOGICAL TRANSP
      DOUBLE PRECISION W1( LW1 ), W2( LW2 )
C
C  Problem name : NYSTROM5  
C
C  -- produced by SIFdecode 2.6.2
C
C  TRANSP = .FALSE. <=> W2 = U * W1
C  TRANSP = .TRUE.  <=> W2 = U(transpose) * W1
C
      INTEGER I
      GO TO (99998,    2,    3,99998,99998,    6,    7,99998,
     *       99998,99998,   11,   12,99998,   14,   15
     *                                                        ), ITYPE
C
C  Element type : 2PRI2     
C
    2 CONTINUE
      IF ( TRANSP ) THEN
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     2 ) 
         W2(     3 ) =   W1(     2 ) 
      ELSE
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     2 ) 
     *                 + W1(     3 ) 
      END IF
      RETURN
C
C  Element type : 2PRI3     
C
    3 CONTINUE
      IF ( TRANSP ) THEN
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     2 ) 
         W2(     3 ) =   W1(     2 ) 
         W2(     4 ) =   W1(     2 ) 
      ELSE
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     2 ) 
     *                 + W1(     3 ) 
     *                 + W1(     4 ) 
      END IF
      RETURN
C
C  Element type : 3PRI2     
C
   11 CONTINUE
      IF ( TRANSP ) THEN
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     2 ) 
         W2(     3 ) =   W1(     3 ) 
         W2(     4 ) =   W1(     3 ) 
      ELSE
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     2 ) 
         W2(     3 ) =   W1(     3 ) 
     *                 + W1(     4 ) 
      END IF
      RETURN
C
C  Element type : 3PRI3     
C
   12 CONTINUE
      IF ( TRANSP ) THEN
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     2 ) 
         W2(     3 ) =   W1(     3 ) 
         W2(     4 ) =   W1(     3 ) 
         W2(     5 ) =   W1(     3 ) 
      ELSE
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     2 ) 
         W2(     3 ) =   W1(     3 ) 
     *                 + W1(     4 ) 
     *                 + W1(     5 ) 
      END IF
      RETURN
C
C  Element type : XY2I2     
C
    6 CONTINUE
      IF ( TRANSP ) THEN
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     2 ) 
         W2(     3 ) =   W1(     2 ) 
      ELSE
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     2 ) 
     *                 + W1(     3 ) 
      END IF
      RETURN
C
C  Element type : XY2I3     
C
    7 CONTINUE
      IF ( TRANSP ) THEN
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     2 ) 
         W2(     3 ) =   W1(     2 ) 
         W2(     4 ) =   W1(     2 ) 
      ELSE
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     2 ) 
     *                 + W1(     3 ) 
     *                 + W1(     4 ) 
      END IF
      RETURN
C
C  Element type : XY2ZI2    
C
   14 CONTINUE
      IF ( TRANSP ) THEN
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     2 ) 
         W2(     3 ) =   W1(     3 ) 
         W2(     4 ) =   W1(     3 ) 
      ELSE
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     2 ) 
         W2(     3 ) =   W1(     3 ) 
     *                 + W1(     4 ) 
      END IF
      RETURN
C
C  Element type : XY2ZI3    
C
   15 CONTINUE
      IF ( TRANSP ) THEN
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     2 ) 
         W2(     3 ) =   W1(     3 ) 
         W2(     4 ) =   W1(     3 ) 
         W2(     5 ) =   W1(     3 ) 
      ELSE
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     2 ) 
         W2(     3 ) =   W1(     3 ) 
     *                 + W1(     4 ) 
     *                 + W1(     5 ) 
      END IF
      RETURN
C
C  Elements without internal variables.
C
99998 CONTINUE
      DO 99999 i = 1, nelvar
         W2( i ) = W1( i )
99999 CONTINUE
      RETURN
      END
