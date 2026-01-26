      SUBROUTINE RANGE( IELEMN, TRANSP, W1, W2, nelvar, ninvar,
     *                  itype, LW1, LW2 )
      INTEGER IELEMN, nelvar, ninvar, itype, LW1, LW2
      LOGICAL TRANSP
      DOUBLE PRECISION W1( LW1 ), W2( LW2 )
C
C  Problem name : BT13      
C
C  -- produced by SIFdecode 2.6.2
C
C  TRANSP = .FALSE. <=> W2 = U * W1
C  TRANSP = .TRUE.  <=> W2 = U(transpose) * W1
C
      INTEGER I
      GO TO (99998,    2,    3,    4
     *                                                        ), ITYPE
C
C  Element type : ISQ2      
C
    2 CONTINUE
      IF ( TRANSP ) THEN
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) = - W1(     1 ) *      2.00000
      ELSE
         W2(     1 ) =   W1(     1 ) 
     *                 - W1(     2 ) *      2.00000
      END IF
      RETURN
C
C  Element type : ISQ3      
C
    3 CONTINUE
      IF ( TRANSP ) THEN
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) = - W1(     1 ) *      3.00000
      ELSE
         W2(     1 ) =   W1(     1 ) 
     *                 - W1(     2 ) *      3.00000
      END IF
      RETURN
C
C  Element type : ISQ4      
C
    4 CONTINUE
      IF ( TRANSP ) THEN
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) = - W1(     1 ) *      4.00000
      ELSE
         W2(     1 ) =   W1(     1 ) 
     *                 - W1(     2 ) *      4.00000
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
