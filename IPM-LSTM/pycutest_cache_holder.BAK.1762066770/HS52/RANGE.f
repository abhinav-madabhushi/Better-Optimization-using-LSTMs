      SUBROUTINE RANGE( IELEMN, TRANSP, W1, W2, nelvar, ninvar,
     *                  itype, LW1, LW2 )
      INTEGER IELEMN, nelvar, ninvar, itype, LW1, LW2
      LOGICAL TRANSP
      DOUBLE PRECISION W1( LW1 ), W2( LW2 )
C
C  Problem name : HS52      
C
C  -- produced by SIFdecode 2.4
C
C  TRANSP = .FALSE. <=> W2 = U * W1
C  TRANSP = .TRUE.  <=> W2 = U(transpose) * W1
C
      INTEGER I
      GO TO (    1,    2,99998
     *                                                        ), ITYPE
C
C  Element type : SQ1M2     
C
    1 CONTINUE
      IF ( TRANSP ) THEN
         W2(     1 ) =   W1(     1 ) *      4.00000
         W2(     2 ) = - W1(     1 ) 
      ELSE
         W2(     1 ) =   W1(     1 ) *      4.00000
     *                 - W1(     2 ) 
      END IF
      RETURN
C
C  Element type : SQ1P2     
C
    2 CONTINUE
      IF ( TRANSP ) THEN
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     1 ) 
      ELSE
         W2(     1 ) =   W1(     1 ) 
     *                 + W1(     2 ) 
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
