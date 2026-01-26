      SUBROUTINE RANGE( IELEMN, TRANSP, W1, W2, nelvar, ninvar,
     *                  itype, LW1, LW2 )
      INTEGER IELEMN, nelvar, ninvar, itype, LW1, LW2
      LOGICAL TRANSP
      DOUBLE PRECISION W1( LW1 ), W2( LW2 )
C
C  Problem name : EG3       
C
C  -- produced by SIFdecode 2.6.2
C
C  TRANSP = .FALSE. <=> W2 = U * W1
C  TRANSP = .TRUE.  <=> W2 = U(transpose) * W1
C
      INTEGER I
      GO TO (    1,99998,99998,    4
     *                                                        ), ITYPE
C
C  Element type : RPROD     
C
    1 CONTINUE
      IF ( TRANSP ) THEN
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) = - W1(     1 ) 
         W2(     3 ) =   W1(     2 ) 
      ELSE
         W2(     1 ) =   W1(     1 ) 
     *                 - W1(     2 ) 
         W2(     2 ) =   W1(     3 ) 
      END IF
      RETURN
C
C  Element type : SQR2      
C
    4 CONTINUE
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
