      SUBROUTINE RANGE( IELEMN, TRANSP, W1, W2, nelvar, ninvar,
     *                  itype, LW1, LW2 )
      INTEGER IELEMN, nelvar, ninvar, itype, LW1, LW2
      LOGICAL TRANSP
      DOUBLE PRECISION W1( LW1 ), W2( LW2 )
C
C  Problem name : STEENBRA  
C
C  -- produced by SIFdecode 2.4
C
C  TRANSP = .FALSE. <=> W2 = U * W1
C  TRANSP = .TRUE.  <=> W2 = U(transpose) * W1
C
      INTEGER I
C
C  Element type : QUAD      
C
    1 CONTINUE
      IF ( TRANSP ) THEN
         W2(     1 ) =   W1(     1 ) 
         W2(     2 ) =   W1(     1 ) 
         W2(     3 ) =   W1(     1 ) 
         W2(     4 ) =   W1(     1 ) 
         W2(     5 ) =   W1(     1 ) 
         W2(     6 ) =   W1(     1 ) 
         W2(     7 ) =   W1(     1 ) 
         W2(     8 ) =   W1(     1 ) 
         W2(     9 ) =   W1(     1 ) 
         W2(    10 ) =   W1(     1 ) 
         W2(    11 ) =   W1(     1 ) 
         W2(    12 ) =   W1(     1 ) 
      ELSE
         W2(     1 ) =   W1(     1 ) 
     *                 + W1(     2 ) 
     *                 + W1(     3 ) 
     *                 + W1(     4 ) 
     *                 + W1(     5 ) 
     *                 + W1(     6 ) 
     *                 + W1(     7 ) 
     *                 + W1(     8 ) 
     *                 + W1(     9 ) 
     *                 + W1(    10 ) 
     *                 + W1(    11 ) 
     *                 + W1(    12 ) 
      END IF
      RETURN
      END
