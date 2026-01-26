      SUBROUTINE GROUP ( GVALUE, LGVALU, FVALUE, GPVALU, NCALCG, 
     *                   ITYPEG, ISTGPA, ICALCG, LTYPEG, LSTGPA, 
     *                   LCALCG, LFVALU, LGPVLU, DERIVS, IGSTAT )
      INTEGER LGVALU, NCALCG, LTYPEG, LSTGPA
      INTEGER LCALCG, LFVALU, LGPVLU, IGSTAT
      LOGICAL DERIVS
      INTEGER ITYPEG(LTYPEG), ISTGPA(LSTGPA), ICALCG(LCALCG)
      DOUBLE PRECISION GVALUE(LGVALU,3), FVALUE(LFVALU), GPVALU(LGPVLU)
C
C  Problem name : TOINTGOR  
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IGRTYP, IGROUP, IPSTRT, JCALCG
      DOUBLE PRECISION T     , AT    , AT1   , LAT   , AG    
      DOUBLE PRECISION TPOS  , TNEG  , AA    , ONE   , ZERO  
      INTRINSIC ABS   , LOG   , SIGN  , MAX   
      IGSTAT = 0
      ONE    = 1.0D0                                    
      ZERO   = 0.0D0                                    
      DO     3 JCALCG = 1, NCALCG
       IGROUP = ICALCG(JCALCG)
       IGRTYP = ITYPEG(IGROUP)
       IF ( IGRTYP == 0 ) GO TO     3
       IPSTRT = ISTGPA(IGROUP) - 1
       GO TO (    1,    2
     *                                                        ), IGRTYP
C
C  Group type : ACT     
C
    1  CONTINUE 
       T     = FVALUE(IGROUP)
       AT    = ABS( T )                                 
       AT1   = AT + ONE                                 
       LAT   = LOG( AT1 )                               
       AA    = AT / AT1                                 
       AG    = AA + LAT                                 
       IF ( .NOT. DERIVS ) THEN
        GVALUE(IGROUP,1)= AT * LAT                                 
       ELSE
        GVALUE(IGROUP,2)= SIGN( AG, T )                            
        GVALUE(IGROUP,3)= ( 2.0 - AA ) / AT1                       
       END IF
       GO TO     3
C
C  Group type : BBT     
C
    2  CONTINUE 
       T     = FVALUE(IGROUP)
       AT    = ABS( T )                                 
       AT1   = AT + ONE                                 
       LAT   = LOG( AT1 )                               
       AA    = AT / AT1                                 
       TPOS  = MAX( SIGN( ONE, T ), ZERO )              
       TNEG  = ONE - TPOS                               
       IF ( .NOT. DERIVS ) THEN
        GVALUE(IGROUP,1)= T * T * ( TNEG + TPOS * LAT )            
       ELSE
        GVALUE(IGROUP,2)= TNEG * 2.0 * T                           
     *                         + TPOS * T * ( AA + 2.0 * LAT )          
        GVALUE(IGROUP,3)= TNEG * 2.0                               
     *                         + TPOS * ( AA * ( 4.0 - AA ) + 2.0*LAT ) 
       END IF
    3 CONTINUE
      RETURN
      END
