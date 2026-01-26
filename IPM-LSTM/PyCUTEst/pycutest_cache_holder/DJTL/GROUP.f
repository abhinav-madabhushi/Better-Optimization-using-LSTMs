      SUBROUTINE GROUP ( GVALUE, LGVALU, FVALUE, GPVALU, NCALCG, 
     *                   ITYPEG, ISTGPA, ICALCG, LTYPEG, LSTGPA, 
     *                   LCALCG, LFVALU, LGPVLU, DERIVS, IGSTAT )
      INTEGER LGVALU, NCALCG, LTYPEG, LSTGPA
      INTEGER LCALCG, LFVALU, LGPVLU, IGSTAT
      LOGICAL DERIVS
      INTEGER ITYPEG(LTYPEG), ISTGPA(LSTGPA), ICALCG(LCALCG)
      DOUBLE PRECISION GVALUE(LGVALU,3), FVALUE(LFVALU), GPVALU(LGPVLU)
C
C  Problem name : DJTL      
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IGRTYP, IGROUP, IPSTRT, JCALCG
      DOUBLE PRECISION ALPHA , P1    , P2    , APP1  , P1P2  
      DOUBLE PRECISION BIG   , FF    , GG    , HH    
      LOGICAL ARG0  
      INTRINSIC LOG   
      IGSTAT = 0
      DO     2 JCALCG = 1, NCALCG
       IGROUP = ICALCG(JCALCG)
       IGRTYP = ITYPEG(IGROUP)
       IF ( IGRTYP == 0 ) GO TO     2
       IPSTRT = ISTGPA(IGROUP) - 1
C
C  Group type : LOG     
C
       ALPHA = FVALUE(IGROUP)
       P1    = GPVALU(IPSTRT+     1)
       P2    = GPVALU(IPSTRT+     2)
       APP1  = ALPHA + P1                               
       P1P2  = P1 * P2                                  
       ARG0  = APP1 .LE. 0.0                            
       BIG   = 1.0000D+10                               
       IF (ARG0  ) FF     = BIG * ALPHA**2                           
       IF (.NOT.ARG0  ) FF    =-P1P2 * LOG(APP1)                        
       IF (ARG0  ) GG     = 2.0 * BIG * ALPHA                        
       IF (.NOT.ARG0  ) GG    =-P1P2 / APP1                             
       IF (ARG0  ) HH     = 2.0 * BIG                                
       IF (.NOT.ARG0  ) HH    =P1P2 / APP1**2                           
       IF ( .NOT. DERIVS ) THEN
        GVALUE(IGROUP,1)= FF                                       
       ELSE
        GVALUE(IGROUP,2)= GG                                       
        GVALUE(IGROUP,3)= HH                                       
       END IF
    2 CONTINUE
      RETURN
      END
