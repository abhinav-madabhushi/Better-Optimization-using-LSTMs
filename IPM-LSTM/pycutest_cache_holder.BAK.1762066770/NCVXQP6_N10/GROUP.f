      SUBROUTINE GROUP ( GVALUE, LGVALU, FVALUE, GPVALU, NCALCG, 
     *                   ITYPEG, ISTGPA, ICALCG, LTYPEG, LSTGPA, 
     *                   LCALCG, LFVALU, LGPVLU, DERIVS, IGSTAT )
      INTEGER LGVALU, NCALCG, LTYPEG, LSTGPA
      INTEGER LCALCG, LFVALU, LGPVLU, IGSTAT
      LOGICAL DERIVS
      INTEGER ITYPEG(LTYPEG), ISTGPA(LSTGPA), ICALCG(LCALCG)
      DOUBLE PRECISION GVALUE(LGVALU,3), FVALUE(LFVALU), GPVALU(LGPVLU)
C
C  Problem name : NCVXQP6   
C
C  -- produced by SIFdecode 2.4
C
      INTEGER IGRTYP, IGROUP, IPSTRT, JCALCG
      DOUBLE PRECISION ALPHA , P     
      IGSTAT = 0
      DO     2 JCALCG = 1, NCALCG
       IGROUP = ICALCG(JCALCG)
       IGRTYP = ITYPEG(IGROUP)
       IF ( IGRTYP == 0 ) GO TO     2
       IPSTRT = ISTGPA(IGROUP) - 1
C
C  Group type : SQR     
C
       ALPHA = FVALUE(IGROUP)
       P     = GPVALU(IPSTRT+     1)
       IF ( .NOT. DERIVS ) THEN
        GVALUE(IGROUP,1)= 0.5 * P * ALPHA * ALPHA                  
       ELSE
        GVALUE(IGROUP,2)= P * ALPHA                                
        GVALUE(IGROUP,3)= P                                        
       END IF
    2 CONTINUE
      RETURN
      END
