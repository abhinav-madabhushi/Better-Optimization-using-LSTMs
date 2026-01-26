      SUBROUTINE GROUP ( GVALUE, LGVALU, FVALUE, GPVALU, NCALCG, 
     *                   ITYPEG, ISTGPA, ICALCG, LTYPEG, LSTGPA, 
     *                   LCALCG, LFVALU, LGPVLU, DERIVS, IGSTAT )
      INTEGER LGVALU, NCALCG, LTYPEG, LSTGPA
      INTEGER LCALCG, LFVALU, LGPVLU, IGSTAT
      LOGICAL DERIVS
      INTEGER ITYPEG(LTYPEG), ISTGPA(LSTGPA), ICALCG(LCALCG)
      DOUBLE PRECISION GVALUE(LGVALU,3), FVALUE(LFVALU), GPVALU(LGPVLU)
C
C  Problem name : EG1       
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IGRTYP, IGROUP, IPSTRT, JCALCG
      DOUBLE PRECISION ALPHA , ALPHA2, TWO   
      IGSTAT = 0
      DO     3 JCALCG = 1, NCALCG
       IGROUP = ICALCG(JCALCG)
       IGRTYP = ITYPEG(IGROUP)
       IF ( IGRTYP == 0 ) GO TO     3
       IPSTRT = ISTGPA(IGROUP) - 1
       GO TO (    1,    2
     *                                                        ), IGRTYP
C
C  Group type : GTYPE1  
C
    1  CONTINUE 
       ALPHA = FVALUE(IGROUP)
       TWO   = 2.0                                      
       IF ( .NOT. DERIVS ) THEN
        GVALUE(IGROUP,1)= ALPHA * ALPHA                            
       ELSE
        GVALUE(IGROUP,2)= TWO * ALPHA                              
        GVALUE(IGROUP,3)= TWO                                      
       END IF
       GO TO     3
C
C  Group type : GTYPE2  
C
    2  CONTINUE 
       ALPHA = FVALUE(IGROUP)
       ALPHA2= ALPHA * ALPHA                            
       IF ( .NOT. DERIVS ) THEN
        GVALUE(IGROUP,1)= ALPHA2 * ALPHA2                          
       ELSE
        GVALUE(IGROUP,2)= 4.0 * ALPHA2 * ALPHA                     
        GVALUE(IGROUP,3)= 12.0 * ALPHA2                            
       END IF
    3 CONTINUE
      RETURN
      END
