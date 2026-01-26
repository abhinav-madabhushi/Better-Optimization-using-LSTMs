      SUBROUTINE GROUP ( GVALUE, LGVALU, FVALUE, GPVALU, NCALCG, 
     *                   ITYPEG, ISTGPA, ICALCG, LTYPEG, LSTGPA, 
     *                   LCALCG, LFVALU, LGPVLU, DERIVS, IGSTAT )
      INTEGER LGVALU, NCALCG, LTYPEG, LSTGPA
      INTEGER LCALCG, LFVALU, LGPVLU, IGSTAT
      LOGICAL DERIVS
      INTEGER ITYPEG(LTYPEG), ISTGPA(LSTGPA), ICALCG(LCALCG)
      DOUBLE PRECISION GVALUE(LGVALU,3), FVALUE(LFVALU), GPVALU(LGPVLU)
C
C  Problem name : LUKVLE14  
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IGRTYP, IGROUP, IPSTRT, JCALCG
      DOUBLE PRECISION ALPHA 
      IGSTAT = 0
      DO     4 JCALCG = 1, NCALCG
       IGROUP = ICALCG(JCALCG)
       IGRTYP = ITYPEG(IGROUP)
       IF ( IGRTYP == 0 ) GO TO     4
       IPSTRT = ISTGPA(IGROUP) - 1
       GO TO (    1,    2,    3
     *                                                        ), IGRTYP
C
C  Group type : L2      
C
    1  CONTINUE 
       ALPHA = FVALUE(IGROUP)
       IF ( .NOT. DERIVS ) THEN
        GVALUE(IGROUP,1)= ALPHA * ALPHA                            
       ELSE
        GVALUE(IGROUP,2)= ALPHA + ALPHA                            
        GVALUE(IGROUP,3)= 2.0                                      
       END IF
       GO TO     4
C
C  Group type : L4      
C
    2  CONTINUE 
       ALPHA = FVALUE(IGROUP)
       IF ( .NOT. DERIVS ) THEN
        GVALUE(IGROUP,1)= ALPHA ** 4                               
       ELSE
        GVALUE(IGROUP,2)= 4.0 * ALPHA ** 3                         
        GVALUE(IGROUP,3)= 12.0 * ALPHA ** 2                        
       END IF
       GO TO     4
C
C  Group type : L6      
C
    3  CONTINUE 
       ALPHA = FVALUE(IGROUP)
       IF ( .NOT. DERIVS ) THEN
        GVALUE(IGROUP,1)= ALPHA ** 6                               
       ELSE
        GVALUE(IGROUP,2)= 6.0 * ALPHA ** 5                         
        GVALUE(IGROUP,3)= 30.0 * ALPHA ** 4                        
       END IF
    4 CONTINUE
      RETURN
      END
