      SUBROUTINE GROUP ( GVALUE, LGVALU, FVALUE, GPVALU, NCALCG, 
     *                   ITYPEG, ISTGPA, ICALCG, LTYPEG, LSTGPA, 
     *                   LCALCG, LFVALU, LGPVLU, DERIVS, IGSTAT )
      INTEGER LGVALU, NCALCG, LTYPEG, LSTGPA
      INTEGER LCALCG, LFVALU, LGPVLU, IGSTAT
      LOGICAL DERIVS
      INTEGER ITYPEG(LTYPEG), ISTGPA(LSTGPA), ICALCG(LCALCG)
      DOUBLE PRECISION GVALUE(LGVALU,3), FVALUE(LFVALU), GPVALU(LGPVLU)
C
C  Problem name : CURLY30   
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IGRTYP, IGROUP, IPSTRT, JCALCG
      DOUBLE PRECISION GVAR  , APB   
      IGSTAT = 0
      DO     2 JCALCG = 1, NCALCG
       IGROUP = ICALCG(JCALCG)
       IGRTYP = ITYPEG(IGROUP)
       IF ( IGRTYP == 0 ) GO TO     2
       IPSTRT = ISTGPA(IGROUP) - 1
C
C  Group type : P4      
C
       GVAR  = FVALUE(IGROUP)
       APB   = 2.0D+1                                   
       IF ( .NOT. DERIVS ) THEN
        GVALUE(IGROUP,1)= GVAR * ( GVAR * ( GVAR ** 2 - APB )      
     *                  - 1.0D-1 )                               
       ELSE
        GVALUE(IGROUP,2)= 2.0D+0 * GVAR * ( 2.0D+0 * GVAR ** 2     
     *                         - APB ) - 1.0D-1                         
        GVALUE(IGROUP,3)= 1.2D+1 * GVAR ** 2 - 2.0D+0 * APB        
       END IF
    2 CONTINUE
      RETURN
      END
