      SUBROUTINE ELFUN ( FUVALS, XVALUE, EPVALU, NCALCF, ITYPEE, 
     *                   ISTAEV, IELVAR, INTVAR, ISTADH, ISTEPA, 
     *                   ICALCF, LTYPEE, LSTAEV, LELVAR, LNTVAR, 
     *                   LSTADH, LSTEPA, LCALCF, LFVALU, LXVALU, 
     *                   LEPVLU, IFFLAG, IFSTAT )
      INTEGER NCALCF, IFFLAG, LTYPEE, LSTAEV, LELVAR, LNTVAR
      INTEGER LSTADH, LSTEPA, LCALCF, LFVALU, LXVALU, LEPVLU
      INTEGER IFSTAT
      INTEGER ITYPEE(LTYPEE), ISTAEV(LSTAEV), IELVAR(LELVAR)
      INTEGER INTVAR(LNTVAR), ISTADH(LSTADH), ISTEPA(LSTEPA)
      INTEGER ICALCF(LCALCF)
      DOUBLE PRECISION FUVALS(LFVALU), XVALUE(LXVALU), EPVALU(LEPVLU)
C
C  Problem name : CHEMRCTA  
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION U     , T     , G     , DADT  , D2ADT2
      DOUBLE PRECISION EX    , UEX   
      INTRINSIC EXP   
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : REAC      
C
       U      = XVALUE(IELVAR(ILSTRT+     1))
       T      = XVALUE(IELVAR(ILSTRT+     2))
       G      = EPVALU(IPSTRT+     1)
       DADT   = G / ( T * T )                            
       D2ADT2 = - 2.0 * DADT / T                         
       EX     = EXP( G - G / T )                         
       UEX    = EX * U                                   
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= UEX                                      
       ELSE
        FUVALS(IGSTRT+     1)= EX                                       
        FUVALS(IGSTRT+     2)= UEX * DADT                               
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     2)=EX * DADT                                
         FUVALS(IHSTRT+     3)=UEX * ( DADT * DADT + D2ADT2 )           
         FUVALS(IHSTRT+     1)=0.0D+0
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
