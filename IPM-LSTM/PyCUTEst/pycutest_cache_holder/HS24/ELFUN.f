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
C  Problem name : HS24      
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION V1    , V2    , FACT  , RT3X27, X1M3  
      DOUBLE PRECISION X1M3M9, V2SQ  , V2CB  
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : OBJ       
C
       V1     = XVALUE(IELVAR(ILSTRT+     1))
       V2     = XVALUE(IELVAR(ILSTRT+     2))
       RT3X27 = 1.0D0 / ( 27.0D0 * SQRT( 3.0D0 ) )       
       V2SQ   = V2 * V2                                  
       V2CB   = V2SQ * V2                                
       X1M3   = V1 - 3.0                                 
       X1M3M9 = X1M3 * X1M3 - 9.0                        
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= RT3X27 * X1M3M9 * V2CB                   
       ELSE
        FUVALS(IGSTRT+     1)= 2.0 * X1M3 * RT3X27 * V2CB               
        FUVALS(IGSTRT+     2)= 3.0 * X1M3M9 * RT3X27 * V2SQ             
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=2.0 * RT3X27 * V2CB                      
         FUVALS(IHSTRT+     2)=6.0 * X1M3 * RT3X27 * V2SQ               
         FUVALS(IHSTRT+     3)=6.0 * X1M3M9 * RT3X27 * V2               
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
