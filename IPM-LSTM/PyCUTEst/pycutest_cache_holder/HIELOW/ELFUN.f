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
C  Problem name : HIELOW    
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION B1    , B2    , B3    , F     , G     
      DOUBLE PRECISION H     
      EXTERNAL F     , G     , H     
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : LIKE      
C
       B1     = XVALUE(IELVAR(ILSTRT+     1))
       B2     = XVALUE(IELVAR(ILSTRT+     2))
       B3     = XVALUE(IELVAR(ILSTRT+     3))
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= F(B1,B2,B3)                              
       ELSE
        FUVALS(IGSTRT+     1)= G(B1,B2,B3,1)                            
        FUVALS(IGSTRT+     2)= G(B1,B2,B3,2)                            
        FUVALS(IGSTRT+     3)= G(B1,B2,B3,3)                            
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=H(B1,B2,B3,1,1)                          
         FUVALS(IHSTRT+     2)=H(B1,B2,B3,1,2)                          
         FUVALS(IHSTRT+     4)=H(B1,B2,B3,1,3)                          
         FUVALS(IHSTRT+     3)=H(B1,B2,B3,2,2)                          
         FUVALS(IHSTRT+     5)=H(B1,B2,B3,2,3)                          
         FUVALS(IHSTRT+     6)=H(B1,B2,B3,3,3)                          
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
