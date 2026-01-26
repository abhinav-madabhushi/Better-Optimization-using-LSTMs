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
C  Problem name : VIBRBEAMNE
C
C  -- produced by SIFdecode 2.6.2
C
      INTEGER IELEMN, IELTYP, IHSTRT, ILSTRT, IGSTRT, IPSTRT
      INTEGER JCALCF
      DOUBLE PRECISION a0    , a1    , a2    , a3    , b     
      DOUBLE PRECISION y     , q     , phi   , cosphi, sinphi
      DOUBLE PRECISION y2    , y3    , y4    , y5    , y6    
      DOUBLE PRECISION bcos  , bsin  
      INTRINSIC sin   , cos   
      IFSTAT = 0
      DO     2 JCALCF = 1, NCALCF
       IELEMN = ICALCF(JCALCF) 
       ILSTRT = ISTAEV(IELEMN) - 1
       IGSTRT = INTVAR(IELEMN) - 1
       IPSTRT = ISTEPA(IELEMN) - 1
       IF ( IFFLAG == 3 ) IHSTRT = ISTADH(IELEMN) - 1
C
C  Element type : fun       
C
       a0     = XVALUE(IELVAR(ILSTRT+     1))
       a1     = XVALUE(IELVAR(ILSTRT+     2))
       a2     = XVALUE(IELVAR(ILSTRT+     3))
       a3     = XVALUE(IELVAR(ILSTRT+     4))
       b      = XVALUE(IELVAR(ILSTRT+     5))
       y      = EPVALU(IPSTRT+     1)
       q      = EPVALU(IPSTRT+     2)
       y2     = y * y                                    
       y3     = y * y2                                   
       y4     = y2 * y2                                  
       y5     = y2 * y3                                  
       y6     = y3 * y3                                  
       phi    = a0+y*(a1+y*(a2+y*a3))-q                  
       cosphi = cos( phi )                               
       sinphi = sin( phi )                               
       bcos   = b * cosphi                               
       bsin   = b * sinphi                               
       IF ( IFFLAG == 1 ) THEN
        FUVALS(IELEMN)= bcos                                     
       ELSE
        FUVALS(IGSTRT+     1)= - bsin                                   
        FUVALS(IGSTRT+     2)= - bsin * y                               
        FUVALS(IGSTRT+     3)= - bsin * y2                              
        FUVALS(IGSTRT+     4)= - bsin * y3                              
        FUVALS(IGSTRT+     5)= cosphi                                   
        IF ( IFFLAG == 3 ) THEN
         FUVALS(IHSTRT+     1)=- bcos                                   
         FUVALS(IHSTRT+     2)=- bcos * y                               
         FUVALS(IHSTRT+     4)=- bcos * y2                              
         FUVALS(IHSTRT+     7)=- bcos * y3                              
         FUVALS(IHSTRT+    11)=- sinphi                                 
         FUVALS(IHSTRT+     3)=- bcos * y2                              
         FUVALS(IHSTRT+     5)=- bcos * y3                              
         FUVALS(IHSTRT+     8)=- bcos * y4                              
         FUVALS(IHSTRT+    12)=- sinphi * y                             
         FUVALS(IHSTRT+     6)=- bcos * y4                              
         FUVALS(IHSTRT+     9)=- bcos * y5                              
         FUVALS(IHSTRT+    13)=- sinphi * y2                            
         FUVALS(IHSTRT+    10)=- bcos * y6                              
         FUVALS(IHSTRT+    14)=- sinphi * y3                            
         FUVALS(IHSTRT+    15)=0.0D+0
        END IF
       END IF
    2 CONTINUE
      RETURN
      END
