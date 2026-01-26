      function ba( v_rx, v_ry, v_rz, v_x, v_y, v_z, v_tx,               
     *             v_ty, v_tz, v_k1, v_k2, v_fl,                        
     *             rx, rx1, rx12, ry, ry1, ry12, copy_y )               
!  ------------------------------------------------------------         
!  compute the Snavely reprojection error for bundle adjustment         
!  variables encoded in v are                                           
!    r1,r2,r3 Rodrigues rotation coordiinates                           
!    x1,x2,x3 position of object                                        
!    s1,s2,s3 translation                                               
!    k1,k2 radial distortion                                            
!    fl focal length                                                    
!  output residuals                                                     
!  rx, ry residuals with their first and second derivatives             
!  rx1, rx2, ry1, ry2. Upper triangle of second derivatives             
!  is held (by columns and increasing row count) in a 1-D array         
!  ------------------------------------------------------------         
      implicit none                                                     
      integer, parameter :: wp = KIND( 1.0D0 )                          
      integer :: ba                                                     
      real (KIND = wp ), INTENT( IN ) :: v_rx, v_ry, v_rz, v_x, v_y, v_z
      real (KIND = wp ), INTENT( IN ) :: v_tx, v_ty, v_tz, v_k1, v_k2   
      real (KIND = wp ), INTENT( IN ) :: v_fl                           
      real (KIND = wp ), INTENT( OUT ) :: rx                            
      real (KIND = wp ), INTENT( OUT ), dimension( 12 ) :: rx1          
      real (KIND = wp ), INTENT( OUT ), dimension( 78 ) :: rx12         
      real (KIND = wp ), INTENT( INOUT ) :: ry                          
      real (KIND = wp ), INTENT( INOUT ), dimension( 12 ) :: ry1        
      real (KIND = wp ), INTENT( INOUT ), dimension( 78 ) :: ry12       
      LOGICAL :: copy_y                                                 
      integer, parameter :: r1 = 1                                      
      integer, parameter :: r2 = 2                                      
      integer, parameter :: r3 = 3                                      
      integer, parameter :: x1 = 4                                      
      integer, parameter :: x2 = 5                                      
      integer, parameter :: x3 = 6                                      
      integer, parameter :: s1 = 7                                      
      integer, parameter :: s2 = 8                                      
      integer, parameter :: s3 = 9                                      
      integer, parameter :: k1 = 10                                     
      integer, parameter :: k2 = 11                                     
      integer, parameter :: fl = 12                                     
C  local variables                                                      
      real (KIND = wp ), dimension( 12 ) :: v                           
      integer :: i, j, k                                                
      real (KIND = wp ) :: tt, t, ct, st, ex, ey, ez, p, cx, cy, cz     
      real (KIND = wp ) :: dx, dy, cc, d                                
      real (KIND = wp ) :: cccc, ccd, ccdk2, omct, sa, f1, f11, f12, f3 
      real (KIND = wp ) :: f2r1, f2r2, f2r3, f22r1, f22r2, f22r3        
      real (KIND = wp ) :: fx, fx1, fx2, fx22, fy, fy1, fy2, fy22, fz1  
      real (KIND = wp ) :: fcx11, fcx12, fcx13, fcx21, fcx22, fcx23     
      real (KIND = wp ) :: fcy11, fcy12, fcy13, fcy21, fcy22, fcy23     
      real (KIND = wp ) :: fcz11, fcz12, fcz13, fcz21, fcz22, fcz23     
      real (KIND = wp ), dimension( r3 ) :: tt1, t1, ct1, st1, ss       
      real (KIND = wp ), dimension( r3 ) :: ex1, ey1, ez1               
      real (KIND = wp ), dimension( x3 ) :: p1                          
      real (KIND = wp ), dimension( s3 ) :: cx1, cy1, cz1, dx1, dy1, cc1
      real (KIND = wp ), dimension( k2 ) :: d1                          
      real (KIND = wp ), dimension( r3, r3 ) :: tt2, t2, ct2, st2       
      real (KIND = wp ), dimension( r3, r3 ) :: ex2, ey2, ez2           
      real (KIND = wp ), dimension( x3, r3 ) :: p2                      
      real (KIND = wp ), dimension( s3, s3 ) :: cx2, cy2, cz2, dx2, dy2 
      real (KIND = wp ), dimension( s3, s3 ) :: cc2                     
      real (KIND = wp ), dimension( k2, k2 ) :: d2                      
      real (KIND = wp ), dimension( fl, fl ) :: rx2, ry2                
      if ( copy_y ) go to 1                                             
      v(r1) = v_rx                                                      
      v(r2) = v_ry                                                      
      v(r3) = v_rz                                                      
      v(x1) = v_x                                                       
      v(x2) = v_y                                                       
      v(x3) = v_z                                                       
      v(s1) = v_tx                                                      
      v(s2) = v_ty                                                      
      v(s3) = v_tz                                                      
      v(k1) = v_k1                                                      
      v(k2) = v_k2                                                      
      v(fl) = v_fl                                                      
C  for any function f, its gradient is f1 and its Hessian is f2         
C  (lower triangle values filled only)                                  
C  convert from world coordinates x to camera coordinates c             
C  use the Rodrigues representation of a rotation of theta radians about
C  the unit vector e; on input theta = ||w|| and e = w / theta          
C  compute ttt = t ** 2                                                 
C == tt                                                                 
      tt = v(r1) * v(r1) + v(r2) * v(r2) + v(r3) * v(r3)                
      tt1(r1) = 2.0_wp * v(r1)                                          
      tt1(r2) = 2.0_wp * v(r2)                                          
      tt1(r3) = 2.0_wp * v(r3)                                          
      tt2(r1,r1) = 2.0_wp                                               
      tt2(r2,r2) = 2.0_wp                                               
      tt2(r3,r3) = 2.0_wp                                               
C rotate the data                                                       
C compute the camera coordinates wrt the world coordinates              
C away from zero, use the rodriguez formula                             
C   c = pt ct + (w x x) * st + w (w . x) (1 - ct)                       
C Be careful to only evaluate the square root if the norm of the w vecto
C is greater than zero as otherwise we may get a division by zero       
      if ( tt > epsilon( 1.0_wp ) ) then                                
C == t                                                                  
        t = sqrt(tt)                                                    
        f1 = 0.5_wp / t                                                 
        f11 = - 0.25_wp / t ** 3                                        
        do j = r1, r3                                                   
          t1(j) = f1 * tt1(j)                                           
          t2(j,j) = f11 * tt1(j) ** 2 + f1 * tt2(j,j)                   
          do i = j+1, r3                                                
            t2(i,j) = f11 * tt1(i) * tt1(j)                             
          end do                                                        
        end do                                                          
C == ct and st                                                          
        ct = cos(t)                                                     
        st = sin(t)                                                     
        do j = r1, r3                                                   
          ct1(j) = - st * t1(j)                                         
          st1(j) = ct * t1(j)                                           
          do i = j, r3                                                  
            ct2(i,j) = - ct * t1(i) * t1(j) - st * t2(i,j)              
            st2(i,j) = - st * t1(i) * t1(j) + ct * t2(i,j)              
          end do                                                        
        end do                                                          
C == ex, ey & ez                                                        
        ex = v(r1) / t                                                  
        ey = v(r2) / t                                                  
        ez = v(r3) / t                                                  
        f1 = 1.0_wp / t                                                 
        f2r1 = - v(r1) / tt                                             
        f2r2 = - v(r2) / tt                                             
        f2r3 = - v(r3) / tt                                             
        f12 = - 1.0_wp / tt                                             
        f22r1 = 2.0_wp * v(r1) / t ** 3                                 
        f22r2 = 2.0_wp * v(r2) / t ** 3                                 
        f22r3 = 2.0_wp * v(r3) / t ** 3                                 
        do j = r1, r3                                                   
          ex1(j) = f2r1 * t1(j)                                         
          ey1(j) = f2r2 * t1(j)                                         
          ez1(j) = f2r3 * t1(j)                                         
          do i = j, r3                                                  
            ex2(i,j) = f22r1 * t1(i) * t1(j) + f2r1 * t2(i,j)           
            ey2(i,j) = f22r2 * t1(i) * t1(j) + f2r2 * t2(i,j)           
            ez2(i,j) = f22r3 * t1(i) * t1(j) + f2r3 * t2(i,j)           
          end do                                                        
        end do                                                          
        ex1(r1) = ex1(r1) + f1                                          
        ey1(r2) = ey1(r2) + f1                                          
        ez1(r3) = ez1(r3) + f1                                          
        ex2(r1,r1) = ex2(r1,r1) + 2.0_wp * f12 * t1(r1)                 
        ex2(r2,r1) = ex2(r2,r1) + f12 * t1(r2)                          
        ex2(r3,r1) = ex2(r3,r1) + f12 * t1(r3)                          
        ey2(r2,r2) = ey2(r2,r2) + 2.0_wp * f12 * t1(r2)                 
        ey2(r2,r1) = ey2(r2,r1) + f12 * t1(r1)                          
        ey2(r3,r2) = ey2(r3,r2) + f12 * t1(r3)                          
        ez2(r3,r3) = ez2(r3,r3) + 2.0_wp * f12 * t1(r3)                 
        ez2(r3,r1) = ez2(r3,r1) + f12 * t1(r1)                          
        ez2(r3,r2) = ez2(r3,r2) + f12 * t1(r2)                          
C == p                                                                  
        omct = 1.0_wp - ct                                              
        sa =  ex * v(x1) + ey * v(x2) + ez * v(x3)                      
        p = sa * omct                                                   
        fx1 = v(x1) * omct                                              
        fy1 = v(x2) * omct                                              
        fz1 = v(x3) * omct                                              
        p1(r1: r3) = fx1 * ex1(r1: r3) + fy1 * ey1(r1: r3)              
     *               + fz1 * ez1(r1: r3) - sa * ct1(r1: r3)             
        p1(x1) = ex * omct ; p1(x2) = ey * omct ; p1(x3) = ez * omct    
        ss(r1:r3) = v(x1) * ex1(r1:r3) + v(x2) * ey1(r1:r3)             
     *               + v(x3) * ez1(r1:r3)                               
        do j = r1, r3                                                   
          do i = j, r3                                                  
            p2(i,j) = - ss(i) * ct1(j) - ss(j) * ct1(i) - sa * ct2(i,j) 
     *                + fx1 * ex2(i,j) + fy1 * ey2(i,j) + fz1 * ez2(i,j)
          end do                                                        
        end do                                                          
        p2(x1,r1: r3) = omct * ex1(r1: r3) - ex * ct1(r1: r3)           
        p2(x2,r1: r3) = omct * ey1(r1: r3) - ey * ct1(r1: r3)           
        p2(x3,r1: r3) = omct * ez1(r1: r3) - ez * ct1(r1: r3)           
C  do j = x1, x3                                                        
C    do i = j, x3                                                       
C      p2(i,j) = 0.0_wp                                                 
C    end do                                                             
C  end do                                                               
C  == cx = v(x1) * ct + ey * v(x3) * st - ez * v(x2) * st + ex * p      
        fcx11 = v(x3) * st                                              
        fcx12 = ey * st                                                 
        fcx13 = ey * v(x3)                                              
        fcx21 = - v(x2) * st                                            
        fcx22 = - ez * st                                               
        fcx23 = - ez * v(x2)                                            
        cx = v(x1) * ct + ( fcx13 + fcx23 ) * st + ex * p               
        do j = r1, r3                                                   
          cx1(j) = v(x1) * ct1(j) + fcx11 * ey1(j) + fcx13 * st1(j)     
     *             + fcx21 * ez1(j) + fcx23 * st1(j)                    
     *             + p * ex1(j) + ex * p1(j)                            
        end do                                                          
        cx1(x1) = ex * p1(x1) + ct                                      
        cx1(x2) = ex * p1(x2) + fcx22                                   
        cx1(x3) = ex * p1(x3) + fcx12                                   
        do j = r1, r3                                                   
          do i = j, r3                                                  
            cx2(i,j) = v(x1) * ct2(i,j) +                               
     *                 v(x3) * ( ey1(i) * st1(j) + st1(i) * ey1(j) ) +  
     *                 fcx11 * ey2(i,j) + fcx13 * st2(i,j) -            
     *                 v(x2) * ( ez1(i) * st1(j) + st1(i) * ez1(j) ) +  
     *                 fcx21 * ez2(i,j) + fcx23 * st2(i,j) +            
     *                 ex1(i) * p1(j) + p1(i) * ex1(j) +                
     *                 p * ex2(i,j) + ex * p2(i,j)                      
          end do                                                        
        end do                                                          
        do j = r1, r3                                                   
          cx2(x1,j) = p1(x1) * ex1(j) + ex * p2(x1,j) + ct1(j)          
          cx2(x2,j) = p1(x2) * ex1(j) + ex * p2(x2,j)                   
     *                - st * ez1(j) - ez * st1(j)                       
          cx2(x3,j) = p1(x3) * ex1(j) + ex * p2(x3,j)                   
     *                + st * ey1(j) + ey * st1(j)                       
        end do                                                          
C  do j = x1, x3                                                        
C    do i = j, x3                                                       
C      cx2(i,j) = 0.0_wp                                                
C    end do                                                             
C  end do                                                               
C  == cy = v(x2) * ct + ez * v(x1) * st - ex * v(x3) * st + ey * p      
        fcy11 = v(x1) * st                                              
        fcy12 = ez * st                                                 
        fcy13 = ez * v(x1)                                              
        fcy21 = - v(x3) * st                                            
        fcy22 = - ex * st                                               
        fcy23 = - ex * v(x3)                                            
        cy = v(x2) * ct + ( fcy13 + fcy23 ) * st + ex * p               
        do j = r1, r3                                                   
          cy1(j) = v(x2) * ct1(j) + fcy11 * ez1(j) + fcy13 * st1(j) +   
     *             fcy21 * ex1(j) + fcy23 * st1(j) +                    
     *             p * ex1(j) + ex * p1(j)                              
        end do                                                          
        cy1(x1) = ex * p1(x1) + fcy12                                   
        cy1(x2) = ex * p1(x2) + ct                                      
        cy1(x3) = ex * p1(x3) + fcy22                                   
        do j = r1, r3                                                   
          do i = j, r3                                                  
            cy2(i,j) = v(x2) * ct2(i,j) +                               
     *                 v(x1) * ( ez1(i) * st1(j) + st1(i) * ez1(j) ) +  
     *                 fcy11 * ez2(i,j) + fcy13 * st2(i,j) -            
     *                 v(x3) * ( ex1(i) * st1(j) + st1(i) * ex1(j) ) +  
     *                 fcy21 * ex2(i,j) + fcy23 * st2(i,j) +            
     *                 ex1(i) * p1(j) + p1(i) * ex1(j) +                
     *                 p * ex2(i,j) + ex * p2(i,j)                      
          end do                                                        
        end do                                                          
        do j = r1, r3                                                   
          cy2(x1,j) = p1(x1) * ex1(j) + ex * p2(x1,j) +                 
     *                st * ez1(j) + ez * st1(j)                         
          cy2(x2,j) = p1(x2) * ex1(j) + ex * p2(x2,j) + ct1(j)          
          cy2(x3,j) = p1(x3) * ex1(j) + ex * p2(x3,j) -                 
     *                st * ex1(j) - ex * st1(j)                         
        end do                                                          
C  do j = x1, x3                                                        
C    do i = j, x3                                                       
C      cy2(i,j) = 0.0_wp                                                
C    end do                                                             
C  end do                                                               
C  == cz = v(x3) * ct + ex * v(x2) * st - ey * v(x1) * st + ez * p      
        fcz11 = v(x2) * st                                              
        fcz12 = ex * st                                                 
        fcz13 = ex * v(x2)                                              
        fcz21 = - v(x1) * st                                            
        fcz22 = - ey * st                                               
        fcz23 = - ey * v(x1)                                            
        cz = v(x3) * ct + ( fcz13 + fcz23 ) * st + ez * p               
        do j = r1, r3                                                   
          cz1(j) = v(x3) * ct1(j) + fcz11 * ex1(j) + fcz13 * st1(j) +   
     *             fcz21 * ey1(j) + fcz23 * st1(j) +                    
     *             p * ez1(j) + ez * p1(j)                              
        end do                                                          
        cz1(x1) = ez * p1(x1) + fcz22                                   
        cz1(x2) = ez * p1(x2) + fcz12                                   
        cz1(x3) = ez * p1(x3) + ct                                      
        do j = r1, r3                                                   
          do i = j, r3                                                  
            cz2(i,j) = v(x3) * ct2(i,j) +                               
     *                 v(x2) * ( ex1(i) * st1(j) + st1(i) * ex1(j) ) +  
     *                 fcz11 * ex2(i,j) + fcz13 * st2(i,j) -            
     *                 v(x1) * ( ey1(i) * st1(j) + st1(i) * ey1(j) ) +  
     *                 fcz21 * ey2(i,j) + fcz23 * st2(i,j) +            
     *                 ez1(i) * p1(j) + p1(i) * ez1(j) +                
     *                 p * ez2(i,j) + ez * p2(i,j)                      
          end do                                                        
        end do                                                          
        do j = r1, r3                                                   
          cz2(x1,j) = p1(x1) * ez1(j) + ez * p2(x1,j)                   
     *                - st * ey1(j) - ey * st1(j)                       
          cz2(x2,j) = p1(x2) * ez1(j) + ez * p2(x2,j)                   
     *                + st * ex1(j) + ex * st1(j)                       
          cz2(x3,j) = p1(x3) * ez1(j) + ez * p2(x3,j) + ct1(j)          
        end do                                                          
C  do j = x1, x3                                                        
C    do i = j, x3                                                       
C      cz2(i,j) = 0.0_wp                                                
C    end do                                                             
C  end do                                                               
C near zero, the first order Taylor approximation of the rotation       
C matrix R corresponding to a vector r and angle t is                   
C   R = I + hat(r) * sin(t)                                             
C But sin t ~ t and t * r = r, which gives us                           
C  R = I + hat(w)                                                       
C and actually performing multiplication with the point x, gives us     
C R * x = x + r x x.                                                    
      else                                                              
C == cx, cy, cz                                                         
        cx = v(x1) + v(r2) * v(x3) - v(r3) * v(x2)                      
        cy = v(x2) + v(r3) * v(x1) - v(r1) * v(x3)                      
        cz = v(x3) + v(r1) * v(x2) - v(r2) * v(x1)                      
        cx1(r1) = 0.0_wp                                                
        cx1(r2) = v(x3)                                                 
        cx1(r3) = - v(x2)                                               
        cx1(x1) = 1.0_wp                                                
        cx1(x2) = - v(r3)                                               
        cx1(x3) =  v(r2)                                                
        do j = r1, r3                                                   
          do i = j, x3                                                  
            cx2(i,j) = 0.0_wp                                           
          end do                                                        
        end do                                                          
        cx2(x3,r2) = 1.0_wp                                             
        cx2(x2,r3) = - 1.0_wp                                           
        cy1(r1) = - v(x3)                                               
        cy1(r2) = 0.0_wp                                                
        cy1(r3) = v(x1)                                                 
        cy1(x1) = v(r3)                                                 
        cy1(x2) = 1.0_wp                                                
        cy1(x3) = - v(r1)                                               
        do j = r1, r3                                                   
          do i = j, x3                                                  
            cy2(i,j) = 0.0_wp                                           
          end do                                                        
        end do                                                          
        cy2(x3,r1) = - 1.0_wp                                           
        cy2(x1,r3) = 1.0_wp                                             
        cz1(r1) = v(x2)                                                 
        cz1(r2) = - v(x1)                                               
        cz1(r3) = 0.0_wp                                                
        cz1(x1) = - v(r2)                                               
        cz1(x2) = v(r1)                                                 
        cz1(x3) = 1.0_wp                                                
        do j = r1, r3                                                   
          do i = j, x3                                                  
            cz2(i,j) = 0.0_wp                                           
          end do                                                        
        end do                                                          
        cz2(x1,r2) = - 1.0_wp                                           
        cz2(x2,r1) = 1.0_wp                                             
      end if                                                            
C translate the rotated data                                            
C == cx, cy, cz with shifts                                             
      cx = cx + v(s1)                                                   
      cx1(s1) = 1.0_wp                                                  
      cx1(s2) = 0.0_wp                                                  
      cx1(s3) = 0.0_wp                                                  
      cy = cy + v(s2)                                                   
      cy1(s1) = 0.0_wp                                                  
      cy1(s2) = 1.0_wp                                                  
      cy1(s3) = 0.0_wp                                                  
      cz = cz + v(s3)                                                   
      cz1(s1) = 0.0_wp                                                  
      cz1(s2) = 0.0_wp                                                  
      cz1(s3) = 1.0_wp                                                  
Cdo i = s1, s3                                                          
C  do j = r1, i                                                         
C    cx2(i,j) = 0.0_wp                                                  
C    cy2(i,j) = 0.0_wp                                                  
C    cz2(i,j) = 0.0_wp                                                  
C  end do                                                               
Cend do                                                                 
C  compute the centre of distortion                                     
C == dx and dy                                                          
      dx = - cx / cz                                                    
      dy = - cy / cz                                                    
      f1 = - 1.0 /cz                                                    
      fx2 = cx / cz ** 2                                                
      fy2 = cy / cz ** 2                                                
      f12 = 1.0 / cz ** 2                                               
      fx22 = - 2.0 * cx / cz ** 3                                       
      fy22 = - 2.0 * cy / cz ** 3                                       
      do j = r1, s3                                                     
        dx1(j) = f1 * cx1(j) + fx2 * cz1(j)                             
        dy1(j) = f1 * cy1(j) + fy2 * cz1(j)                             
      end do                                                            
      do j = r1, r3                                                     
        do i = j, x3                                                    
          dx2(i,j) = f12 * ( cx1(i) * cz1(j) + cz1(i) * cx1(j) ) +      
     *                     fx22 * cz1(i) * cz1(j) +                     
     *                     f1 * cx2(i,j) + fx2 * cz2(i,j)               
          dy2(i,j) = f12 * ( cy1(i) * cz1(j) + cz1(i) * cy1(j) ) +      
     *                     fy22 * cz1(i) * cz1(j) +                     
     *                     f1 * cy2(i,j) + fy2 * cz2(i,j)               
        end do                                                          
        do i = s1, s3                                                   
          dx2(i,j) = f12 * ( cx1(i) * cz1(j) + cz1(i) * cx1(j) ) +      
     *                     fx22 * cz1(i) * cz1(j)                       
          dy2(i,j) = f12 * ( cy1(i) * cz1(j) + cz1(i) * cy1(j) ) +      
     *                     fy22 * cz1(i) * cz1(j)                       
        end do                                                          
      end do                                                            
      do j = x1, s3                                                     
        do i = j, s3                                                    
          dx2(i,j) = f12 * ( cx1(i) * cz1(j) + cz1(i) * cx1(j) ) +      
     *                     fx22 * cz1(i) * cz1(j)                       
          dy2(i,j) = f12 * ( cy1(i) * cz1(j) + cz1(i) * cy1(j) ) +      
     *                     fy22 * cz1(i) * cz1(j)                       
        end do                                                          
      end do                                                            
C  compute the scaling to account for radial distortion                 
C == cc                                                                 
      cc = dx ** 2 + dy ** 2                                            
      fx = 2.0_wp * dx                                                  
      fy = 2.0_wp * dy                                                  
      do j = r1, s3                                                     
        cc1(j) = fx * dx1(j) + fy * dy1(j)                              
      end do                                                            
      do j = r1, s3                                                     
        do i = j, s3                                                    
          cc2(i,j) = 2.0_wp * dx1(i) * dx1(j) + fx * dx2(i,j) +         
     *                     2.0_wp * dy1(i) * dy1(j) + fy * dy2(i,j)     
        end do                                                          
      end do                                                            
C == d                                                                  
      cccc = cc * cc                                                    
      ccd = 2.0_wp * cc                                                 
      ccdk2 = ccd * v(k2)                                               
      d = 1.0_wp + v(k1) * cc + v(k2) * cccc                            
      do j = r1, s3                                                     
        d1(j) = v(k1) * cc1(j) + ccdk2 * cc1(j)                         
      end do                                                            
      d1(k1) = cc                                                       
      d1(k2) = cccc                                                     
      do j = r1, s3                                                     
        do i = j, s3                                                    
          d2(i,j) = v(k1) * cc2(i,j) +                                  
     *              2.0_wp * v(k2) * cc1(i) * cc1(j) + ccdk2 * cc2(i,j) 
        end do                                                          
        d2(k1,j) = cc1(j)                                               
        d2(k2,j) = ccd * cc1(j)                                         
      end do                                                            
Cdo j = k1, k2                                                          
C  do i = j, k2                                                         
C    d2(i,j) = 0.0_wp                                                   
C  end do                                                               
Cend do                                                                 
C  convert to pixel coordinates to obtain the required residuals        
C  == rx and ry                                                         
      fx1 = d * dx                                                      
      fy1 = d * dy                                                      
      fx2 = v(fl) * dx                                                  
      fy2 = v(fl) * dy                                                  
      f3 = v(fl) * d                                                    
      rx = v(fl) * fx1                                                  
      ry = v(fl) * fy1                                                  
      do j = 1, s3                                                      
        rx1(j) = fx2 * d1(j) + f3 * dx1(j)                              
        ry1(j) = fy2 * d1(j) + f3 * dy1(j)                              
      end do                                                            
      do j = k1, k2                                                     
        rx1(j) = fx2 * d1(j)                                            
        ry1(j) = fy2 * d1(j)                                            
      end do                                                            
      rx1(fl) = fx1                                                     
      ry1(fl) = fy1                                                     
      do j = r1, s3                                                     
        do i = j, s3                                                    
          rx2(i,j) = v(fl) * ( d1(i) * dx1(j) + dx1(i) * d1(j) ) +      
     *                     fx2 * d2(i,j) + f3 * dx2(i,j)                
          ry2(i,j) = v(fl) * ( d1(i) * dy1(j) + dy1(i) * d1(j) ) +      
     *                     fy2 * d2(i,j) + f3 * dy2(i,j)                
        end do                                                          
        do i = k1, k2                                                   
          rx2(i,j) = v(fl) * d1(i) * dx1(j) + fx2 * d2(i,j)             
          ry2(i,j) = v(fl) * d1(i) * dy1(j) + fy2 * d2(i,j)             
        end do                                                          
        rx2(fl,j) = dx * d1(j) + d * dx1(j)                             
        ry2(fl,j) = dy * d1(j) + d * dy1(j)                             
      end do                                                            
      do j = k1, k2                                                     
        do i = j, k2                                                    
          rx2(i,j) = 0.0_wp                                             
          ry2(i,j) = 0.0_wp                                             
        end do                                                          
        rx2(fl,j) = dx * d1(j)                                          
        ry2(fl,j) = dy * d1(j)                                          
      end do                                                            
      rx2(fl,fl) = 0.0_wp                                               
      ry2(fl,fl) = 0.0_wp                                               
      k = 0                                                             
      do j = 1, fl                                                      
        do i = 1, j                                                     
          k = k + 1                                                     
          rx12(k) = rx2(j,i)                                            
          ry12(k) = ry2(j,i)                                            
        end do                                                          
      end do                                                            
      ba = 0                                                            
      return                                                            
C  copy the ry components to rx rather than recomputing them            
    1 continue                                                          
      rx = ry                                                           
      rx1( : 12 ) = ry1( : 12 )                                         
      rx12( : 78 ) = ry12( : 78 )                                       
      ba = 0                                                            
      return                                                            
      end function ba                                                   
