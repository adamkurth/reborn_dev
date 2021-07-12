   subroutine rotate3dfftw(f,euler,nin)
   use fftwrotate_m
   implicit none
   integer(kind=4), intent(in) :: nin
   real(kind=8), intent(in) :: euler(3)
   complex(kind=8), intent(inout) :: f(nin,nin,nin)
   if (nin.ne.nfft()) then
      call initfft(nin)
   endif
! negative euler to agree with pure python order for x,y,x rotations
   call rotate3dz(f,-euler(1),nin)
   call rotate3dy(f,-euler(2),nin)
   call rotate3dz(f,-euler(3),nin)
   end subroutine rotate3dfftw
