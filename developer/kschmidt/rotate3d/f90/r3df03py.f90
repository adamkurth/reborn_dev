!module rotate_m
!   use, intrinsic :: iso_c_binding
!   implicit none
!   integer, private, parameter :: i4=4
!   integer, private, parameter :: r8=8
!   integer, private, parameter :: c8=8
!   integer, private, parameter :: i8=8

!   public :: rotate3dfftw
!
!contains

   subroutine rotate3dfftw(f,euler,nin)
   implicit none
   integer(kind=4), intent(in) :: nin
   real(kind=8), intent(in) :: euler(3)
   complex(kind=8), intent(inout) :: f(nin,nin,nin)
   integer(kind=4), external :: nfft
   if (nin.ne.nfft()) then
      call initfft(nin)
   endif
   call rotate3dz(f,euler(1),nin)
   call rotate3dy(f,euler(2),nin)
   call rotate3dz(f,euler(3),nin)
   end subroutine rotate3dfftw

!end module rotate_m
