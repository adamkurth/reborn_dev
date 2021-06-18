   program r3ddriver
   use random
   use fftwrotate_m
   implicit none
   integer, parameter :: i4=4
   integer, parameter :: r8=8
   integer, parameter :: c8=8
   integer, parameter :: i8=8
   complex(kind=c8), parameter :: czero=(0.0_r8,0.0_r8)
   complex(kind=c8), parameter :: cone=(1.0_r8,0.0_r8)
   complex(kind=c8), parameter :: ci=(0.0_r8,1.0_r8)
   complex(kind=c8), allocatable :: f(:,:,:)
   real(kind=r8) :: euler(3)
   real :: t1,t2
   integer(kind=i4) :: n
   n=384
   allocate(f(n,n,n))
   euler=[0.17_r8, 0.35_r8, -0.21_r8]
   f=reshape(crandn(n**3),shape(f))
   if (n.ne.nfft()) then
      call initfft(n)
   endif
   call cpu_time(t1)
   call rotate3dz(f,euler(1),n)
   call rotate3dy(f,euler(2),n)
   call rotate3dz(f,euler(3),n)
   call cpu_time(t2)
   write (*,*) 'rotation time',t2-t1
   contains
      function crandn(n)
      integer(kind=i4) :: n
      complex(kind=r8) :: crandn(n)
      real(kind=r8) :: r1(n),r2(n)
      r1=randn(n)
      r2=randn(n)
      crandn=cmplx(r1,r2)
      end function crandn
   end program r3ddriver
