module fftwrotate_m
   use, intrinsic :: iso_c_binding
   implicit none
   integer, private, parameter :: i4=c_int
   integer, private, parameter :: r8=c_double
   integer, private, parameter :: c8=c_double_complex
   integer, private, parameter :: i8=c_long_long
   complex(kind=c8), private, parameter :: czero=(0.0_r8,0.0_r8)
   complex(kind=c8), private, parameter :: cone=(1.0_r8,0.0_r8)
   complex(kind=c8), private, parameter :: ci=(0.0_r8,1.0_r8)

   real(kind=r8), private, save :: pi
   integer, private, save :: n=0
   type(c_ptr), private, save :: iplanx,iplanxi,iplany,iplanyi
   complex(kind=c8), private, pointer, save :: ftmp1(:,:),ftmp2(:,:)
   type(c_ptr), private, save :: p1,p2

   public :: rotate3dz,rotate3dy,initfft,cleanfft,nfft

contains

   function nfft()
   integer(kind=i4) :: nfft
   nfft=n
   end function nfft

   subroutine rotate3dz(f,ang,nin)
   real(kind=r8), intent(in) :: ang
   integer(kind=i4), intent(in) :: nin
   complex(kind=c8), intent(inout) :: f(nin,nin,nin)
   integer(kind=i4) :: n90,ix,iy,iz
   real(kind=r8) :: dang
   complex(kind=c8) :: fftshift(nin)
   complex(kind=c8) :: xkfac(nin,nin),xfac(nin,nin)
   complex(kind=c8) :: ykfac(nin,nin),yfac(nin,nin)
   n90=modulo(nint(ang*2.0_r8/pi),4)
   dang=ang-n90*0.5_r8*pi
   if ((dang.eq.0.0_r8).and.(n90.eq.0)) return
   if (dang.ne.0.0_r8) call setupfacs(dang,xkfac,xfac,ykfac,yfac,fftshift)
   do iz=1,n
      select case (n90)
         case(0)
            ftmp1=f(:,:,iz)
         case(1)
            do iy=1,n
               do ix=1,n
                  ftmp1(n+1-iy,ix)=f(ix,iy,iz)
               enddo
            enddo
         case(2)
            do iy=1,n
               do ix=1,n
                  ftmp1(n+1-ix,n+1-iy)=f(ix,iy,iz)
               enddo
            enddo
         case(3)
            do iy=1,n
               do ix=1,n
                  ftmp1(iy,n+1-ix)=f(ix,iy,iz)
               enddo
            enddo
      end select
      if (dang.eq.0.0) then
         f(:,:,iz)=ftmp1
         cycle
      endif
      do iy=1,n
         ftmp1(:,iy)=ftmp1(:,iy)*fftshift
      enddo
      call dfftw_execute_dft(iplanx,ftmp1,ftmp2)
      ftmp2=ftmp2*xkfac
      call dfftw_execute_dft(iplanxi,ftmp2,ftmp1)
      ftmp1=ftmp1*xfac
      do iy=1,n
         ftmp1(:,iy)=ftmp1(:,iy)*fftshift(iy)
      enddo
      call dfftw_execute_dft(iplany,ftmp1,ftmp2)
      ftmp2=ftmp2*ykfac
      call dfftw_execute_dft(iplanyi,ftmp2,ftmp1)
      ftmp1=ftmp1*yfac
      do iy=1,n
         ftmp1(:,iy)=ftmp1(:,iy)*fftshift
      enddo
      call dfftw_execute_dft(iplanx,ftmp1,ftmp2)
      ftmp2=ftmp2*xkfac
      call dfftw_execute_dft(iplanxi,ftmp2,ftmp1)
      f(:,:,iz)=ftmp1*xfac
   enddo
   end subroutine rotate3dz

   subroutine rotate3dy(f,ang,nin)
   real(kind=r8), intent(in) :: ang
   integer(kind=i4), intent(in) :: nin
   complex(kind=c8), intent(inout) :: f(nin,nin,nin)
   integer(kind=i4) :: n90,ix,iy,iz
   real(kind=r8) :: dang
   complex(kind=c8) :: fftshift(nin)
   complex(kind=c8) :: xkfac(nin,nin),xfac(nin,nin)
   complex(kind=c8) :: ykfac(nin,nin),yfac(nin,nin)
   n90=modulo(nint(ang*2.0_r8/pi),4)
   dang=ang-n90*0.5_r8*pi
   if ((dang.eq.0.0_r8).and.(n90.eq.0)) return
   if (dang.ne.0.0_r8) call setupfacs(dang,xkfac,xfac,ykfac,yfac,fftshift)
! same as rotate3dz but put cycle coordinates x->y->z->x of f
   do iz=1,n
      select case (n90)
         case(0)
            do ix=1,n
               ftmp1(ix,:)=f(:,iz,ix)
            enddo
         case(1)
            do ix=1,n
               do iy=1,n
                  ftmp1(n+1-iy,ix)=f(iy,iz,ix)
               enddo
            enddo
         case(2)
            do ix=1,n
               do iy=1,n
                  ftmp1(n+1-ix,n+1-iy)=f(iy,iz,ix)
               enddo
            enddo
         case(3)
            do ix=1,n
               do iy=1,n
                  ftmp1(iy,n+1-ix)=f(iy,iz,ix)
               enddo
            enddo
      end select
      if (dang.eq.0.0_r8) then
         do ix=1,n
            f(:,iz,ix)=ftmp1(ix,:)
         enddo
         cycle
      endif
      do iy=1,n
         ftmp1(:,iy)=ftmp1(:,iy)*fftshift
      enddo
      call dfftw_execute_dft(iplanx,ftmp1,ftmp2)
      ftmp2=ftmp2*xkfac
      call dfftw_execute_dft(iplanxi,ftmp2,ftmp1)
      ftmp1=ftmp1*xfac
      do iy=1,n
         ftmp1(:,iy)=ftmp1(:,iy)*fftshift(iy)
      enddo
      call dfftw_execute_dft(iplany,ftmp1,ftmp2)
      ftmp2=ftmp2*ykfac
      call dfftw_execute_dft(iplanyi,ftmp2,ftmp1)
      ftmp1=ftmp1*yfac
      do iy=1,n
         ftmp1(:,iy)=ftmp1(:,iy)*fftshift
      enddo
      call dfftw_execute_dft(iplanx,ftmp1,ftmp2)
      ftmp2=ftmp2*xkfac
      call dfftw_execute_dft(iplanxi,ftmp2,ftmp1)
      do ix=1,n
         f(:,iz,ix)=ftmp1(ix,:)*xfac(ix,:)
      enddo
   enddo
   end subroutine rotate3dy

   subroutine setupfacs(dang,xkfac,xfac,ykfac,yfac,fftshift)
   complex(kind=c8) :: xkfac(:,:),xfac(:,:),ykfac(:,:),yfac(:,:),fftshift(:)
   real(kind=r8) :: dang
   real(kind=r8) :: scalex,scaley,c0,eni
   complex(kind=c8) :: ex1,ex2,ex3
   integer(kind=i4) :: ix,iy
   c0=0.5_r8*(n-1)
   eni=1.0_r8/n
   ex1=exp(ci*pi*(1.0_r8-mod(n,2)*eni))
   fftshift(1)=cone
   do ix=2,n
      fftshift(ix)=fftshift(ix-1)*ex1
   enddo
   scalex=-tan(0.5_r8*dang)
   scaley=sin(dang)
!slower but equivalent:
!   do iy=1,n
!      fftshift(iy)=exp(ci*pi*(1.0_r8-mod(n,2)*eni)*(iy-1))
!      do ix=1,n
!         xkfac(ix,iy)=exp(-ci*2.0_r8*pi*eni*scalex*(ix-1)*(iy-1-c0))
!         xfac(ix,iy)=exp(-ci*pi*(1.0_r8-mod(n,2)*eni)*(ix-1-scalex*(iy-1-c0)))
!         ykfac(ix,iy)=exp(-ci*2.0_r8*pi*eni*scaley*(iy-1)*(ix-1-c0))
!         yfac(ix,iy)=exp(-ci*pi*(1.0_r8-mod(n,2)*eni)*(iy-1-scaley*(ix-1-c0)))
!         xfac(ix,iy)=xfac(ix,iy)*eni
!         yfac(ix,iy)=yfac(ix,iy)*eni
!      enddo
!   enddo
   ex1=exp(-ci*2.0_r8*pi*eni*scalex)
   ex2=exp(ci*2.0_r8*pi*eni*scalex*c0)
   xkfac(1,:)=cone
   xkfac(2,1)=ex2
   do iy=2,n
      xkfac(2,iy)=xkfac(2,iy-1)*ex1
   enddo
   do ix=3,n
      xkfac(ix,:)=xkfac(ix-1,:)*xkfac(2,:)
   enddo
   ex1=exp(-ci*pi*(1.0_r8-mod(n,2)*eni))
   ex2=exp(ci*pi*(1.0_r8-mod(n,2)*eni)*scalex)
   ex3=exp(-ci*pi*(1.0_r8-mod(n,2)*eni)*scalex*c0)
   xfac(1,1)=ex3*eni
   do ix=2,n
      xfac(ix,1)=xfac(ix-1,1)*ex1
   enddo
   do iy=2,n
      xfac(:,iy)=xfac(:,iy-1)*ex2
   enddo
   ex1=exp(-ci*2.0_r8*pi*eni*scaley)
   ex2=exp(ci*2.0_r8*pi*eni*scaley*c0)
   ykfac(:,1)=cone
   ykfac(1,2)=ex2
   do ix=2,n
      ykfac(ix,2)=ykfac(ix-1,2)*ex1
   enddo
   do iy=3,n
      ykfac(:,iy)=ykfac(:,iy-1)*ykfac(:,2)
   enddo
   ex1=exp(-ci*pi*(1.0_r8-mod(n,2)*eni))
   ex2=exp(ci*pi*(1.0_r8-mod(n,2)*eni)*scaley)
   ex3=exp(-ci*pi*(1.0_r8-mod(n,2)*eni)*scaley*c0)
   yfac(1,1)=ex3*eni
   do iy=2,n
      yfac(1,iy)=yfac(1,iy-1)*ex1
   enddo
   do ix=2,n
      yfac(ix,:)=yfac(ix-1,:)*ex2
   enddo
   end subroutine setupfacs

   subroutine initfft(nin)
   integer(kind=i4), intent(in) :: nin
   include 'fftw3.f03'
   integer :: iflags
   if (n.ne.0) call cleanfft
   n=nin
   pi=4.0_r8*atan(1.0_r8)
   p1=fftw_alloc_complex(int(n*n,c_size_t))
   call c_f_pointer(p1,ftmp1,[n,n])
   p2=fftw_alloc_complex(int(n*n,c_size_t))
   call c_f_pointer(p2,ftmp2,[n,n])
   iflags=FFTW_MEASURE
   iflags=ior(iflags,FFTW_DESTROY_INPUT)
   call dfftw_plan_many_dft(iplanx,1,n,n,ftmp1,n,1,n,ftmp2,n,1,n, &
      FFTW_FORWARD,iflags)
   call dfftw_plan_many_dft(iplanxi,1,n,n,ftmp1,n,1,n,ftmp2,n,1,n, &
      FFTW_BACKWARD,iflags)
   call dfftw_plan_many_dft(iplany,1,n,n,ftmp1,n,n,1,ftmp2,n,n,1, &
      FFTW_FORWARD,iflags)
   call dfftw_plan_many_dft(iplanyi,1,n,n,ftmp1,n,n,1,ftmp2,n,n,1, &
      FFTW_BACKWARD,iflags)
   end subroutine initfft

   subroutine cleanfft
   include 'fftw3.f03'
   n=0
   call dfftw_destroy_plan(iplanx)
   call dfftw_destroy_plan(iplanxi)
   call dfftw_destroy_plan(iplany)
   call dfftw_destroy_plan(iplanyi)
   call fftw_free(p1)
   call fftw_free(p2)
   end subroutine cleanfft

end module fftwrotate_m
