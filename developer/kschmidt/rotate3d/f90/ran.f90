module random2
   implicit none
   integer, private, parameter :: i4=selected_int_kind(9)
   integer, private, parameter :: i8=selected_int_kind(15)
   integer, private, parameter :: r8=selected_real_kind(15,9)

contains
   subroutine ran1(rn,irn)
!
! multiplicative congruential with additive constant
!
   integer(kind=i8), parameter :: mask24 = ishft(1_i8,24)-1
   integer(kind=i8), parameter :: mask48 = ishft(1_i8,48_i8)-1_i8
   real(kind=r8),    parameter :: twom48=2.0_r8**(-48)
   integer(kind=i8), parameter :: mult1 = 44485709377909_i8
   integer(kind=i8), parameter :: m11 = iand(mult1,mask24)
   integer(kind=i8), parameter :: m12 = iand(ishft(mult1,-24),mask24)
   integer(kind=i8), parameter :: iadd1 = 96309754297_i8
   integer(kind=i8) :: irn
   real(kind=r8) :: rn
   integer(kind=i8) :: is1,is2
   is2 = iand(ishft(irn,-24),mask24)
   is1 = iand(irn,mask24)
   irn = iand(ishft(iand(is1*m12+is2*m11,mask24),24)+is1*m11+iadd1,mask48)
   rn = ior(irn,1_i8)*twom48
   return
   end subroutine ran1

   subroutine ran2(rn,irn)
!
! multiplicative congruential with additive constant
!
   integer(kind=i8), parameter :: mask24 = ishft(1_i8,24)-1
   integer(kind=i8), parameter :: mask48 = ishft(1_i8,48)-1
   real(kind=r8),    parameter :: twom48=2.0_r8**(-48)
   integer(kind=i8), parameter :: mult2 = 34522712143931_i8
   integer(kind=i8), parameter :: m21 = iand(mult2,mask24)
   integer(kind=i8), parameter :: m22 = iand(ishft(mult2,-24),mask24)
   integer(kind=i8), parameter :: iadd2 = 55789347517_i8
   integer(kind=i8) :: irn
   real(kind=r8) :: rn
   integer(kind=i8) :: is1,is2
   is2 = iand(ishft(irn,-24),mask24)
   is1 = iand(irn,mask24)
   irn = iand(ishft(iand(is1*m22+is2*m21,mask24),24)+is1*m21+iadd2,mask48)
   rn = ior(irn,1_i8)*twom48
   return
   end subroutine ran2
end module random2

module random
   implicit none
   integer, private, parameter :: i4=selected_int_kind(9)
   integer, private, parameter :: i8=selected_int_kind(15)
   integer, private, parameter :: r8=selected_real_kind(15,9)
   integer(kind=i8), private, parameter :: mask48 = ishft(1_i8,48)-1
   integer(kind=i8), private, save :: irn = 1_i8

contains   
   function randn(n)
!
! return an array of random variates (0,1)
!
   use random2
   integer(kind=i4) :: n
   real(kind=r8), dimension(n) :: randn
   integer(kind=i4) :: i
   do i=1,n
      call ran1(randn(i),irn)
   enddo
   return
   end function randn

   function gaussian(n)
!
! return an array of gaussian random variates with variance 1
!
   integer(kind=i4) :: n
   real(kind=r8), dimension(n) :: gaussian
   real(kind=r8), dimension(2*((n+1)/2)) :: rn
   real(kind=r8) :: rn1,rn2,arg,x1,x2,x3,pi
   integer(kind=i4) :: i
   pi=4.0_r8*atan(1.0_r8)
   rn=randn(size(rn))
   do i=1,n/2
      rn1=rn(2*i-1)
      rn2=rn(2*i)
      arg=2.0_r8*pi*rn1
      x1=sin(arg)
      x2=cos(arg)
      x3=sqrt(-2.0_r8*log(rn2))
      gaussian(2*i-1)=x1*x3
      gaussian(2*i)=x2*x3
   enddo
   if (mod(n,2).ne.0) then
      rn1=rn(n)
      rn2=rn(n+1)
      arg=2.0_r8*pi*rn1
      x2=cos(arg)
      x3=sqrt(-2.0_r8*log(rn2))
      gaussian(n)=x2*x3
      endif
   return
   end function gaussian

   subroutine setrn(irnin)
!
! set the seed
!
   integer(kind=i8) :: irnin
   irn=iand(irnin,mask48)
   return
   end subroutine setrn

   subroutine savern(irnout)
!
! save the seed
!
   integer(kind=i8) :: irnout
   irnout=irn
   return
   end subroutine savern

end module random
