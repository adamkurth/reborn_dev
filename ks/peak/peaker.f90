module peaker
   implicit none
   integer, private, parameter :: i4=selected_int_kind(9)
   integer, private, parameter :: r8=selected_real_kind(15,9)

contains
   subroutine squarediff(p,npx,npy,nosx,nosy,nisx,nisy)
!
! p(npx,npy) input array, overwritten to give output array
! sums pixels in outer square from i-nosx to i+nosx, j-nosy to j+nosy
! not contained in inner square from i-nisx to i+nisx, j-nisy to j+nisy
!
      real(kind=r8), intent(inout) :: p(npx,npy)
      integer(kind=i4), intent(in) :: npx,npy,nosx,nosy,nisx,nisy
      real(kind=r8) :: cumx(0:npx,npy),cum1y(npx,0:npy),cum2y(npx,0:npy)
      real(kind=r8) :: sqox(npx,npy),sqix(npx,npy)
      integer(kind=i4) :: ix,iy
      cumx(0,:)=0.0_r8
      do ix=1,npx
         cumx(ix,:)=cumx(ix-1,:)+p(ix,:)
      enddo
      do ix=1,npx
         sqix(ix,:)=cumx(min(npx,ix+nisx),:)-cumx(max(0,ix-nisx-1),:)
         sqox(ix,:)=cumx(min(npx,ix+nosx),:)-cumx(max(0,ix-nosx-1),:)
      enddo
      cum1y(:,1)=0.0_r8
      cum2y(:,1)=0.0_r8
      do iy=1,npy
         cum1y(:,iy)=cum1y(:,iy-1)+sqix(:,iy)
         cum2y(:,iy)=cum2y(:,iy-1)+sqox(:,iy)
      enddo
      do iy=1,npy
         p(:,iy)=cum2y(:,min(npy,iy+nosy))-cum2y(:,max(0,iy-nosy-1)) &
            -cum1y(:,min(npy,iy+nisy))+cum1y(:,max(0,iy-nisy-1))
      enddo
   end subroutine squarediff

end module peaker

   program junk
   use peaker
   implicit none
   integer, parameter :: i4=selected_int_kind(9)
   integer, parameter :: r8=selected_real_kind(15,9)
   real(kind=r8), allocatable :: p(:,:),p0(:,:)
   integer(kind=i4) :: n,i,j
   n=10
   allocate(p(n,n),p0(n,n))
   do i=1,n
      do j=1,n
         p(j,i)=n*j+i
      enddo
   enddo
!   p=1.0_r8
   p0=p
   call squarediff(p,n,n,3,3,2,2)
   write (*,*) 3,2
   write (*,*) ' p '
   write (*,'(10f7.1)' ) p
   write (*,*) ' p0 '
   write (*,'(10f7.1)' ) p0
   end program junk

