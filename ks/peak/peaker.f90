module peaker
   implicit none
!   integer, private, parameter :: 4=selected_int_kind(9)
!   integer, private, parameter :: 8=selected_real_kind(15,9)

contains
   subroutine squarediff(p,npx,npy,nosx,nosy,nisx,nisy)
!
! p(npx,npy) input array, overwritten to give output array
! sums pixels in outer square from i-nosx to i+nosx, j-nosy to j+nosy
! not contained in inner square from i-nisx to i+nisx, j-nisy to j+nisy
!
      real(kind=8), intent(inout) :: p(npx,npy)
      integer(kind=4), intent(in) :: npx,npy,nosx,nosy,nisx,nisy
      real(kind=8) :: cumx(0:npx,npy),cum1y(npx,0:npy),cum2y(npx,0:npy)
      real(kind=8) :: sqox(npx,npy),sqix(npx,npy)
      integer(kind=4) :: ix,iy
      cumx(0,:)=0.0_8
      do ix=1,npx
         cumx(ix,:)=cumx(ix-1,:)+p(ix,:)
      enddo
      do ix=1,npx
         sqix(ix,:)=cumx(min(npx,ix+nisx),:)-cumx(max(0,ix-nisx-1),:)
         sqox(ix,:)=cumx(min(npx,ix+nosx),:)-cumx(max(0,ix-nosx-1),:)
      enddo
      cum1y(:,1)=0.0_8
      cum2y(:,1)=0.0_8
      do iy=1,npy
         cum1y(:,iy)=cum1y(:,iy-1)+sqix(1:npx,iy)
         cum2y(:,iy)=cum2y(:,iy-1)+sqox(1:npx,iy)
      enddo
      do iy=1,npy
         p(:,iy)=cum2y(:,min(npy,iy+nosy))-cum2y(:,max(0,iy-nosy-1)) &
            -cum1y(:,min(npy,iy+nisy))+cum1y(:,max(0,iy-nisy-1))
      enddo

   end subroutine squarediff

end module peaker
