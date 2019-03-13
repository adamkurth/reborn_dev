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
   p=1.0_r8
   p0=p
   call squarediff(p,n,n,3,3,2,2)
   write (*,*) 3,2
   write (*,*) ' p '
   write (*,'(10f7.1)' ) p
   write (*,*) ' p0 '
   write (*,'(10f7.1)' ) p0
   end program junk

