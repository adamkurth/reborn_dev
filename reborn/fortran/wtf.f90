! Suddenly, it became necessary to have this main program in order for these to compile with f2py.  Nothing in the
! python or fortran code changed in reborn.  Not sure if this is due to numpy or conda, but the failure began on
! May 23, 2020 when compiling via the gitlab runner.  This is possibly helpful:
!
!  https://github.com/numpy/numpy/issues/14222
!
! I wonder if the main program will now break compilation on other systems.....
!PROGRAM MAIN
!PRINT *,'Hello world'
!END PROGRAM MAIN

subroutine wtf(out1, out2, out3)
    implicit none
    real(kind=8), intent(inout) :: out1(:), out2(:,:), out3(:,:,:)
    out1(2) = 10
    out2(2,1) = 10
    out3(2,1,1) = 10
end subroutine wtf