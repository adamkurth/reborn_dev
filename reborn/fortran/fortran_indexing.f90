subroutine wtf(out1, out2, out3)
    implicit none
    real(kind=8), intent(inout) :: out1(:), out2(:,:), out3(:,:,:)
    out1(2) = 10
    out2(2,1) = 10
    out3(2,1,1) = 10
end subroutine wtf