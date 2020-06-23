subroutine max_pair_distance(vectors, out)
    implicit none
    real(kind=8), intent(inout) :: out(:)
    real(kind=8), intent(in) :: vectors(:,:)
    real(kind=8) :: d_max, d_vec(3), d
    integer(kind=4) :: n, i, j
    n = size(vectors, 2)
    d_max = 0
    do i=1,n
        do j=i,n
            d_vec = vectors(:,i) - vectors(:,j)
            d = d_vec(1)*d_vec(1) + d_vec(2)*d_vec(2) + d_vec(3)*d_vec(3)
            if (d > d_max) then
                d_max = d
            end if
        end do
    enddo
    out(1) = sqrt(d_max)
end subroutine max_pair_distance
