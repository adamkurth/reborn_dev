subroutine polar_binning(nq, np, qindex, pindex, data, dsum, count)
    implicit none
    integer(kind=8), intent(in) :: nq, np
    integer(kind=8), intent(in) :: qindex(:), pindex(:)
    real(kind=8), intent(in) :: data(:)
    integer(kind=8), intent(out) :: count(0:nq * np - 1)
    real(kind=8), intent(out) :: dsum(0:nq * np - 1)
    integer(kind=8) :: i, ii, q, p
    dsum = 0
    count = 0
    do i = 0, size(data)
        q = qindex(i + 1)
        p = pindex(i + 1)
        ii = q + nq * p
        count(ii) = count(ii) + 1
        dsum(ii) = dsum(ii) + data(i + 1)
    end do
end subroutine polar_binning