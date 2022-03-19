! This file is part of reborn <https://kirianlab.gitlab.io/reborn/>.
!
! reborn is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! reborn is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License
! along with reborn.  If not, see <https://www.gnu.org/licenses/>.

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