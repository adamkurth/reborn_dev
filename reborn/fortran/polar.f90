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

subroutine polar_binning(nq, qbin_size, qmin, np, pbin_size, pmin, qs, phis, data, mask, dsum, count)
    implicit none
    real(kind=8), parameter :: pi = 4.d0 * datan(1.d0)
    integer(kind=8), intent(in) :: nq, np
    real(kind=8), intent(in) :: qbin_size, qmin, pbin_size, pmin
    real(kind=8), intent(in) :: qs(:), phis(:), data(:), mask(:)
    integer(kind=8), intent(out), dimension(nq * np) :: count
    real(kind=8), intent(out), dimension(nq * np) :: dsum
    integer(kind=8) :: i, ii, q_index, p_index
    real(kind=8) :: q, p
    dsum = 0.d0
    count = 0.d0
    do i = 1, size(data)
        if (mask(i) == 0) cycle
        ! compute q index
        q = qs(i)
        q_index = int(floor((q - qmin) / qbin_size))
        if (q_index < 0) cycle
        if (q_index >= nq) cycle
        ! compute phi index
        ! p = modulo(phis(i), 2 * pi)
        ! p_index = int(floor((p - pmin) / pbin_size))
        p = phis(i)
        p_index = modulo(int(floor((p - pmin) / pbin_size)), np)
        if (p_index < 0) cycle
        if (p_index >= np) cycle
        ! bin data
        ii = np * q_index + p_index + 1
        count(ii) = count(ii) + 1
        dsum(ii) = dsum(ii) + data(i)
    end do
end subroutine polar_binning