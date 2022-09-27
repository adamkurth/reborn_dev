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

module polar_binning
!**********************************************************************
! Subroutines to polar bin data with solid angle corrections.
!
! EDITS
! ---------------------------------------------------------------------
! 2022/09/16 (rca): Created module from standalone subroutine.
!**********************************************************************
    implicit none
    contains

    subroutine polar_mean(nq, qbin_size, qmin, np, pbin_size, pmin, qs, phis, weight, data, mask, pmean, pmask)
        implicit none
        real(kind=8), parameter :: pi = 4.d0 * datan(1.d0)
        integer(kind=8), intent(in) :: nq, np
        real(kind=8), intent(in) :: qbin_size, qmin, pbin_size, pmin, qs(:), phis(:), weight(:), data(:), mask(:)
        real(kind=8), intent(out), dimension(nq * np) :: pmask, pmean
        integer(kind=8) :: count(nq * np), i, ii, p_index, q_index
        real(kind=8) :: dsum(nq * np), p, q, weights(nq * np)
        dsum = 0.d0
        count = 0
        weights = 0.d0
        do i = 1, size(data)
            if (mask(i) == 0) cycle
            q = qs(i)
            q_index = int(floor((q - qmin) / qbin_size))
            if (q_index < 0) cycle
            if (q_index >= nq) cycle
!            p = phis(i)
!            p_index = modulo(int(floor((p - pmin) / pbin_size)), np)
            p = modulo(phis(i), 2 * pi)
            p_index = int(floor((p - pmin) / pbin_size))
            if (p_index < 0) cycle
            if (p_index >= np) cycle
            ! bin data
            ii = np * q_index + p_index + 1
            count(ii) = count(ii) + 1
            dsum(ii) = dsum(ii) + data(i)
            weights(ii) = weights(ii) + weight(i)
        end do
        where (count /= 0)
            pmask = weights / count
        elsewhere
            pmask = 0
        end where
        where (pmask /= 0)
            pmean = dsum / pmask
        elsewhere
            pmean = 0
        end where
    end subroutine polar_mean

end module polar_binning
