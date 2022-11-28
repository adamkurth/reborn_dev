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
        ! TODO: Needs documentation
        ! It appears that if both mask and weights are equal to 1 everywhere, then the returned pmean is actually the
        ! sum of intensities.  Is that expected?  The use case should be explained.
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

    subroutine polar_stats(pattern, q, p, weights, n_q_bins, q_min, q_max, n_p_bins, p_min, p_max, sum_, sum2, w_sum, initialize)
    ! Calculate polar-binned statistics:
    ! 1) The weighted sum of intensities.
    ! 2) The weighted sum of squared intensities.
    ! 3) The sum of weights.
    ! With the above you may calculate a weighted average and standard deviation.
    !
    ! INPUTS:
    ! pattern: Flattened 1D array of scattering intensities (do not correct for polarization, solid angle)
    ! q: Flattened 1D array of q vectors that correspond to the above scattering intensities
    ! p: Flattened 1D array of phi angles that correspond to the above scattering intensities
    ! weights: For weighted average.  This should be the product of the mask (if 0 means ignore), the polarization
    !          factor, the solid angle of the pixel, and any other relevant weighting.
    ! n_q_bins: Number of q bins.
    ! q_min: The *center* position of the minimum q bin.
    ! q_max: The *center* position of the maximum q bin.
    ! n_p_bins: Number of phi bins.
    ! p_min: The *center* position of the minimum phi bin.
    ! p_max: The *center* position of the maximum phi bin.
    ! sum: Sum of intensities multiplied by the weights.
    ! sum2: Sum of intensities squared multiplied by the weights.
    ! w_sum: Sum of the weights.
    ! initialize: Set to 0 if you do not want to zero out the output arrays (e.g. if you want to add to existing arrays)
    implicit none
    real(kind=8), parameter :: tp = 8.d0 * datan(1.d0)
    integer(kind=4), intent(in)    :: n_q_bins, n_p_bins, initialize
    real(kind=8),    intent(in)    :: pattern(:), q(:), p(:), weights(:), q_min, q_max, p_min, p_max
    real(kind=8),    intent(inout) :: sum_(:,:), sum2(:,:), w_sum(:,:)
    real(kind=8)                   :: qm, dq, pm, dp, pp
    integer(kind=4)                :: i, j, k, npat
    if (initialize /= 0) then
        sum_ = 0
        sum2 = 0
        w_sum = 0
    end if
    npat = size(pattern, 1)
    dq = (q_max - q_min) / (n_q_bins - 1)
    qm = q_min - dq/2
    dp = (p_max - p_min) / (n_p_bins - 1)
    pm = p_min - dp/2
    do i=1, npat
        pp = modulo(p(i), tp)
        j = floor((q(i)-qm)/dq) + 1
        k = floor((pp-pm)/dp) + 1
        if (j > n_q_bins) cycle
        if (j < 1) cycle
        if (k > n_p_bins) cycle
        if (k < 1) cycle
        sum_(k, j) = sum_(k, j) + pattern(i)*weights(i)
        sum2(k, j) = sum2(k, j) + pattern(i)**2*weights(i)
        w_sum(k, j) = w_sum(k, j) + weights(i)
    end do
    end subroutine polar_stats

    subroutine polar_stats_avg(sum_, sum2, w_sum, meen, std)
    ! Given output of profile_stats, calculate the weighted mean and weighted standard deviation
    implicit none
    real(kind=8),    intent(in) :: sum_(:,:), sum2(:,:), w_sum(:,:)
    real(kind=8), intent(inout) :: meen(:,:), std(:,:)
    real(kind=8)                :: m, s
    integer(kind=4)             :: i, j, n, p
    n = size(sum_, 1)
    p = size(sum_, 2)
    do j=1, p
    do i=1, n
        if (w_sum(i,j) == 0) then
            meen(i,j) = 0
            std(i,j) = 0
        else
            m = sum_(i,j) / w_sum(i,j)
            meen(i,j) = m
            s = sum2(i,j) / w_sum(i,j)
            std(i,j) = sqrt(s - m*m)
        end if
    end do
    end do
    end subroutine polar_stats_avg

end module polar_binning
