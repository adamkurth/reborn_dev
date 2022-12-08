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
! 2022/12/06 (rca): Fixed polar_mean logic to calculate weighted mean
!                   correctly. Added subroutine to compute bin indices.
!**********************************************************************
    implicit none
    contains

    subroutine polar_bin_indices(nq, qbin_size, qmin, np, pbin_size, pmin, qs, phis, q_index, p_index)
    ! Fetch polar binning index mapping
    !
    ! INPUTS:
    ! nq: Number of q bins.
    ! qbin_size: Size of q bin.
    ! qmin: The *center* position of the minimum q bin.
    ! np: Number of phi bins.
    ! pbin_size: Size of phi bin.
    ! pmin: The *center* position of the minimum phi bin.
    ! qs: Flattened 1D array of q vectors that correspond to the above scattering intensities.
    ! phis: Flattened 1D array of phi angles that correspond to the above scattering intensities.
    !
    ! RETURNS:
    ! q_index: q polar mapping index (value of 0 means out of bounds)
    ! p_index: phi polar mapping index (value of 0 means out of bounds)
        implicit none
        real(kind=8), parameter :: tp = 8.d0 * datan(1.d0)  ! 2 * pi
        integer(kind=8), intent(in) :: nq, np
        real(kind=8), intent(in) :: qbin_size, qmin, pbin_size, pmin, qs(:), phis(:)
        integer(kind=8), intent(out) :: q_index(size(qs)), p_index(size(phis))
        integer(kind=8) :: i, pi, qi
        real(kind=8) :: p, q
        do i = 1, size(qs)
            q = qs(i)
            qi = int(floor((q - qmin) / qbin_size))
            if (qi < 0) then
                qi = 0
                cycle
            end if
            if (qi > nq) then
                qi = 0
                cycle
            end if
            q_index(i) = qi
        end do
        do i = 1, size(phis)
            p = modulo(phis(i), tp)
            pi = int(floor((p - pmin) / pbin_size))
            if (pi < 0) then
                pi = 0
                cycle
            end if
            if (pi > np) then
                pi = 0
                cycle
            end if
            p_index(i) = pi
        end do
    end subroutine polar_bin_indices

    subroutine polar_bin_avg(nq, qbin_size, qmin, np, pbin_size, pmin, qs, phis, weights, data, mask, pmean, pmask)
    ! Calculate polar-binned mean:
    ! 1) weighted intensities.
    ! 2) weights.
    !
    ! INPUTS:
    ! nq: Number of q bins.
    ! qbin_size: Size of q bin.
    ! qmin: The *center* position of the minimum q bin.
    ! np: Number of phi bins.
    ! pbin_size: Size of phi bin.
    ! pmin: The *center* position of the minimum phi bin.
    ! qs: Flattened 1D array of q vectors that correspond to the above scattering intensities.
    ! phis: Flattened 1D array of phi angles that correspond to the above scattering intensities.
    ! weights: For weighted average. This should be the product of the mask (if 0 means ignore), the polarization
    !          factor, the solid angle of the pixel, and any other relevant weighting.
    ! data: Flattened 1D array of scattering intensities (do not correct for polarization, solid angle).
    ! mask: Flattened 1D array of mask (0 to ignore; 1 to include).
    ! pmean: Polar binned mean of data (each data point is multiplied by the corresponding weight).
    ! pmask: Polar binned mean of weights multiplied by mask.
        implicit none
        real(kind=8), parameter :: pi = 4.d0 * datan(1.d0)
        integer(kind=8), intent(in) :: nq, np
        real(kind=8), intent(in) :: qbin_size, qmin, pbin_size, pmin, qs(:), phis(:), weights(:), data(:), mask(:)
        real(kind=8), intent(out), dimension(nq * np) :: pmask, pmean
        integer(kind=8) :: count(nq * np), i, ii, p_index(size(qs)), q_index(size(phis))
        real(kind=8) :: dsum(nq * np), p, q, wsum(nq * np)
        dsum = 0.d0
        wsum = 0.d0
        count = 0
        call polar_bin_indices(nq, qbin_size, qmin, np, pbin_size, pmin, qs, phis, q_index, p_index)
        do i = 1, size(data)
            if (mask(i) == 0) cycle
            q = q_index(i)
            if (q == 0) cycle
            p = p_index(i)
            if (p == 0) cycle
            ! bin data
            ii = np * q + p + 1
            count(ii) = count(ii) + 1
            dsum(ii) = dsum(ii) + data(i) * weights(i)
            wsum(ii) = wsum(ii) + weights(i)
        end do
        where (count /= 0)
            pmean = dsum / count
            pmask = weights / count
        elsewhere
            pmean = 0
            pmask = 0
        end where
    end subroutine polar_bin_avg

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
        sum_(k, j) = sum_(k, j) + pattern(i) * weights(i)
        sum2(k, j) = sum2(k, j) + pattern(i) ** 2 * weights(i)
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
