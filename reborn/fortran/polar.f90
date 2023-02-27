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
! 2023/02/27 (rca): Refactored to be more consistent.
!**********************************************************************
    implicit none
    contains

    subroutine q_bin_indices(nq, qbin_size, qmin, qs, q_index)
    ! Fetch polar binning q index mapping
    !
    ! INPUTS:
    ! nq: Number of q bins.
    ! qbin_size: Size of q bin.
    ! qmin: The *center* position of the minimum q bin.
    ! qs: Flattened 1D array of q vectors that correspond to the above scattering intensities.
    !
    ! RETURNS:
    ! q_index: q polar mapping index (value of 0 means out of bounds)
        implicit none
        real(kind=8), parameter :: tp = 8.d0 * datan(1.d0)  ! 2 * pi
        integer(kind=8), intent(in) :: nq
        real(kind=8), intent(in) :: qbin_size, qmin, qs(:)
        integer(kind=8), intent(out) :: q_index(size(qs))
        integer(kind=8) :: i, qi
        real(kind=8) :: q
        q_index = 0.d0
        do i = 1, size(qs)
            q = qs(i)
            qi = int(floor((q - qmin) / qbin_size)) + 1
            q_index(i) = qi
        end do
        where (q_index < 0)
            q_index = -1
        end where
        where (q_index > nq)
            q_index = -1
        end where
    end subroutine q_bin_indices

    subroutine p_bin_indices(np, pbin_size, pmin, phis, p_index)
    ! Fetch polar binning p index mapping
    !
    ! INPUTS:
    ! np: Number of phi bins.
    ! pbin_size: Size of phi bin.
    ! pmin: The *center* position of the minimum phi bin.
    ! phis: Flattened 1D array of phi angles that correspond to the above scattering intensities.
    !
    ! RETURNS:
    ! p_index: phi polar mapping index (value of -1 means out of bounds)
        implicit none
        real(kind=8), parameter :: tp = 8.d0 * datan(1.d0)  ! 2 * pi
        integer(kind=8), intent(in) :: np
        real(kind=8), intent(in) :: pbin_size, pmin, phis(:)
        integer(kind=8), intent(out) :: p_index(size(phis))
        integer(kind=8) :: i, pi
        real(kind=8) :: p
        p_index = 0.d0
        do i = 1, size(phis)
            p = modulo(phis(i), tp)
            pi = int(floor((p - pmin) / pbin_size)) + 1
            p_index(i) = pi
        end do
        where (p_index < 0)
            p_index = -1
        end where
        where (p_index > np)
            p_index = -1
        end where
    end subroutine p_bin_indices

    subroutine bin_sum(nq, np, q_index, p_index, array, a_sum)
    ! Calculate polar-binned sum of given array.
    !
    ! INPUTS:
    ! nq: Number of q bins.
    ! np: Number of phi bins.
    ! array: Flattened 1D array to bin.
    ! bsum: Polar binned sum of array.
        implicit none
        integer(kind=8), intent(in) :: nq, q_index(:), np, p_index(:)
        real(kind=8), intent(in) :: array(:)
        real(kind=8), intent(out) :: a_sum(nq * np)
        integer(kind=8) :: i, ii
        a_sum = 0.d0
        do i = 1, size(array)
            ii = np * (q_index(i) - 1) + p_index(i)
            a_sum(ii) = a_sum(ii) + array(i)
        end do
    end subroutine bin_sum

    subroutine bin_mean(nq, qbin_size, qmin, np, pbin_size, pmin, qs, phis, weights, data, pmean, pmask)
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
    ! pmean: Binned mean of data.
    ! pmask: Binned mean of weights.
        implicit none
        real(kind=8), parameter :: pi = 4.d0 * datan(1.d0)
        integer(kind=8), intent(in) :: nq, np
        real(kind=8), intent(in) :: qbin_size, qmin, pbin_size, pmin, qs(:), phis(:), weights(:), data(:)
        real(kind=8), intent(out), dimension(nq * np) :: pmask, pmean
        integer(kind=8) :: p_index(size(qs)), q_index(size(phis))
        real(kind=8) :: csum(nq * np), dsum(nq * np), wsum(nq * np), mask(size(data))
        call q_bin_indices(nq, qbin_size, qmin, qs, q_index)
        call p_bin_indices(np, pbin_size, pmin, phis, p_index)
        mask = merge(0, 1, weights <= 0) * &
                merge(0, 1, q_index < 0) * merge(0, 1, q_index > nq) * &
                merge(0, 1, p_index < 0) * merge(0, 1, p_index > np)
        call bin_sum(nq, np, q_index, p_index, mask, csum)
        call bin_sum(nq, np, q_index, p_index, data * mask, dsum)
        call bin_sum(nq, np, q_index, p_index, weights * mask, wsum)
        where (csum /= 0)
            pmean = dsum / csum
            pmask = wsum / csum
        elsewhere
            pmean = 0
            pmask = 0
        end where
    end subroutine bin_mean

    subroutine stats(pattern, q, p, weights, n_q_bins, q_min, q_max, n_p_bins, p_min, p_max, sum_, sum2, w_sum, initialize)
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
    end subroutine stats

    subroutine stats_mean(sum_, sum2, w_sum, meen, std)
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
    end subroutine stats_mean

end module polar_binning
