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

subroutine debye(rvecs, qmags, fidx, ff, out)
    ! Calculate intensity from Debye formula: I(q) = sum_ij f_i(q) f_j(q)* sin(qr)/qr
    ! rvecs is an array of vectors size (3, Nr) (xyz contiguous)
    ! qmags is an array of q magnitudes size (Nq)
    ! fidx is an array of integer indices, one for each r vector.  size(Nr)
    ! ff is an array of complex scattering factors of size (Nq, Nz).
    ! out is the intensity to be calculated
    implicit none
    integer(kind=4), intent(in) :: fidx(:)
    real(kind=8),    intent(in) :: rvecs(:,:), qmags(:)
    complex(kind=8), intent(in) :: ff(:, :)
    real(kind=8),    intent(inout) :: out(:)
    real(kind=8) :: r1(3), r2(3), rij, sinc, qr
    complex(kind=8), allocatable :: f1(:), f2(:)
    integer(kind=4) :: nq, nr, ri, rj, qi
    nr = size(rvecs, 2)
    nq = size(qmags, 1)
    allocate(f1(nq))
    allocate(f2(nq))
    do ri=1,nr
        r1 = rvecs(:,ri)
        f1(:) = ff(:, fidx(ri)+1)
        out = out + real(f1*conjg(f1))
        do rj=ri+1,nr
            r2 = rvecs(:,rj)
            f2(:) = ff(:, fidx(rj)+1)
            rij = sqrt(sum((r1-r2)**2))
            !$OMP parallel default(None) &
            !$OMP private(qi,qr,sinc) &
            !$OMP shared(out,rij,f1,f2,nq,qmags)
            !$OMP do schedule(static)
            do qi=1,nq
                sinc = 1.0
                qr = qmags(qi)*rij
                if (qr /= 0.0) sinc = sin(qr)/qr
                out(qi) = out(qi) + 2*real(f1(qi)*conjg(f2(qi)))*sinc
            end do
            !$OMP enddo
            !$OMP end parallel
        end do
    end do
    deallocate(f1)
    deallocate(f2)
end subroutine debye


subroutine profile_stats(pattern, q, weights, n_bins, q_min, q_max, sum, sum2, w_sum)
    ! Calculate radial profile statistics.
    !
    ! pattern: Flattened 1D array of scattering intensities (do not correct for polarization, solid angle)
    ! q: Flattened 1D array of q vectors that correspond to the above scattering intensities
    ! weights: For weighted average.  This should be the product of the mask (if 0 means ignore), the polarization
    !          factor, the solid angle of the pixel, and any other relevant weighting.
    ! n_bins: How many bins in the 1D profile.
    ! q_min: The *center* position of the minimum q bin.
    ! q_max: The *center* position of the maximum q bin.
    ! sum: Profile with sum of intensities.
    ! sum2: Profile with sum of intensities squared.
    ! w_sum: Sum of the weights.
    implicit none
    integer(kind=4), intent(in)    :: n_bins
    real(kind=8),    intent(in)    :: pattern(:), q(:), weights(:), q_min, q_max
    real(kind=8),    intent(inout) :: sum(:), sum2(:), w_sum(:)
    real(kind=8)                   :: qm, dq
    integer(kind=4)                :: i, j, npat
    npat = size(pattern, 1)
    dq = (q_max - q_min) / (n_bins - 1)
    qm = q_min - dq/2
    do i=1, npat
        j = floor((q(i)-qm)/dq) + 1
        if (j > n_bins) then
            cycle
        end if
        if (j < 1) then
            cycle
        end if
!        j = min(j, n_bins)
!        j = max(j, 1)
        sum(j) = sum(j) + pattern(i)*weights(i)
        sum2(j) = sum2(j) + pattern(i)**2*weights(i)
        w_sum(j) = w_sum(j) + weights(i)
    end do
end subroutine profile_stats


subroutine profile_indices(q, n_bins, q_min, q_max, indices)
    ! Fetch the indices that go along with profile_stats
    implicit none
    integer(kind=4), intent(in)    :: n_bins
    real(kind=8),    intent(in)    :: q(:), q_min, q_max
    integer(kind=4), intent(inout) :: indices(:)
    real(kind=8)                   :: dq, qm
    integer(kind=4)                :: i, j, npat
    npat = size(q, 1)
    dq = (q_max - q_min) / (n_bins - 1)
    qm = q_min - dq/2
    do i=1, npat
        j = floor((q(i)-qm)/dq) + 1
        if (j > n_bins) then
            j = 0
            cycle
        end if
        if (j < 1) then
            j = 0
            cycle
        end if
!        j = min(j, n_bins)
!        j = max(j, 1)
        indices(i) = j
    end do
end subroutine profile_indices


subroutine profile_stats_indexed(pattern, indices, weights, sum, sum2, w_sum)
    ! Same as profile_stats but with indices pre-calculated
    implicit none
    integer(kind=4), intent(in)    :: indices(:)
    real(kind=8),    intent(in)    :: pattern(:), weights(:)
    real(kind=8),    intent(inout) :: sum(:), sum2(:), w_sum(:)
    integer(kind=4)                :: i, j, npat
    npat = size(pattern, 1)
    do i=1, npat
        j = indices(i)
        if (j < 1) then
            cycle
        end if
        sum(j) = sum(j) + pattern(i)*weights(i)
        sum2(j) = sum2(j) + pattern(i)**2*weights(i)
        w_sum(j) = w_sum(j) + weights(i)
    end do
end subroutine profile_stats_indexed


subroutine profile_stats_avg(sum, sum2, w_sum, meen, std)
    ! Given output of profile_stats, calculate the weighted mean and weighted standard deviation
    implicit none
    real(kind=8),    intent(in) :: sum(:), sum2(:), w_sum(:)
    real(kind=8), intent(inout) :: meen(:), std(:)
    real(kind=8)                :: m, s
    integer(kind=4)             :: i, n
    n = size(sum, 1)
    do i=1, n
        if (w_sum(i) == 0) then
            meen(i) = 0
            std(i) = 0
        else
            m = sum(i) / w_sum(i)
            meen(i) = m
            s = sum2(i) / w_sum(i)
            std(i) = sqrt(s - m*m)
        end if
    end do
end subroutine profile_stats_avg