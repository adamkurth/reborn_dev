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
!        print *, 'f f1', f1
        out = out + real(f1*conjg(f1))
        do rj=ri+1,nr
            r2 = rvecs(:,rj)
            f2(:) = ff(:, fidx(rj)+1)
!            print *, 'f f2', f2
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