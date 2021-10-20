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

subroutine trilinear_interpolation(densities, vectors, corners, deltas, out)
    implicit none
    real(kind=8), intent(inout) :: out(:)
    real(kind=8), intent(in) :: densities(:,:,:), vectors(:,:), corners(3), deltas(3)
    real(kind=8) :: i_f,j_f,k_f,x0,y0,z0,x1,y1,z1
    integer(kind=4) :: nn,i0,j0,k0,i1,j1,k1,ii,nx,ny,nz
    nn = size(out, 1)
    nx = size(densities, 1)
    ny = size(densities, 2)
    nz = size(densities, 3)
    !$OMP parallel default(None) private(ii,k_f,j_f,i_f,i0,j0,k0,i1,j1,k1,x0,y0,z0,x1,y1,z1) &
    !$OMP & shared(vectors,corners,deltas,densities,out,nn,nx,ny,nz)

    !$OMP do schedule(static)
    do ii=1,nn
        k_f = 1.0 + (vectors(1, ii) - corners(1)) / deltas(1)
        j_f = 1.0 + (vectors(2, ii) - corners(2)) / deltas(2)
        i_f = 1.0 + (vectors(3, ii) - corners(3)) / deltas(3)
        i0 = modulo(floor(i_f) - 1, nx) + 1
        j0 = modulo(floor(j_f) - 1, ny) + 1
        k0 = modulo(floor(k_f) - 1, nz) + 1
        i1 = modulo(i0, nx) + 1
        j1 = modulo(j0, ny) + 1
        k1 = modulo(k0, nz) + 1
        x0 = i_f - floor(i_f)
        y0 = j_f - floor(j_f)
        z0 = k_f - floor(k_f)
        x1 = 1.0 - x0
        y1 = 1.0 - y0
        z1 = 1.0 - z0
        out(ii) = out(ii) + &
                  densities(i0, j0, k0) * x1 * y1 * z1 + &
                  densities(i1, j0, k0) * x0 * y1 * z1 + &
                  densities(i0, j1, k0) * x1 * y0 * z1 + &
                  densities(i0, j0, k1) * x1 * y1 * z0 + &
                  densities(i1, j0, k1) * x0 * y1 * z0 + &
                  densities(i0, j1, k1) * x1 * y0 * z0 + &
                  densities(i1, j1, k0) * x0 * y0 * z1 + &
                  densities(i1, j1, k1) * x0 * y0 * z0
    enddo
    !$OMP enddo nowait
    !$OMP end parallel
end subroutine trilinear_interpolation

subroutine trilinear_interpolation_complex(densities, vectors, corners, deltas, out)
    implicit none
    complex(kind=8), intent(inout) :: out(:)
    complex(kind=8), intent(in) :: densities(:,:,:)
    real(kind=8), intent(in) :: vectors(:,:), corners(3), deltas(3)
    real(kind=8) :: i_f,j_f,k_f,x0,y0,z0,x1,y1,z1
    integer(kind=4) :: nn,i0,j0,k0,i1,j1,k1,ii,nx,ny,nz
    nn = size(out, 1)
    nx = size(densities, 1)
    ny = size(densities, 2)
    nz = size(densities, 3)
    !$OMP parallel default(None) private(ii,k_f,j_f,i_f,i0,j0,k0,i1,j1,k1,x0,y0,z0,x1,y1,z1) &
    !$OMP & shared(vectors,corners,deltas,densities,out,nn,nx,ny,nz)

    !$OMP do schedule(static)
    do ii=1,nn
        k_f = 1.0 + (vectors(1, ii) - corners(1)) / deltas(1)
        j_f = 1.0 + (vectors(2, ii) - corners(2)) / deltas(2)
        i_f = 1.0 + (vectors(3, ii) - corners(3)) / deltas(3)
        i0 = modulo(floor(i_f) - 1, nx) + 1
        j0 = modulo(floor(j_f) - 1, ny) + 1
        k0 = modulo(floor(k_f) - 1, nz) + 1
        i1 = modulo(i0, nx) + 1
        j1 = modulo(j0, ny) + 1
        k1 = modulo(k0, nz) + 1
        x0 = i_f - floor(i_f)
        y0 = j_f - floor(j_f)
        z0 = k_f - floor(k_f)
        x1 = 1.0 - x0
        y1 = 1.0 - y0
        z1 = 1.0 - z0
        out(ii) = out(ii) + &
                  densities(i0, j0, k0) * x1 * y1 * z1 + &
                  densities(i1, j0, k0) * x0 * y1 * z1 + &
                  densities(i0, j1, k0) * x1 * y0 * z1 + &
                  densities(i0, j0, k1) * x1 * y1 * z0 + &
                  densities(i1, j0, k1) * x0 * y1 * z0 + &
                  densities(i0, j1, k1) * x1 * y0 * z0 + &
                  densities(i1, j1, k0) * x0 * y0 * z1 + &
                  densities(i1, j1, k1) * x0 * y0 * z0
    enddo
    !$OMP enddo nowait
    !$OMP end parallel
end subroutine trilinear_interpolation_complex

subroutine trilinear_insertion_real(densities, weights, vectors, vals, corners, deltas)
    implicit none
    real(kind=8), intent(inout) :: densities(:,:,:)
    real(kind=8), intent(in) :: vals(:)
    real(kind=8) :: val
    real(kind=8), intent(inout) :: weights(:,:,:)
    real(kind=8), intent(in) :: vectors(:,:), corners(3), deltas(3)
    real(kind=8) :: i_f,j_f,k_f,x0,y0,z0,x1,y1,z1
    integer(kind=4) :: i0,j0,k0,i1,j1,k1,ii,nx,ny,nz,nn
    nn = size(vals, 1)
    nx = size(densities, 1)
    ny = size(densities, 2)
    nz = size(densities, 3)
    do ii=1,nn
        k_f = 1.0 + (vectors(1, ii) - corners(1)) / deltas(1)
        j_f = 1.0 + (vectors(2, ii) - corners(2)) / deltas(2)
        i_f = 1.0 + (vectors(3, ii) - corners(3)) / deltas(3)
        i0 = modulo(floor(i_f) - 1, nz) + 1
        j0 = modulo(floor(j_f) - 1, ny) + 1
        k0 = modulo(floor(k_f) - 1, nx) + 1
        i1 = modulo(i0, nz) + 1
        j1 = modulo(j0, ny) + 1
        k1 = modulo(k0, nx) + 1
        x0 = i_f - floor(i_f)
        y0 = j_f - floor(j_f)
        z0 = k_f - floor(k_f)
        x1 = 1.0 - x0
        y1 = 1.0 - y0
        z1 = 1.0 - z0
        val = vals(ii)
        densities(i0, j0, k0) = densities(i0, j0, k0) + val * x1 * y1 * z1
        densities(i1, j0, k0) = densities(i1, j0, k0) + val * x0 * y1 * z1
        densities(i0, j1, k0) = densities(i0, j1, k0) + val * x1 * y0 * z1
        densities(i0, j0, k1) = densities(i0, j0, k1) + val * x1 * y1 * z0
        densities(i1, j0, k1) = densities(i1, j0, k1) + val * x0 * y1 * z0
        densities(i0, j1, k1) = densities(i0, j1, k1) + val * x1 * y0 * z0
        densities(i1, j1, k0) = densities(i1, j1, k0) + val * x0 * y0 * z1
        densities(i1, j1, k1) = densities(i1, j1, k1) + val * x0 * y0 * z0
        weights(i0, j0, k0) = weights(i0, j0, k0) + x1 * y1 * z1
        weights(i1, j0, k0) = weights(i1, j0, k0) + x0 * y1 * z1
        weights(i0, j1, k0) = weights(i0, j1, k0) + x1 * y0 * z1
        weights(i0, j0, k1) = weights(i0, j0, k1) + x1 * y1 * z0
        weights(i1, j0, k1) = weights(i1, j0, k1) + x0 * y1 * z0
        weights(i0, j1, k1) = weights(i0, j1, k1) + x1 * y0 * z0
        weights(i1, j1, k0) = weights(i1, j1, k0) + x0 * y0 * z1
        weights(i1, j1, k1) = weights(i1, j1, k1) + x0 * y0 * z0
    enddo
end subroutine trilinear_insertion_real

subroutine trilinear_insertion_complex(densities, weights, vectors, vals, corners, deltas)
    implicit none
    complex(kind=8), intent(inout) :: densities(:,:,:)
    complex(kind=8), intent(in) :: vals(:)
    complex(kind=8) :: val
    real(kind=8), intent(inout) :: weights(:,:,:)
    real(kind=8), intent(in) :: vectors(:,:), corners(3), deltas(3)
    real(kind=8) :: i_f,j_f,k_f,x0,y0,z0,x1,y1,z1
    integer(kind=4) :: i0,j0,k0,i1,j1,k1,ii,nx,ny,nz,nn
    nn = size(vals, 1)
    nx = size(densities, 1)
    ny = size(densities, 2)
    nz = size(densities, 3)
    do ii=1,nn
        k_f = 1.0 + (vectors(1, ii) - corners(1)) / deltas(1)
        j_f = 1.0 + (vectors(2, ii) - corners(2)) / deltas(2)
        i_f = 1.0 + (vectors(3, ii) - corners(3)) / deltas(3)
        i0 = modulo(floor(i_f) - 1, nz) + 1
        j0 = modulo(floor(j_f) - 1, ny) + 1
        k0 = modulo(floor(k_f) - 1, nx) + 1
        i1 = modulo(i0, nz) + 1
        j1 = modulo(j0, ny) + 1
        k1 = modulo(k0, nx) + 1
        x0 = i_f - floor(i_f)
        y0 = j_f - floor(j_f)
        z0 = k_f - floor(k_f)
        x1 = 1.0 - x0
        y1 = 1.0 - y0
        z1 = 1.0 - z0
        val = vals(ii)
        densities(i0, j0, k0) = densities(i0, j0, k0) + val * x1 * y1 * z1
        densities(i1, j0, k0) = densities(i1, j0, k0) + val * x0 * y1 * z1
        densities(i0, j1, k0) = densities(i0, j1, k0) + val * x1 * y0 * z1
        densities(i0, j0, k1) = densities(i0, j0, k1) + val * x1 * y1 * z0
        densities(i1, j0, k1) = densities(i1, j0, k1) + val * x0 * y1 * z0
        densities(i0, j1, k1) = densities(i0, j1, k1) + val * x1 * y0 * z0
        densities(i1, j1, k0) = densities(i1, j1, k0) + val * x0 * y0 * z1
        densities(i1, j1, k1) = densities(i1, j1, k1) + val * x0 * y0 * z0
        weights(i0, j0, k0) = weights(i0, j0, k0) + x1 * y1 * z1
        weights(i1, j0, k0) = weights(i1, j0, k0) + x0 * y1 * z1
        weights(i0, j1, k0) = weights(i0, j1, k0) + x1 * y0 * z1
        weights(i0, j0, k1) = weights(i0, j0, k1) + x1 * y1 * z0
        weights(i1, j0, k1) = weights(i1, j0, k1) + x0 * y1 * z0
        weights(i0, j1, k1) = weights(i0, j1, k1) + x1 * y0 * z0
        weights(i1, j1, k0) = weights(i1, j1, k0) + x0 * y0 * z1
        weights(i1, j1, k1) = weights(i1, j1, k1) + x0 * y0 * z0
    enddo
end subroutine trilinear_insertion_complex

subroutine trilinear_insertions_real(densities, vectors, vals, corners, deltas)
    implicit none
    real(kind=8), intent(inout) :: densities(:,:,:,:)
    real(kind=8), intent(in) :: vals(:,:)
    real(kind=8) :: val(size(densities,1))
    real(kind=8), intent(in) :: vectors(:,:), corners(3), deltas(3)
    real(kind=8) :: i_f,j_f,k_f,x0,y0,z0,x1,y1,z1
    integer(kind=4) :: i0,j0,k0,i1,j1,k1,ii,nx,ny,nz,nn
    nn = size(vals, 2)
    nx = size(densities, 2)
    ny = size(densities, 3)
    nz = size(densities, 4)
    do ii=1,nn
        k_f = 1.0 + (vectors(1, ii) - corners(1)) / deltas(1)
        j_f = 1.0 + (vectors(2, ii) - corners(2)) / deltas(2)
        i_f = 1.0 + (vectors(3, ii) - corners(3)) / deltas(3)
        i0 = modulo(floor(i_f) - 1, nz) + 1
        j0 = modulo(floor(j_f) - 1, ny) + 1
        k0 = modulo(floor(k_f) - 1, nx) + 1
        i1 = modulo(i0, nz) + 1
        j1 = modulo(j0, ny) + 1
        k1 = modulo(k0, nx) + 1
        x0 = i_f - floor(i_f)
        y0 = j_f - floor(j_f)
        z0 = k_f - floor(k_f)
        x1 = 1.0 - x0
        y1 = 1.0 - y0
        z1 = 1.0 - z0
        val = vals(:,ii)
        densities(:,i0, j0, k0) = densities(:,i0, j0, k0) + val * x1 * y1 * z1
        densities(:,i1, j0, k0) = densities(:,i1, j0, k0) + val * x0 * y1 * z1
        densities(:,i0, j1, k0) = densities(:,i0, j1, k0) + val * x1 * y0 * z1
        densities(:,i0, j0, k1) = densities(:,i0, j0, k1) + val * x1 * y1 * z0
        densities(:,i1, j0, k1) = densities(:,i1, j0, k1) + val * x0 * y1 * z0
        densities(:,i0, j1, k1) = densities(:,i0, j1, k1) + val * x1 * y0 * z0
        densities(:,i1, j1, k0) = densities(:,i1, j1, k0) + val * x0 * y0 * z1
        densities(:,i1, j1, k1) = densities(:,i1, j1, k1) + val * x0 * y0 * z0
    enddo
end subroutine trilinear_insertions_real

subroutine trilinear_insertions_complex(densities, vectors, vals, corners, deltas)
    implicit none
    complex(kind=8), intent(inout) :: densities(:,:,:,:)
    complex(kind=8), intent(in) :: vals(:,:)
    complex(kind=8) :: val(size(densities,1))
    real(kind=8), intent(in) :: vectors(:,:), corners(3), deltas(3)
    real(kind=8) :: i_f,j_f,k_f,x0,y0,z0,x1,y1,z1
    integer(kind=4) :: i0,j0,k0,i1,j1,k1,ii,nx,ny,nz,nn
    nn = size(vals, 2)
    nx = size(densities, 2)
    ny = size(densities, 3)
    nz = size(densities, 4)
    do ii=1,nn
        k_f = 1.0 + (vectors(1, ii) - corners(1)) / deltas(1)
        j_f = 1.0 + (vectors(2, ii) - corners(2)) / deltas(2)
        i_f = 1.0 + (vectors(3, ii) - corners(3)) / deltas(3)
        i0 = modulo(floor(i_f) - 1, nz) + 1
        j0 = modulo(floor(j_f) - 1, ny) + 1
        k0 = modulo(floor(k_f) - 1, nx) + 1
        i1 = modulo(i0, nz) + 1
        j1 = modulo(j0, ny) + 1
        k1 = modulo(k0, nx) + 1
        x0 = i_f - floor(i_f)
        y0 = j_f - floor(j_f)
        z0 = k_f - floor(k_f)
        x1 = 1.0 - x0
        y1 = 1.0 - y0
        z1 = 1.0 - z0
        val = vals(:,ii)
        densities(:,i0, j0, k0) = densities(:,i0, j0, k0) + val * x1 * y1 * z1
        densities(:,i1, j0, k0) = densities(:,i1, j0, k0) + val * x0 * y1 * z1
        densities(:,i0, j1, k0) = densities(:,i0, j1, k0) + val * x1 * y0 * z1
        densities(:,i0, j0, k1) = densities(:,i0, j0, k1) + val * x1 * y1 * z0
        densities(:,i1, j0, k1) = densities(:,i1, j0, k1) + val * x0 * y1 * z0
        densities(:,i0, j1, k1) = densities(:,i0, j1, k1) + val * x1 * y0 * z0
        densities(:,i1, j1, k0) = densities(:,i1, j1, k0) + val * x0 * y0 * z1
        densities(:,i1, j1, k1) = densities(:,i1, j1, k1) + val * x0 * y0 * z0
    enddo
end subroutine trilinear_insertions_complex


subroutine trilinear_insertion_factor_real(summedvalues, vectors, vals, corners, deltas,factor2)
    implicit none
    real(kind=8), intent(inout) :: summedvalues(:,:,:,:)
    real(kind=8), intent(in) :: vals(:),factor2
    real(kind=8) :: val
    real(kind=8), intent(in) :: vectors(:,:), corners(3), deltas(3)
    real(kind=8) :: i_f,j_f,k_f,x0,y0,z0,x1,y1,z1
    real(kind=8) :: f1,f2,f3,f4,f5,f6,f7,f8
    integer(kind=4) :: i0,j0,k0,i1,j1,k1,ii,nx,ny,nz,nn

    nn = size(vals, 1)
    if (size(summedvalues,1).ne.2) then
       stop 'fix me'
    endif
    nx = size(summedvalues, 4)  ! Rick: Check this (4,3,2) for your previous codes
    ny = size(summedvalues, 3)
    nz = size(summedvalues, 2)

    do ii=1,nn
        k_f = 1.0 + (vectors(1, ii) - corners(1)) / deltas(1)
        j_f = 1.0 + (vectors(2, ii) - corners(2)) / deltas(2)
        i_f = 1.0 + (vectors(3, ii) - corners(3)) / deltas(3)
        i0 = modulo(floor(i_f) - 1, nz) + 1
        j0 = modulo(floor(j_f) - 1, ny) + 1
        k0 = modulo(floor(k_f) - 1, nx) + 1
        i1 = modulo(i0, nz) + 1
        j1 = modulo(j0, ny) + 1
        k1 = modulo(k0, nx) + 1
        x0 = i_f - floor(i_f)
        y0 = j_f - floor(j_f)
        z0 = k_f - floor(k_f)
        x1 = 1.0 - x0
        y1 = 1.0 - y0
        z1 = 1.0 - z0
        f1 = x1 * y1 * z1
        f2 = x0 * y1 * z1
        f3 = x1 * y0 * z1
        f4 = x1 * y1 * z0
        f5 = x0 * y1 * z0
        f6 = x1 * y0 * z0
        f7 = x0 * y0 * z1
        f8 = x0 * y0 * z0

        val = vals(ii)

        summedvalues(1, i0, j0, k0) = summedvalues(1, i0, j0, k0) + val * f1
        summedvalues(2, i0, j0, k0) = summedvalues(2, i0, j0, k0) + factor2 * f1
        summedvalues(1, i1, j0, k0) = summedvalues(1, i1, j0, k0) + val * f2
        summedvalues(2, i1, j0, k0) = summedvalues(2, i1, j0, k0) + factor2 * f2
        summedvalues(1, i0, j1, k0) = summedvalues(1, i0, j1, k0) + val * f3
        summedvalues(2, i0, j1, k0) = summedvalues(2, i0, j1, k0) + factor2 * f3
        summedvalues(1, i0, j0, k1) = summedvalues(1, i0, j0, k1) + val * f4
        summedvalues(2, i0, j0, k1) = summedvalues(2, i0, j0, k1) + factor2 * f4
        summedvalues(1, i1, j0, k1) = summedvalues(1, i1, j0, k1) + val * f5
        summedvalues(2, i1, j0, k1) = summedvalues(2, i1, j0, k1) + factor2 * f5
        summedvalues(1, i0, j1, k1) = summedvalues(1, i0, j1, k1) + val * f6
        summedvalues(2, i0, j1, k1) = summedvalues(2, i0, j1, k1) + factor2 * f6
        summedvalues(1, i1, j1, k0) = summedvalues(1, i1, j1, k0) + val * f7
        summedvalues(2, i1, j1, k0) = summedvalues(2, i1, j1, k0) + factor2 * f7
        summedvalues(1, i1, j1, k1) = summedvalues(1, i1, j1, k1) + val * f8
        summedvalues(2, i1, j1, k1) = summedvalues(2, i1, j1, k1) + factor2 * f8
    enddo
end subroutine trilinear_insertion_factor_real



!subroutine build_atomic_scattering_density_map(densities, weights, vectors, vals, corners, deltas, weight)
!    implicit none
!    real(kind=8), intent(inout) :: densities(:,:,:), weights(:,:,:)
!    real(kind=8), intent(in) :: vectors(:,:), vals(:), corners(3), deltas(3), weight
!    real(kind=8) :: i_f,j_f,k_f,x0,y0,z0,x1,y1,z1,val
!    integer(kind=4) :: i0,j0,k0,i1,j1,k1,ii,nx,ny,nz,nn
!    nn = size(vals, 1)
!    nx = size(densities, 1)
!    ny = size(densities, 2)
!    nz = size(densities, 3)
!    do ii=1,nn
!        k_f = 1.0 + (vectors(1, ii) - corners(1)) / deltas(1)
!        j_f = 1.0 + (vectors(2, ii) - corners(2)) / deltas(2)
!        i_f = 1.0 + (vectors(3, ii) - corners(3)) / deltas(3)
!        i0 = modulo(floor(i_f) - 1, nz) + 1
!        j0 = modulo(floor(j_f) - 1, ny) + 1
!        k0 = modulo(floor(k_f) - 1, nx) + 1
!        i1 = modulo(i0, nz) + 1
!        j1 = modulo(j0, ny) + 1
!        k1 = modulo(k0, nx) + 1
!        x0 = i_f - floor(i_f)
!        y0 = j_f - floor(j_f)
!        z0 = k_f - floor(k_f)
!        x1 = 1.0 - x0
!        y1 = 1.0 - y0
!        z1 = 1.0 - z0
!        val = vals(ii)
!        densities(i0, j0, k0) = densities(i0, j0, k0) + val * x1 * y1 * z1 * weight
!        densities(i1, j0, k0) = densities(i1, j0, k0) + val * x0 * y1 * z1 * weight
!        densities(i0, j1, k0) = densities(i0, j1, k0) + val * x1 * y0 * z1 * weight
!        densities(i0, j0, k1) = densities(i0, j0, k1) + val * x1 * y1 * z0 * weight
!        densities(i1, j0, k1) = densities(i1, j0, k1) + val * x0 * y1 * z0 * weight
!        densities(i0, j1, k1) = densities(i0, j1, k1) + val * x1 * y0 * z0 * weight
!        densities(i1, j1, k0) = densities(i1, j1, k0) + val * x0 * y0 * z1 * weight
!        densities(i1, j1, k1) = densities(i1, j1, k1) + val * x0 * y0 * z0 * weight
!        weights(i0, j0, k0) = weights(i0, j0, k0) + x1 * y1 * z1 * weight
!        weights(i1, j0, k0) = weights(i1, j0, k0) + x0 * y1 * z1 * weight
!        weights(i0, j1, k0) = weights(i0, j1, k0) + x1 * y0 * z1 * weight
!        weights(i0, j0, k1) = weights(i0, j0, k1) + x1 * y1 * z0 * weight
!        weights(i1, j0, k1) = weights(i1, j0, k1) + x0 * y1 * z0 * weight
!        weights(i0, j1, k1) = weights(i0, j1, k1) + x1 * y0 * z0 * weight
!        weights(i1, j1, k0) = weights(i1, j1, k0) + x0 * y0 * z1 * weight
!        weights(i1, j1, k1) = weights(i1, j1, k1) + x0 * y0 * z0 * weight
!    enddo
!end subroutine build_atomic_scattering_density_map
