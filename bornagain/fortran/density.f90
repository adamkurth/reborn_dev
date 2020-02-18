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
        out(ii) = densities(i0, j0, k0) * x1 * y1 * z1 + &
                  densities(i1, j0, k0) * x0 * y1 * z1 + &
                  densities(i0, j1, k0) * x1 * y0 * z1 + &
                  densities(i0, j0, k1) * x1 * y1 * z0 + &
                  densities(i1, j0, k1) * x0 * y1 * z0 + &
                  densities(i0, j1, k1) * x1 * y0 * z0 + &
                  densities(i1, j1, k0) * x0 * y0 * z1 + &
                  densities(i1, j1, k1) * x0 * y0 * z0
    enddo
end subroutine trilinear_interpolation

subroutine trilinear_insertion(densities, weights, vectors, vals, corners, deltas, weight)
    implicit none
    real(kind=8), intent(inout) :: densities(:,:,:), weights(:,:,:)
    real(kind=8), intent(in) :: vectors(:,:), vals(:), corners(3), deltas(3), weight
    real(kind=8) :: i_f,j_f,k_f,x0,y0,z0,x1,y1,z1,val
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
        densities(i0, j0, k0) = densities(i0, j0, k0) + val * x1 * y1 * z1 * weight
        densities(i1, j0, k0) = densities(i1, j0, k0) + val * x0 * y1 * z1 * weight
        densities(i0, j1, k0) = densities(i0, j1, k0) + val * x1 * y0 * z1 * weight
        densities(i0, j0, k1) = densities(i0, j0, k1) + val * x1 * y1 * z0 * weight
        densities(i1, j0, k1) = densities(i1, j0, k1) + val * x0 * y1 * z0 * weight
        densities(i0, j1, k1) = densities(i0, j1, k1) + val * x1 * y0 * z0 * weight
        densities(i1, j1, k0) = densities(i1, j1, k0) + val * x0 * y0 * z1 * weight
        densities(i1, j1, k1) = densities(i1, j1, k1) + val * x0 * y0 * z0 * weight
        weights(i0, j0, k0) = weights(i0, j0, k0) + x1 * y1 * z1 * weight
        weights(i1, j0, k0) = weights(i1, j0, k0) + x0 * y1 * z1 * weight
        weights(i0, j1, k0) = weights(i0, j1, k0) + x1 * y0 * z1 * weight
        weights(i0, j0, k1) = weights(i0, j0, k1) + x1 * y1 * z0 * weight
        weights(i1, j0, k1) = weights(i1, j0, k1) + x0 * y1 * z0 * weight
        weights(i0, j1, k1) = weights(i0, j1, k1) + x1 * y0 * z0 * weight
        weights(i1, j1, k0) = weights(i1, j1, k0) + x0 * y0 * z1 * weight
        weights(i1, j1, k1) = weights(i1, j1, k1) + x0 * y0 * z0 * weight
    enddo
end subroutine trilinear_insertion

subroutine build_atomic_scattering_density_map(densities, weights, vectors, vals, corners, deltas, weight)
    implicit none
    real(kind=8), intent(inout) :: densities(:,:,:), weights(:,:,:)
    real(kind=8), intent(in) :: vectors(:,:), vals(:), corners(3), deltas(3), weight
    real(kind=8) :: i_f,j_f,k_f,x0,y0,z0,x1,y1,z1,val
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
        densities(i0, j0, k0) = densities(i0, j0, k0) + val * x1 * y1 * z1 * weight
        densities(i1, j0, k0) = densities(i1, j0, k0) + val * x0 * y1 * z1 * weight
        densities(i0, j1, k0) = densities(i0, j1, k0) + val * x1 * y0 * z1 * weight
        densities(i0, j0, k1) = densities(i0, j0, k1) + val * x1 * y1 * z0 * weight
        densities(i1, j0, k1) = densities(i1, j0, k1) + val * x0 * y1 * z0 * weight
        densities(i0, j1, k1) = densities(i0, j1, k1) + val * x1 * y0 * z0 * weight
        densities(i1, j1, k0) = densities(i1, j1, k0) + val * x0 * y0 * z1 * weight
        densities(i1, j1, k1) = densities(i1, j1, k1) + val * x0 * y0 * z0 * weight
        weights(i0, j0, k0) = weights(i0, j0, k0) + x1 * y1 * z1 * weight
        weights(i1, j0, k0) = weights(i1, j0, k0) + x0 * y1 * z1 * weight
        weights(i0, j1, k0) = weights(i0, j1, k0) + x1 * y0 * z1 * weight
        weights(i0, j0, k1) = weights(i0, j0, k1) + x1 * y1 * z0 * weight
        weights(i1, j0, k1) = weights(i1, j0, k1) + x0 * y1 * z0 * weight
        weights(i0, j1, k1) = weights(i0, j1, k1) + x1 * y0 * z0 * weight
        weights(i1, j1, k0) = weights(i1, j1, k0) + x0 * y0 * z1 * weight
        weights(i1, j1, k1) = weights(i1, j1, k1) + x0 * y0 * z0 * weight
    enddo
end subroutine trilinear_insertion