subroutine trilinear_interpolation(densities, vectors, limits, out)
    implicit none
    real(kind=8), intent(inout) :: out(:)
    real(kind=8), intent(in) :: densities(:,:,:), vectors(:,:), limits(:,:)
    real(kind=8) :: dx,dy,dz,i_f,j_f,k_f,x0,y0,z0,x1,y1,z1
    integer(kind=4) :: nx,ny,nz,nn,i0,j0,k0,i1,j1,k1,ii
    nn = size(out, 1)
    nx = size(densities, 1)
    ny = size(densities, 2)
    nz = size(densities, 3)
    dx = (limits(2, 1) - limits(1, 1)) / (nx - 1)
    dy = (limits(2, 2) - limits(1, 2)) / (ny - 1)
    dz = (limits(2, 3) - limits(1, 3)) / (nz - 1)
    do ii=1,nn
        i_f = 1.0 + (vectors(1, ii) - limits(1, 1)) / dx
        j_f = 1.0 + (vectors(2, ii) - limits(1, 2)) / dy
        k_f = 1.0 + (vectors(3, ii) - limits(1, 3)) / dz
        i0 = max(floor(i_f), 1)
        j0 = max(floor(j_f), 1)
        k0 = max(floor(k_f), 1)
        k1 = min(k0+1, nx)
        j1 = min(j0+1, ny)
        i1 = min(i0+1, nz)
        x0 = i_f - floor(i_f)
        y0 = j_f - floor(j_f)
        z0 = k_f - floor(k_f)
        x1 = 1.0 - x0
        y1 = 1.0 - y0
        z1 = 1.0 - z0
        out(ii) = densities(k0, j0, i0) * x1 * y1 * z1 + &
                  densities(k0, j0, i1) * x0 * y1 * z1 + &
                  densities(k0, j1, i0) * x1 * y0 * z1 + &
                  densities(k1, j0, i0) * x1 * y1 * z0 + &
                  densities(k1, j0, i1) * x0 * y1 * z0 + &
                  densities(k1, j1, i0) * x1 * y0 * z0 + &
                  densities(k0, j1, i1) * x0 * y0 * z1 + &
                  densities(i1, j1, k1) * x0 * y0 * z0
    enddo
end subroutine trilinear_interpolation


subroutine wtf(out1, out2, out3)
    implicit none
    real(kind=8), intent(inout) :: out1(:), out2(:,:), out3(:,:,:)
    out1(2) = 10
    out2(2,1) = 10
    out3(2,1,1) = 10
end subroutine wtf


subroutine trilinear_insertion(densities, weights, vectors, vals, limits)
    implicit none
    real(kind=8), intent(inout) :: densities(:,:,:), weights(:,:,:)
    real(kind=8), intent(in) :: vectors(:,:), vals(:), limits(:,:)
    real(kind=8) :: dx,dy,dz,i_f,j_f,k_f,x0,y0,z0,x1,y1,z1,val
    integer(kind=4) :: i0,j0,k0,i1,j1,k1,ii,nx,ny,nz,nn
    nn = size(vals, 1)
    nx = size(densities, 1)
    ny = size(densities, 2)
    nz = size(densities, 3)
    dx = (limits(2, 3) - limits(1, 3)) / (nx - 1)
    dy = (limits(2, 2) - limits(1, 2)) / (ny - 1)
    dz = (limits(2, 1) - limits(1, 1)) / (nz - 1)
    do ii=1,nn
        i_f = 1.0 + (vectors(1, ii) - limits(1, 1)) / dx
        j_f = 1.0 + (vectors(2, ii) - limits(1, 2)) / dy
        k_f = 1.0 + (vectors(3, ii) - limits(1, 3)) / dz
        i0 = max(floor(i_f), 1)
        j0 = max(floor(j_f), 1)
        k0 = max(floor(k_f), 1)
        k1 = min(k0+1, nx)
        j1 = min(j0+1, ny)
        i1 = min(i0+1, nz)
        x0 = i_f - floor(i_f)
        y0 = j_f - floor(j_f)
        z0 = k_f - floor(k_f)
        x1 = 1.0 - x0
        y1 = 1.0 - y0
        z1 = 1.0 - z0
        val = vals(ii)
        densities(k0, j0, i0) = densities(k0, j0, i0) + val * x1 * y1 * z1
        densities(k0, j0, i1) = densities(k0, j0, i1) + val * x0 * y1 * z1
        densities(k0, j1, i0) = densities(k0, j1, i0) + val * x1 * y0 * z1
        densities(k1, j0, i0) = densities(k1, j0, i0) + val * x1 * y1 * z0
        densities(k1, j0, i1) = densities(k1, j0, i1) + val * x0 * y1 * z0
        densities(k1, j1, i0) = densities(k1, j1, i0) + val * x1 * y0 * z0
        densities(k0, j1, i1) = densities(k0, j1, i1) + val * x0 * y0 * z1
        densities(k1, j1, i1) = densities(k1, j1, i1) + val * x0 * y0 * z0
        weights(k0, j0, i0) = weights(k0, j0, i0) + x1 * y1 * z1
        weights(k0, j0, i1) = weights(k0, j0, i1) + x0 * y1 * z1
        weights(k0, j1, i0) = weights(k0, j1, i0) + x1 * y0 * z1
        weights(k1, j0, i0) = weights(k1, j0, i0) + x1 * y1 * z0
        weights(k1, j0, i1) = weights(k1, j0, i1) + x0 * y1 * z0
        weights(k1, j1, i0) = weights(k1, j1, i0) + x1 * y0 * z0
        weights(k0, j1, i1) = weights(k0, j1, i1) + x0 * y0 * z1
        weights(k1, j1, i1) = weights(k1, j1, i1) + x0 * y0 * z0
    enddo
end subroutine trilinear_insertion
