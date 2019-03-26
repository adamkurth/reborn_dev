subroutine trilinear_interpolation(densities, vectors, limits, out, nx, ny, nz, nn)
    implicit none
    real(kind=8), intent(inout) :: out(nn)
    real(kind=8), intent(in) :: densities(nx,ny,nz), vectors(nn,3), limits(3,2)
    integer(kind=4), intent(in) :: nx,ny,nz,nn
    real(kind=8) :: dx,dy,dz,i_f,j_f,k_f,x0,y0,z0,x1,y1,z1
    integer(kind=4) :: i0,j0,k0,i1,j1,k1,ii
    dx = (limits(1, 2) - limits(1, 1)) / nx
    dy = (limits(2, 2) - limits(2, 1)) / ny
    dz = (limits(3, 2) - limits(3, 1)) / nz
    do ii=1,nn
        i_f = (vectors(ii, 1) - limits(1, 1)) / dx
        j_f = (vectors(ii, 2) - limits(2, 1)) / dy
        k_f = (vectors(ii, 3) - limits(3, 1)) / dz
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


subroutine trilinear_insertion(densities, weights, vectors, vals, limits, nx, ny, nz, nn)
    implicit none
    real(kind=8), intent(inout) :: densities(nx,ny,nz), weights(nx,ny,nz)
    real(kind=8), intent(in) :: vectors(nn,3), vals(nn), limits(3,2)
    integer(kind=4), intent(in) :: nx,ny,nz,nn
    real(kind=8) :: dx,dy,dz,i_f,j_f,k_f,x0,y0,z0,x1,y1,z1,val
    integer(kind=4) :: i0,j0,k0,i1,j1,k1,ii
    dx = (limits(1, 2) - limits(1, 1)) / nx
    dy = (limits(2, 2) - limits(2, 1)) / ny
    dz = (limits(3, 2) - limits(3, 1)) / nz
    do ii=1,nn
        i_f = (vectors(ii, 1) - limits(1, 1)) / dx
        j_f = (vectors(ii, 2) - limits(2, 1)) / dy
        k_f = (vectors(ii, 3) - limits(3, 1)) / dz
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
        densities(i0, j0, k0) = densities(i0, j0, k0) + val
        densities(i1, j0, k0) = densities(i1, j0, k0) + val
        densities(i0, j1, k0) = densities(i0, j1, k0) + val
        densities(i0, j0, k1) = densities(i0, j0, k1) + val
        densities(i1, j0, k1) = densities(i1, j0, k1) + val
        densities(i0, j1, k1) = densities(i0, j1, k1) + val
        densities(i1, j1, k0) = densities(i1, j1, k0) + val
        densities(i1, j1, k1) = densities(i1, j1, k1) + val
        weights(i0, j0, k0) = weights(i0, j0, k0) + x1 * y1 * z1
        weights(i1, j0, k0) = weights(i1, j0, k0) + x0 * y1 * z1
        weights(i0, j1, k0) = weights(i0, j1, k0) + x1 * y0 * z1
        weights(i0, j0, k1) = weights(i0, j0, k1) + x1 * y1 * z0
        weights(i1, j0, k1) = weights(i1, j0, k1) + x0 * y1 * z0
        weights(i0, j1, k1) = weights(i0, j1, k1) + x1 * y0 * z0
        weights(i1, j1, k0) = weights(i1, j1, k0) + x0 * y0 * z1
        weights(i1, j1, k1) = weights(i1, j1, k1) + x0 * y0 * z0
    enddo
end subroutine trilinear_insertion