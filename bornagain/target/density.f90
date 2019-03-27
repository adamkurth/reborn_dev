subroutine trilinear_interpolation(densities, vectors, limits, out)
    implicit none
    real(kind=8), intent(inout) :: out(:)
    real(kind=8), intent(in) :: densities(:,:,:), vectors(:,:), limits(:,:)
    real(kind=8) :: dx,dy,dz,i_f,j_f,k_f,x0,y0,z0,x1,y1,z1
    integer(kind=4) :: nx,ny,nz,nn,i0,j0,k0,i1,j1,k1,ii,shap1(1),shap3(3)
    shap1 = shape(out)
    nn = shap1(1)
    shap3 = shape(densities)
    nx = shap3(1)
    ny = shap3(2)
    nz = shap3(3)
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


!subroutine trilinear_insertion(densities, vectors, limits, out)
!    implicit none
!    real(kind=8), intent(inout) :: out(:)
!    real(kind=8), intent(in) :: densities(:,:,:), vectors(:,:), limits(:,:)
!    real(kind=8) :: dx,dy,dz,i_f,j_f,k_f,x0,y0,z0,x1,y1,z1
!    integer(kind=4) :: nx,ny,nz,nn,i0,j0,k0,i1,j1,k1,ii,shap1(1),shap3(3)
!    shap1 = shape(out)
!    nn = shap1(1)
!    shap3 = shape(densities)
!    nx = shap3(1)
!    ny = shap3(2)
!    nz = shap3(3)
!    dx = (limits(1, 2) - limits(1, 1)) / nx
!    dy = (limits(2, 2) - limits(2, 1)) / ny
!    dz = (limits(3, 2) - limits(3, 1)) / nz
!    do ii=1,nn
!        i_f = (vectors(ii, 1) - limits(1, 1)) / dx
!        j_f = (vectors(ii, 2) - limits(2, 1)) / dy
!        k_f = (vectors(ii, 3) - limits(3, 1)) / dz
!        i0 = max(floor(i_f), 1)
!        j0 = max(floor(j_f), 1)
!        k0 = max(floor(k_f), 1)
!        k1 = min(k0+1, nx)
!        j1 = min(j0+1, ny)
!        i1 = min(i0+1, nz)
!        x0 = i_f - floor(i_f)
!        y0 = j_f - floor(j_f)
!        z0 = k_f - floor(k_f)
!        x1 = 1.0 - x0
!        y1 = 1.0 - y0
!        z1 = 1.0 - z0
!        out(ii) = densities(i0, j0, k0) * x1 * y1 * z1 + &
!                densities(i1, j0, k0) * x0 * y1 * z1 + &
!                densities(i0, j1, k0) * x1 * y0 * z1 + &
!                densities(i0, j0, k1) * x1 * y1 * z0 + &
!                densities(i1, j0, k1) * x0 * y1 * z0 + &
!                densities(i0, j1, k1) * x1 * y0 * z0 + &
!                densities(i1, j1, k0) * x0 * y0 * z1 + &
!                densities(i1, j1, k1) * x0 * y0 * z0
!    enddo
!end subroutine trilinear_insertion


!subroutine trilinear_insertion(densities, weights, vectors, vals, limits, nx, ny, nz, nn)
!    implicit none
!    real(kind=8), intent(inout) :: densities(:,:,:), weights(:,:,:)
!    real(kind=8), intent(in) :: vectors(:,:), vals(:), limits(:,:)
!    integer(kind=4), intent(in) :: nx,ny,nz,nn
!    real(kind=8) :: dx,dy,dz,i_f,j_f,k_f,x0,y0,z0,x1,y1,z1,val
!    integer(kind=4) :: i0,j0,k0,i1,j1,k1,ii
!    densities(1,1,1) = 1
!    dx = (limits(1, 2) - limits(1, 1)) / nx
!    dy = (limits(2, 2) - limits(2, 1)) / ny
!    dz = (limits(3, 2) - limits(3, 1)) / nz
!    write (*,*) nn, shape(densities)
!    do ii=1,nn
!        i_f = (vectors(ii, 1) - limits(1, 1)) / dx
!        j_f = (vectors(ii, 2) - limits(2, 1)) / dy
!        k_f = (vectors(ii, 3) - limits(3, 1)) / dz
!        i0 = max(floor(i_f), 1)
!        j0 = max(floor(j_f), 1)
!        k0 = max(floor(k_f), 1)
!        k1 = min(k0+1, nx)
!        j1 = min(j0+1, ny)
!        i1 = min(i0+1, nz)
!        x0 = i_f - floor(i_f)
!        y0 = j_f - floor(j_f)
!        z0 = k_f - floor(k_f)
!        x1 = 1.0 - x0
!        y1 = 1.0 - y0
!        z1 = 1.0 - z0
!        val = vals(ii)
!        write(*,*)'hello',i0, j0, k0, val, densities(i0, j0, k0)
!        densities(i0, j0, k0) = densities(i0, j0, k0) + val
!        densities(i1, j0, k0) = densities(i1, j0, k0) + val
!        densities(i0, j1, k0) = densities(i0, j1, k0) + val
!        densities(i0, j0, k1) = densities(i0, j0, k1) + val
!        densities(i1, j0, k1) = densities(i1, j0, k1) + val
!        densities(i0, j1, k1) = densities(i0, j1, k1) + val
!        densities(i1, j1, k0) = densities(i1, j1, k0) + val
!        densities(i1, j1, k1) = densities(i1, j1, k1) + val
!        weights(i0, j0, k0) = weights(i0, j0, k0) + x1 * y1 * z1
!        weights(i1, j0, k0) = weights(i1, j0, k0) + x0 * y1 * z1
!        weights(i0, j1, k0) = weights(i0, j1, k0) + x1 * y0 * z1
!        weights(i0, j0, k1) = weights(i0, j0, k1) + x1 * y1 * z0
!        weights(i1, j0, k1) = weights(i1, j0, k1) + x0 * y1 * z0
!        weights(i0, j1, k1) = weights(i0, j1, k1) + x1 * y0 * z0
!        weights(i1, j1, k0) = weights(i1, j1, k0) + x0 * y0 * z1
!        weights(i1, j1, k1) = weights(i1, j1, k1) + x0 * y0 * z0
!        write(*,*) densities(i0, j0, k0)
!    enddo
!end subroutine trilinear_insertion