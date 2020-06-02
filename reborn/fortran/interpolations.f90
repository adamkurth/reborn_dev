! Suddenly, it became necessary to have this main program in order for these to compile with f2py.  Nothing in the
! python or fortran code changed in reborn.  Not sure if this is due to numpy or conda, but the failure began on
! May 23, 2020 when compiling via the gitlab runner.  This is possibly helpful:
!
!  https://github.com/numpy/numpy/issues/14222
!
! I wonder if the main program will now break compilation on other systems.....
PROGRAM MAIN
PRINT *,'Hello world'
END PROGRAM MAIN

subroutine trilinear_interpolation(datin, datin_corner, datin_dx, datout_coords, datout)
    ! Interpolate a 3D grid of data onto arbitrary points
    ! datin is the input data with corner coordinate datin_corner (3-vector) and grid step size datin_dx (3-vector)
    ! datout_coords are the coordinates of the interpolation points.  Interpolated values go into datout.
    ! Works with doubles only
    implicit none
    real(kind=8), intent(inout) :: datout(:)
    real(kind=8), intent(in) :: datin(:,:,:), datout_coords(:,:), datin_corner(3), datin_dx(3)
    real(kind=8) :: i_f,j_f,k_f,x0,y0,z0,x1,y1,z1
    integer(kind=4) :: nn,i0,j0,k0,i1,j1,k1,ii,nx,ny,nz
    nn = size(datout, 1)
    nx = size(datin, 1)
    ny = size(datin, 2)
    nz = size(datin, 3)
    do ii=1,nn
        k_f = 1.0 + (datout_coords(1, ii) - datin_corner(1)) / datin_dx(1)
        j_f = 1.0 + (datout_coords(2, ii) - datin_corner(2)) / datin_dx(2)
        i_f = 1.0 + (datout_coords(3, ii) - datin_corner(3)) / datin_dx(3)
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
        datout(ii) = datin(i0, j0, k0) * x1 * y1 * z1 + &
                     datin(i1, j0, k0) * x0 * y1 * z1 + &
                     datin(i0, j1, k0) * x1 * y0 * z1 + &
                     datin(i0, j0, k1) * x1 * y1 * z0 + &
                     datin(i1, j0, k1) * x0 * y1 * z0 + &
                     datin(i0, j1, k1) * x1 * y0 * z0 + &
                     datin(i1, j1, k0) * x0 * y0 * z1 + &
                     datin(i1, j1, k1) * x0 * y0 * z0
    enddo
end subroutine trilinear_interpolation


subroutine trilinear_insert(data_coord, data_val, x_min, &
                            N_data, Delta_x, one_over_bin_volume, c1, &
                            dataout, weightout)
    ! Note this Fortran funtion populates dataout which is defined to be of shape N+2  
    ! where the addition of 2 is for the boundary samples. The Python code in utils.py then 
    ! crops this N+2 array out to the correct shape of N.
    implicit none
    complex(kind=8), intent(inout) :: dataout(:,:,:)
    real(kind=8),    intent(inout) :: weightout(:,:,:)
    complex(kind=8), intent(in)    :: data_val(:)
    real(kind=8),    intent(in)    :: data_coord(:,:), x_min(3), &
                                      Delta_x(3), one_over_bin_volume, c1(3)
    integer(kind=4), intent(in)    :: N_data

    complex(kind=8) :: data_val_curr_scaled
    real(kind=8)    :: data_coord_curr(3), x_ind_fl(3), x_ind_cl(3), Delta_x_1(3), Delta_x_0(3), &
                       N_000, N_100, N_010, N_110, N_001, N_101, N_011, N_111
    integer(kind=4) :: i, ind_fl(3), ind_cl(3)

    do i=1,N_data
        data_coord_curr = data_coord(i,:)
        data_val_curr_scaled = data_val(i) * one_over_bin_volume ! Multiply the data value by the inverse bin volume here to save computations later.

        ! Bin index
        ind_fl = floor(data_coord_curr / Delta_x + c1)
        ind_cl = ind_fl + 1

        ! Bin position
        x_ind_fl = x_min + ind_fl * Delta_x
        x_ind_cl = x_ind_fl + Delta_x ! This is the same as x_min + ind_cl*Delta_x

        ! Distances from the data point to the fl and cl bins
        Delta_x_1 = x_ind_cl - data_coord_curr
        Delta_x_0 = data_coord_curr - x_ind_fl

        ! The trilinear weights
        N_000 = Delta_x_1(1) * Delta_x_1(2) * Delta_x_1(3)
        N_100 = Delta_x_0(1) * Delta_x_1(2) * Delta_x_1(3)
        N_010 = Delta_x_1(1) * Delta_x_0(2) * Delta_x_1(3)
        N_110 = Delta_x_0(1) * Delta_x_0(2) * Delta_x_1(3)
        N_001 = Delta_x_1(1) * Delta_x_1(2) * Delta_x_0(3)
        N_101 = Delta_x_0(1) * Delta_x_1(2) * Delta_x_0(3)
        N_011 = Delta_x_1(1) * Delta_x_0(2) * Delta_x_0(3)
        N_111 = Delta_x_0(1) * Delta_x_0(2) * Delta_x_0(3)

        ! Add 1 to the bin indices - this is to correspond to the plus two boundary padding for the edge cases.
        ! Add another 1 for default Fortran indexing starting at 1.
        ind_fl = ind_fl + 2
        ind_cl = ind_cl + 2

        ! Accumulate the data values
        dataout(ind_fl(1), ind_fl(2), ind_fl(3)) = dataout(ind_fl(1), ind_fl(2), ind_fl(3)) & 
                                                   + N_000 * data_val_curr_scaled
        dataout(ind_cl(1), ind_fl(2), ind_fl(3)) = dataout(ind_cl(1), ind_fl(2), ind_fl(3)) &
                                                   + N_100 * data_val_curr_scaled
        dataout(ind_fl(1), ind_cl(2), ind_fl(3)) = dataout(ind_fl(1), ind_cl(2), ind_fl(3)) &
                                                   + N_010 * data_val_curr_scaled
        dataout(ind_cl(1), ind_cl(2), ind_fl(3)) = dataout(ind_cl(1), ind_cl(2), ind_fl(3)) &
                                                   + N_110 * data_val_curr_scaled
        dataout(ind_fl(1), ind_fl(2), ind_cl(3)) = dataout(ind_fl(1), ind_fl(2), ind_cl(3)) &
                                                   + N_001 * data_val_curr_scaled
        dataout(ind_cl(1), ind_fl(2), ind_cl(3)) = dataout(ind_cl(1), ind_fl(2), ind_cl(3)) &
                                                   + N_101 * data_val_curr_scaled
        dataout(ind_fl(1), ind_cl(2), ind_cl(3)) = dataout(ind_fl(1), ind_cl(2), ind_cl(3)) &
                                                   + N_011 * data_val_curr_scaled
        dataout(ind_cl(1), ind_cl(2), ind_cl(3)) = dataout(ind_cl(1), ind_cl(2), ind_cl(3)) &
                                                   + N_111 * data_val_curr_scaled

        ! Accumulate the number of times data values had been placed into these bins.
        ! The trilinear weights are by definition bewteen 0 and 1 so use ceiling to deal with
        ! data points that situate excatly on the centre of a bin.
        ! The -1e-10 is to safeguard against small values close to zero.
        weightout(ind_fl(1), ind_fl(2), ind_fl(3)) = weightout(ind_fl(1), ind_fl(2), ind_fl(3)) & 
                                                   + ceiling(N_000 * one_over_bin_volume - 1e-10)
        weightout(ind_cl(1), ind_fl(2), ind_fl(3)) = weightout(ind_cl(1), ind_fl(2), ind_fl(3)) &
                                                   + ceiling(N_100 * one_over_bin_volume - 1e-10)
        weightout(ind_fl(1), ind_cl(2), ind_fl(3)) = weightout(ind_fl(1), ind_cl(2), ind_fl(3)) &
                                                   + ceiling(N_010 * one_over_bin_volume - 1e-10)
        weightout(ind_cl(1), ind_cl(2), ind_fl(3)) = weightout(ind_cl(1), ind_cl(2), ind_fl(3)) &
                                                   + ceiling(N_110 * one_over_bin_volume - 1e-10)
        weightout(ind_fl(1), ind_fl(2), ind_cl(3)) = weightout(ind_fl(1), ind_fl(2), ind_cl(3)) &
                                                   + ceiling(N_001 * one_over_bin_volume - 1e-10)
        weightout(ind_cl(1), ind_fl(2), ind_cl(3)) = weightout(ind_cl(1), ind_fl(2), ind_cl(3)) &
                                                   + ceiling(N_101 * one_over_bin_volume - 1e-10)
        weightout(ind_fl(1), ind_cl(2), ind_cl(3)) = weightout(ind_fl(1), ind_cl(2), ind_cl(3)) &
                                                   + ceiling(N_011 * one_over_bin_volume - 1e-10)
        weightout(ind_cl(1), ind_cl(2), ind_cl(3)) = weightout(ind_cl(1), ind_cl(2), ind_cl(3)) &
                                                   + ceiling(N_111 * one_over_bin_volume - 1e-10)
    enddo
end subroutine trilinear_insert


subroutine trilinear_insert_with_wraparound(data_coord, data_val, x_min, &
                                            N_data, Delta_x, one_over_bin_volume, c1, N_bin, &
                                            dataout, weightout)
    implicit none
    complex(kind=8), intent(inout) :: dataout(:,:,:)
    real(kind=8),    intent(inout) :: weightout(:,:,:)
    complex(kind=8), intent(in)    :: data_val(:)
    real(kind=8),    intent(in)    :: data_coord(:,:), x_min(3), &
                                      Delta_x(3), one_over_bin_volume, c1(3)
    integer(kind=4), intent(in)    :: N_data, N_bin(3)

    complex(kind=8) :: data_val_curr_scaled
    real(kind=8)    :: data_coord_curr(3), x_ind_fl(3), x_ind_cl(3), Delta_x_1(3), Delta_x_0(3), &
                       N_000, N_100, N_010, N_110, N_001, N_101, N_011, N_111
    integer(kind=4) :: i, ind_fl(3), ind_cl(3)

    do i=1,N_data
        data_coord_curr = data_coord(i,:)
        data_val_curr_scaled = data_val(i) * one_over_bin_volume ! Multiply the data value by the inverse bin volume here to save computations later.

        ! Bin index
        ind_fl = floor(data_coord_curr / Delta_x + c1)

        ! Bin position
        x_ind_fl = x_min + ind_fl * Delta_x
        x_ind_cl = x_ind_fl + Delta_x ! This is the same as x_min + ind_cl*Delta_x

        ! Distances from the data point to the fl and cl bins
        Delta_x_0 = data_coord_curr - x_ind_fl
        Delta_x_1 = x_ind_cl - data_coord_curr

        ! Take the modulo
        ind_fl = modulo(ind_fl, N_bin)

        ! Calculate the ceiling
        ind_cl = ind_fl + 1

        ind_cl = modulo(ind_cl, N_bin)

        ! Add one to convert Python indexing to Fortran indexing
        ind_fl = ind_fl + 1
        ind_cl = ind_cl + 1
        
        ! The trilinear weights
        N_000 = Delta_x_1(1) * Delta_x_1(2) * Delta_x_1(3)
        N_100 = Delta_x_0(1) * Delta_x_1(2) * Delta_x_1(3)
        N_010 = Delta_x_1(1) * Delta_x_0(2) * Delta_x_1(3)
        N_110 = Delta_x_0(1) * Delta_x_0(2) * Delta_x_1(3)
        N_001 = Delta_x_1(1) * Delta_x_1(2) * Delta_x_0(3)
        N_101 = Delta_x_0(1) * Delta_x_1(2) * Delta_x_0(3)
        N_011 = Delta_x_1(1) * Delta_x_0(2) * Delta_x_0(3)
        N_111 = Delta_x_0(1) * Delta_x_0(2) * Delta_x_0(3)

        ! Accumulate the data values
        dataout(ind_fl(1), ind_fl(2), ind_fl(3)) = dataout(ind_fl(1), ind_fl(2), ind_fl(3)) & 
                                                   + N_000 * data_val_curr_scaled
        dataout(ind_cl(1), ind_fl(2), ind_fl(3)) = dataout(ind_cl(1), ind_fl(2), ind_fl(3)) &
                                                   + N_100 * data_val_curr_scaled
        dataout(ind_fl(1), ind_cl(2), ind_fl(3)) = dataout(ind_fl(1), ind_cl(2), ind_fl(3)) &
                                                   + N_010 * data_val_curr_scaled
        dataout(ind_cl(1), ind_cl(2), ind_fl(3)) = dataout(ind_cl(1), ind_cl(2), ind_fl(3)) &
                                                   + N_110 * data_val_curr_scaled
        dataout(ind_fl(1), ind_fl(2), ind_cl(3)) = dataout(ind_fl(1), ind_fl(2), ind_cl(3)) &
                                                   + N_001 * data_val_curr_scaled
        dataout(ind_cl(1), ind_fl(2), ind_cl(3)) = dataout(ind_cl(1), ind_fl(2), ind_cl(3)) &
                                                   + N_101 * data_val_curr_scaled
        dataout(ind_fl(1), ind_cl(2), ind_cl(3)) = dataout(ind_fl(1), ind_cl(2), ind_cl(3)) &
                                                   + N_011 * data_val_curr_scaled
        dataout(ind_cl(1), ind_cl(2), ind_cl(3)) = dataout(ind_cl(1), ind_cl(2), ind_cl(3)) &
                                                   + N_111 * data_val_curr_scaled

        ! Accumulate the number of times data values had been placed into these bins.
        ! The trilinear weights are by definition bewteen 0 and 1 so use ceiling to deal with
        ! data points that situate excatly on the centre of a bin.
        ! The -1e-10 is to safeguard against small values close to zero.
        weightout(ind_fl(1), ind_fl(2), ind_fl(3)) = weightout(ind_fl(1), ind_fl(2), ind_fl(3)) & 
                                                   + ceiling(N_000 * one_over_bin_volume - 1e-10)
        weightout(ind_cl(1), ind_fl(2), ind_fl(3)) = weightout(ind_cl(1), ind_fl(2), ind_fl(3)) &
                                                   + ceiling(N_100 * one_over_bin_volume - 1e-10)
        weightout(ind_fl(1), ind_cl(2), ind_fl(3)) = weightout(ind_fl(1), ind_cl(2), ind_fl(3)) &
                                                   + ceiling(N_010 * one_over_bin_volume - 1e-10)
        weightout(ind_cl(1), ind_cl(2), ind_fl(3)) = weightout(ind_cl(1), ind_cl(2), ind_fl(3)) &
                                                   + ceiling(N_110 * one_over_bin_volume - 1e-10)
        weightout(ind_fl(1), ind_fl(2), ind_cl(3)) = weightout(ind_fl(1), ind_fl(2), ind_cl(3)) &
                                                   + ceiling(N_001 * one_over_bin_volume - 1e-10)
        weightout(ind_cl(1), ind_fl(2), ind_cl(3)) = weightout(ind_cl(1), ind_fl(2), ind_cl(3)) &
                                                   + ceiling(N_101 * one_over_bin_volume - 1e-10)
        weightout(ind_fl(1), ind_cl(2), ind_cl(3)) = weightout(ind_fl(1), ind_cl(2), ind_cl(3)) &
                                                   + ceiling(N_011 * one_over_bin_volume - 1e-10)
        weightout(ind_cl(1), ind_cl(2), ind_cl(3)) = weightout(ind_cl(1), ind_cl(2), ind_cl(3)) &
                                                   + ceiling(N_111 * one_over_bin_volume - 1e-10)
    enddo
end subroutine trilinear_insert_with_wraparound



! subroutine nn_binning(data, samples, min_corners, max_corners, shape, out)
! subroutine trilinear_binning(data, samples, min_corners, max_corners, shape, out)
