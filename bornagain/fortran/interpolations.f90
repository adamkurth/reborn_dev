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
