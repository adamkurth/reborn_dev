subroutine density1d(x_max, x_min, &
                     n_bin, n_atoms, n_profs, n_subarray, &
                     zs, z_loc, cromer_mann_densities, prof_dr, density_map)
    ! Make Comer-Mann atomic density map.
    ! 
    ! Coordinates refer to the centers of bins, not the edges.
    ! 
    ! Example: n_bin = 4 and atom position r = 0.75
    !          * is the bin center
    !          | is the bin edge
    !          X is the atom location
    ! 
    ! |_____*_____|__X__*_____|_____*_____|_____*_____|
    implicit none
    ! inputs
    integer(4), intent(in) :: n_bin, n_atoms, n_profs, n_subarray, zs(:)
    real(8), intent(in) :: x_max, x_min, prof_dr, z_loc(:), cromer_mann_densities(:,:)
    ! outputs
    real(8), intent(out) :: density_map(n_bin)
    ! subroutine variables
    integer(4) :: z, atom_n, r_indx, global_indx, pir, i, md, sub_map_size
    real(8) :: dx, r, f, prof_indx, r_map, prof(n_profs), sub_map(2 * n_subarray + 1)
    sub_map_size = size(sub_map)
    density_map = 0.0  ! Comer-Mann density map
    dx = (x_max - x_min) / (n_bin - 1)
    do atom_n = 1, n_atoms
        z = zs(atom_n)  ! pick out atom
        r = z_loc(atom_n)  ! get atom location
        f = z  ! scattering factor. atom should integrate to this value
        prof(:) = cromer_mann_densities(:, z + 1)  ! Crommer-Mann density of atom
        r_indx = int(nint((r - x_min) / dx))  ! nearest index (atom can be anywhere in the bin)
        ! instead of calculating the atom density over the whole domain
        ! we only integrate the density in the neighborhood of the atom
        sub_map = 0.0  ! sub-array for building up and normalizing one-atom density
        do i = 1, sub_map_size  ! i is index in temp map
            global_indx = r_indx - n_subarray + i  ! global index in density_map
            r_map = global_indx * dx + x_min  ! coordinate at global index
            ! radial profile index: distance from this pixel to atom / prof_dr
            prof_indx = sqrt((r - r_map) ** 2) / prof_dr
            pir = int(floor(prof_indx))  ! Index rounded down
            if (pir > n_profs) then  ! Skip when out of range of provided radial profile
                continue
            endif
            ! Linear interpolation
            sub_map(i) = prof(pir + 1) * (1 - (prof_indx - pir)) &
                         + prof(pir + 2) * (prof_indx - pir)
        enddo
        sub_map = sub_map * f / sum(sub_map)  ! Normalize the sampled atom density
        do i = 1, sub_map_size  ! i is index in sub array
            global_indx = r_indx - n_subarray + i  ! global index in map
            ! Note the modulus here - we wrap around when positions are out of bounds
            md = modulo(global_indx, n_bin)
            density_map(md + 1) = density_map(md + 1) + sub_map(i)
        enddo
    enddo
end subroutine density1d