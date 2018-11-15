subroutine peak_snr_filter(data, a, b, c, mask, local_max_only, snr, signal)
    implicit none
    real(kind=8), intent(in)     :: data(:, :), mask(:, :)
    integer(kind=4), intent(in)  :: local_max_only, a, b, c
    real(kind=8), intent(inout)  :: snr(:, :), signal(:, :)
    integer(kind=4) :: i, j, n_local, n_annulus, q, r, ii, q2, jj, nss, nfs
    real(kind=8)    :: this_val, local_signal, local_signal2, annulus_signal, annulus_signal2, rad, rad2, noise
    nfs = size(data,1)
    nss = size(data,2)
    do i = 2,nfs-1
        do j = 2,nss-1
            if (mask(j,i) == 0) cycle
            if (local_max_only == 1) then
                this_val = data(j,i)  ! compare to max
                if (data(j  , i-1) > this_val) cycle
                if (data(j  , i+1) > this_val) cycle
                if (data(j-1,   i) > this_val) cycle
                if (data(j+1,   i) > this_val) cycle
                if (data(j-1, i-1) > this_val) cycle
                if (data(j-1, i+1) > this_val) cycle
                if (data(j+1, i-1) > this_val) cycle
                if (data(j+1, i+1) > this_val) cycle
            end if
            local_signal = 0
            local_signal2 = 0
            n_local = 0
            annulus_signal = 0
            annulus_signal2 = 0
            n_annulus = 0
            do q = -c,c
                ii = i + q
                if (ii < 1) cycle
                if (ii > nss) cycle
                q2 = q**2
                do r = -c,c
                    jj = j+r
                    if (jj < 1) cycle
                    if (jj > nfs) cycle
                    if (mask(jj,ii) == 0) cycle
                    rad2 = q2+r**2
                    rad = sqrt(rad2)
                    if (rad <= a) then
                        n_local = n_local +1
                        local_signal = local_signal + data(jj,ii)
                        local_signal2 = local_signal2 + data(jj,ii)**2
                    end if
                    if ((rad >= b).and.(rad <= c)) then
                        n_annulus = n_annulus +1
                        annulus_signal = annulus_signal + data(jj,ii)
                        annulus_signal2 = annulus_signal2 + data(jj,ii)**2
                    end if
                end do
            end do
            if ((n_local==0).or.(n_annulus < 2)) then
                signal(j,i) = 0
                snr(j,i) = 0
            else
                signal(j,i) = local_signal/n_local - annulus_signal/n_annulus
                noise = sqrt(annulus_signal2/n_annulus - (annulus_signal/n_annulus)**2)
                if (noise == 0) then
                    snr(j,i) = 0
                else
                    snr(j,i) = signal(j,i)/noise
                end if
            end if
        end do
    end do
end subroutine peak_snr_filter


subroutine peak_snr_filter2(data, bind, cind, mask, local_max_only, snr, signal)
    implicit none
    real(kind=8), intent(in)     :: data(:, :), mask(:, :)
    integer(kind=4), intent(in)  :: local_max_only, bind(:), cind(:)
    real(kind=8), intent(inout)  :: snr(:, :), signal(:, :)
    integer(kind=4) :: i, j, n_local, n_annulus, q, ii, jj, nss, nfs, n_bk
    real(kind=8)    :: this_val, annulus_signal, annulus_signal2, noise
    nfs = size(data,1)
    nss = size(data,2)
    n_bk = size(bind,1)
    do i = 2,nfs-1
        do j = 2,nss-1
            if (mask(j,i) == 0) cycle
            if (local_max_only == 1) then
                this_val = data(j,i)
                if (data(j  , i-1) > this_val) cycle
                if (data(j  , i+1) > this_val) cycle
                if (data(j-1,   i) > this_val) cycle
                if (data(j+1,   i) > this_val) cycle
                if (data(j-1, i-1) > this_val) cycle
                if (data(j-1, i+1) > this_val) cycle
                if (data(j+1, i-1) > this_val) cycle
                if (data(j+1, i+1) > this_val) cycle
            end if
            annulus_signal = 0
            annulus_signal2 = 0
            n_annulus = 0
            do q = 1,n_bk
                ii = i + bind(q)
                if (ii < 1) cycle
                if (ii > nss) cycle
                jj = j + cind(q)
                if (jj < 1) cycle
                if (jj > nfs) cycle
                if (mask(jj,ii) == 0) cycle
                n_annulus = n_annulus +1
                annulus_signal = annulus_signal + data(jj,ii)
                annulus_signal2 = annulus_signal2 + data(jj,ii)**2
            end do
            if ((n_local==0).or.(n_annulus < 2)) then
                signal(j,i) = 0
                snr(j,i) = 0
            else
                signal(j,i) = data(j,i) - annulus_signal/n_annulus
                noise = sqrt(annulus_signal2/n_annulus - (annulus_signal/n_annulus)**2)
                if (noise == 0) then
                    snr(j,i) = 0
                else
                    snr(j,i) = signal(j,i)/noise
                end if
            end if
        end do
    end do
end subroutine peak_snr_filter2