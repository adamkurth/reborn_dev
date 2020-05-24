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

subroutine boxsnr(dat,mask,mask2,snr,signal,npx,npy,n_inner,n_center,n_outer)
!
! This routine computes a local signal-to-noise ratio by the following equivalent steps:
!
!   (1) For every pixel in the input data, do a local signal integration within a square region of size n_inner*2+1.
!   (2) Estimate background via a local integration within a square annulus of outer size n_outer*2 + 1 and inner size
!       n_center*2 - 1.
!   (3) From every pixel in the local signal integration square, subtract the average background value from step (2).
!   (4) Compute the standard deviation (sigma) in the square annulus.
!   (5) Divide the locally-integrated signal-minus-background by the standard error.  This standard error is equal to
!       sigma*sqrt(M) where M is the number of pixels in the locally integratied signal region, and sigma comes from
!       step (4).
!
!   Note: There are two masks that are used: one mask is for the local signal integration, and the other mask is used
!         for the local background and error annulus calculations.  The value M above, for example, is computed
!         according to the number of valid pixels, which are indicated by mask == 1.  The use of two masks allows for
!         multi-pass SNR computations in which the results of the first pass may be used to exclude high-SNR regions
!         from contributing to error estimates in the annulus.
!
!   Note: This routine will attempt to use openmp to parallelize the computations.  It is affected by the environment
!         variable OMP_NUM_THREADS
!
!                        ----------------------------------------
!                        |                                      |
!                        |Compute NOISE & BACKGROUND in annulus |
!                        |                                      |
!                        |      |------------------------|      |
!                        |      |  ignore this region    |      |
!                        |      |    |--------------|    |      |
!                        |      |    | Compute      |    |      |
!                        |      |    | integrated   |    |      |
!                        |      |    | SIGNAL minus |    |      |
!                        |      |    | BACKGROUND   |    |      |
!                        |      |    | here         |    |      |
!                        |      |    |--------------|    |      |
!                        |      |        n_inner         |      |
!                        |      |------------------------|      |
!                        |               n_center               |
!                        |                                      |
!                        |                                      |
!                        ----------------------------------------
!                                        n_outer
!
! dat : The 2D array to run the SNR filter over.
! mask : Ignore signal pixels where mask == 0 (the signal is multiplied by this mask).
! mask2 : Ignore background/noise pixels where mask2 == 0 (the background is multiplied by this mask).
! snr : The output SNR
! signal : The output background subtracted signal
! n_inner, n_center, n_outer : define the integration regions: see figure above

    real(kind=8), intent(in) :: dat(npx,npy), mask(npx,npy), mask2(npx,npy)
    real(kind=8), intent(inout) :: snr(npx,npy), signal(npx,npy)
    integer(kind=4), intent(in) :: n_inner, n_center, n_outer
    real(kind=8) :: cumix(0:npx,npy), cum2ix(0:npx,npy), cummix(0:npx,npy), &
                    cumox(0:npx,npy), cum2ox(0:npx,npy), cummox(0:npx,npy), &
                    cumiy(npx,0:npy), cum2iy(npx,0:npy), cummiy(npx,0:npy), &
                    cumcy(npx,0:npy), cum2cy(npx,0:npy), cummcy(npx,0:npy), &
                    cumoy(npx,0:npy), cum2oy(npx,0:npy), cummoy(npx,0:npy), &
                    sqix(npx,npy), sq2ix(npx,npy), sqmix(npx,npy), &
                    sqcx(npx,npy), sq2cx(npx,npy), sqmcx(npx,npy), &
                    sqox(npx,npy), sq2ox(npx,npy), sqmox(npx,npy)
    real(kind=8) :: small
    integer(kind=4) :: ix,iy,mn,mx,npx,npy

    small=1.0e-15_8

    !$OMP parallel default(None) private(ix,iy,mn,mx) &
    !$OMP shared(dat,mask,mask2,cumix,cum2ix,cummix,cumox,cum2ox,cummox,cumiy,sqix,cum2iy,sq2ix,cummiy,sqmix,cumcy, &
    !$OMP sqcx,cum2cy,sq2cx,cummcy,sqmcx,cumoy,sqox,cum2oy,sq2ox,cummoy,sqmox,npx,npy,n_inner,n_center,n_outer, &
    !$OMP small,snr,signal)

    !$OMP do schedule(static)
    do iy=1,npy
        cumix(0,iy)=0.0_8  ! cumulative data inner x
        cum2ix(0,iy)=0.0_8 ! cumulative squared inner x
        cummix(0,iy)=0.0_8 ! cumulative mask inner x

        cumox(0,iy)=0.0_8  ! outer
        cum2ox(0,iy)=0.0_8
        cummox(0,iy)=0.0_8
    enddo
    !$OMP enddo nowait

    !$OMP do schedule(static)
    do ix=1,npx
        cumiy(ix,0)=0.0_8  ! inner
        cum2iy(ix,0)=0.0_8
        cummiy(ix,0)=0.0_8

        cumcy(ix,0)=0.0_8  ! center
        cum2cy(ix,0)=0.0_8
        cummcy(ix,0)=0.0_8

        cumoy(ix,0)=0.0_8  ! outer
        cum2oy(ix,0)=0.0_8
        cummoy(ix,0)=0.0_8
    enddo
    !$OMP enddo

    !$OMP do schedule(static)
    do iy=1,npy
        do ix=1,npx  ! cumulative sums
            cumix(ix,iy)=cumix(ix-1,iy)+dat(ix,iy)*mask(ix,iy)
            cum2ix(ix,iy)=cum2ix(ix-1,iy)+(dat(ix,iy)*mask(ix,iy))**2
            cummix(ix,iy)=cummix(ix-1,iy)+mask(ix,iy)

            cumox(ix,iy)=cumox(ix-1,iy)+dat(ix,iy)*mask2(ix,iy)
            cum2ox(ix,iy)=cum2ox(ix-1,iy)+(dat(ix,iy)*mask2(ix,iy))**2
            cummox(ix,iy)=cummox(ix-1,iy)+mask2(ix,iy)
        enddo
    enddo
    !$OMP enddo


    !$OMP do schedule(static)
    do ix=1,npx  ! windowed sums on one axis
        do iy=1,npy
            mn=min(npx,ix+n_inner)
            mx=max(0,ix-n_inner-1)
            sqix(ix,iy) =cumix(mn,iy)-cumix(mx,iy)
            sq2ix(ix,iy)=cum2ix(mn,iy)-cum2ix(mx,iy)
            sqmix(ix,iy)=cummix(mn,iy)-cummix(mx,iy)

            mn=min(npx,ix+n_center)
            mx=max(0,ix-n_center-1)
            sqcx(ix,iy) =cumox(mn,iy)-cumox(mx,iy)
            sq2cx(ix,iy)=cum2ox(mn,iy)-cum2ox(mx,iy)
            sqmcx(ix,iy)=cummox(mn,iy)-cummox(mx,iy)

            mn=min(npx,ix+n_outer)
            mx=max(0,ix-n_outer-1)
            sqox(ix,iy) =cumox(mn,iy)-cumox(mx,iy)
            sq2ox(ix,iy)=cum2ox(mn,iy)-cum2ox(mx,iy)
            sqmox(ix,iy)=cummox(mn,iy)-cummox(mx,iy)
        enddo
    enddo
    !$OMP enddo

    !$OMP do schedule(static)
    do ix=1,npx
        do iy=1,npy
            cumiy(ix,iy) =cumiy(ix,iy-1) +sqix(ix,iy)
            cum2iy(ix,iy) =cum2iy(ix,iy-1)+sq2ix(ix,iy)
            cummiy(ix,iy)=cummiy(ix,iy-1)+sqmix(ix,iy)

            cumcy(ix,iy) =cumcy(ix,iy-1) +sqcx(ix,iy)
            cum2cy(ix,iy)=cum2cy(ix,iy-1)+sq2cx(ix,iy)
            cummcy(ix,iy)=cummcy(ix,iy-1)+sqmcx(ix,iy)

            cumoy(ix,iy) =cumoy(ix,iy-1) +sqox(ix,iy)
            cum2oy(ix,iy)=cum2oy(ix,iy-1)+sq2ox(ix,iy)
            cummoy(ix,iy)=cummoy(ix,iy-1)+sqmox(ix,iy)
        enddo
    enddo
    !$OMP enddo

    !$OMP do schedule(static)
    do iy=1,npy
        do ix=1,npx
            mn=min(npy,iy+n_inner)
            mx=max(0,iy-n_inner-1)
            sqix(ix,iy) =cumiy (ix,mn)-cumiy(ix,mx)
            sq2ix(ix,iy) =cum2iy (ix,mn)-cum2iy(ix,mx)
            sqmix(ix,iy)=cummiy(ix,mn)-cummiy(ix,mx)

            mn=min(npy,iy+n_center)
            mx=max(0,iy-n_center-1)
            sqcx(ix,iy) =cumcy (ix,mn)-cumcy(ix,mx)
            sq2cx(ix,iy) =cum2cy (ix,mn)-cum2cy(ix,mx)
            sqmcx(ix,iy)=cummcy(ix,mn)-cummcy(ix,mx)

            mn=min(npy,iy+n_outer)
            mx=max(0,iy-n_outer-1)
            sqox(ix,iy) =cumoy (ix,mn)-cumoy(ix,mx)
            sq2ox(ix,iy)=cum2oy(ix,mn)-cum2oy(ix,mx)
            sqmox(ix,iy)=cummoy(ix,mn)-cummoy(ix,mx)
        enddo
    enddo
    !$OMP enddo

    !$OMP do schedule(static)
    do iy=1,npy
        do ix=1,npx
            sqox(ix,iy) = sqox(ix,iy) - sqcx(ix,iy)
            sq2ox(ix,iy) = sq2ox(ix,iy) - sq2cx(ix,iy)
            sqmox(ix,iy) = sqmox(ix,iy) - sqmcx(ix,iy) + small ! avoid divide by zero
            sqmix(ix,iy) = sqmix(ix,iy) + small         ! avoid divide by zero
            signal(ix,iy) = sqix(ix,iy) - sqox(ix,iy)*sqmix(ix,iy)/sqmox(ix,iy)
            snr(ix,iy) = signal(ix,iy)/(sqrt(sqmix(ix,iy))*(sqrt(sq2ox(ix,iy)/sqmox(ix,iy) - (sqox(ix,iy)/sqmox(ix,iy))**2)+small))
        enddo
    enddo
    !$OMP enddo
    !$OMP end parallel

end subroutine boxsnr


subroutine boxconv(dat,datconv,n, npx, npy)

!
! dat : The 2D array to convolve
! datconv : The output convolved array
! n : The width of the convolution kernel
!

    real(kind=8), intent(in) :: dat(npx,npy)
    real(kind=8), intent(inout) :: datconv(npx,npy)
    integer(kind=4), intent(in) :: n
    real(kind=8) :: cumix(0:npx,npy), cumiy(npx,0:npy)
    integer(kind=4) :: ix,iy,mn,mx,npx,npy

    !$OMP parallel default(None) private(ix,iy,mn,mx) &
    !$OMP shared(dat,cumix,cumiy,datconv,sqcx,cum2cy,sq2cx,npx,npy,n)

    !$OMP do schedule(static)
    do iy=1,npy
        cumix(0,iy) = 0.0_8  ! cumulative data inner x
    enddo
    !$OMP enddo nowait

    !$OMP do schedule(static)
    do ix=1,npx
        cumiy(ix,0) = 0.0_8  ! inner
    enddo
    !$OMP enddo

    !$OMP do schedule(static)
    do iy=1,npy
        do ix=1,npx  ! cumulative sums
            cumix(ix,iy) = cumix(ix-1,iy)+dat(ix,iy)
        enddo
    enddo
    !$OMP enddo

    !$OMP do schedule(static)
    do ix=1,npx  ! windowed sums on one axis
        do iy=1,npy
            mn=min(npx,ix+n)
            mx=max(0,ix-n-1)
            datconv(ix,iy) = cumix(mn,iy)-cumix(mx,iy)
        enddo
    enddo
    !$OMP enddo

    !$OMP do schedule(static)
    do ix=1,npx
        do iy=1,npy
            cumiy(ix,iy) = cumiy(ix,iy-1) + datconv(ix,iy)
        enddo
    enddo
    !$OMP enddo

    !$OMP do schedule(static)
    do iy=1,npy
        do ix=1,npx
            mn=min(npy,iy+n)
            mx=max(0,iy-n-1)
            datconv(ix,iy) = cumiy(ix,mn)-cumiy(ix,mx)
        enddo
    enddo
    !$OMP enddo
    !$OMP end parallel

end subroutine boxconv

!module peaker
!   implicit none
!!   integer, private, parameter :: 4=selected_int_kind(9)
!!   integer, private, parameter :: 8=selected_real_kind(15,9)
!
!contains
!   subroutine squarediff(p,npx,npy,nosx,nosy,nisx,nisy)
!!
!! p(npx,npy) input array, overwritten to give output array
!! sums pixels in outer square from i-nosx to i+nosx, j-nosy to j+nosy
!! not contained in inner square from i-nisx to i+nisx, j-nisy to j+nisy
!!
!      real(kind=8), intent(inout) :: p(npx,npy)
!      integer(kind=4), intent(in) :: npx,npy,nosx,nosy,nisx,nisy
!      real(kind=8) :: cumx(0:npx,npy),cum1y(npx,0:npy),cum2y(npx,0:npy)
!      real(kind=8) :: sqox(npx,npy),sqix(npx,npy)
!      integer(kind=4) :: ix,iy
!      cumx(0,:)=0.0_8
!      do ix=1,npx
!         cumx(ix,:)=cumx(ix-1,:)+p(ix,:)
!      enddo
!      do ix=1,npx
!         sqix(ix,:)=cumx(min(npx,ix+nisx),:)-cumx(max(0,ix-nisx-1),:)
!         sqox(ix,:)=cumx(min(npx,ix+nosx),:)-cumx(max(0,ix-nosx-1),:)
!      enddo
!      cum1y(:,1)=0.0_8
!      cum2y(:,1)=0.0_8
!      do iy=1,npy
!         cum1y(:,iy)=cum1y(:,iy-1)+sqix(1:npx,iy)
!         cum2y(:,iy)=cum2y(:,iy-1)+sqox(1:npx,iy)
!      enddo
!      do iy=1,npy
!         p(:,iy)=cum2y(:,min(npy,iy+nosy))-cum2y(:,max(0,iy-nosy-1)) &
!            -cum1y(:,min(npy,iy+nisy))+cum1y(:,max(0,iy-nisy-1))
!      enddo
!
!   end subroutine squarediff
!
!
!    subroutine boxsnr(dat,mask,snr,signal,npx,npy,nin,ncent,nout)
!!
!! p(npx,npy) input array, overwritten to give output array
!! sums pixels in outer square from i-nosx to i+nosx, j-nosy to j+nosy
!! not contained in inner square from i-nisx to i+nisx, j-nisy to j+nisy
!!
!        real(kind=8), intent(in) :: dat(npx,npy), mask(npx,npy)
!        real(kind=8), intent(inout) :: snr(npx,npy), signal(npx,npy)
!        integer(kind=4), intent(in) :: npx, npy, nin, ncent, nout
!        real(kind=8) :: cumx(0:npx,npy), cum2x(0:npx,npy), cummx(0:npx,npy), &
!                        sqix(npx,npy), sq2ix(npx,npy), sqmix(npx,npy), &
!                        sqcx(npx,npy), sq2cx(npx,npy), sqmcx(npx,npy), &
!                        sqox(npx,npy), sq2ox(npx,npy), sqmox(npx,npy), &
!                        cumiy(npx,0:npy), cum2iy(npx,0:npy), cummiy(npx,0:npy), &
!                        cumcy(npx,0:npy), cum2cy(npx,0:npy), cummcy(npx,0:npy), &
!                        cumoy(npx,0:npy), cum2oy(npx,0:npy), cummoy(npx,0:npy)
!        real(kind=8) :: small
!        integer(kind=4) :: ix,iy,mn,mx
!        small=1.0e-15_8
!
!        !$OMP parallel default(None) shared(dat, mask, cumx, cum2x, cummx, cumiy,sqix, &
!        !$OMP & cum2iy,sq2ix,cummiy,sqmix,cumcy,sqcx, cum2cy,sq2cx,cummcy,sqmcx, &
!        !$OMP & cumoy,sqox,cum2oy,sq2ox,cummoy,sqmox, npx, npy, nin, ncent, nout,small,snr,signal) private(ix,iy,mn,mx)
!
!        !$OMP do schedule(static)
!        do iy=1,npy
!            cumx(0,iy)=0.0_8
!            cum2x(0,iy)=0.0_8
!            cummx(0,iy)=0.0_8
!        enddo
!        !$OMP enddo nowait
!
!        !$OMP do schedule(static)
!        do ix=1,npx
!            cumiy(ix,0)=0.0_8
!            cum2iy(ix,0)=0.0_8
!            cummiy(ix,0)=0.0_8
!            cumcy(ix,0)=0.0_8
!            cum2cy(ix,0)=0.0_8
!            cummcy(ix,0)=0.0_8
!            cumoy(ix,0)=0.0_8
!            cum2oy(ix,0)=0.0_8
!            cummoy(ix,0)=0.0_8
!        enddo
!        !$OMP enddo
!
!        !$OMP do schedule(static)
!        do ix=1,npx  ! cumulative sums
!            cumx(ix,:)=cumx(ix-1,:)+dat(ix,:)*mask(ix,:)
!            cum2x(ix,:)=cum2x(ix-1,:)+(dat(ix,:)*mask(ix,:))**2
!            cummx(ix,:)=cummx(ix-1,:)+mask(ix,:)
!        enddo
!        !$OMP enddo
!
!        !$OMP do schedule(static)
!        do ix=1,npx  ! windowed sums on one axis
!            mn=min(npx,ix+nin)
!            mx=max(0,ix-nin-1)
!            sqix(ix,:) =cumx(mn,:)-cumx(mx,:)
!            sq2ix(ix,:)=cum2x(mn,:)-cum2x(mx,:)
!            sqmix(ix,:)=cummx(mn,:)-cummx(mx,:)
!            mn=min(npx,ix+ncent)
!            mx=max(0,ix-ncent-1)
!            sqcx(ix,:) =cumx(mn,:) -cumx(mx,:)
!            sq2cx(ix,:)=cum2x(mn,:)-cum2x(mx,:)
!            sqmcx(ix,:)=cummx(mn,:)-cummx(mx,:)
!            mn=min(npx,ix+nout)
!            mx=max(0,ix-nout-1)
!            sqox(ix,:) =cumx(mn,:) -cumx(mx,:)
!            sq2ox(ix,:)=cum2x(mn,:)-cum2x(mx,:)
!            sqmox(ix,:)=cummx(mn,:)-cummx(mx,:)
!        enddo
!        !$OMP enddo
!
!
!
!        !$OMP do schedule(static)
!        do iy=1,npy
!            cumiy(:,iy) =cumiy(:,iy-1) +sqix(1:npx,iy)
!            cum2iy(:,iy) =cum2iy(:,iy-1)+sq2ix(1:npx,iy)
!            cummiy(:,iy)=cummiy(:,iy-1)+sqmix(1:npx,iy)
!            cumcy(:,iy) =cumcy(:,iy-1) +sqcx(1:npx,iy)
!
!            cum2cy(:,iy)=cum2cy(:,iy-1)+sq2cx(1:npx,iy)
!            cummcy(:,iy)=cummcy(:,iy-1)+sqmcx(1:npx,iy)
!
!            cumoy(:,iy) =cumoy(:,iy-1) +sqox(1:npx,iy)
!            cum2oy(:,iy)=cum2oy(:,iy-1)+sq2ox(1:npx,iy)
!            cummoy(:,iy)=cummoy(:,iy-1)+sqmox(1:npx,iy)
!        enddo
!        !$OMP end do
!
!        !$OMP do schedule(static)
!        do iy=1,npy
!            mn=min(npy,iy+nin)
!            mx=max(0,iy-nin-1)
!            sqix(:,iy) =cumiy (:,mn)-cumiy(:,mx)
!            sq2ix(:,iy) =cum2iy (:,mn)-cum2iy(:,mx)
!            sqmix(:,iy)=cummiy(:,mn)-cummiy(:,mx)
!            mn=min(npy,iy+ncent)
!            mx=max(0,iy-ncent-1)
!            sqcx(:,iy) =cumcy (:,mn)-cumcy(:,mx)
!            sq2cx(:,iy) =cum2cy (:,mn)-cum2cy(:,mx)
!            sqmcx(:,iy)=cummcy(:,mn)-cummcy(:,mx)
!            mn=min(npy,iy+nout)
!            mx=max(0,iy-nout-1)
!            sqox(:,iy) =cumoy (:,mn)-cumoy(:,mx)
!            sq2ox(:,iy)=cum2oy(:,mn)-cum2oy(:,mx)
!            sqmox(:,iy)=cummoy(:,mn)-cummoy(:,mx)
!        enddo
!        !$OMP end do
!
!
!        !$OMP do schedule(static)
!        do iy=1,npy
!            sqox(:,iy) = sqox(:,iy) - sqcx(:,iy)
!            sq2ox(:,iy) = sq2ox(:,iy) - sq2cx(:,iy)
!            sqmox(:,iy) = sqmox(:,iy) - sqmcx(:,iy) + small ! avoid divide by zero
!            sqmix(:,iy) = sqmix(:,iy) + small         ! avoid divide by zero
!            signal(:,iy) = sqix(:,iy) - sqox(:,iy)*sqmix(:,iy)/sqmox(:,iy)
!            snr(:,iy) = signal(:,iy)/(sqrt(sqmix(:,iy))*(sqrt(sq2ox(:,iy)/sqmox(:,iy) - (sqox(:,iy)/sqmox(:,iy))**2)+small))
!        enddo
!        !$OMP end do
!        !$OMP end parallel
!    end subroutine boxsnr
!
!end module peaker

!subroutine peak_snr_filter(data, a, b, c, mask, local_max_only, snr, signal)
!    implicit none
!    real(kind=8), intent(in)     :: data(:, :), mask(:, :)
!    integer(kind=4), intent(in)  :: local_max_only, a, b, c
!    real(kind=8), intent(inout)  :: snr(:, :), signal(:, :)
!    integer(kind=4) :: i, j, n_local, n_annulus, q, r, ii, q2, jj, nss, nfs
!    real(kind=8)    :: this_val, local_signal, local_signal2, annulus_signal, annulus_signal2, rad, rad2, noise
!    nfs = size(data,1)
!    nss = size(data,2)
!    do i = 2,nfs-1
!        do j = 2,nss-1
!            if (mask(j,i) == 0) cycle
!            if (local_max_only == 1) then
!                this_val = data(j,i)  ! compare to max
!                if (data(j  , i-1) > this_val) cycle
!                if (data(j  , i+1) > this_val) cycle
!                if (data(j-1,   i) > this_val) cycle
!                if (data(j+1,   i) > this_val) cycle
!                if (data(j-1, i-1) > this_val) cycle
!                if (data(j-1, i+1) > this_val) cycle
!                if (data(j+1, i-1) > this_val) cycle
!                if (data(j+1, i+1) > this_val) cycle
!            end if
!            local_signal = 0
!            local_signal2 = 0
!            n_local = 0
!            annulus_signal = 0
!            annulus_signal2 = 0
!            n_annulus = 0
!            do q = -c,c
!                ii = i + q
!                if (ii < 1) cycle
!                if (ii > nss) cycle
!                q2 = q**2
!                do r = -c,c
!                    jj = j+r
!                    if (jj < 1) cycle
!                    if (jj > nfs) cycle
!                    if (mask(jj,ii) == 0) cycle
!                    rad2 = q2+r**2
!                    rad = sqrt(rad2)
!                    if (rad <= a) then
!                        n_local = n_local +1
!                        local_signal = local_signal + data(jj,ii)
!                        local_signal2 = local_signal2 + data(jj,ii)**2
!                    end if
!                    if ((rad >= b).and.(rad <= c)) then
!                        n_annulus = n_annulus +1
!                        annulus_signal = annulus_signal + data(jj,ii)
!                        annulus_signal2 = annulus_signal2 + data(jj,ii)**2
!                    end if
!                end do
!            end do
!            if ((n_local==0).or.(n_annulus < 2)) then
!                signal(j,i) = 0
!                snr(j,i) = 0
!            else
!                signal(j,i) = local_signal/n_local - annulus_signal/n_annulus
!                noise = sqrt(annulus_signal2/n_annulus - (annulus_signal/n_annulus)**2)
!                if (noise == 0) then
!                    snr(j,i) = 0
!                else
!                    snr(j,i) = signal(j,i)/noise
!                end if
!            end if
!        end do
!    end do
!end subroutine peak_snr_filter

!subroutine peak_snr_filter2(data, bind, cind, mask, local_max_only, snr, signal)
!    implicit none
!    real(kind=8), intent(in)     :: data(:, :), mask(:, :)
!    integer(kind=4), intent(in)  :: local_max_only, bind(:), cind(:)
!    real(kind=8), intent(inout)  :: snr(:, :), signal(:, :)
!    integer(kind=4) :: i, j, n_local, n_annulus, q, ii, jj, nss, nfs, n_bk
!    real(kind=8)    :: this_val, annulus_signal, annulus_signal2, noise
!    nfs = size(data,1)
!    nss = size(data,2)
!    n_bk = size(bind,1)
!    do i = 2,nfs-1
!        do j = 2,nss-1
!            if (mask(j,i) == 0) cycle
!            if (local_max_only == 1) then
!                this_val = data(j,i)
!                if (data(j  , i-1) > this_val) cycle
!                if (data(j  , i+1) > this_val) cycle
!                if (data(j-1,   i) > this_val) cycle
!                if (data(j+1,   i) > this_val) cycle
!                if (data(j-1, i-1) > this_val) cycle
!                if (data(j-1, i+1) > this_val) cycle
!                if (data(j+1, i-1) > this_val) cycle
!                if (data(j+1, i+1) > this_val) cycle
!            end if
!            annulus_signal = 0
!            annulus_signal2 = 0
!            n_annulus = 0
!            do q = 1,n_bk
!                ii = i + bind(q)
!                if (ii < 1) cycle
!                if (ii > nss) cycle
!                jj = j + cind(q)
!                if (jj < 1) cycle
!                if (jj > nfs) cycle
!                if (mask(jj,ii) == 0) cycle
!                n_annulus = n_annulus +1
!                annulus_signal = annulus_signal + data(jj,ii)
!                annulus_signal2 = annulus_signal2 + data(jj,ii)**2
!            end do
!            if ((n_local==0).or.(n_annulus < 2)) then
!                signal(j,i) = 0
!                snr(j,i) = 0
!            else
!                signal(j,i) = data(j,i) - annulus_signal/n_annulus
!                noise = sqrt(annulus_signal2/n_annulus - (annulus_signal/n_annulus)**2)
!                if (noise == 0) then
!                    snr(j,i) = 0
!                else
!                    snr(j,i) = signal(j,i)/noise
!                end if
!            end if
!        end do
!    end do
!end subroutine peak_snr_filter2
