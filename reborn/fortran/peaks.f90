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
    real(kind=8) :: cumix(0:npx,npy), cummix(0:npx,npy), &
                    cumox(0:npx,npy), cum2ox(0:npx,npy), cummox(0:npx,npy), &
                    cumiy(npx,0:npy), cummiy(npx,0:npy), &
                    cumcy(npx,0:npy), cum2cy(npx,0:npy), cummcy(npx,0:npy), &
                    cumoy(npx,0:npy), cum2oy(npx,0:npy), cummoy(npx,0:npy), &
                    sqix(npx,npy), sqmix(npx,npy), &
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
        cummix(0,iy)=0.0_8 ! cumulative mask inner x

        cumox(0,iy)=0.0_8  ! outer
        cum2ox(0,iy)=0.0_8
        cummox(0,iy)=0.0_8
    enddo
    !$OMP enddo nowait

    !$OMP do schedule(static)
    do ix=1,npx
        cumiy(ix,0)=0.0_8  ! inner
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
