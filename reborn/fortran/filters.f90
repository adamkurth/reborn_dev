module utils
contains
subroutine convolve(dat,datconv,n,cumx,cumy)
    ! dat : The 2D array to convolve
    ! datconv : The output convolved array
    ! n : The width of the convolution kernel
    real(kind=8), intent(in) :: dat(:,:)
    real(kind=8), intent(inout) :: datconv(:,:),cumx(0:,:),cumy(:,0:)
    integer(kind=4), intent(in) :: n
    integer(kind=4) :: ix,iy,mn,mx,npx,npy
    !$OMP parallel default(None) private(ix,iy,mn,mx) &
    !$OMP shared(dat,cumx,cumy,datconv,sqcx,cum2cy,sq2cx,npx,npy,n)
    npx = size(dat,1)
    npy = size(dat,2)
    !$OMP do schedule(static)
    do iy=1,npy
        cumx(0,iy) = 0.0_8  ! cumulative data inner x
    enddo
    !$OMP enddo nowait
    !$OMP do schedule(static)
    do ix=1,npx
        cumy(ix,0) = 0.0_8  ! inner
    enddo
    !$OMP enddo
    !$OMP do schedule(static)
    do iy=1,npy
        do ix=1,npx  ! cumulative sums
            cumx(ix,iy) = cumx(ix-1,iy)+dat(ix,iy)
        enddo
    enddo
    !$OMP enddo
    !$OMP do schedule(static)
    do ix=1,npx  ! windowed sums on one axis
        do iy=1,npy
            mn=min(npx,ix+n)
            mx=max(0,ix-n-1)
            datconv(ix,iy) = cumx(mn,iy)-cumx(mx,iy)
        enddo
    enddo
    !$OMP enddo
    !$OMP do schedule(static)
    do ix=1,npx
        do iy=1,npy
            cumy(ix,iy) = cumy(ix,iy-1) + datconv(ix,iy)
        enddo
    enddo
    !$OMP enddo
    !$OMP do schedule(static)
    do iy=1,npy
        do ix=1,npx
            mn=min(npy,iy+n)
            mx=max(0,iy-n-1)
            datconv(ix,iy) = cumy(ix,mn)-cumy(ix,mx)
        enddo
    enddo
    !$OMP enddo
    !$OMP end parallel
end subroutine convolve
subroutine divide_inplace(a,b,n,m)
    real(kind=8), intent(in) :: a(n,m), b(n,m)

end subroutine divide_inplace
end module utils


subroutine boxsnr(dat,maskin,maskout,snr,signal,npx,npy,n_inner,n_center,n_outer)
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
    real(kind=8), intent(in) :: dat(npx,npy), maskin(npx,npy), maskout(npx,npy)
    real(kind=8), intent(inout) :: snr(npx,npy), signal(npx,npy)
    integer(kind=4), intent(in) :: n_inner, n_center, n_outer
    real(kind=8) :: cumx(0:npx,npy),cumy(npx,0:npy),datin(npx,npy),datout(npx,npy),dat2out(npx,npy)
    real(kind=8) :: back(npx,npy),countin(npx,npy),countan(npx,npy)
    real(kind=8) :: small
    integer(kind=4) :: ix,iy,mn,mx,npx,npy
    small=1.0e-15_8
    cumx = 0.0_8
    cumy = 0.0_8
    datin = dat*maskin
    datout = dat*maskout
    dat2out = dat**2*maskout
    call convolve(datin,back,n_inner,cumx,cumy)
    call convolve(maskin,countin,n_inner,cumx,cumy)
    call convolve(maskout,countout,n_inner,cumx,cumy)

end subroutine boxsnr
