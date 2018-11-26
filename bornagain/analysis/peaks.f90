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


module peaker
   implicit none
!   integer, private, parameter :: 4=selected_int_kind(9)
!   integer, private, parameter :: 8=selected_real_kind(15,9)

contains
   subroutine squarediff(p,npx,npy,nosx,nosy,nisx,nisy)
!
! p(npx,npy) input array, overwritten to give output array
! sums pixels in outer square from i-nosx to i+nosx, j-nosy to j+nosy
! not contained in inner square from i-nisx to i+nisx, j-nisy to j+nisy
!
      real(kind=8), intent(inout) :: p(npx,npy)
      integer(kind=4), intent(in) :: npx,npy,nosx,nosy,nisx,nisy
      real(kind=8) :: cumx(0:npx,npy),cum1y(npx,0:npy),cum2y(npx,0:npy)
      real(kind=8) :: sqox(npx,npy),sqix(npx,npy)
      integer(kind=4) :: ix,iy
      cumx(0,:)=0.0_8
      do ix=1,npx
         cumx(ix,:)=cumx(ix-1,:)+p(ix,:)
      enddo
      do ix=1,npx
         sqix(ix,:)=cumx(min(npx,ix+nisx),:)-cumx(max(0,ix-nisx-1),:)
         sqox(ix,:)=cumx(min(npx,ix+nosx),:)-cumx(max(0,ix-nosx-1),:)
      enddo
      cum1y(:,1)=0.0_8
      cum2y(:,1)=0.0_8
      do iy=1,npy
         cum1y(:,iy)=cum1y(:,iy-1)+sqix(1:npx,iy)
         cum2y(:,iy)=cum2y(:,iy-1)+sqox(1:npx,iy)
      enddo
      do iy=1,npy
         p(:,iy)=cum2y(:,min(npy,iy+nosy))-cum2y(:,max(0,iy-nosy-1)) &
            -cum1y(:,min(npy,iy+nisy))+cum1y(:,max(0,iy-nisy-1))
      enddo

   end subroutine squarediff


    subroutine boxsnr(dat,mask,out,npx,npy,nin,ncent,nout)
!
! p(npx,npy) input array, overwritten to give output array
! sums pixels in outer square from i-nosx to i+nosx, j-nosy to j+nosy
! not contained in inner square from i-nisx to i+nisx, j-nisy to j+nisy
!
        real(kind=8), intent(in) :: dat(npx,npy), mask(npx,npy)
        real(kind=8), intent(inout) :: out(npx,npy)
        integer(kind=4), intent(in) :: npx, npy, nin, ncent, nout
        real(kind=8) :: cumx(0:npx,npy), cum2x(0:npx,npy), cummx(0:npx,npy), &
                        sqix(npx,npy), sq2ix(npx,npy), sqmix(npx,npy), &
                        sqcx(npx,npy), sq2cx(npx,npy), sqmcx(npx,npy), &
                        sqox(npx,npy), sq2ox(npx,npy), sqmox(npx,npy), &
                        cumiy(npx,0:npy), cum2iy(npx,0:npy), cummiy(npx,0:npy), &
                        cumcy(npx,0:npy), cum2cy(npx,0:npy), cummcy(npx,0:npy), &
                        cumoy(npx,0:npy), cum2oy(npx,0:npy), cummoy(npx,0:npy)
        real(kind=8) :: small
        integer(kind=4) :: ix,iy,mn,mx
        small=1.0e-15_8
        cumx(0,:)=0.0_8
        cum2x(0,:)=0.0_8
        cummx(0,:)=0.0_8
        do ix=1,npx  ! cumulative sums
            cumx(ix,:)=cumx(ix-1,:)+dat(ix,:)*mask(ix,:)
            cum2x(ix,:)=cum2x(ix-1,:)+(dat(ix,:)*mask(ix,:))**2
            cummx(ix,:)=cummx(ix-1,:)+mask(ix,:)
        enddo
        do ix=1,npx  ! windowed sums on one axis
            mn=min(npx,ix+nin)
            mx=max(0,ix-nin-1)
            sqix(ix,:) =cumx(mn,:)-cumx(mx,:)
            sq2ix(ix,:)=cum2x(mn,:)-cum2x(mx,:)
            sqmix(ix,:)=cummx(mn,:)-cummx(mx,:)
            mn=min(npx,ix+ncent)
            mx=max(0,ix-ncent-1)
            sqcx(ix,:) =cumx(mn,:) -cumx(mx,:)
            sq2cx(ix,:)=cum2x(mn,:)-cum2x(mx,:)
            sqmcx(ix,:)=cummx(mn,:)-cummx(mx,:)
            mn=min(npx,ix+nout)
            mx=max(0,ix-nout-1)
            sqox(ix,:) =cumx(mn,:) -cumx(mx,:)
            sq2ox(ix,:)=cum2x(mn,:)-cum2x(mx,:)
            sqmox(ix,:)=cummx(mn,:)-cummx(mx,:)
        enddo
        cumiy(:,0)=0.0_8
        cum2iy(:,0)=0.0_8
        cummiy(:,0)=0.0_8
        cumcy(:,0)=0.0_8
        cum2cy(:,0)=0.0_8
        cummcy(:,0)=0.0_8
        cumoy(:,0)=0.0_8
        cum2oy(:,0)=0.0_8
        cummoy(:,0)=0.0_8
        do iy=1,npy
            cumiy(:,iy) =cumiy(:,iy-1) +sqix(1:npx,iy)
            cum2iy(:,iy) =cum2iy(:,iy-1)+sq2ix(1:npx,iy)
            cummiy(:,iy)=cummiy(:,iy-1)+sqmix(1:npx,iy)
            cumcy(:,iy) =cumcy(:,iy-1) +sqcx(1:npx,iy)
            cum2cy(:,iy)=cum2cy(:,iy-1)+sq2cx(1:npx,iy)
            cummcy(:,iy)=cummcy(:,iy-1)+sqmcx(1:npx,iy)
            cumoy(:,iy) =cumoy(:,iy-1) +sqox(1:npx,iy)
            cum2oy(:,iy)=cum2oy(:,iy-1)+sq2ox(1:npx,iy)
            cummoy(:,iy)=cummoy(:,iy-1)+sqmox(1:npx,iy)
        enddo
        do iy=1,npy
            mn=min(npy,iy+nin)
            mx=max(0,iy-nin-1)
            sqix(:,iy) =cumiy (:,mn)-cumiy(:,mx)
            sq2ix(:,iy) =cum2iy (:,mn)-cum2iy(:,mx)
            sqmix(:,iy)=cummiy(:,mn)-cummiy(:,mx)
            mn=min(npy,iy+ncent)
            mx=max(0,iy-ncent-1)
            sqcx(:,iy) =cumcy (:,mn)-cumcy(:,mx)
            sq2cx(:,iy) =cum2cy (:,mn)-cum2cy(:,mx)
            sqmcx(:,iy)=cummcy(:,mn)-cummcy(:,mx)
            mn=min(npy,iy+nout)
            mx=max(0,iy-nout-1)
            sqox(:,iy) =cumoy (:,mn)-cumoy(:,mx)
            sq2ox(:,iy)=cum2oy(:,mn)-cum2oy(:,mx)
            sqmox(:,iy)=cummoy(:,mn)-cummoy(:,mx)
        enddo
        sqox = sqox - sqcx
        sq2ox = sq2ox - sq2cx
        sqmox = sqmox - sqmcx + small ! avoid divide by zero
        sqmix = sqmix + small         ! avoid divide by zero
        out = (sqix-sqox*sqmix/sqmox)/(sqrt(sqmix)*(sqrt(sq2ox/sqmox - (sqox/sqmox)**2)+small))
    end subroutine boxsnr

end module peaker




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