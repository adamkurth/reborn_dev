module utils
contains
function matinv3(A)
    !! Performs a direct calculation of the inverse of a 3×3 matrix.
    real(kind=8), intent(in)    :: A(3,3)
    real(kind=8) :: matinv3(3,3)
    real(kind=8)                :: detinv
    ! Calculate the inverse determinant of the matrix
    detinv = 1/(A(1,1)*A(2,2)*A(3,3) - A(1,1)*A(2,3)*A(3,2)&
              - A(1,2)*A(2,1)*A(3,3) + A(1,2)*A(2,3)*A(3,1)&
              + A(1,3)*A(2,1)*A(3,2) - A(1,3)*A(2,2)*A(3,1))
    ! Calculate the inverse of the matrix
    matinv3(1,1) = +detinv * (A(2,2)*A(3,3) - A(2,3)*A(3,2))
    matinv3(2,1) = -detinv * (A(2,1)*A(3,3) - A(2,3)*A(3,1))
    matinv3(3,1) = +detinv * (A(2,1)*A(3,2) - A(2,2)*A(3,1))
    matinv3(1,2) = -detinv * (A(1,2)*A(3,3) - A(1,3)*A(3,2))
    matinv3(2,2) = +detinv * (A(1,1)*A(3,3) - A(1,3)*A(3,1))
    matinv3(3,2) = -detinv * (A(1,1)*A(3,2) - A(1,2)*A(3,1))
    matinv3(1,3) = +detinv * (A(1,2)*A(2,3) - A(1,3)*A(2,2))
    matinv3(2,3) = -detinv * (A(1,1)*A(2,3) - A(1,3)*A(2,1))
    matinv3(3,3) = +detinv * (A(1,1)*A(2,2) - A(1,2)*A(2,1))
end function matinv3
function identity(val)
    !! make an identity (populate diagonal with value)
    real(kind=8), intent(in) :: val
    real(kind=8) :: identity(3,3)
    integer(kind=4) :: j
    A = 0
    do j=1,3
        identity(j,j) = val
    end do
end function identity
function outer(v)
    !! takes the outer product of 3 vectors: Aij = vi vj
    real(kind=8), intent(in) :: v(3)
    real(kind=8) :: outer(3,3)
    integer(kind=4) :: j,k
    do k=1,3
    do j=1,3
        outer(k, j)=v(k)*v(j)
    end do
    end do
end function outer
function normvec(v)
    !! Normalize the vector
    real(kind=8), intent(in) :: v(3)
    real(kind=8) :: normvec(3)
    normvec = v / sqrt(v(1)*v(1)+v(2)*v(2)+v(3)*v(3))
end function normvec
function vecmag(v)
    !! Normalize the vector
    real(kind=8), intent(in) :: v(3)
    real(kind=8) :: vecmag
    vecmag = sqrt(v(1)*v(1)+v(2)*v(2)+v(3)*v(3))
end function vecmag
end module utils

subroutine gaussian_crystal(qin, kin, A, F, R, var_siz, var_mos, var_wav, var_div, Iout)
    ! qin: Inpug q vectors.  Vector components contiguous.
    ! A: Reciprocal lattice vectors.  Vector components contiguous.
    ! F: Structure factors.  3D array.  Last component "l" of "hkl" is contiguous.
    ! var_siz: Variance due to crystal size.
    ! var_mos: Variance due to crystal mosaicity.  Radian units.
    ! var_wav: Variance of dE/E.
    ! Iout: Output intensities.
    use utils
    implicit none
    real(kind=8), intent(in) :: qin(:,:),kin(:),A(3,3),F(:,:,:),R(3,3),var_siz,var_mos,var_wav,var_div
    real(kind=8), intent(inout) :: Iout(:)
    real(kind=8) :: q(3),h(3),hh(3),Ainv(3, 3),RA(3,3),RAinv(3,3),g(3),dq(3),kout(3) !, dqn(3)
    real(kind=8) :: cov_siz(3,3),cov_mos(3,3),cov_wav(3,3),cov_div(3,3),cov(3,3),id(3,3)
    integer(kind=4) :: nq, i, nh, nk, nl, hp, kp, lp
    nq = size(qin, 2)
    nh = size(F, 3)
    nk = size(F, 2)
    nl = size(F, 1)
    Ainv = matinv3(A)
    id = identity(1.0_8)
    cov_siz = 0._8
    cov_div = 0._8
    cov_wav = 0._8
    cov_mos = 0._8
    if (var_siz > 0._8) then
        cov_siz = identity(var_siz)
    end if
    if (var_div > 0._8) then
        cov_div = (id - outer(normvec(kin))) * vecmag(kin)**2 * var_div
    end if
    !$OMP parallel default(None) private(i,q,kout,h,g,dq,cov_wav,cov_mos,v,c,cov,cov_inv,hp,kp,lp,hh) &
    !$OMP shared(var_mos,var_siz,var_wav,qin,id,nq,Iout,A,Ainv,kin,R,RA,RAinv,cov_siz,cov_div)
    !$OMP do schedule(static)
    RA = matmul(R, A)
    RAinv = matmul(R, Ainv)
    do i=1,nq
!        Iout(i) = 0
        q = qin(:,i)
        kout = q + kin
        h = nint(matmul(RAinv, q))
!        do hp=-1,1
!        do kp=-1,1
!        do lp=-1,1
!        hh = h
!        hh(1) = hh(1)+hp
!        hh(2) = hh(2)+kp
!        hh(3) = hh(3)+lp
!        g = matmul(RA,hh)
        g = matmul(RA,h)
        dq = q - g
        if (var_wav > 0._8) then
            cov_wav = outer(q)*var_wav
        end if
        if (var_mos > 0._8) then
            cov_mos = (id - outer(normvec(g))) * vecmag(g)**2 * var_mos
        end if
        cov = cov_siz + cov_wav + cov_mos + cov_div
        Iout(i) = exp(-dot_product(dq,matmul(matinv3(cov),  dq)))
!        Iout(i) = Iout(i) + exp(-dot_product(dq,matmul(matinv3(cov),  dq)))
!        end do
!        end do
!        end do
    end do
    !$OMP enddo
    !$OMP end parallel
end subroutine gaussian_crystal
