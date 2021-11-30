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

subroutine max_pair_distance(vectors, out)
    implicit none
    real(kind=8), intent(inout) :: out(:)
    real(kind=8), intent(in) :: vectors(:,:)
    real(kind=8) :: d_max, d_vec(3), d
    integer(kind=4) :: n, i, j
    n = size(vectors, 2)
    d_max = 0
    do i=1,n
        do j=i,n
            d_vec = vectors(:,i) - vectors(:,j)
            d = d_vec(1)*d_vec(1) + d_vec(2)*d_vec(2) + d_vec(3)*d_vec(3)
            if (d > d_max) then
                d_max = d
            end if
        end do
    enddo
    out(1) = sqrt(d_max)
end subroutine max_pair_distance
