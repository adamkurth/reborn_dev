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

subroutine fortran_indexing(out1, out2, out3)
    implicit none
    real(kind=8), intent(inout) :: out1(:), out2(:,:), out3(:,:,:)
    out1(2) = 10
    out2(2,1) = 10
    out3(2,1,1) = 10
end subroutine fortran_indexing