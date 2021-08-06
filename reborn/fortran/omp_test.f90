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

subroutine get_max_threads(n)
    !$ use omp_lib
    implicit none
    integer(kind=4), intent(inout) :: n(1)
    integer(kind=4) :: m
    n(1) = 1
    m = 1
    !$ n(1) = omp_get_max_threads()
    !$ m = omp_get_thread_num()
end subroutine get_max_threads


subroutine omp_test()
    !$ use omp_lib
    integer(kind=4) :: n
    n = 1
    !$ n = omp_get_max_threads()
    !$ m = omp_get_thread_num()
    write(*,*) n
    !$omp parallel
    write(*,*) 'Hello World'
    !$omp end parallel
end subroutine omp_test
