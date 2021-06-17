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
