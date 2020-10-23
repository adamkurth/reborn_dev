subroutine omp_test()
    !$omp parallel
    write(*,*) 'Hello World'
    !$omp end parallel
end subroutine omp_test