! Copyright 2025 Pasteur Labs. All Rights Reserved.
! SPDX-License-Identifier: Apache-2.0
!
! 1D Transient Heat Equation Solver
! =================================
! Solves: dT/dt = alpha * d^2T/dx^2
! Method: Explicit finite difference (forward Euler in time, central difference in space)
!
! Usage: heat_solver <input_file> <output_file>
!
! Input file format (text, one parameter per line):
!   n_points <integer>
!   n_steps <integer>
!   alpha <real>
!   length <real>
!   dt <real>
!   t_left <real>
!   t_right <real>
!   initial_temperature <real>
!
! Output file format (binary):
!   n_points (int32)
!   n_steps (int32)
!   x(1:n_points) (float64)
!   T_history(1:n_points, 1:n_steps+1) (float64, column-major)

program heat_solver
    implicit none

    ! Parameters
    integer :: n_points
    integer :: n_steps
    real(8) :: alpha
    real(8) :: length
    real(8) :: dt
    real(8) :: t_left
    real(8) :: t_right
    real(8) :: initial_temperature

    ! Arrays
    real(8), allocatable :: x(:)
    real(8), allocatable :: T(:)
    real(8), allocatable :: T_new(:)
    real(8), allocatable :: T_history(:,:)

    ! Local variables
    integer :: i, step
    real(8) :: dx, r
    character(256) :: input_file, output_file
    integer :: nargs

    ! Check command line arguments
    nargs = command_argument_count()
    if (nargs /= 2) then
        write(*,*) 'Usage: heat_solver <input_file> <output_file>'
        stop 1
    end if

    call get_command_argument(1, input_file)
    call get_command_argument(2, output_file)

    ! Read input parameters
    call read_input(trim(input_file), n_points, n_steps, alpha, length, dt, &
                    t_left, t_right, initial_temperature)

    ! Allocate arrays
    allocate(x(n_points))
    allocate(T(n_points))
    allocate(T_new(n_points))
    allocate(T_history(n_points, n_steps + 1))

    ! Initialize spatial grid
    dx = length / dble(n_points - 1)
    do i = 1, n_points
        x(i) = dble(i - 1) * dx
    end do

    ! Compute stability parameter
    r = alpha * dt / (dx * dx)

    ! Initialize temperature field
    do i = 1, n_points
        T(i) = initial_temperature
    end do

    ! Apply boundary conditions
    T(1) = t_left
    T(n_points) = t_right

    ! Store initial state
    T_history(:, 1) = T

    ! Time stepping loop (explicit scheme)
    do step = 1, n_steps
        ! Update interior points
        do i = 2, n_points - 1
            T_new(i) = T(i) + r * (T(i-1) - 2.0d0 * T(i) + T(i+1))
        end do

        ! Apply boundary conditions
        T_new(1) = t_left
        T_new(n_points) = t_right

        ! Update solution
        T = T_new

        ! Store in history
        T_history(:, step + 1) = T
    end do

    ! Write output
    call write_output(trim(output_file), n_points, n_steps, x, T_history)

    ! Cleanup
    deallocate(x, T, T_new, T_history)

end program heat_solver


subroutine read_input(filename, n_points, n_steps, alpha, length, dt, &
                      t_left, t_right, initial_temperature)
    implicit none
    character(*), intent(in) :: filename
    integer, intent(out) :: n_points, n_steps
    real(8), intent(out) :: alpha, length, dt, t_left, t_right, initial_temperature

    integer :: ios, unit_num
    character(64) :: key
    real(8) :: value

    ! Set defaults
    n_points = 51
    n_steps = 100
    alpha = 0.01d0
    length = 1.0d0
    dt = 0.001d0
    t_left = 100.0d0
    t_right = 0.0d0
    initial_temperature = 0.0d0

    unit_num = 10
    open(unit=unit_num, file=filename, status='old', action='read', iostat=ios)
    if (ios /= 0) then
        write(*,*) 'Error: Cannot open input file: ', trim(filename)
        stop 1
    end if

    ! Read key-value pairs
    do
        read(unit_num, *, iostat=ios) key, value
        if (ios /= 0) exit

        select case (trim(key))
        case ('n_points')
            n_points = nint(value)
        case ('n_steps')
            n_steps = nint(value)
        case ('alpha')
            alpha = value
        case ('length')
            length = value
        case ('dt')
            dt = value
        case ('t_left')
            t_left = value
        case ('t_right')
            t_right = value
        case ('initial_temperature')
            initial_temperature = value
        case default
            ! Ignore unknown keys
        end select
    end do

    close(unit_num)

end subroutine read_input


subroutine write_output(filename, n_points, n_steps, x, T_history)
    implicit none
    character(*), intent(in) :: filename
    integer, intent(in) :: n_points, n_steps
    real(8), intent(in) :: x(n_points)
    real(8), intent(in) :: T_history(n_points, n_steps + 1)

    integer :: ios, unit_num

    unit_num = 20
    open(unit=unit_num, file=filename, status='replace', action='write', &
         form='unformatted', access='stream', iostat=ios)
    if (ios /= 0) then
        write(*,*) 'Error: Cannot open output file: ', trim(filename)
        stop 1
    end if

    ! Write header
    write(unit_num) n_points
    write(unit_num) n_steps

    ! Write spatial coordinates
    write(unit_num) x

    ! Write temperature history (column-major, Fortran native)
    write(unit_num) T_history

    close(unit_num)

end subroutine write_output
