! Copyright 2025 Pasteur Labs. All Rights Reserved.
! SPDX-License-Identifier: Apache-2.0
!
! 2D transient heat conduction solver with temperature-dependent conductivity.
!
! Solves:
!
!   rho * cp * dT/dt = div( k(T) * grad(T) ) + Q
!
! on a rectangular domain [0, Lx] x [0, Ly] with a structured grid.
!
! Material model:
!   k(T) = k0 + k1 * T
!
! Boundary conditions:
!   Bottom (y=0):  Dirichlet, T = T_hot
!   Top    (y=Ly): Convection, -k dT/dn = h_conv * (T - T_inf)
!   Left   (x=0):  Neumann (insulated), dT/dx = 0
!   Right  (x=Lx): Neumann (insulated), dT/dx = 0
!
! Time integration: explicit Euler with n_steps steps of size dt.
!
! Work arrays T_cur and T_new are passed in from the caller to avoid
! dynamic allocation (LFortran emits _lfortran_malloc for VLAs, which
! Enzyme cannot differentiate through).
!
! This subroutine is compiled to LLVM IR via LFortran, then
! differentiated by Enzyme to obtain exact JVP and VJP functions.

subroutine thermal_2d_solve(n, nx, ny, n_steps, &
                            T_init, T_final, T_cur, T_new, &
                            k0, k1, rho, cp, &
                            h_conv, T_inf, T_hot, &
                            Q, Lx, Ly, dt)
  implicit none
  integer, intent(in) :: n, nx, ny, n_steps
  double precision, intent(in) :: T_init(n)
  double precision, intent(out) :: T_final(n)
  double precision, intent(out) :: T_cur(n)
  double precision, intent(out) :: T_new(n)
  double precision, intent(in) :: k0, k1, rho, cp
  double precision, intent(in) :: h_conv, T_inf, T_hot
  double precision, intent(in) :: Q(n)
  double precision, intent(in) :: Lx, Ly, dt

  double precision :: dx, dy
  double precision :: kx_east, kx_west, ky_north, ky_south
  double precision :: T_c, T_e, T_w, T_nn, T_s
  double precision :: flux_x, flux_y
  integer :: i, j, idx, step

  dx = Lx / dble(nx - 1)
  dy = Ly / dble(ny - 1)

  ! Copy initial condition
  do idx = 1, n
    T_cur(idx) = T_init(idx)
  end do

  ! Time integration loop
  do step = 1, n_steps

    ! --- Interior points ---
    do j = 2, ny - 1
      do i = 2, nx - 1
        idx = (j - 1) * nx + i

        T_c = T_cur(idx)
        T_e = T_cur(idx + 1)
        T_w = T_cur(idx - 1)
        T_nn = T_cur(idx + nx)
        T_s = T_cur(idx - nx)

        ! Harmonic-mean conductivity at cell faces
        kx_east = 2.0d0 * (k0 + k1 * T_c) * (k0 + k1 * T_e) &
                  / ((k0 + k1 * T_c) + (k0 + k1 * T_e))
        kx_west = 2.0d0 * (k0 + k1 * T_c) * (k0 + k1 * T_w) &
                  / ((k0 + k1 * T_c) + (k0 + k1 * T_w))
        ky_north = 2.0d0 * (k0 + k1 * T_c) * (k0 + k1 * T_nn) &
                   / ((k0 + k1 * T_c) + (k0 + k1 * T_nn))
        ky_south = 2.0d0 * (k0 + k1 * T_c) * (k0 + k1 * T_s) &
                   / ((k0 + k1 * T_c) + (k0 + k1 * T_s))

        flux_x = (kx_east * (T_e - T_c) - kx_west * (T_c - T_w)) / (dx * dx)
        flux_y = (ky_north * (T_nn - T_c) - ky_south * (T_c - T_s)) / (dy * dy)

        T_new(idx) = T_c + dt / (rho * cp) * (flux_x + flux_y + Q(idx))
      end do
    end do

    ! --- Bottom boundary (j=1): Dirichlet T = T_hot ---
    do i = 1, nx
      idx = i
      T_new(idx) = T_hot
    end do

    ! --- Top boundary (j=ny): Convection BC ---
    ! -k dT/dn = h_conv * (T - T_inf)
    ! (one-sided difference for the normal derivative)
    do i = 2, nx - 1
      idx = (ny - 1) * nx + i

      T_c = T_cur(idx)
      T_e = T_cur(idx + 1)
      T_w = T_cur(idx - 1)
      T_s = T_cur(idx - nx)

      kx_east = 2.0d0 * (k0 + k1 * T_c) * (k0 + k1 * T_e) &
                / ((k0 + k1 * T_c) + (k0 + k1 * T_e))
      kx_west = 2.0d0 * (k0 + k1 * T_c) * (k0 + k1 * T_w) &
                / ((k0 + k1 * T_c) + (k0 + k1 * T_w))
      ky_south = 2.0d0 * (k0 + k1 * T_c) * (k0 + k1 * T_s) &
                 / ((k0 + k1 * T_c) + (k0 + k1 * T_s))

      flux_x = (kx_east * (T_e - T_c) - kx_west * (T_c - T_w)) / (dx * dx)

      T_new(idx) = T_c + dt / (rho * cp) * ( &
          flux_x &
          + ky_south * (T_s - T_c) / (dy * dy) &
          - h_conv * (T_c - T_inf) / dy &
          + Q(idx))
    end do

    ! --- Left boundary (i=1): insulated (zero flux) ---
    ! Mirror: T_w = T_e => flux_x uses only east neighbor
    do j = 2, ny - 1
      idx = (j - 1) * nx + 1

      T_c = T_cur(idx)
      T_e = T_cur(idx + 1)
      T_nn = T_cur(idx + nx)
      T_s = T_cur(idx - nx)

      kx_east = 2.0d0 * (k0 + k1 * T_c) * (k0 + k1 * T_e) &
                / ((k0 + k1 * T_c) + (k0 + k1 * T_e))
      ky_north = 2.0d0 * (k0 + k1 * T_c) * (k0 + k1 * T_nn) &
                 / ((k0 + k1 * T_c) + (k0 + k1 * T_nn))
      ky_south = 2.0d0 * (k0 + k1 * T_c) * (k0 + k1 * T_s) &
                 / ((k0 + k1 * T_c) + (k0 + k1 * T_s))

      ! Zero-flux left: symmetric difference gives 2*(T_e - T_c)/(2*dx^2)
      flux_x = kx_east * (T_e - T_c) / (dx * dx)
      flux_y = (ky_north * (T_nn - T_c) - ky_south * (T_c - T_s)) / (dy * dy)

      T_new(idx) = T_c + dt / (rho * cp) * (flux_x + flux_y + Q(idx))
    end do

    ! --- Right boundary (i=nx): insulated (zero flux) ---
    do j = 2, ny - 1
      idx = (j - 1) * nx + nx

      T_c = T_cur(idx)
      T_w = T_cur(idx - 1)
      T_nn = T_cur(idx + nx)
      T_s = T_cur(idx - nx)

      kx_west = 2.0d0 * (k0 + k1 * T_c) * (k0 + k1 * T_w) &
                / ((k0 + k1 * T_c) + (k0 + k1 * T_w))
      ky_north = 2.0d0 * (k0 + k1 * T_c) * (k0 + k1 * T_nn) &
                 / ((k0 + k1 * T_c) + (k0 + k1 * T_nn))
      ky_south = 2.0d0 * (k0 + k1 * T_c) * (k0 + k1 * T_s) &
                 / ((k0 + k1 * T_c) + (k0 + k1 * T_s))

      flux_x = kx_west * (T_w - T_c) / (dx * dx)
      flux_y = (ky_north * (T_nn - T_c) - ky_south * (T_c - T_s)) / (dy * dy)

      T_new(idx) = T_c + dt / (rho * cp) * (flux_x + flux_y + Q(idx))
    end do

    ! --- Corners ---
    ! Bottom-left and bottom-right: Dirichlet (already set above)
    ! Top-left corner (i=1, j=ny)
    idx = (ny - 1) * nx + 1
    T_c = T_cur(idx)
    T_e = T_cur(idx + 1)
    T_s = T_cur(idx - nx)

    kx_east = 2.0d0 * (k0 + k1 * T_c) * (k0 + k1 * T_e) &
              / ((k0 + k1 * T_c) + (k0 + k1 * T_e))
    ky_south = 2.0d0 * (k0 + k1 * T_c) * (k0 + k1 * T_s) &
               / ((k0 + k1 * T_c) + (k0 + k1 * T_s))

    T_new(idx) = T_c + dt / (rho * cp) * ( &
        kx_east * (T_e - T_c) / (dx * dx) &
        + ky_south * (T_s - T_c) / (dy * dy) &
        - h_conv * (T_c - T_inf) / dy &
        + Q(idx))

    ! Top-right corner (i=nx, j=ny)
    idx = ny * nx
    T_c = T_cur(idx)
    T_w = T_cur(idx - 1)
    T_s = T_cur(idx - nx)

    kx_west = 2.0d0 * (k0 + k1 * T_c) * (k0 + k1 * T_w) &
              / ((k0 + k1 * T_c) + (k0 + k1 * T_w))
    ky_south = 2.0d0 * (k0 + k1 * T_c) * (k0 + k1 * T_s) &
               / ((k0 + k1 * T_c) + (k0 + k1 * T_s))

    T_new(idx) = T_c + dt / (rho * cp) * ( &
        kx_west * (T_w - T_c) / (dx * dx) &
        + ky_south * (T_s - T_c) / (dy * dy) &
        - h_conv * (T_c - T_inf) / dy &
        + Q(idx))

    ! Swap: T_cur <- T_new
    do idx = 1, n
      T_cur(idx) = T_new(idx)
    end do

  end do

  ! Copy result
  do idx = 1, n
    T_final(idx) = T_cur(idx)
  end do

end subroutine
