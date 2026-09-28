! Copyright 2025 Pasteur Labs. All Rights Reserved.
! SPDX-License-Identifier: Apache-2.0
!
! Single explicit Euler step for the 1D heat equation:
!
!   dT/dt = alpha * d^2T/dx^2
!
! Discretized with central differences in space:
!
!   T_out(i) = T_in(i) + r * (T_in(i-1) - 2*T_in(i) + T_in(i+1))
!
! where r = alpha * dt / dx^2.
!
! Boundary conditions are Dirichlet (fixed): T_out(1) = T_in(1),
! T_out(n) = T_in(n).
!
! This subroutine is compiled to LLVM IR via LFortran, then
! differentiated by Enzyme to obtain exact (not finite-difference)
! JVP and VJP functions.

subroutine heat_step(n, T_in, T_out, alpha, dx, dt)
  implicit none
  integer, intent(in) :: n
  double precision, intent(in) :: T_in(n), alpha, dx, dt
  double precision, intent(out) :: T_out(n)
  integer :: i
  double precision :: r

  r = alpha * dt / (dx * dx)

  ! Dirichlet boundary conditions
  T_out(1) = T_in(1)
  T_out(n) = T_in(n)

  ! Interior points: explicit finite difference stencil
  do i = 2, n-1
    T_out(i) = T_in(i) + r * (T_in(i-1) - 2.0d0*T_in(i) + T_in(i+1))
  end do
end subroutine
