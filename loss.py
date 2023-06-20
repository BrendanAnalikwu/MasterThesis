from typing import Tuple

import torch
import torch.utils.data


def delta(vx_x: torch.Tensor, vx_y: torch.Tensor, vy_x: torch.Tensor, vy_y: torch.Tensor, e_2: float = .25,
          dmin: float = 2e-9) -> torch.Tensor:
    """
    Computes the nonlinear ∆ function.

    Parameters
    ----------
    vx_x : torch.Tensor
        Derivative of the x-component of the ice velocity w.r.t. x. Tensor of shape (N, H, W)
    vx_y : torch.Tensor
        Derivative of the x-component of the ice velocity w.r.t. y. Tensor of shape (N, H, W)
    vy_x : torch.Tensor
        Derivative of the y-component of the ice velocity w.r.t. x. Tensor of shape (N, H, W)
    vy_y : torch.Tensor
        Derivative of the y-component of the ice velocity w.r.t. y. Tensor of shape (N, H, W)
    e_2 : float
        Yield curve eccentricity e^-2
    dmin : float
        Minimal value of ∆, to avoid degeneration to 0.


    Returns
    -------
    torch.Tensor
        ∆ values
    """
    return ((1 + e_2) * (torch.square(vx_x) + torch.square(vy_y))
            + 2 * (1 - e_2) * torch.mul(vx_x, vy_y)
            + e_2 * torch.square(vx_y + vy_x)
            + dmin ** 2)


def finite_differences(v: torch.Tensor, dx: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Finite difference computation using the central difference. The computed differences are on the cell midpoints.

    Parameters
    ----------
    v : torch.Tensor
        Ice velocity tensor of shape (N, 2, H+1, W+1)
    dx : float
        Grid spacing

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple of tensors with finite differences with shape (N, 2, H, W)
    """
    v_x = (v[:, :, :-1, 1:] + v[:, :, 1:, 1:] - v[:, :, :-1, :-1] - v[:, :, 1:, :-1]) / dx / 2.
    v_y = (v[:, :, 1:, :-1] + v[:, :, 1:, 1:] - v[:, :, :-1, :-1] - v[:, :, :-1, 1:]) / dx / 2.

    return v_x, v_y


def internal_stress(v: torch.Tensor, H: torch.Tensor, A: torch.Tensor, dx: float, C: float, e_2: float) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Internal stress tensor computation. Returns s_xx, s_yy, s_xy components of s=[[s_xx, s_xy], [s_xy, s_yy]].

    Parameters
    ----------
    v : torch.Tensor
        Ice velocity tensor of shape (N, 2, H+1, W+1)
    H: torch.Tensor
        Ice height tensor of shape (N, H, W)
    A: torch.Tensor
        Ice concentration tensor of shape (N, H, W)
    e_2 : float
        Yield curve eccentricity e^-2
    C : float
        Ice concentration parameter
    dx : float
        Grid spacing

    Returns
    -------
    s_xx, s_yy, s_xy : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple of stress tensor components with shape (N, H, W)
    """
    # Compute derivatives
    v_x, v_y = finite_differences(v, dx)

    # Pre-compute ∆ and trace
    delta_ = delta(v_x[:, 0, :, :], v_y[:, 0, :, :], v_x[:, 1, :, :], v_y[:, 1, :, :])
    trace = v_x[:, 0, :, :] + v_y[:, 1, :, :]

    # Stress component computation
    s_xx = H * torch.exp(-C * (1 - A)) * ((2 * e_2 * v_x[:, 0, :, :] + (1 - e_2) * trace) / delta_ - 1)
    s_yy = H * torch.exp(-C * (1 - A)) * ((2 * e_2 * v_y[:, 1, :, :] + (1 - e_2) * trace) / delta_ - 1)
    s_xy = H * torch.exp(-C * (1 - A)) * e_2 * (v_x[:, 1, :, :] + v_y[:, 0, :, :]) / delta_

    return s_xx, s_yy, s_xy


def form(v: torch.Tensor, H: torch.Tensor, A: torch.Tensor, dx: float, C: float, e_2: float, dt: float, T: float,
         f_c: float, C_r: float) -> torch.Tensor:
    """
    Left hand side vector computation of the system of equations in the weak formulation. The Coriolis term and ocean
    velocity are neglected (for now).

    Parameters
    ----------
    v : torch.Tensor
        Ice velocity tensor of shape (N, 2, H+1, W+1)
    H: torch.Tensor
        Ice height tensor of shape (N, H, W)
    A: torch.Tensor
        Ice concentration tensor of shape (N, H, W)
    T : float
        Characteristic time
    e_2 : float
        Yield curve eccentricity e^-2
    C : float
        Ice concentration parameter
    f_c : float
        Coriolis number
    dt : float
        Time step
    dx : float
        Grid spacing
    C_r : float
        Mehlmann number

    Returns
    -------
    A : torch.Tensor
        Left hand side vector with shape (N, 2, H-1, W-1)
    """
    # Stress computation
    s_xx, s_yy, s_xy = internal_stress(v, H, A, dx, C, e_2)

    # Set convolutional filters for easy computation
    # v_filter = dx ** 2 / 36 * torch.tensor([[1, 4, 1], [4, 16, 4], [1, 4, 1]], dtype=torch.float32)
    v_filter = dx ** 2 / 36 * torch.tensor([[2, 4], [1, 2]], dtype=torch.float32)[None, :, :]
    v_filter = torch.stack((v_filter, v_filter))

    # Compute first term in LHS (H v / dt, ϕ)
    A = (torch.mul(torch.nn.functional.conv2d(v[:, :, 1:, :-1], v_filter, groups=2), H[:, None, 1:, :-1])
         + torch.mul(torch.nn.functional.conv2d(v[:, :, 1:, 1:], torch.flip(v_filter, [3]), groups=2),
                     H[:, None, 1:, 1:])
         + torch.mul(torch.nn.functional.conv2d(v[:, :, :-1, :-1], torch.flip(v_filter, [2]), groups=2),
                     H[:, None, :-1, :-1])
         + torch.mul(torch.nn.functional.conv2d(v[:, :, :-1, 1:], torch.flip(v_filter, [2, 3]), groups=2),
                     H[:, None, :-1, 1:])) / dt

    # Compute stress term in LHS (C_r σ, ∇ϕ)
    s_filter = dx ** 2 / 3 * torch.tensor([[1, 1], [1, 1]], dtype=torch.float32)[None, None, :, :]
    A[:, 0:1, :, :] -= C_r * torch.nn.functional.conv2d((s_xx + s_xy)[:, None, ...], s_filter)
    A[:, 1:2, :, :] -= C_r * torch.nn.functional.conv2d((s_xy + s_yy)[:, None, ...], s_filter)

    return A


def vector(H: torch.Tensor, A: torch.Tensor, v_old: torch.Tensor, v_a: torch.Tensor, C_a: float,
           dx: float, dt: float) -> torch.Tensor:
    """
    Right hand side vector computation of the system of equations in the weak formulation.

    Parameters
    ----------
    H: torch.Tensor
        Ice height tensor of shape (N, H, W)
    A: torch.Tensor
        Ice concentration tensor of shape (N, H, W)
    v_old : torch.Tensor
        Ice velocity tensor of shape (N, 2, H+1, W+1) from previous time step
    v_a : torch.Tensor
        Wind velocity tensor of shape (N, 2, H+1, W+1) from previous time step
    C_a : float
        Wind number
    dt : float
        Time step
    dx : float
        Grid spacing

    Returns
    -------
    F : torch.Tensor
        Right hand side vector with shape (N, 2, H, W)
    """
    v_filter = dx ** 2 / 36 * torch.tensor([[2, 4], [1, 2]], dtype=torch.float32)[None, :, :]
    v_filter = torch.stack((v_filter, v_filter))

    # Compute first term in RHS (H v_old / dt, ϕ)
    F = (torch.mul(torch.nn.functional.conv2d(v_old[:, :, 1:, :-1], v_filter, groups=2), H[:, None, 1:, :-1])
         + torch.mul(torch.nn.functional.conv2d(v_old[:, :, 1:, 1:], torch.flip(v_filter, [3]), groups=2),
                     H[:, None, 1:, 1:])
         + torch.mul(torch.nn.functional.conv2d(v_old[:, :, :-1, :-1], torch.flip(v_filter, [2]), groups=2),
                     H[:, None, :-1, :-1])
         + torch.mul(torch.nn.functional.conv2d(v_old[:, :, :-1, 1:], torch.flip(v_filter, [2, 3]), groups=2),
                     H[:, None, :-1, 1:])) / dt

    v_a_filter = dx ** 2 / 36 * torch.tensor([[2, 4], [1, 2]], dtype=torch.float32)[None, :, :]
    v_a_filter = torch.stack((v_a_filter, v_a_filter))

    # Compute wind term in RHS (C_a A τ_a, ϕ)
    t_a = C_a * torch.mul(torch.linalg.norm(v_a, dim=0, keepdim=True), v_a)
    F += (torch.mul(torch.nn.functional.conv2d(t_a[:, :, 1:, :-1], v_a_filter, groups=2), A[:, None, 1:, :-1])
          + torch.mul(torch.nn.functional.conv2d(t_a[:, :, 1:, 1:], torch.flip(v_a_filter, [3]), groups=2),
                      A[:, None, 1:, 1:])
          + torch.mul(torch.nn.functional.conv2d(t_a[:, :, :-1, :-1], torch.flip(v_a_filter, [2]), groups=2),
                      A[:, None, :-1, :-1])
          + torch.mul(torch.nn.functional.conv2d(t_a[:, :, :-1, 1:], torch.flip(v_a_filter, [2, 3]), groups=2),
                      A[:, None, :-1, 1:])) / dt

    return F


def loss_func(v: torch.Tensor, H: torch.Tensor, A: torch.Tensor, v_old: torch.Tensor, v_a: torch.Tensor, C_r, C_a, T,
              e_2, C, f_c, dx, dt) -> torch.Tensor:
    """
    Loss computation using the PINN-like approach. The PINN loss is weighted with the boundary condition loss, such that
    they are of approximate same importance.

    Parameters
    ----------
    v : torch.Tensor
        Ice velocity tensor of shape (N, 2, H+1, W+1)
    H: torch.Tensor
        Ice height tensor of shape (N, H, W)
    A: torch.Tensor
        Ice concentration tensor of shape (N, H, W)
    v_old : torch.Tensor
        Ice velocity tensor of shape (N, 2, H+1, W+1) from previous time step
    v_a : torch.Tensor
        Wind velocity tensor of shape (N, 2, H+1, W+1) from previous time step
    C_r : float
        Mehlmann number
    C_a : float
        Wind number
    T : float
        Characteristic time
    e_2 : float
        Yield curve eccentricity e^-2
    C : float
        Ice concentration parameter
    f_c : float
        Coriolis number
    dt : float
        Time step
    dx : float
        Grid spacing

    Returns
    -------
    loss : torch.Tensor
        Total loss value
    """
    return (1e4 * torch.sum(torch.pow(vector(H, A, v_old, v_a, C_a, dx, dt)
                                      - form(v, H, A, dx, C, e_2, dt, T, f_c, C_r), 2))
            + torch.sum(torch.pow(v[:, :, 0, :], 2)) + torch.sum(torch.pow(v[:, :, -1, :], 2))
            + torch.sum(torch.pow(v[:, :, 0, 1:-1], 2)) + torch.sum(torch.pow(v[:, :, -1, 1:-1], 2))) * 1e6
