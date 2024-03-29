from math import sqrt
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
    return torch.sqrt((1 + e_2) * (torch.square(vx_x) + torch.square(vy_y))
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


integration_filter = torch.tensor([[2, 4], [1, 2]], dtype=torch.float32)[None, :, :]
integration_filter = torch.stack((integration_filter, integration_filter))


def integration_B_grid(vertex: torch.Tensor, cell: torch.Tensor, dx: float):
    filter = integration_filter.to(vertex.device)

    res = torch.mul(torch.nn.functional.conv2d(vertex[:, :, 1:, :-1], filter, groups=2), cell[:, None, 1:, :-1])
    res += torch.mul(torch.nn.functional.conv2d(vertex[:, :, 1:, 1:], torch.flip(filter, [3]), groups=2),
                     cell[:, None, 1:, 1:])
    res += torch.mul(torch.nn.functional.conv2d(vertex[:, :, :-1, :-1], torch.flip(filter, [2]), groups=2),
                     cell[:, None, :-1, :-1])
    res += torch.mul(torch.nn.functional.conv2d(vertex[:, :, :-1, 1:], torch.flip(filter, [2, 3]), groups=2),
                     cell[:, None, :-1, 1:])

    return res * dx ** 2 / 36


def stress(v: torch.Tensor, H: torch.Tensor, A: torch.Tensor, dx: float, C: float, e_2: float, C_r: float,
           dmin: float = 2e-6):
    P = H * torch.exp(-C*(1-A))

    v1 = v[:, :, :-1, :-1]
    v2 = v[:, :, 1:, :-1]
    v3 = v[:, :, :-1, 1:]
    v4 = v[:, :, 1:, 1:]

    v__ = v1 - v2 - v3 + v4

    g = (1 + e_2) * v__[:, 1].square() + e_2 * v__[:, 0].square()
    h = (1 + e_2) * v__[:, 0].square() + e_2 * v__[:, 1].square()
    i = 2 * v__[:, 0] * v__[:, 1]
    j = 2 * (((1 + e_2) * (v3[:, 1] - v1[:, 1]) + (1 - e_2) * (v2[:, 0] - v1[:, 0])) * v__[:, 1]
             + e_2 * v__[:, 0] * (v3[:, 0] - v1[:, 0] + v2[:, 1] - v1[:, 1]))
    k = 2 * (((1 + e_2) * (v2[:, 0] - v1[:, 0]) + (1 - e_2) * (v3[:, 1] - v1[:, 1])) * v__[:, 0]
             + e_2 * v__[:, 1] * (v3[:, 0] - v1[:, 0] + v2[:, 1] - v1[:, 1]))
    l = (1 + e_2) * ((v2[:, 0] - v1[:, 0]).square() + (v3[:, 1] - v1[:, 1]).square()) + 2 * (1 - e_2) * (
                v2[:, 0] - v1[:, 0]) * (v3[:, 1] - v1[:, 1]) + e_2 * (
                    (v2[:, 1] - v1[:, 1]).square() + (v3[:, 0] - v1[:, 0]).square() + 2 * (v2[:, 1] - v1[:, 1]) * (
                        v3[:, 0] - v1[:, 0]))
    a = (1 - e_2) * v__[:, 1]
    b = (1 + e_2) * v__[:, 0]
    c = (1 + e_2) * (v2[:, 0] - v1[:, 0]) + (1 - e_2) * (v3[:, 1] - v1[:, 1])

    coords = torch.tensor([.5 - .5 / sqrt(3), .5 + .5 / sqrt(3)], device=H.device)
    x, y = torch.meshgrid(coords, coords, indexing='xy')
    x = x.reshape(1, 4, 1, 1)
    y = y.reshape(1, 4, 1, 1)

    delta_ = torch.sqrt(g[:, None] * x.square() + h[:, None] * y.square() + i[:, None] * x * y + j[:, None] * x
                        + k[:, None] * y + l[:, None] + dmin ** 2)
    sxx = P[:, None] * ((a[:, None] * x + b[:, None] * y + c[:, None]) / delta_ - 1)

    sxy = P[:, None] * e_2 * (v__[:, :1] * x + v__[:, 1:] * y + v2[:, 1:] - v1[:, 1:] + v3[:, :1] - v1[:, :1]) / delta_

    filter_dx = torch.empty(1, 4, 2, 2, device=H.device)
    filter_dx[0, :, 0, 0] = y.flatten()
    filter_dx[0, :, 1, 0] = -y.flatten()
    filter_dx[0, :, 0, 1] = 1 - y.flatten()
    filter_dx[0, :, 1, 1] = y.flatten() - 1
    filter_dy = torch.empty(1, 4, 2, 2, device=H.device)
    filter_dy[0, :, 0, 0] = x.flatten()
    filter_dy[0, :, 1, 0] = 1 - x.flatten()
    filter_dy[0, :, 0, 1] = -x.flatten()
    filter_dy[0, :, 1, 1] = x.flatten() - 1

    res1 = torch.nn.functional.conv2d(sxx, filter_dx) / 4
    res1 += torch.nn.functional.conv2d(sxy, filter_dy) / 4

    d = (1 + e_2) * v__[:, 1]
    e = (1 - e_2) * v__[:, 0]
    f = (1 - e_2) * (v2[:, 0] - v1[:, 0]) + (1 + e_2) * (v3[:, 1] - v1[:, 1])
    syy = P[:, None] * ((d[:, None] * x + e[:, None] * y + f[:, None]) / delta_ - 1)

    res2 = torch.nn.functional.conv2d(sxy, filter_dx)/4
    res2 += torch.nn.functional.conv2d(syy, filter_dy)/4

    return - torch.cat((res1, res2), dim=1) * dx * C_r


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

    # Compute first term in LHS (H v / dt, ϕ)
    mat = integration_B_grid(v, H, dx) / dt

    # Compute stress term in LHS (C_r σ, ∇ϕ)
    s_filter = dx ** 2 / 3 * torch.tensor([[1, 1], [1, 1]], dtype=torch.float32, device=H.device)[None, None, :, :]
    mat[:, 0:1, :, :] -= C_r * torch.nn.functional.conv2d((s_xx + s_xy)[:, None, ...], s_filter)
    mat[:, 1:2, :, :] -= C_r * torch.nn.functional.conv2d((s_xy + s_yy)[:, None, ...], s_filter)

    return mat


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

    # Compute first term in RHS (H v_old / dt, ϕ)
    F = integration_B_grid(v_old, H, dx) / dt

    # Compute wind term in RHS (C_a A τ_a, ϕ)
    t_a = C_a * torch.mul(torch.linalg.norm(v_a, dim=1, keepdim=True), v_a)
    F += integration_B_grid(t_a, A, dx)

    return F


def loss_func(dv: torch.Tensor, H: torch.Tensor, A: torch.Tensor, v_old: torch.Tensor, v_a: torch.Tensor,
              v_o: torch.Tensor, C_r, C_a, C_o, T, e_2, C, f_c, dx, dt) -> torch.Tensor:
    """
    Loss computation using the PINN-like approach. The PINN loss is weighted with the boundary condition loss, such that
    they are of approximate same importance.

    Parameters
    ----------
    dv : torch.Tensor
        Ice velocity tensor of shape (N, 2, H+1, W+1)
    H: torch.Tensor
        Ice height tensor of shape (N, H, W)
    A: torch.Tensor
        Ice concentration tensor of shape (N, H, W)
    v_old : torch.Tensor
        Ice velocity tensor of shape (N, 2, H+1, W+1) from previous time step
    v_a : torch.Tensor
        Wind velocity tensor of shape (N, 2, H+1, W+1)
    v_o : torch.Tensor
        Water velocity tensor of shape (N, 2, H+1, W+1) or (1, 2, H+1, W+1)
    C_r : float
        Mehlmann number
    C_a : float
        Wind number
    C_o : float
        Water number
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
    time_deriv = integration_B_grid(dv, H, dx) / dt

    v_c = torch.empty_like(dv)
    v_o_diff = v_o - v_old - dv
    v_c[:, 0] = v_o_diff[:, 1]
    v_c[:, 1] = -v_o_diff[:, 0]
    coriolis_term = integration_B_grid(T*f_c*v_c, H, dx)

    # v_o_diff = v_o - v_old - dv
    t_o = C_o * torch.mul(torch.sqrt(v_o_diff.square().sum(1, keepdim=True) + 1e-8), v_o_diff)
    ocean_term = integration_B_grid(t_o, torch.ones_like(A), dx)

    # Compute wind term in RHS (C_a A τ_a, ϕ)
    t_a = C_a * torch.mul(torch.linalg.norm(v_a, dim=1, keepdim=True), v_a)
    wind_term = integration_B_grid(t_a, torch.ones_like(A), dx)

    # Stress computation
    stress_term = stress(v_old + dv, H, A, dx, C, e_2, C_r)

    # A - F
    fem_total = time_deriv + coriolis_term - stress_term - ocean_term - wind_term

    return (dx ** -2 * torch.sum(torch.pow(fem_total, 2))
            + torch.sum(torch.pow(dv[:, :, 0, :], 2)) + torch.sum(torch.pow(dv[:, :, -1, :], 2))
            + torch.sum(torch.pow(dv[:, :, 0, 1:-1], 2)) + torch.sum(torch.pow(dv[:, :, -1, 1:-1], 2)))


def strain_rate(v: torch.tensor):
    if v.dim() == 3:
        v = v[None]
    e_x, e_y = finite_differences(v, 1.)
    return e_x[:, 0], e_y[:, 1], e_x[:, 1] + e_y[:, 0]


def strain_rate_loss(v: torch.tensor, label: torch.tensor):
    error = v - label
    if error.dim() == 3:
        error = error[None]
    e_x, e_y = finite_differences(error, 1.)
    return (e_x[:, 0].square() + .5 * (e_x[:, 1] + e_y[:, 0]).square() + e_y[:, 1].square()).mean()
