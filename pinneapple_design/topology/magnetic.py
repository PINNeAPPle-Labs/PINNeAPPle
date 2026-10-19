"""Topology optimization of rotor iron for torque, on the 2D A_z machine model.

Design domain: the rotor-core elements of a
:class:`~pinneapple_simulation.electromagnetics.machine.SPMMachine` (with ``magnet_thickness=0`` it is a
reluctance rotor; with magnets, the iron under them). Material interpolation on the reluctivity,
``nu = nu0 + rho^p (nu_iron(|B|^2) - nu0)`` (linear or nonlinear iron). The torque at each rotor angle is
the Maxwell-stress (Arkkio) torque of the model and its sensitivity comes from one adjoint solve per
angle with the Newton tangent (:meth:`SPMMachine.torque_and_design_gradient`).

Objectives (``evaluate`` returns a value to minimise):

* ``"mean"``: ``-mean_k T(theta_k) / T_ref`` -- maximise the average torque over the given angles;
* ``"ripple"``: ``std_k T(theta_k) / T_ref`` -- use with :meth:`RotorTorqueProblem.mean_torque_constraint`.

Currents follow the rotor synchronously: ``i_k = I cos(p theta + gamma - 2 pi k / 3)`` (``gamma`` is the
electrical angle of the current vector relative to the rotor's reference axis at ``theta = 0``).
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = ["RotorTorqueProblem"]


class RotorTorqueProblem:
    def __init__(self, machine, thetas: Sequence[float], I_peak: float, gamma: float,
                 objective: str = "mean", T_ref: float = 1.0):
        self.m = machine
        self.thetas = np.asarray(thetas, float)
        self.I, self.gamma = I_peak, gamma
        self.objective = objective
        self.T_ref = T_ref
        mesh = machine.mesh(0.0)
        self.elements = machine.design_elements(mesh)
        self.n = len(self.elements)
        from pinneapple_simulation.electromagnetics.fem import Geometry

        self.volumes = Geometry.of(mesh).area[self.elements]
        self.centroids = mesh.centroids()[self.elements]
        self._key = None
        self._A0 = {}

    def currents(self, theta: float):
        x = self.m.pole_pairs * theta + self.gamma
        return tuple(self.I * math.cos(x - 2 * math.pi * k / 3) for k in range(3))

    def torques(self, x: np.ndarray):
        """Torque at every angle and its gradient (n_angles, n) w.r.t. the physical densities."""
        key = x.tobytes()
        if key != self._key:
            T, G = [], []
            for k, th in enumerate(self.thetas):
                t, g, ms = self.m.torque_and_design_gradient(th, self.currents(th), x, A0=self._A0.get(k))
                self._A0[k] = ms.A
                T.append(t)
                G.append(g)
            self._key, self._T, self._G = key, np.array(T), np.array(G)
        return self._T, self._G

    def evaluate(self, x: np.ndarray):
        T, G = self.torques(x)
        if self.objective == "mean":
            return -T.mean() / self.T_ref, -G.mean(0) / self.T_ref
        if self.objective == "ripple":
            d = T - T.mean()
            s = math.sqrt(max(np.mean(d * d), 1e-300))
            return s / self.T_ref, (d @ (G - G.mean(0))) / (len(T) * s * self.T_ref)
        raise ValueError(self.objective)

    def mean_torque_constraint(self, T_min: float):
        """Constraint ``1 - mean(T) / T_min <= 0`` for :func:`~pinneapple_design.topology.optimize`."""
        def con(x):
            T, G = self.torques(x)
            return 1.0 - T.mean() / T_min, -G.mean(0) / T_min
        return con
