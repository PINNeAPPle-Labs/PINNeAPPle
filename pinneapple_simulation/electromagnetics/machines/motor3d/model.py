"""The multiphysics reference model of one motor design: electromagnetics, losses, cooling and temperature, coupled.

One call to :meth:`Motor3DModel.solve` runs the chain an engineer runs by hand:

1. **electromagnetic** -- nonlinear 2-D magnetostatic FEA of the cross-section at the operating point
   (:class:`~..pmsm.IPMMachine`), over a set of rotor angles, with the magnet remanence at a first-guess magnet
   temperature;
2. **skew** -- the stack is cut into ``n_segments`` axial slices twisted by ``skew_deg``: each slice contributes the
   torque of the cross-section at its own electrical angle, so the slot harmonics cancel the way they do in a skewed
   machine, and the fundamental is reduced by the skew factor;
3. **losses** -- stator and rotor iron loss from the flux density over an electrical period, resistance-limited
   magnet eddy loss, and the copper loss of the slot and end-winding parts of the winding, with the end turns of the
   axial geometry (the winding resistance depends on the stack length, so it is a 3-D quantity);
4. **cooling and temperature** -- the axial lumped network of :mod:`.thermal3d`, whose jacket conductance follows the
   coolant flow as ``flow^0.8`` (turbulent convection), giving a temperature per part and per axial segment;
5. **back to the electromagnetics** -- the magnet temperature changes the remanence (``Br(T)`` of the grade), so
   torque, iron loss (through ``B^2``) and the demagnetisation margin are corrected, and the copper loss is
   re-evaluated at the winding temperature with the network solved again until the temperatures stop moving.

Two costs are kept honest. Step 5 is a one-step fixed point around one FEA solve per design (``fea_iterations=1``);
setting it to 2 re-solves the field at the converged magnet temperature. The difference between the two is measured,
not assumed: :func:`~.workflow.check_temperature_coupling` reports it on a set of designs and the model card records
it.

Everything here is SI: metres, amperes, volts, webers, watts, degrees Celsius, kilograms. The outputs are listed in
:data:`OUTPUTS`, each with its unit and whether a better design makes it larger or smaller.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..drive import copper_resistance
from ..pmsm import IPMMachine
from ..thermal import NDFEB_HCJ_MIN_KAM, MachineThermalData, demagnetisation_margin, intrinsic_coercivity
from .design import Motor3DDesign, from_matrix
from .thermal3d import AxialThermalModel

__all__ = ["MotorOutput", "OUTPUTS", "SCALARS", "PROFILES", "Motor3DModel", "N_SEGMENTS", "MAGNET_GRADE_HCJ",
           "PRIUS"]

#: Axial segments of the thermal network, and the length of every axial profile the surrogate predicts.
N_SEGMENTS = 4

#: Intrinsic coercivity at 20 degC of the NdFeB grade the study uses (UH: 25 kOe), A/m. The magnet material of
#: :data:`~..steel.MAGNET_PRIUS` is a sintered NdFeB; the grade is a design assumption, stated in every result.
MAGNET_GRADE_HCJ = NDFEB_HCJ_MIN_KAM["UH"] * 1e3      # the table is in kA/m

# winding data of pyleecan's Toyota_Prius.json, the same as :func:`~..drive.prius_phase_resistance`
_SERIES_TURNS = 72.0
_CONDUCTOR_AREA = 13 * math.pi * (0.912e-3 / 2.0) ** 2
_OVERHANG = 0.019366
_ALPHA_CU = 0.00393


@dataclass(frozen=True)
class MotorOutput:
    """One output of the model: name, unit, whether a better design makes it larger, and what it is."""
    name: str
    unit: str
    larger_is_better: bool
    description: str
    group: str = "performance"        # performance | loss | thermal | magnet | mass


OUTPUTS: tuple[MotorOutput, ...] = (
    MotorOutput("torque_Nm", "N m", True,
                "mean torque over the load period, skewed and at the converged magnet temperature"),
    MotorOutput("ripple", "-", False, "peak to peak torque over the mean, over the skewed stack"),
    MotorOutput("torque_density_Nm_kg", "N m/kg", True, "mean torque per kilogram of active mass"),
    MotorOutput("efficiency", "-", True, "mechanical power over mechanical plus electromagnetic loss", "loss"),
    MotorOutput("iron_loss_W", "W", False, "stator and rotor iron loss at the operating speed", "loss"),
    MotorOutput("magnet_loss_W", "W", False, "resistance-limited eddy loss in the magnets", "loss"),
    MotorOutput("copper_loss_W", "W", False, "slot and end-winding copper loss at the winding temperature", "loss"),
    MotorOutput("total_loss_W", "W", False, "iron + magnet + copper loss", "loss"),
    MotorOutput("T_winding_C", "degC", False, "hottest winding node of the axial network", "thermal"),
    MotorOutput("T_magnet_C", "degC", False, "hottest magnet node: it sets the remanence and the margin", "thermal"),
    MotorOutput("T_end_winding_C", "degC", False, "end-winding node, the usual hot spot of a jacket-cooled machine",
                "thermal"),
    MotorOutput("demag_ratio", "-", False,
                "worst reverse field over the intrinsic coercivity at the magnet temperature (< 1 keeps the magnet)",
                "magnet"),
    MotorOutput("magnet_mass_kg", "kg", False, "sintered NdFeB in the rotor", "mass"),
    MotorOutput("active_mass_kg", "kg", False, "steel, copper and magnets of the active part", "mass"),
)

SCALARS: tuple[str, ...] = tuple(o.name for o in OUTPUTS)
PROFILES: tuple[str, ...] = ("torque_slice_Nm", "T_winding_profile_C", "T_magnet_profile_C")


def _interp_periodic(theta: np.ndarray, values: np.ndarray, x: np.ndarray, period: float) -> np.ndarray:
    """``values(theta)`` at ``x`` for a period-long sample (linear, wrap-around)."""
    th = np.mod(np.asarray(theta, float) - theta[0], period)
    order = np.argsort(th)
    th, v = th[order], np.asarray(values, float)[order]
    xe = np.mod(np.asarray(x, float) - theta[0], period)
    return np.interp(xe, np.concatenate([th, [period]]), np.concatenate([v, [v[0]]]))


class Motor3DModel:
    """The reference multiphysics model (see the module docstring).

    ``n_gap`` is the air-gap mesh of the cross-section (a multiple of twice the slot and pole counts), ``n_positions``
    the rotor angles the torque is averaged over, ``n_segments`` the axial segments of the skew model and of the
    thermal network, ``magnet_T_guess`` the magnet temperature the field is first solved at and
    ``fea_iterations`` how many times the field is re-solved at the converged magnet temperature (1 = the one-step
    correction, 2 = one extra solve per design).

    ``count`` is the number of FEA operating points solved so far and ``seconds`` the time spent in them: a study
    reports what it cost, and the point of the surrogate is the ratio between these numbers and its own.
    """

    fidelity = "reference"

    def __init__(self, n_gap: int = 192, n_positions: int = 8, n_segments: int = N_SEGMENTS,
                 magnet_T_guess: float = 90.0, fea_iterations: int = 1, coolant_flow_ref: float = 6.0,
                 thermal: MachineThermalData | None = None, Hcj20: float = MAGNET_GRADE_HCJ, v_dc: float = 500.0,
                 loss_steps: int = 6, magnet_T_limit: float = 180.0, name: str = "motor3d-fea"):
        self.n_gap, self.n_positions = int(n_gap), int(n_positions)
        self.n_segments = int(n_segments)
        self.magnet_T_guess, self.fea_iterations = float(magnet_T_guess), max(1, int(fea_iterations))
        self.coolant_flow_ref, self.thermal = float(coolant_flow_ref), thermal or MachineThermalData()
        self.Hcj20, self.v_dc, self.loss_steps = float(Hcj20), float(v_dc), int(loss_steps)
        #: Above this magnet temperature the linear Hcj(T) of the grade is out of its reversible range (the
        #: coercivity would fall through zero), so the design is reported as not assessable rather than scored.
        self.magnet_T_limit = float(magnet_T_limit)
        self.name = name
        self.outputs = list(SCALARS) + list(PROFILES)
        self.description = (f"3-D multiphysics: {n_gap}-node gap FEA x {n_positions} rotor angles, "
                            f"{n_segments} axial segments (skew and thermal), magnet temperature by "
                            f"{self.fea_iterations} field iteration(s)")
        self.count = 0
        self.seconds = 0.0
        self._cache: dict[tuple, IPMMachine] = {}

    # --------------------------------------------------------------------------------------------- helpers
    def machine(self, d: Motor3DDesign, magnet_temperature: float | None = None) -> IPMMachine:
        """The FEA machine of a design (cached per geometry and magnet temperature: the mesh is the expensive part)."""
        T = self.magnet_T_guess if magnet_temperature is None else float(magnet_temperature)
        key = (round(d.W4, 9), round(d.H3, 9), round(d.H0, 9), round(d.W0, 9), round(d.H1, 9), round(d.L, 9),
               round(T, 6))
        if key not in self._cache:
            if len(self._cache) > 32:                       # a long study keeps a few meshes, not thousands
                self._cache.clear()
            self._cache[key] = IPMMachine(d.geometry(), n_gap=self.n_gap, magnet_temperature=T)
        return self._cache[key]

    def winding(self, d: Motor3DDesign) -> dict[str, float]:
        """Winding quantities of the axial geometry: end-turn length per side (m), the phase resistance at 20 degC
        (Ohm) and its split between the slot and the ends (the resistance grows with the stack length)."""
        st = d.geometry().stator
        r_mid = st.r_in + 0.5 * (st.H0 + st.H2)
        pitch = 6 * 2.0 * math.pi * r_mid / st.slots             # 6-slot coil pitch at the mid-slot radius
        end = 0.5 * math.pi * pitch + 2.0 * _OVERHANG
        R20 = copper_resistance(_SERIES_TURNS, _CONDUCTOR_AREA, d.L, end, 20.0, parallel_paths=1)
        frac_slot = d.L / (d.L + end)
        return {"end_turn_m": end, "R_phase_20C": R20, "slot_fraction": frac_slot,
                "R_slot_20C": R20 * frac_slot, "R_end_20C": R20 * (1.0 - frac_slot)}

    def _iron_split(self, m: IPMMachine, loss: dict) -> dict[str, float]:
        """Iron loss in the yoke, the teeth and the rotor (W), splitting the model's totals with the loss density of
        the solution the way :func:`~..thermal.iron_split` does."""
        mesh = m.mm.mesh(0.0)
        area, c = mesh.areas(), mesh.centroids()
        rc = np.hypot(c[:, 0], c[:, 1])
        st = m._stator_iron
        s = m.geometry.stator
        r_slot_bot = s.r_in + s.H0 + s.H1 + s.H2
        dens = loss["stator_loss_density"]
        yoke = float(np.sum((area[st] * dens)[rc[st] >= r_slot_bot]))
        teeth = float(np.sum((area[st] * dens)[rc[st] < r_slot_bot]))
        rotor = float(np.sum(area[m._rotor_iron] * loss["rotor_loss_density"]))
        tot = yoke + teeth + rotor
        total_W = float(loss["stator_iron_W"] + loss["rotor_iron_W"])
        k = total_W / tot if tot > 0 else 0.0
        return {"yoke_W": yoke * k, "teeth_W": teeth * k, "rotor_W": rotor * k, "total_W": total_W}

    def thermal_model(self, d: Motor3DDesign) -> AxialThermalModel:
        """The axial network of a design, with the jacket conductance scaled by the coolant flow (``flow^0.8``)."""
        data = MachineThermalData(**dict(self.thermal.__dict__))
        data.jacket_GA = self.thermal.jacket_GA * (max(d.flow_L_min, 1e-3) / self.coolant_flow_ref) ** 0.8
        return AxialThermalModel(d.geometry(), data, n_segments=self.n_segments,
                                 end_length=2.0 * self.winding(d)["end_turn_m"])

    # --------------------------------------------------------------------------------------------- one design
    def solve(self, d: Motor3DDesign, keep_fields: bool = False) -> dict[str, Any]:
        """Every quantity of one design, with the intermediate state (losses, node temperatures, fields)."""
        errors = d.sector_errors()
        if errors:
            return {"valid": False, "errors": errors}
        t0 = time.perf_counter()
        try:
            out = self._solve(d, keep_fields)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:      # no convergence, degenerate mesh
            return {"valid": False, "errors": [f"{type(e).__name__}: {e}"]}
        if out.get("invalid_reason"):
            return {"valid": False, "errors": [out["invalid_reason"]]}
        out["seconds"] = time.perf_counter() - t0
        out["valid"] = True
        return out

    def _solve(self, d: Motor3DDesign, keep_fields: bool) -> dict[str, Any]:
        gamma = math.radians(d.gamma_deg)
        i_d, i_q = -d.I_peak * math.sin(gamma), d.I_peak * math.cos(gamma)
        w = self.winding(d)

        magnet_T = self.magnet_T_guess
        for it in range(self.fea_iterations):
            last = it == self.fea_iterations - 1
            m = self.machine(d, magnet_T)
            t_fea = time.perf_counter()
            op = m.operating_point(i_d=i_d, i_q=i_q, n_positions=self.n_positions, keep=last)
            loss = m.losses(d.rpm, i_d, i_q, steps=self.loss_steps)
            self.count += 1
            self.seconds += time.perf_counter() - t_fea
            split = self._iron_split(m, loss)
            P_cu_20 = 1.5 * d.I_peak ** 2 * w["R_phase_20C"]
            state = self._thermal_loop(d, split, loss["magnet_W"], P_cu_20)
            magnet_T = state["T_magnet_max_C"]

        # --- skew: every axial slice contributes at its own electrical angle
        period, p = m.load_period, m.pole_pairs
        K = max(self.n_segments, 2)
        deltas = (np.arange(K) / (K - 1) - 0.5) * d.skew_rad * p       # electrical angle of each slice
        torque_t, th = np.asarray(op["torque"], float), np.asarray(op["thetas"], float)
        slice_t = np.array([_interp_periodic(th, torque_t, np.array([dd]), period)[0] for dd in deltas])
        skew_all = np.concatenate([_interp_periodic(th, torque_t, th + dd, period) for dd in deltas])
        ripple_skew = float((skew_all.max() - skew_all.min()) / abs(skew_all.mean()))

        # --- magnet temperature coupling: Br(T) scales the magnet part of the field
        br_ratio = m.magnet.Br(magnet_T) / m.magnet.Br(self.magnet_T_guess)
        k_skew = d.skew_factor(p)
        torque = op["torque_mean"] * br_ratio * k_skew
        split_T = {k: v * br_ratio ** 2 for k, v in split.items()}
        p_magnet_loss = loss["magnet_W"] * br_ratio ** 2
        state = self._thermal_loop(d, split_T, p_magnet_loss, 1.5 * d.I_peak ** 2 * w["R_phase_20C"])
        p_cu = state["P_cu_W"]
        p_iron = split_T["total_W"]

        if state["T_magnet_max_C"] > self.magnet_T_limit:
            return {"invalid_reason": f"magnet at {state['T_magnet_max_C']:.0f} degC is past the reversible range of "
                                      f"the grade ({self.magnet_T_limit:.0f} degC): Hcj(T) would be extrapolated "
                                      f"through zero and the demagnetisation margin has no meaning"}

        # --- demagnetisation at the magnet temperature (the reverse field scales with Br too)
        Hcj = intrinsic_coercivity(self.Hcj20, state["T_magnet_max_C"])
        ratio = 0.0
        for sol in op.get("solutions") or []:
            ratio = max(ratio, float(demagnetisation_margin(m, sol, Hcj)["ratio"].max()))
        ratio *= br_ratio

        # --- drive limits, masses, efficiency
        v_phase = d.omega_mech * p * math.hypot(op["psi_d"], op["psi_q"])
        v_limit = self.v_dc / math.sqrt(3)
        p_mech = torque * d.omega_mech
        p_loss = p_iron + p_magnet_loss + p_cu
        eff = p_mech / (p_mech + p_loss) if p_mech > 0 else float("nan")

        geom = d.geometry()
        rt, rho_fe, rho_cu = geom.rotor, 7650.0, 8900.0
        V_magnet = 2 * rt.poles * rt.W4 * rt.H3 * d.L
        magnet_mass = V_magnet * 7500.0
        V_iron = math.pi * (geom.stator.r_out ** 2 - rt.r_in ** 2) * d.L - V_magnet
        copper_mass = 3.0 * _SERIES_TURNS * 2.0 * (d.L + w["end_turn_m"]) * _CONDUCTOR_AREA * rho_cu
        active_mass = V_iron * rho_fe * geom.stacking_factor + magnet_mass + copper_mass

        res = {"torque_Nm": float(torque), "ripple": ripple_skew,
               "torque_density_Nm_kg": float(torque / active_mass),
               "efficiency": float(eff), "iron_loss_W": float(p_iron), "magnet_loss_W": float(p_magnet_loss),
               "copper_loss_W": float(p_cu), "total_loss_W": float(p_loss),
               "T_winding_C": state["T_winding_max_C"], "T_magnet_C": state["T_magnet_max_C"],
               "T_end_winding_C": state["T_end_winding_C"],
               "demag_ratio": float(ratio), "magnet_mass_kg": float(magnet_mass),
               "active_mass_kg": float(active_mass),
               "torque_slice_Nm": slice_t,
               "T_winding_profile_C": state["T_winding_profile_C"],
               "T_magnet_profile_C": state["T_magnet_profile_C"],
               "extra": {"Br_ratio": float(br_ratio), "skew_factor": float(k_skew), "i_d": float(i_d),
                         "i_q": float(i_q), "v_phase_V": float(v_phase), "v_limit_V": float(v_limit),
                         "voltage_margin": float(1.0 - v_phase / v_limit), "omega_mech": float(d.omega_mech),
                         "R_phase_ohm": float(state["R_phase_ohm"]), "Hcj_A_m": float(Hcj),
                         "iron_split": split, "T_nodes": state["T"], "T_names": state["names"],
                         "psi_d": float(op["psi_d"]), "psi_q": float(op["psi_q"])}}
        if keep_fields:
            res["fields"] = {"machine": m, "op": op, "loss": loss}
        return res

    def _thermal_loop(self, d: Motor3DDesign, split: dict[str, float], p_magnet_loss: float, P_cu_20: float,
                      n_iter: int = 3) -> dict[str, Any]:
        """The axial network with the copper loss at the winding temperature, until the two agree (the resistance
        depends on the temperature the loss produces). No FEA: one linear solve per sweep."""
        net, w = self.thermal_model(d), self.winding(d)

        def levels(T: np.ndarray) -> np.ndarray:
            return np.array([T[net.index("winding", k)] for k in range(net.K)])

        def run(P_cu: float) -> np.ndarray:
            return net.network.steady(net.powers(split["yoke_W"], split["teeth_W"],
                                                 P_cu * w["slot_fraction"], P_cu * (1.0 - w["slot_fraction"]),
                                                 split["rotor_W"], p_magnet_loss))

        T = run(P_cu_20)
        for _ in range(n_iter):
            P_cu = 1.5 * d.I_peak ** 2 * w["R_phase_20C"] * (1.0 + _ALPHA_CU * (float(levels(T).max()) - 20.0))
            T = run(P_cu)
        out: dict[str, Any] = {"T": T, "names": list(net.names), "P_cu_W": float(P_cu),
                               "R_phase_ohm": float(w["R_phase_20C"] * (1.0 + _ALPHA_CU * (float(levels(T).max())
                                                                                          - 20.0))),
                               "T_end_winding_C": float(T[net.index("end_winding")])}
        for part, tag in (("winding", "winding"), ("magnets", "magnet")):
            prof = np.array([T[net.index(part, k)] for k in range(net.K)])
            out[f"T_{tag}_profile_C"] = prof
            out[f"T_{tag}_max_C"] = float(prof.max())
            out[f"T_{tag}_mean_C"] = float(prof.mean())
        return out

    # ----------------------------------------------------------------------------------------- whole batches
    def evaluate(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """The outputs of every design of ``X`` (``(n, d)``, columns in the order of :data:`~.design.NAMES`), as
        arrays: the scalars of :data:`OUTPUTS` and the profiles of :data:`PROFILES` (``(n, n_segments)``). Designs
        the mesher cannot build come back as NaN."""
        X = np.atleast_2d(np.asarray(X, float))
        n = len(X)
        out: dict[str, np.ndarray] = {k: np.full(n, np.nan) for k in SCALARS}
        for k in PROFILES:
            out[k] = np.full((n, self.n_segments), np.nan)
        out["_valid"] = np.zeros(n, bool)
        for i, d in enumerate(from_matrix(X)):
            r = self.solve(d)
            if not r.get("valid"):
                continue
            for k in SCALARS:
                out[k][i] = r[k]
            for k in PROFILES:
                out[k][i] = r[k]
            out["_valid"][i] = True
        return out

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "fidelity": "reference", "outputs": list(self.outputs),
                "description": self.description, "n_fea_solves": self.count, "fea_seconds": self.seconds}


#: The 2004 Prius design at its published operating point, the reference every study is compared with.
PRIUS = Motor3DDesign()
