# 07 Conjugate cold plate: microchannels, coolant and a three-channel FNO

**What.** A 100 x 100 mm copper cold plate with microchannels slot-milled into it, two 110 W chips (14 x 14 mm) on
its face and water at 40 C through the slots. The problem is the *conjugate* one, not a conduction plate: heat
conducts in the plane of the slots and leaves through the walls of each slot into the coolant, whose own temperature
rises as it takes that heat. Per unit plate area,

```
-div(G grad T) + sigma (T - T_f) = q''        T = T_in on the four edges
```

with `G` the thickness-integrated in-plane conductance (full metal over a wall, the fin plus the water over a slot),
`sigma = h (1 + 2 depth/width)` the slot's coupling to the coolant, and `T_f = T_in + dT_f/2`, `dT_f = Q_coolant /
(m_dot c_p)` closed against the heat the slots actually take. `h` comes from the fully developed Nusselt number of a
lidded slot (Shah & London 1978, `Nu_Dh = 5.385`) with Hausen's entry correction, water properties from Incropera
and DeWitt Table A.6, and the pressure drop from Muzychka-Yovanovich with the same rectangular-duct `f Re` fit and
Kays & London losses the plate-fin model uses.

**Why three input channels.** The operator solved is `-div(a grad u) + s u = f0 + s theta_f`, and the source a slot
gives back to the plate carries the coolant's own rise `theta_f`, which no map of the conductance and the coupling
alone can recover. A two-channel FNO reaches 5.2 % on the held-out fields; handing it the source it actually solves
takes that to 1.1 %. The error is visible because PINNeAPPle measures it, not because somebody guessed the inputs.

| Step | Who | Ran here |
|---|---|---|
| 250 designs (channel count, slot width, flow, film coefficient), each solved coupled | PINNeAPPle | yes |
| FNO (2.4 M parameters) on (conductance, coupling, source) -> temperature field | PhysicsNeMo 2.2.2 | yes (CPU) |
| FullyConnected on the design variables -> hot spot, dp, pumping power | PhysicsNeMo 2.2.2 | yes (CPU) |
| certified residual bounds for the reacting operator, conjugate energy balance, maximum principle, reference, envelope | PINNeAPPle | yes |
| Monte-Carlo of the plant (flow, film coefficient, duty) on two designs | PINNeAPPle | yes |
| OpenUSD stage: temperature, conductance, coupling, reference and estimated error as primvars, trust on the prim | OpenUSD (`usd-core`) | yes, validated with `pxr.UsdValidation` |
| RTX render in Omniverse Kit / USD Composer | Omniverse | **no** (needs Omniverse and an RTX GPU): `results/kit_render.py` |

```bash
pip install nvidia-physicsnemo usd-core
python examples/nvidia/07_omniverse_physicsnemo_coldplate/run.py           # ~10 min on 10 CPU cores, no GPU
python examples/nvidia/07_omniverse_physicsnemo_coldplate/run.py --smoke   # seconds (CI)
```

Two thirds of that is the FNO: 2400 steps of a three-channel operator at 32 x 32 on the CPU.

## Results (this run: macOS arm64, 10 CPU cores, no GPU, PhysicsNeMo 2.2.2, torch 2.15.0, usd-core 26.11)

- Every one of the 250 coupled solves closes its energy balance to 1e-14 (heat to the coolant + heat to the walls =
  chip power). The guard ring around the plate keeps the outer walls adiabatic: the rim takes 0.2 % of the power,
  and without it a quarter of the heat would leave through the manifold and the hot spot would be optimistic by 5 K.
- FNO, 2400 Adam steps (batch 8, 96 epochs), 200 training / 50 held-out designs, 32 x 32:
  **1.1 %** relative L2 on the held-out fields, worst hot-spot error **1.2 K**.
- **Decision: APPROVED.** The certified energy-error bound is 2.3 % (limit 10 %), the L2 bound 1.9 % (limit 5 %), the
  conjugate energy balance closes to 5e-15 (limit 2 %), the flux balance 2.0 % (limit 2 %), the boundary condition
  0.46 %, the maximum principle holds, and the
  coefficient envelope flags none of the 50 held-out designs. The reference solver's own fields pass every check, so
  the tolerances are not the reason the surrogate passes.
- Design surrogate on the held-out designs: 3.0 % (hot spot), 3.9 % (dp), 4.2 % (pumping power); all five physical
  directions hold across the whole box (hot spot falls with flow, with the film coefficient and with the channel
  count; dp and pumping power grow with flow), and the field surrogate never flips the design decision on the 50
  held-out designs.
- Chosen design (lightest pumping power among the ones with margin): 27 slots of 0.91 mm at 0.58 L/min --
  **74.6 C** hot spot, dp 497 Pa, Re 453 (laminar), h 3654 W/m2 K, coolant rise 5.5 K, pumping power 4.8 mW.
- **Monte-Carlo of the plant** (flow lognormal 10 %, film coefficient +/-15 %, duty +/-5 %, 400 draws). The robust
  pick the study ships fails in P = 0.00, from the surrogate and from the coupled solves alike. The lightest design
  the surrogate puts 2 K under the limit fails in **P = 0.30 by the surrogate and P = 1.00 by the coupled solves**:
  selecting on a surrogate's own prediction also selects the designs where that prediction is optimistic, so the
  check is written as "the design at the edge really is worse" rather than as "the two probabilities match". That
  gap is the whole argument for shipping a design with margin instead of the lightest one that fits.

Figures in `results/`: `architecture.png`, `fields.png` (conductance, coupling, both temperature fields, the true
error and the error estimated from the residual alone), `training.png`, `design_space.png` (pumping power against hot
spot, the two picks), `uncertainty.png` (both designs, surrogate against coupled solves), `checks.png`,
`usd_preview.png` (the stage read back with OpenUSD; not an RTX render). `summary.json` and `run_info.json` hold
every number and the environment. The `.usda` is regenerated by the script and not committed.
