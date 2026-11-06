`pinneapple_flow.wrappers.car_wind_tunnel_3d`: the Ahmed body (Ahmed, Ramm & Faltin 1984) in the 3-D lattice-Boltzmann
wind tunnel (`WindTunnelLBM`, D3Q19 + Smagorinsky, moving road) as one JSON-argument call, so a flow or the engineering
app runs a real 3-D external-aerodynamics case with no mesh file. It takes the slant angle, the speed and a resolution,
builds the body's solid mask from its own definition, runs, and returns the drag and lift coefficients, the force
history, the mean field on three slices (mid-y and mid-z planes) and its checks (`simulate.converged` on the force
settling, `simulate.balance` on the flux closure). The Reynolds number is clamped to what the lattice holds at that
resolution (`tau >= 0.52`), and the value states the regime: the reference coefficient of the body (0.285 at 25 deg)
belongs to Re 4.3e6, which this grid cannot reach.
