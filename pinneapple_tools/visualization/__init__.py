"""
pinneapple_viz — CFD-style visualisation for PINNeAPPle.

Quick start
-----------
from pinneapple_tools.visualization import use_cfd_style, plot_scalar, plot_vectors, plot_streamlines
from pinneapple_tools.visualization import plot_loss_history, plot_collocation, plot_pinn_prediction
from pinneapple_tools.visualization import animate_scalar_field, plot_solver_output

use_theme()   # the house theme (#918); use_cfd_style() keeps the older dark CFD look
"""
from .style import (
    use_cfd_style,
    use_theme,
    get_cmap,
    make_figure,
    make_panel_figure,
    figure_sizes,
    field_cmap,
    colorbar,
    label,
    annotate_verdict,
    panel_letters,
    source_note,
    save_figure,
    CMAPS,
    FIELD_CMAPS,
    PALETTE,
    DEFAULT_CMAP,
)

from .fields import (
    plot_scalar,
    plot_scalar_3d,
    plot_vectors,
    plot_streamlines,
    compare_fields,
    plot_error,
)

from .pinn import (
    plot_loss_history,
    plot_multi_loss,
    plot_collocation,
    plot_pinn_prediction,
    plot_pde_residual,
    plot_gradient_magnitude,
    predict_and_plot,
)

from .solver import (
    plot_solver_output,
    plot_fem_result,
    plot_fvm_result,
    plot_residuals,
)

# PyVista 3-D rendering (optional dependency: pip install "pinneapple[pyvista]")
from .pyvista_bridge import (
    calculix_grid,
    scene_to_multiblock,
    grid_from_arrays,
    von_mises,
    export_vtk,
    can_render,
    render_png,
    render_gif,
)

from .mesh import (
    plot_mesh,
    plot_boundary,
    plot_point_cloud,
)

from .voxel import (
    plot_voxel_slice,
    plot_voxel_3d,
    plot_voxel_histogram,
)

from .animation import (
    animate_scalar_field,
    animate_streamlines,
    make_gif,
)

# Orbiting camera and advancing time in one call, GIF or MP4, 1080p (#64)
from .orbit import (
    DEFAULT_AZIMUTH,
    DEFAULT_ORBIT,
    DEFAULT_SIZE,
    turntable,
)

from .vortex import (
    compute_vorticity_2d,
    compute_q_criterion_2d,
    compute_q_criterion_3d,
    compute_lambda2_3d,
    plot_vorticity,
    plot_q_criterion_2d,
    plot_q_criterion_3d,
    plot_vortex_identification,
    plot_lbm_flow,
    plot_flow_panel,
)

__all__ = [
    # Style
    "use_cfd_style", "get_cmap", "make_figure", "CMAPS", "DEFAULT_CMAP",
    # House theme (#918): one palette, one map per quantity, labelled axes
    "use_theme", "make_panel_figure", "figure_sizes", "field_cmap",
    "colorbar", "label", "annotate_verdict", "panel_letters", "source_note",
    "save_figure", "FIELD_CMAPS", "PALETTE",
    # Scalar / vector fields
    "plot_scalar", "plot_scalar_3d", "plot_vectors", "plot_streamlines",
    "compare_fields", "plot_error",
    # PINN
    "plot_loss_history", "plot_multi_loss", "plot_collocation",
    "plot_pinn_prediction", "plot_pde_residual", "plot_gradient_magnitude",
    "predict_and_plot",
    # Solvers
    "plot_solver_output", "plot_fem_result", "plot_fvm_result", "plot_residuals",
    # Mesh
    "plot_mesh", "plot_boundary", "plot_point_cloud",
    # Voxel
    "plot_voxel_slice", "plot_voxel_3d", "plot_voxel_histogram",
    # Animation
    "animate_scalar_field", "animate_streamlines", "make_gif",
    # Turntable (orbiting camera + time) and its defaults
    "turntable", "DEFAULT_SIZE", "DEFAULT_ORBIT", "DEFAULT_AZIMUTH",
    # Vortex / Q-criterion
    "compute_vorticity_2d", "compute_q_criterion_2d",
    "compute_q_criterion_3d", "compute_lambda2_3d",
    "plot_vorticity", "plot_q_criterion_2d", "plot_q_criterion_3d",
    "plot_vortex_identification", "plot_lbm_flow", "plot_flow_panel",
    # PyVista bridge
    "calculix_grid", "scene_to_multiblock", "grid_from_arrays", "von_mises",
    "export_vtk", "can_render", "render_png", "render_gif",
]
