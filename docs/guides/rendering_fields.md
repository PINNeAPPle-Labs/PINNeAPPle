# A field on any domain: the picture and the VTU

Roadmap item #61. A model is evaluated where the question is, not where it was trained: a mesh, a point cloud, a box
that is not the training box. `render_field` evaluates it there and writes the same numbers twice — as a figure to
look at and as a `.vtu` the solvers and ParaView read.

```python
import pinneapple as pp

written = pp.viz.render_field(
    model,                                   # anything with .predict, or a callable taking the points
    {"x": (0.0, 1.0), "y": (0.0, 1.0)},      # a problem, {name: (low, high)}, (lo, hi), or points=
    out="figures/plate",
    title="Temperature, steady state",
)
written        # {"image": "figures/plate.png", "vtu": "figures/plate.vtu"}
```

The name is `render_field` and not `render(model, domain)`: `pp.viz.render` is the Blender render of a scene, and one
name for two things would make both ambiguous.

## The domain

| You have | Pass |
|---|---|
| a `PhysicalProblem` | the problem: its `domain_bounds` are used |
| the box of the study | `{"x": (0, 1), "y": (0, 2), "z": (0, 0.1)}` |
| one, two or three extents | `(lo, hi)` with two arrays |
| the points themselves | `points=X`, shape `(n, 2)` or `(n, 3)` |

`n` is the point budget shared between the axes (default 4096), so a 2-D box gets 64 x 64 and a 3-D one 16 x 16 x 16.
With `points=` nothing is generated and the grid is what you passed.

## What the file carries

`write_vtu` writes VTK XML by hand, ASCII, with no optional dependency: every point, its coordinates and one data
array per field. A point cloud becomes one vertex cell per point, because that is what evaluating a model on an
arbitrary domain produces; a mesh passes its own connectivity (`cells=[[0, 1, 2], [0, 2, 3]]`, or the three VTK
arrays), and a three-node cell is written as a triangle.

```python
pp.viz.write_vtu("cloud.vtu", X, {"T": temperature, "u": velocity_vectors})
```

A 2-D domain is written with `z = 0`: VTK is three-dimensional, and a reader that has to guess the missing coordinate
is a reader that draws nothing.

## The picture

A 2-D grid is a filled contour of the first field, on the house theme, with the axes named after the coordinates of
the domain. A 3-D grid is drawn as the **mid-plane slice in the last coordinate** — the VTU holds the whole volume,
and the picture is the one a reader can look at without a 3-D viewer.

## What it refuses

A model that returns a shape that does not match the points, a number of field names that does not match the columns,
a points array that is not `(n, 2)` or `(n, 3)`, and a field whose length is not the number of points. Each refusal
names the shape it got and the shape it needed, because a silently truncated field is a figure that lies.
