- A Blender extension for STEP parts, `pinneapple_blender/step_extension` (#1119). Its **PINNeAPPle** sidebar panel
  covers:
  - import of exact bodies at their real size;
  - the mesher and quality, and re-meshing;
  - the quality report;
  - curvature, zebra and highlight-line shading, plus a real-time zebra material and a stripe environment;
  - bake to GLB, OBJ, STL or USDA;
  - "Send to PINNeAPPle" as `MeshData` arrays.

  `python -m pinneapple_blender.step_extension.build --wheels` builds the installable zip, with a copy of the
  library and the per-platform wheels.
