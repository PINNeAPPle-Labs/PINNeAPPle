"""Blender operators and the sidebar panel. Imported by :func:`register` only (it needs ``bpy``)."""
from __future__ import annotations

import json
import os

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, IntProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

from . import core

ATTR = "pnp_diagnostic"
MESHERS = [("auto", "Auto", "Fast when every face allows it, else the kernel"),
           ("fast", "Fast", "Analytic: planes, cylinders, cones, spheres; watertight by construction"),
           ("kernel", "Kernel", "The exact kernel's tessellation, any surface")]
QUALITIES = [("draft", "Draft", "Chord 2e-3 of the size, 30 deg"), ("normal", "Normal", "Chord 5e-4, 15 deg"),
             ("fine", "Fine", "Chord 1e-4, 7.5 deg")]

# object name -> (CadBody, TessMesh) for this session; rebuilt from the object's STEP path when missing
_MESHES: dict[str, tuple] = {}


class PNP_StepSettings(bpy.types.PropertyGroup):
    mesher: EnumProperty(name="Mesher", items=MESHERS, default="auto")
    quality: EnumProperty(name="Quality", items=QUALITIES, default="normal")
    diagnostic: EnumProperty(name="Diagnostic", items=[(k, v, v) for k, v in core.DIAGNOSTICS.items()],
                             default="MEAN")
    refine: BoolProperty(name="Refine for diagnostics", default=True,
                         description="Split triangles so planar faces get interior vertices to colour")
    stripes: IntProperty(name="Stripes", default=16, min=2, max=200)


def _settings(context) -> PNP_StepSettings:
    return context.scene.pinneapple_step


def _set_mesh(obj, tm) -> None:
    """Replace the object's mesh with a TessMesh (metres, Blender's default unit)."""
    me = obj.data
    me.clear_geometry()
    V, T = np.asarray(tm.vertices, np.float32), np.asarray(tm.triangles, np.int32)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.ravel())
    me.loops.add(T.size)
    me.loops.foreach_set("vertex_index", T.ravel())
    me.polygons.add(len(T))
    me.polygons.foreach_set("loop_start", np.arange(0, T.size, 3, dtype=np.int32))
    me.polygons.foreach_set("loop_total", np.full(len(T), 3, np.int32))
    me.update(calc_edges=True)
    me.validate()
    me.polygons.foreach_set("use_smooth", np.zeros(len(T), bool))
    if "pnp_face" in me.attributes:
        me.attributes.remove(me.attributes["pnp_face"])
    fa = me.attributes.new("pnp_face", "INT", "FACE")
    fa.data.foreach_set("value", np.asarray(tm.face_id, np.int32))


def _entry(obj, settings=None):
    """(CadBody, TessMesh) of a STEP object, re-read and re-meshed when the session does not have it."""
    if obj.name in _MESHES:
        return _MESHES[obj.name]
    path, idx = obj.get("pnp_step_path"), obj.get("pnp_body")
    if path is None or idx is None:
        return None
    body = core.bodies(path)[int(idx)]
    tm = core.brep().tessellate(body, obj.get("pnp_quality", "normal"), obj.get("pnp_mesher", "auto"))
    _MESHES[obj.name] = (body, tm)
    return _MESHES[obj.name]


def _step_objects(context):
    return [o for o in context.selected_objects if o.type == "MESH" and "pnp_step_path" in o]


class PNP_OT_import_step(bpy.types.Operator, ImportHelper):
    """Import every body of a STEP file at its real size, keeping the exact body for re-meshing"""
    bl_idname = "pinneapple.import_step"
    bl_label = "Import STEP"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".step"
    filter_glob: StringProperty(default="*.step;*.stp;*.STEP;*.STP", options={"HIDDEN"})

    def execute(self, context):
        s = _settings(context)
        try:
            bodies = core.bodies(self.filepath)
        except Exception as e:                                     # report, do not crash Blender
            self.report({"ERROR"}, f"STEP import failed: {e}")
            return {"CANCELLED"}
        stem = os.path.splitext(os.path.basename(self.filepath))[0]
        col = bpy.data.collections.new(stem)
        context.scene.collection.children.link(col)
        for i, b in enumerate(bodies):
            tm = core.brep().tessellate(b, s.quality, s.mesher)
            name = b.name or f"{stem}_{i}"
            obj = bpy.data.objects.new(name, bpy.data.meshes.new(name))
            col.objects.link(obj)
            _set_mesh(obj, tm)
            obj["pnp_step_path"] = os.path.abspath(self.filepath)
            obj["pnp_body"] = i
            obj["pnp_quality"], obj["pnp_mesher"] = s.quality, tm.mesher
            obj["pnp_volume_m3"], obj["pnp_area_m2"] = b.volume_m3, b.area_m2
            _MESHES[obj.name] = (b, tm)
        self.report({"INFO"}, f"{len(bodies)} bodies from {os.path.basename(self.filepath)}")
        return {"FINISHED"}


class PNP_OT_remesh(bpy.types.Operator):
    """Mesh the selected STEP objects again with the mesher and quality of the panel"""
    bl_idname = "pinneapple.remesh"
    bl_label = "Re-mesh"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = _settings(context)
        objs = _step_objects(context)
        for o in objs:
            body = (_entry(o) or (None,))[0]
            try:
                tm = core.brep().tessellate(body, s.quality, s.mesher)
            except ValueError as e:                                # the fast mesher refused this body
                self.report({"WARNING"}, f"{o.name}: {e}")
                continue
            _set_mesh(o, tm)
            o["pnp_quality"], o["pnp_mesher"] = s.quality, tm.mesher
            _MESHES[o.name] = (body, tm)
        self.report({"INFO"}, f"re-meshed {len(objs)} objects")
        return {"FINISHED"}


class PNP_OT_quality(bpy.types.Operator):
    """Mesh quality of the selected STEP objects against their exact bodies"""
    bl_idname = "pinneapple.quality_report"
    bl_label = "Quality report"

    def execute(self, context):
        lines = []
        for o in _step_objects(context):
            body, tm = _entry(o)
            r = core.brep().mesh_quality(tm, body)
            o["pnp_quality_report"] = json.dumps(r)
            lines.append(f"{o.name}: {r['n_triangles']} tris, watertight {r['watertight']}, "
                         f"self-int {r['self_intersections']}, min angle {r['min_angle_deg']:.2f} deg, "
                         f"chord {r['chordal_error_over_tolerance']:.2f} x tol, volume err {r['volume_error_rel']:.1e}")
        for ln in lines:
            self.report({"INFO"}, ln)
        return {"FINISHED"}


def _attribute_material(name: str):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    if hasattr(mat, "use_nodes") and not mat.use_nodes:
        mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = ATTR
    emit = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def _assign(obj, mat) -> None:
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


class PNP_OT_diagnostic(bpy.types.Operator):
    """Colour the selected STEP objects by a surface diagnostic computed on the exact body"""
    bl_idname = "pinneapple.diagnostic_shading"
    bl_label = "Diagnostic shading"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = _settings(context)
        mat = _attribute_material("PNP Diagnostic")
        for o in _step_objects(context):
            body, tm = _entry(o)
            if s.refine:
                tm = core.brep().diagnostics.refine(tm, max_triangles=150_000)
                _set_mesh(o, tm)
                _MESHES[o.name] = (body, tm)
            eye = None
            if context.region_data is not None:                    # the viewport's eye, for zebra
                eye = list(context.region_data.view_matrix.inverted().translation)
            v, rgb, summary = core.diagnostic(body, tm, s.diagnostic, stripes=s.stripes, eye=eye)
            me = o.data
            if ATTR in me.color_attributes:
                me.color_attributes.remove(me.color_attributes[ATTR])
            ca = me.color_attributes.new(ATTR, "FLOAT_COLOR", "POINT")
            ca.data.foreach_set("color", np.c_[rgb, np.ones(len(rgb))].astype(np.float32).ravel())
            me.color_attributes.active_color = ca
            _assign(o, mat)
            o["pnp_diagnostic"] = json.dumps(summary)
        for area in getattr(context.screen, "areas", []):
            if area.type == "VIEW_3D":
                area.spaces[0].shading.color_type = "VERTEX"
        return {"FINISHED"}


class PNP_OT_zebra_material(bpy.types.Operator):
    """Real-time zebra material: stripes from the azimuth of the reflection vector, as the view moves"""
    bl_idname = "pinneapple.zebra_material"
    bl_label = "Zebra material"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = _settings(context)
        mat = bpy.data.materials.get("PNP Zebra") or bpy.data.materials.new("PNP Zebra")
        if hasattr(mat, "use_nodes") and not mat.use_nodes:
            mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        tc = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        atan = nt.nodes.new("ShaderNodeMath")
        atan.operation = "ARCTAN2"
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = float(s.stripes)
        sin = nt.nodes.new("ShaderNodeMath")
        sin.operation = "SINE"
        step = nt.nodes.new("ShaderNodeMath")
        step.operation = "GREATER_THAN"
        step.inputs[1].default_value = 0.0
        emit = nt.nodes.new("ShaderNodeEmission")
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(tc.outputs["Reflection"], sep.inputs[0])
        nt.links.new(sep.outputs["Y"], atan.inputs[0])
        nt.links.new(sep.outputs["X"], atan.inputs[1])
        nt.links.new(atan.outputs[0], mul.inputs[0])
        nt.links.new(mul.outputs[0], sin.inputs[0])
        nt.links.new(sin.outputs[0], step.inputs[0])
        nt.links.new(step.outputs[0], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
        for o in _step_objects(context):
            _assign(o, mat)
        return {"FINISHED"}


class PNP_OT_stripe_world(bpy.types.Operator):
    """A world of horizontal light stripes, so glossy materials show highlight lines in Material Preview"""
    bl_idname = "pinneapple.stripe_world"
    bl_label = "Stripe environment"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = _settings(context)
        world = bpy.data.worlds.get("PNP Stripes") or bpy.data.worlds.new("PNP Stripes")
        if hasattr(world, "use_nodes") and not world.use_nodes:
            world.use_nodes = True
        nt = world.node_tree
        nt.nodes.clear()
        tc = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = float(s.stripes) * np.pi
        sin = nt.nodes.new("ShaderNodeMath")
        sin.operation = "SINE"
        step = nt.nodes.new("ShaderNodeMath")
        step.operation = "GREATER_THAN"
        step.inputs[1].default_value = 0.6
        bg = nt.nodes.new("ShaderNodeBackground")
        out = nt.nodes.new("ShaderNodeOutputWorld")
        nt.links.new(tc.outputs["Generated"], sep.inputs[0])       # the view direction in a world shader
        nt.links.new(sep.outputs["Z"], mul.inputs[0])
        nt.links.new(mul.outputs[0], sin.inputs[0])
        nt.links.new(sin.outputs[0], step.inputs[0])
        nt.links.new(step.outputs[0], bg.inputs["Strength"])
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
        context.scene.world = world
        return {"FINISHED"}


class PNP_OT_bake(bpy.types.Operator, ExportHelper):
    """Write the selected STEP objects to a mesh file (format from the extension), with diagnostic colours"""
    bl_idname = "pinneapple.bake"
    bl_label = "Bake to mesh"
    filename_ext = ".glb"
    format: EnumProperty(name="Format", items=[(e, e[1:].upper(), "") for e in (".glb", ".stl", ".obj", ".usda")],
                         default=".glb")
    colors: BoolProperty(name="Diagnostic colours", default=True)

    def check(self, context):
        changed = super().check(context) if hasattr(super(), "check") else False
        root = os.path.splitext(self.filepath)[0]
        if self.filepath != root + self.format:
            self.filepath = root + self.format
            changed = True
        return changed

    def execute(self, context):
        objs = _step_objects(context)
        if not objs:
            self.report({"ERROR"}, "select imported STEP objects")
            return {"CANCELLED"}
        ext = os.path.splitext(self.filepath)[1].lower()
        meshes, cols = [], []
        for o in objs:
            tm = _entry(o)[1]
            meshes.append(tm)
            c = None
            if self.colors and ext != ".stl" and ATTR in o.data.color_attributes \
                    and len(o.data.vertices) == len(tm.vertices):
                buf = np.empty(4 * len(tm.vertices), np.float32)
                o.data.color_attributes[ATTR].data.foreach_get("color", buf)
                c = buf.reshape(-1, 4)[:, :3].astype(float)
            cols.append(c)
        info = core.brep().bake(meshes, self.filepath, colors=cols if len(cols) > 1 else cols[0],
                                names=[o.name for o in objs])
        self.report({"INFO"}, f"{info['n_triangles']} triangles to {info['path']} ({info['unit']})")
        return {"FINISHED"}


class PNP_OT_send(bpy.types.Operator):
    """Write the selected STEP objects as MeshData arrays (.npz, metres) for a simulation or PINN run"""
    bl_idname = "pinneapple.send_meshdata"
    bl_label = "Send to PINNeAPPle"
    directory: StringProperty(subtype="DIR_PATH")

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        out = []
        for o in _step_objects(context):
            tm = _entry(o)[1]
            out.append(core.write_meshdata(tm, os.path.join(self.directory, f"{bpy.path.clean_name(o.name)}.npz"),
                                           o.name)["path"])
        self.report({"INFO"}, f"wrote {len(out)} MeshData files to {self.directory}")
        return {"FINISHED"}


class PNP_PT_step(bpy.types.Panel):
    bl_label = "STEP"
    bl_idname = "PNP_PT_step"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PINNeAPPle"

    def draw(self, context):
        s = _settings(context)
        col = self.layout.column(align=True)
        col.operator(PNP_OT_import_step.bl_idname, icon="IMPORT")
        box = self.layout.box()
        box.prop(s, "mesher")
        box.prop(s, "quality")
        box.operator(PNP_OT_remesh.bl_idname, icon="MOD_TRIANGULATE")
        box.operator(PNP_OT_quality.bl_idname, icon="CHECKMARK")
        box = self.layout.box()
        box.prop(s, "diagnostic")
        box.prop(s, "refine")
        box.prop(s, "stripes")
        box.operator(PNP_OT_diagnostic.bl_idname, icon="SHADING_RENDERED")
        row = box.row(align=True)
        row.operator(PNP_OT_zebra_material.bl_idname)
        row.operator(PNP_OT_stripe_world.bl_idname)
        o = context.active_object
        if o is not None and "pnp_diagnostic" in o:
            d = json.loads(o["pnp_diagnostic"])
            box.label(text=f"{d['label']}: {d['min']:.4g} .. {d['max']:.4g}")
        col = self.layout.column(align=True)
        col.operator(PNP_OT_bake.bl_idname, icon="EXPORT")
        col.operator(PNP_OT_send.bl_idname, icon="FORWARD")


CLASSES = (PNP_StepSettings, PNP_OT_import_step, PNP_OT_remesh, PNP_OT_quality, PNP_OT_diagnostic,
           PNP_OT_zebra_material, PNP_OT_stripe_world, PNP_OT_bake, PNP_OT_send, PNP_PT_step)


def _menu_import(self, context):
    self.layout.operator(PNP_OT_import_step.bl_idname, text="STEP (PINNeAPPle exact B-rep)")


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.pinneapple_step = PointerProperty(type=PNP_StepSettings)
    bpy.types.TOPBAR_MT_file_import.append(_menu_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(_menu_import)
    del bpy.types.Scene.pinneapple_step
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
    _MESHES.clear()
    core.clear_cache()
