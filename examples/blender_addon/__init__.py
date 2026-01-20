"""Blender addon for previewing Gaussian splats.

Based on https://github.com/ReshotAI/gaussian-splatting-blender-addon
but simplified to use RGB colors (no spherical harmonics).

Installation:
    Zip the examples/blender_addon folder and install via Edit > Preferences > Add-ons > Install
"""

import os
import time

import bpy
import mathutils
import numpy as np

from .nodes import setup_geometry_nodes, setup_material
from .splat_io import SUPPORTED_EXTENSIONS, GaussianData, load_splat

bl_info = {
    "name": "USD Gaussian Splats Previewer",
    "author": "World Labs",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "3D Viewport > Sidebar > Gaussian Splatting",
    "description": "Preview Gaussian splats from USDZ files",
}


class SplatPreviewPreferences(bpy.types.AddonPreferences):
    """Addon preferences for Gaussian Splat Preview."""

    bl_idname = __name__

    max_splat_count: bpy.props.IntProperty(
        name="Max Splat Count",
        description="Maximum number of splats to display on import",
        default=500000,
        min=1000,
        max=2000000,
    )

    opacity_threshold: bpy.props.FloatProperty(
        name="Opacity Threshold",
        description="Opacity threshold for filtering splats to maxcount on import",
        default=0.2,
        min=0.0,
        max=1.0,
    )

    scale_multiplier: bpy.props.FloatProperty(
        name="Scale Multiplier",
        description="Scale multiplier for splat size on import",
        default=1.0,
        min=0.1,
        max=10.0,
    )

    point_cloud_mode: bpy.props.BoolProperty(
        name="Point Cloud Mode",
        description="Use point cloud mode instead of ellipsoids on import",
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "max_splat_count")
        layout.prop(self, "opacity_threshold")
        layout.prop(self, "scale_multiplier")
        layout.prop(self, "point_cloud_mode")


def get_preferences() -> SplatPreviewPreferences:
    """Get addon preferences."""
    return bpy.context.preferences.addons[__name__].preferences


class ImportGaussianSplatting(bpy.types.Operator):
    """Import a Gaussian splat file."""

    bl_idname = "object.preview_gaussian_splatting"
    bl_label = "Preview Gaussian Splatting"
    bl_description = "Preview a Gaussian splat file"
    bl_options = {"REGISTER", "UNDO"}

    filepath: bpy.props.StringProperty(
        name="File Path",
        description="Path to the splat file",
        subtype="FILE_PATH",
    )

    filter_glob: bpy.props.StringProperty(
        default="*.usdz",
        options={"HIDDEN"},
    )

    def execute(self, context):
        if not self.filepath:
            self.report({"WARNING"}, "No file selected")
            return {"CANCELLED"}

        start_time = time.time()

        try:
            gaussians = load_splat(self.filepath)
        except Exception as e:
            self.report({"ERROR"}, f"Failed to load: {e}")
            return {"CANCELLED"}

        n_gaussians = len(gaussians.positions)
        print(f"Loaded {n_gaussians:,} Gaussians in {time.time() - start_time:.2f}s")

        # Set up EEVEE renderer (more stable for large gaussian counts)
        bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
        # Enable transparency
        bpy.context.scene.eevee.use_raytracing = False

        # Get name from filename
        splat_name = os.path.splitext(os.path.basename(self.filepath))[0]

        # Create mesh from positions
        mesh = bpy.data.meshes.new(name=f"{splat_name}_mesh")
        mesh.from_pydata(gaussians.positions.tolist(), [], [])
        mesh.update()

        # Add attributes
        self._add_mesh_attributes(mesh, gaussians)

        # Create object
        obj = bpy.data.objects.new(splat_name, mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Apply rotation to match Blender coordinate systems
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = (np.pi / 2, 0, 0)
        obj["gaussian_splatting"] = True

        # Set up material and geometry nodes with preferences
        prefs = get_preferences()
        setup_material(obj)
        setup_geometry_nodes(
            obj,
            max_splat_count=prefs.max_splat_count,
            opacity_threshold=prefs.opacity_threshold,
            scale_multiplier=prefs.scale_multiplier,
            point_cloud_mode=prefs.point_cloud_mode,
        )

        print(f"Total import time: {time.time() - start_time:.2f}s")
        return {"FINISHED"}

    def _add_mesh_attributes(self, mesh: bpy.types.Mesh, data: GaussianData) -> None:
        """Add Gaussian attributes to mesh."""
        n = len(data.positions)

        # Opacity
        opacity_attr = mesh.attributes.new(name="opacity", type="FLOAT", domain="POINT")
        opacity_attr.data.foreach_set("value", data.opacities.flatten())

        # Scale
        scale_attr = mesh.attributes.new(
            name="scale", type="FLOAT_VECTOR", domain="POINT"
        )
        scale_attr.data.foreach_set("vector", data.scales.flatten())

        # Color (RGB)
        color_attr = mesh.attributes.new(
            name="color", type="FLOAT_VECTOR", domain="POINT"
        )
        color_attr.data.foreach_set("vector", data.colors.flatten())

        # Rotation as euler (for geometry nodes instancing)
        euler_rotations = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            w, x, y, z = data.rotations[i]
            quat = mathutils.Quaternion((w, x, y, z))
            euler = quat.to_euler()
            euler_rotations[i] = (euler.x, euler.y, euler.z)

        rot_attr = mesh.attributes.new(
            name="rotation_euler", type="FLOAT_VECTOR", domain="POINT"
        )
        rot_attr.data.foreach_set("vector", euler_rotations.flatten())

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class GaussianSplattingPanel(bpy.types.Panel):
    """Panel for Gaussian Splatting tools."""

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_idname = "VIEW3D_PT_preview_gaussian_splatting"
    bl_category = "Splats Preview"
    bl_label = "Splats Preview"

    def draw(self, context):
        layout = self.layout

        # Import button
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        layout.operator(
            ImportGaussianSplatting.bl_idname,
            text="Import Splat",
            icon="IMPORT",
        )
        layout.label(text=f"Formats: {supported}")

        # Show controls if a gaussian splat object is selected
        obj = context.active_object
        if obj and obj.get("gaussian_splatting"):
            mod = obj.modifiers.get("Geometry Nodes")
            if mod and mod.node_group:
                layout.separator()
                layout.label(
                    text=f"Selected: {obj.name}", icon="OUTLINER_OB_POINTCLOUD"
                )

                # Splat Filtering section
                box = layout.box()
                box.label(text="Splat Filtering:")

                # Find the modifier input identifiers for our custom sockets
                for item in mod.node_group.interface.items_tree:
                    if item.item_type == "SOCKET" and item.in_out == "INPUT":
                        if item.name == "Max Splat Count":
                            box.prop(mod, f'["{item.identifier}"]', text="Max Splats")
                        elif item.name == "Opacity Threshold":
                            box.prop(
                                mod, f'["{item.identifier}"]', text="Opacity Threshold"
                            )

                # Display Options section
                box = layout.box()
                box.label(text="Display Options:")

                # Scale Multiplier
                for item in mod.node_group.interface.items_tree:
                    if item.item_type == "SOCKET" and item.in_out == "INPUT":
                        if item.name == "Scale Multiplier":
                            box.prop(
                                mod, f'["{item.identifier}"]', text="Scale Multiplier"
                            )

                # Point cloud toggle (from geometry nodes)
                bool_node = mod.node_group.nodes.get("Boolean")
                if bool_node:
                    box.prop(
                        bool_node,
                        "boolean",
                        text="Point Cloud (faster)",
                    )


class GaussianSplattingPrefsPanel(bpy.types.Panel):
    """Collapsible preferences subpanel."""

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_idname = "VIEW3D_PT_splat_preview_prefs"
    bl_parent_id = "VIEW3D_PT_preview_gaussian_splatting"
    bl_label = "Import Defaults"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        prefs = get_preferences()

        layout.prop(prefs, "max_splat_count", text="Max Splats")
        layout.prop(prefs, "opacity_threshold", text="Opacity Threshold")
        layout.prop(prefs, "scale_multiplier", text="Scale Multiplier")
        layout.prop(prefs, "point_cloud_mode", text="Point Cloud Mode")


classes = [
    SplatPreviewPreferences,
    ImportGaussianSplatting,
    GaussianSplattingPanel,
    GaussianSplattingPrefsPanel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
