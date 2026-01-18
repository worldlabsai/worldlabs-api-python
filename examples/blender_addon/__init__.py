"""Splat file I/O for Blender.

This module provides file format loaders for Gaussian splats that work in
Blender's Python environment without requiring torch.

USDZ format matches NVIDIA 3dgrut: scales and densities are stored in
pre-activation form and converted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _vt_array_to_numpy(vt_array, dtype=np.float32) -> np.ndarray:
    """Convert USD Vt array to numpy, handling Vec3f/Color3f/Quatf types.

    USD's Vec3fArray/Color3fArray sometimes don't convert to (N, 3) directly
    with np.array(). This function ensures proper shape.
    """
    arr = np.array(vt_array, dtype=dtype)

    # If already the right shape, return as-is
    if arr.ndim == 2:
        return arr

    # If 1D array of compound types, try to reshape
    if arr.ndim == 1 and len(arr) > 0:
        # Check if elements are iterable (Vec3f, Color3f, etc.)
        try:
            first = arr[0]
            if hasattr(first, "__len__"):
                n_components = len(first)
                return np.array(
                    [[v[i] for i in range(n_components)] for v in arr], dtype=dtype
                )
        except (TypeError, IndexError):
            pass

    return arr


@dataclass
class GaussianData:
    """Gaussian splat data in numpy format (post-activation)."""

    positions: np.ndarray  # (N, 3)
    scales: np.ndarray  # (N, 3) - activated (post-exp)
    rotations: np.ndarray  # (N, 4) quaternions (w, x, y, z)
    opacities: np.ndarray  # (N,) - activated (post-sigmoid)
    colors: np.ndarray  # (N, 3) RGB


SUPPORTED_EXTENSIONS = {".usdz"}


def load_splat(filepath: str) -> GaussianData:
    """Load Gaussian splat data from a file.

    Automatically dispatches to the appropriate loader based on file extension.

    Args:
        filepath: Path to the splat file.

    Returns:
        GaussianData with the loaded splat (all values post-activation).

    Raises:
        ValueError: If the file format is not supported.
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext == ".usdz":
        return _load_usdz(filepath)
    else:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file format: {ext}. Supported: {supported}")


def _load_usdz(filepath: str) -> GaussianData:
    """Load Gaussian splat data from a USDZ file.

    USDZ format stores pre-activation values (matching NVIDIA 3dgrut):
        - scales: log-scale -> apply exp
        - densities: logit -> apply sigmoid
    """
    try:
        from pxr import Usd
    except ImportError as e:
        raise ImportError(
            "`from pxr import Usd` failed. Upgrade your Blender version to 4.0+,\n"
            "or install usd-core in Blender's Python:\n"
            "  /path/to/blender/python -m pip install usd-core"
        ) from e

    stage = Usd.Stage.Open(filepath)
    prim = stage.GetDefaultPrim()

    # Positions (N, 3) - direct
    positions = _vt_array_to_numpy(prim.GetAttribute("positions").Get())
    print(f"[DEBUG] positions shape: {positions.shape}")

    # Scales (N, 3) - stored as log-scale, apply exp
    scales_raw = _vt_array_to_numpy(prim.GetAttribute("scales").Get())
    scales = np.exp(scales_raw)
    print(f"[DEBUG] scales shape: {scales.shape}")

    # Rotations (N, 4) - quaternions (w, x, y, z), direct
    rotations = _vt_array_to_numpy(prim.GetAttribute("rotations").Get())
    print(f"[DEBUG] rotations shape: {rotations.shape}")

    # Densities (N,) - stored as logit, apply sigmoid
    densities_raw = np.array(prim.GetAttribute("densities").Get(), dtype=np.float32)
    opacities = _sigmoid(densities_raw)
    print(f"[DEBUG] opacities shape: {opacities.shape}")

    # Colors (N, 3) - RGB, direct
    colors = _vt_array_to_numpy(prim.GetAttribute("features_albedo").Get())
    print(f"[DEBUG] colors shape: {colors.shape}")
    if colors.ndim == 2 and colors.shape[0] > 0:
        print(f"[DEBUG] first color: {colors[0]}, last color: {colors[-1]}")
        print(f"[DEBUG] color range: min={colors.min():.3f}, max={colors.max():.3f}")

    return GaussianData(
        positions=positions,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        colors=colors,
    )


"""Blender addon for importing Gaussian splats.

Inspired by https://github.com/ReshotAI/gaussian-splatting-blender-addon
Simplified to use RGB colors (no spherical harmonics).

Installation:
    Zip the examples/blender_addon folder and install via Edit > Preferences > Add-ons > Install
"""

bl_info = {
    "name": "Gaussian Splatting",
    "author": "World Labs",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "3D Viewport > Sidebar > Gaussian Splatting",
    "description": "Import Gaussian splats",
}

import os
import time

import bpy
import mathutils
import numpy as np


class ImportGaussianSplatting(bpy.types.Operator):
    """Import a Gaussian splat file."""

    bl_idname = "object.import_gaussian_splatting"
    bl_label = "Import Gaussian Splatting"
    bl_description = "Import a Gaussian splat file"
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

        # Apply rotation to match coordinate systems (Y-up to Z-up)
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = (-np.pi / 2, 0, 0)
        obj["gaussian_splatting"] = True

        # Set up material and geometry nodes
        self._setup_material(obj)
        self._setup_geometry_nodes(obj)

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

        # Color (RGB) - use FLOAT_VECTOR like ReshotAI does
        print(f"[DEBUG] colors shape: {data.colors.shape}")
        print(f"[DEBUG] first color RGB: {data.colors[0]}")

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

    def _setup_material(self, obj: bpy.types.Object) -> None:
        """Create material with vertex colors and opacity.

        Material graph:
            Geometry.Normal ───┐
                               ├─► Dot Product ──┐
            Geometry.Incoming ─┘                 ├─► Multiply ──► BSDF.Alpha
            opacity.Fac ─────────────────────────┘

            color.Color ──► Gamma(2.2) ──► BSDF.Emission
        """
        mat = bpy.data.materials.new(name="GaussianSplatMaterial")
        mat.use_nodes = True
        mat.blend_method = "HASHED"

        tree = mat.node_tree
        tree.nodes.clear()

        # === Geometry node (for Normal) ===
        geometry_node = tree.nodes.new("ShaderNodeNewGeometry")
        geometry_node.location = (-600, 200)

        # === Opacity attribute ===
        opacity_attr = tree.nodes.new("ShaderNodeAttribute")
        opacity_attr.location = (-600, 0)
        opacity_attr.attribute_name = "opacity"
        opacity_attr.attribute_type = "GEOMETRY"

        # === Dot Product: Normal · Incoming ===
        dot_product = tree.nodes.new("ShaderNodeVectorMath")
        dot_product.location = (-350, 100)
        dot_product.operation = "DOT_PRODUCT"
        tree.links.new(geometry_node.outputs["Normal"], dot_product.inputs[0])
        tree.links.new(geometry_node.outputs["Incoming"], dot_product.inputs[1])

        # === Multiply: Dot Product × opacity.Fac ===
        multiply = tree.nodes.new("ShaderNodeMath")
        multiply.location = (-100, 50)
        multiply.operation = "MULTIPLY"
        tree.links.new(dot_product.outputs["Value"], multiply.inputs[0])
        tree.links.new(opacity_attr.outputs["Fac"], multiply.inputs[1])

        # === Color attribute ===
        color_attr = tree.nodes.new("ShaderNodeAttribute")
        color_attr.location = (-600, -250)
        color_attr.attribute_name = "color"
        color_attr.attribute_type = "GEOMETRY"

        # === Gamma correction (2.2) ===
        gamma = tree.nodes.new("ShaderNodeGamma")
        gamma.location = (-350, -250)
        gamma.inputs["Gamma"].default_value = 2.2
        tree.links.new(color_attr.outputs["Color"], gamma.inputs["Color"])

        # === Principled BSDF ===
        bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (150, 0)
        # Base Color: black (emission-only)
        bsdf.inputs["Base Color"].default_value = (0, 0, 0, 1)
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 0.0
        # Emission from gamma-corrected color
        tree.links.new(gamma.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = 1.0
        # Alpha from dot product × opacity
        tree.links.new(multiply.outputs["Value"], bsdf.inputs["Alpha"])

        # === Output ===
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        output.location = (450, 0)
        tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

        obj.data.materials.append(mat)

    def _create_splat_limit_node_group(self) -> bpy.types.NodeTree:
        """Create a custom node group for limiting splat count.

        Filtering logic:
        1. Delete splats with opacity < threshold
        2. Sort remaining by max scale dimension (descending)
        3. Keep only the top N splats

        Inputs:
            - Geometry: Input geometry with opacity/scale attributes
            - Max Count: Maximum number of splats to keep (default 200000)
            - Opacity Threshold: Minimum opacity to keep (default 0.1)

        Output:
            - Geometry: Filtered geometry
        """
        # Check if already exists
        if "SplatLimitFilter" in bpy.data.node_groups:
            return bpy.data.node_groups["SplatLimitFilter"]

        tree = bpy.data.node_groups.new(
            name="SplatLimitFilter", type="GeometryNodeTree"
        )

        # === Interface ===
        tree.interface.new_socket(
            name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
        )
        max_count_socket = tree.interface.new_socket(
            name="Max Count", in_out="INPUT", socket_type="NodeSocketInt"
        )
        max_count_socket.default_value = 200000
        max_count_socket.min_value = 1

        opacity_thresh_socket = tree.interface.new_socket(
            name="Opacity Threshold", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        opacity_thresh_socket.default_value = 0.1
        opacity_thresh_socket.min_value = 0.0
        opacity_thresh_socket.max_value = 1.0

        tree.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # === Nodes ===
        input_node = tree.nodes.new("NodeGroupInput")
        input_node.location = (-1200, 0)

        output_node = tree.nodes.new("NodeGroupOutput")
        output_node.location = (800, 0)

        # --- Step 1: Delete by opacity threshold ---
        opacity_attr = tree.nodes.new("GeometryNodeInputNamedAttribute")
        opacity_attr.location = (-1000, -150)
        opacity_attr.data_type = "FLOAT"
        opacity_attr.inputs["Name"].default_value = "opacity"

        opacity_compare = tree.nodes.new("FunctionNodeCompare")
        opacity_compare.location = (-800, -100)
        opacity_compare.data_type = "FLOAT"
        opacity_compare.operation = "GREATER_EQUAL"
        tree.links.new(opacity_attr.outputs["Attribute"], opacity_compare.inputs["A"])
        tree.links.new(
            input_node.outputs["Opacity Threshold"], opacity_compare.inputs["B"]
        )

        delete_by_opacity = tree.nodes.new("GeometryNodeDeleteGeometry")
        delete_by_opacity.location = (-600, 0)
        delete_by_opacity.domain = "POINT"
        delete_by_opacity.mode = "ALL"
        tree.links.new(
            input_node.outputs["Geometry"], delete_by_opacity.inputs["Geometry"]
        )
        # Delete where NOT (opacity >= threshold), i.e., delete where opacity < threshold
        # So we need to invert the selection
        invert_opacity = tree.nodes.new("FunctionNodeBooleanMath")
        invert_opacity.location = (-700, -100)
        invert_opacity.operation = "NOT"
        tree.links.new(opacity_compare.outputs["Result"], invert_opacity.inputs[0])
        tree.links.new(
            invert_opacity.outputs["Boolean"], delete_by_opacity.inputs["Selection"]
        )

        # --- Step 2: Compute max scale per point ---
        scale_attr = tree.nodes.new("GeometryNodeInputNamedAttribute")
        scale_attr.location = (-400, -200)
        scale_attr.data_type = "FLOAT_VECTOR"
        scale_attr.inputs["Name"].default_value = "scale"

        separate_xyz = tree.nodes.new("ShaderNodeSeparateXYZ")
        separate_xyz.location = (-200, -200)
        tree.links.new(scale_attr.outputs["Attribute"], separate_xyz.inputs["Vector"])

        max_xy = tree.nodes.new("ShaderNodeMath")
        max_xy.location = (0, -150)
        max_xy.operation = "MAXIMUM"
        tree.links.new(separate_xyz.outputs["X"], max_xy.inputs[0])
        tree.links.new(separate_xyz.outputs["Y"], max_xy.inputs[1])

        max_xyz = tree.nodes.new("ShaderNodeMath")
        max_xyz.location = (150, -150)
        max_xyz.operation = "MAXIMUM"
        tree.links.new(max_xy.outputs["Value"], max_xyz.inputs[0])
        tree.links.new(separate_xyz.outputs["Z"], max_xyz.inputs[1])

        # --- Step 3: Sort by max scale (descending = negate then sort ascending) ---
        negate_scale = tree.nodes.new("ShaderNodeMath")
        negate_scale.location = (300, -150)
        negate_scale.operation = "MULTIPLY"
        negate_scale.inputs[1].default_value = -1.0
        tree.links.new(max_xyz.outputs["Value"], negate_scale.inputs[0])

        sort_elements = tree.nodes.new("GeometryNodeSortElements")
        sort_elements.location = (-200, 0)
        sort_elements.domain = "POINT"
        tree.links.new(
            delete_by_opacity.outputs["Geometry"], sort_elements.inputs["Geometry"]
        )
        tree.links.new(
            negate_scale.outputs["Value"], sort_elements.inputs["Sort Weight"]
        )

        # --- Step 4: Delete where index >= max_count ---
        index_node = tree.nodes.new("GeometryNodeInputIndex")
        index_node.location = (200, 100)

        index_compare = tree.nodes.new("FunctionNodeCompare")
        index_compare.location = (400, 50)
        index_compare.data_type = "INT"
        index_compare.operation = "GREATER_EQUAL"
        tree.links.new(index_node.outputs["Index"], index_compare.inputs["A"])
        tree.links.new(input_node.outputs["Max Count"], index_compare.inputs["B"])

        delete_by_count = tree.nodes.new("GeometryNodeDeleteGeometry")
        delete_by_count.location = (600, 0)
        delete_by_count.domain = "POINT"
        delete_by_count.mode = "ALL"
        tree.links.new(
            sort_elements.outputs["Geometry"], delete_by_count.inputs["Geometry"]
        )
        tree.links.new(
            index_compare.outputs["Result"], delete_by_count.inputs["Selection"]
        )

        tree.links.new(
            delete_by_count.outputs["Geometry"], output_node.inputs["Geometry"]
        )

        return tree

    def _setup_geometry_nodes(self, obj: bpy.types.Object) -> None:
        """Create geometry nodes with point cloud / ellipsoid toggle.

        Uses Set Material node to apply material in geometry nodes
        (like ReshotAI does).
        """
        # Get the material we created
        mat = obj.data.materials[0] if obj.data.materials else None

        geo_tree = bpy.data.node_groups.new(
            name="GaussianSplatGeometry", type="GeometryNodeTree"
        )

        mod = obj.modifiers.new(name="Geometry Nodes", type="NODES")
        mod.node_group = geo_tree

        # === Interface sockets ===
        geo_tree.interface.new_socket(
            name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
        )
        # Expose splat limit parameters
        max_count_socket = geo_tree.interface.new_socket(
            name="Max Splat Count", in_out="INPUT", socket_type="NodeSocketInt"
        )
        max_count_socket.default_value = 200000
        max_count_socket.min_value = 1

        opacity_thresh_socket = geo_tree.interface.new_socket(
            name="Opacity Threshold", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        opacity_thresh_socket.default_value = 0.1
        opacity_thresh_socket.min_value = 0.0
        opacity_thresh_socket.max_value = 1.0

        scale_mult_socket = geo_tree.interface.new_socket(
            name="Scale Multiplier", in_out="INPUT", socket_type="NodeSocketFloat"
        )
        scale_mult_socket.default_value = 1.0
        scale_mult_socket.min_value = 1.0
        scale_mult_socket.max_value = 10.0

        geo_tree.interface.new_socket(
            name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
        )

        # Set defaults on the modifier explicitly (interface defaults don't always apply)
        for item in geo_tree.interface.items_tree:
            if item.item_type == "SOCKET" and item.in_out == "INPUT":
                if item.name == "Max Splat Count":
                    mod[item.identifier] = 200000
                elif item.name == "Opacity Threshold":
                    mod[item.identifier] = 0.1
                elif item.name == "Scale Multiplier":
                    mod[item.identifier] = 1.0

        # =====================================================
        # NODE LAYOUT (matching screenshot):
        # Left column: Input, Splat Filter, Multiply
        # Top row: Ellipsoid branch (Mesh to Points, Ico, Shade Smooth, Instance, Realize)
        # Bottom row: Point cloud branch (scale processing, Mesh to Points)
        # Right column: Boolean, Switch, Set Material, Output
        # =====================================================

        # === LEFT COLUMN ===
        input_node = geo_tree.nodes.new("NodeGroupInput")
        input_node.location = (-800, 0)

        splat_limit_group = self._create_splat_limit_node_group()
        splat_limit_node = geo_tree.nodes.new("GeometryNodeGroup")
        splat_limit_node.location = (-600, 0)
        splat_limit_node.node_tree = splat_limit_group
        splat_limit_node.label = "Splat Limit Filter"

        geo_tree.links.new(
            input_node.outputs["Geometry"], splat_limit_node.inputs["Geometry"]
        )
        geo_tree.links.new(
            input_node.outputs["Max Splat Count"], splat_limit_node.inputs["Max Count"]
        )
        geo_tree.links.new(
            input_node.outputs["Opacity Threshold"],
            splat_limit_node.inputs["Opacity Threshold"],
        )

        # Scale Multiplier ×2
        scale_mult_x2 = geo_tree.nodes.new("ShaderNodeMath")
        scale_mult_x2.location = (-600, -150)
        scale_mult_x2.operation = "MULTIPLY"
        scale_mult_x2.inputs[1].default_value = 2.0
        geo_tree.links.new(
            input_node.outputs["Scale Multiplier"], scale_mult_x2.inputs[0]
        )

        # === RIGHT COLUMN ===
        output_node = geo_tree.nodes.new("NodeGroupOutput")
        output_node.location = (1400, 100)

        set_material = geo_tree.nodes.new("GeometryNodeSetMaterial")
        set_material.location = (1200, 100)
        if mat:
            set_material.inputs["Material"].default_value = mat

        point_cloud_switch = geo_tree.nodes.new("GeometryNodeSwitch")
        point_cloud_switch.location = (1000, 100)
        point_cloud_switch.input_type = "GEOMETRY"
        point_cloud_switch.label = "Point Cloud Mode"

        bool_node = geo_tree.nodes.new("FunctionNodeInputBool")
        bool_node.location = (1000, 250)
        bool_node.boolean = True  # Default: point cloud mode
        bool_node.label = "Point Cloud (faster)"

        geo_tree.links.new(
            bool_node.outputs["Boolean"], point_cloud_switch.inputs["Switch"]
        )
        geo_tree.links.new(
            point_cloud_switch.outputs["Output"], set_material.inputs["Geometry"]
        )
        geo_tree.links.new(
            set_material.outputs["Geometry"], output_node.inputs["Geometry"]
        )

        # === TOP ROW: ELLIPSOID BRANCH ===
        # Mesh to Points (for ellipsoid instancing)
        mesh_to_points_inst = geo_tree.nodes.new("GeometryNodeMeshToPoints")
        mesh_to_points_inst.location = (-400, 200)
        geo_tree.links.new(
            splat_limit_node.outputs["Geometry"], mesh_to_points_inst.inputs["Mesh"]
        )

        # Ico Sphere
        ico_sphere = geo_tree.nodes.new("GeometryNodeMeshIcoSphere")
        ico_sphere.location = (-400, 50)
        ico_sphere.inputs["Radius"].default_value = 1.0
        ico_sphere.inputs["Subdivisions"].default_value = 2

        # Set Shade Smooth
        set_shade_smooth = geo_tree.nodes.new("GeometryNodeSetShadeSmooth")
        set_shade_smooth.location = (-200, 50)
        geo_tree.links.new(
            ico_sphere.outputs["Mesh"], set_shade_smooth.inputs["Geometry"]
        )

        # Named Attribute: rotation_euler
        rot_attr = geo_tree.nodes.new("GeometryNodeInputNamedAttribute")
        rot_attr.location = (200, 200)
        rot_attr.data_type = "FLOAT_VECTOR"
        rot_attr.inputs["Name"].default_value = "rotation_euler"

        # Named Attribute: scale (for ellipsoid)
        scale_attr = geo_tree.nodes.new("GeometryNodeInputNamedAttribute")
        scale_attr.location = (200, 50)
        scale_attr.data_type = "FLOAT_VECTOR"
        scale_attr.inputs["Name"].default_value = "scale"

        # Scale (vector math - multiply by scale multiplier)
        scale_mult_inst = geo_tree.nodes.new("ShaderNodeVectorMath")
        scale_mult_inst.location = (400, 50)
        scale_mult_inst.operation = "SCALE"
        geo_tree.links.new(scale_attr.outputs["Attribute"], scale_mult_inst.inputs[0])
        geo_tree.links.new(
            scale_mult_x2.outputs["Value"], scale_mult_inst.inputs["Scale"]
        )

        # Instance on Points
        instance_node = geo_tree.nodes.new("GeometryNodeInstanceOnPoints")
        instance_node.location = (600, 150)
        geo_tree.links.new(
            mesh_to_points_inst.outputs["Points"], instance_node.inputs["Points"]
        )
        geo_tree.links.new(
            set_shade_smooth.outputs["Geometry"], instance_node.inputs["Instance"]
        )
        geo_tree.links.new(
            rot_attr.outputs["Attribute"], instance_node.inputs["Rotation"]
        )
        geo_tree.links.new(
            scale_mult_inst.outputs["Vector"], instance_node.inputs["Scale"]
        )

        # Realize Instances
        realize_instances = geo_tree.nodes.new("GeometryNodeRealizeInstances")
        realize_instances.location = (800, 150)
        geo_tree.links.new(
            instance_node.outputs["Instances"], realize_instances.inputs["Geometry"]
        )

        # Connect to switch (False = ellipsoid mode)
        geo_tree.links.new(
            realize_instances.outputs["Geometry"], point_cloud_switch.inputs["False"]
        )

        # === BOTTOM ROW: POINT CLOUD BRANCH ===
        # Named Attribute: scale (for point cloud)
        scale_attr_pc = geo_tree.nodes.new("GeometryNodeInputNamedAttribute")
        scale_attr_pc.location = (-200, -300)
        scale_attr_pc.data_type = "FLOAT_VECTOR"
        scale_attr_pc.inputs["Name"].default_value = "scale"

        # Separate XYZ
        separate_xyz = geo_tree.nodes.new("ShaderNodeSeparateXYZ")
        separate_xyz.location = (0, -300)
        geo_tree.links.new(
            scale_attr_pc.outputs["Attribute"], separate_xyz.inputs["Vector"]
        )

        # Maximum (X, Y)
        max_xy = geo_tree.nodes.new("ShaderNodeMath")
        max_xy.location = (200, -250)
        max_xy.operation = "MAXIMUM"
        geo_tree.links.new(separate_xyz.outputs["X"], max_xy.inputs[0])
        geo_tree.links.new(separate_xyz.outputs["Y"], max_xy.inputs[1])

        # Maximum (XY, Z)
        max_xyz = geo_tree.nodes.new("ShaderNodeMath")
        max_xyz.location = (400, -250)
        max_xyz.operation = "MAXIMUM"
        geo_tree.links.new(max_xy.outputs["Value"], max_xyz.inputs[0])
        geo_tree.links.new(separate_xyz.outputs["Z"], max_xyz.inputs[1])

        # Multiply by scale multiplier
        scale_mult_pc = geo_tree.nodes.new("ShaderNodeMath")
        scale_mult_pc.location = (600, -250)
        scale_mult_pc.operation = "MULTIPLY"
        geo_tree.links.new(max_xyz.outputs["Value"], scale_mult_pc.inputs[0])
        geo_tree.links.new(scale_mult_x2.outputs["Value"], scale_mult_pc.inputs[1])

        # Mesh to Points (for point cloud)
        mesh_to_points = geo_tree.nodes.new("GeometryNodeMeshToPoints")
        mesh_to_points.location = (800, -300)
        geo_tree.links.new(
            splat_limit_node.outputs["Geometry"], mesh_to_points.inputs["Mesh"]
        )
        geo_tree.links.new(
            scale_mult_pc.outputs["Value"], mesh_to_points.inputs["Radius"]
        )

        # Connect to switch (True = point cloud mode)
        geo_tree.links.new(
            mesh_to_points.outputs["Points"], point_cloud_switch.inputs["True"]
        )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class GaussianSplattingPanel(bpy.types.Panel):
    """Panel for Gaussian Splatting tools."""

    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_idname = "VIEW3D_PT_gaussian_splatting"
    bl_category = "Gaussian Splatting"
    bl_label = "Gaussian Splatting"

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

        # Show defaults for initial import
        layout.separator()
        box = layout.box()
        box.label(text="Import Defaults:", icon="INFO")
        box.label(text="  Max Splats: 200,000")
        box.label(text="  Opacity Threshold: 0.1")
        box.label(text="  Scale Multiplier: 1.0")

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


classes = [
    ImportGaussianSplatting,
    GaussianSplattingPanel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
