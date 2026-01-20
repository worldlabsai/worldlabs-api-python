"""Blender node graph builders for Gaussian splat visualization.

This module provides functions to create:
- Material shader graphs for splat rendering
- Geometry node graphs for splat instancing and point cloud modes
"""

from __future__ import annotations

import bpy


def setup_material(obj: bpy.types.Object) -> None:
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

    # === Multiply: Dot Product × opacity.Fac (clamped to 0-1) ===
    multiply = tree.nodes.new("ShaderNodeMath")
    multiply.location = (-100, 50)
    multiply.operation = "MULTIPLY"
    multiply.use_clamp = True
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
    bsdf.inputs["Roughness"].default_value = 1.0
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


def _create_splat_limit_node_group() -> bpy.types.NodeTree:
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

    tree = bpy.data.node_groups.new(name="SplatLimitFilter", type="GeometryNodeTree")

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
    tree.links.new(input_node.outputs["Opacity Threshold"], opacity_compare.inputs["B"])

    delete_by_opacity = tree.nodes.new("GeometryNodeDeleteGeometry")
    delete_by_opacity.location = (-600, 0)
    delete_by_opacity.domain = "POINT"
    delete_by_opacity.mode = "ALL"
    tree.links.new(input_node.outputs["Geometry"], delete_by_opacity.inputs["Geometry"])
    # Delete where NOT (opacity >= threshold), i.e., delete where opacity < threshold
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
    tree.links.new(negate_scale.outputs["Value"], sort_elements.inputs["Sort Weight"])

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
    tree.links.new(index_compare.outputs["Result"], delete_by_count.inputs["Selection"])

    tree.links.new(delete_by_count.outputs["Geometry"], output_node.inputs["Geometry"])

    return tree


def setup_geometry_nodes(
    obj: bpy.types.Object,
    max_splat_count: int = 500000,
    opacity_threshold: float = 0.1,
    scale_multiplier: float = 1.0,
    point_cloud_mode: bool = True,
) -> None:
    """Create geometry nodes with point cloud / ellipsoid toggle.

    Args:
        obj: The Blender object to add geometry nodes to.
        max_splat_count: Maximum number of splats to display.
        opacity_threshold: Minimum opacity to display.
        scale_multiplier: Scale multiplier for splat size.
        point_cloud_mode: If True, use point cloud mode (faster).

    Node layout:
        Left column: Input, Splat Filter, Multiply
        Top row: Ellipsoid branch (Mesh to Points, Ico, Shade Smooth, Instance, Realize)
        Bottom row: Point cloud branch (scale processing, Mesh to Points)
        Right column: Boolean, Switch, Set Material, Output
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
    max_count_socket.default_value = max_splat_count
    max_count_socket.min_value = 1

    opacity_thresh_socket = geo_tree.interface.new_socket(
        name="Opacity Threshold", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    opacity_thresh_socket.default_value = opacity_threshold
    opacity_thresh_socket.min_value = 0.0
    opacity_thresh_socket.max_value = 1.0

    scale_mult_socket = geo_tree.interface.new_socket(
        name="Scale Multiplier", in_out="INPUT", socket_type="NodeSocketFloat"
    )
    scale_mult_socket.default_value = scale_multiplier
    scale_mult_socket.min_value = 1.0
    scale_mult_socket.max_value = 10.0

    geo_tree.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )

    # Set defaults on the modifier explicitly (interface defaults don't always apply)
    for item in geo_tree.interface.items_tree:
        if item.item_type == "SOCKET" and item.in_out == "INPUT":
            if item.name == "Max Splat Count":
                mod[item.identifier] = max_splat_count
            elif item.name == "Opacity Threshold":
                mod[item.identifier] = opacity_threshold
            elif item.name == "Scale Multiplier":
                mod[item.identifier] = scale_multiplier

    # === LEFT COLUMN ===
    input_node = geo_tree.nodes.new("NodeGroupInput")
    input_node.location = (-800, 0)

    splat_limit_group = _create_splat_limit_node_group()
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
    geo_tree.links.new(input_node.outputs["Scale Multiplier"], scale_mult_x2.inputs[0])

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
    bool_node.boolean = point_cloud_mode
    bool_node.label = "Point Cloud (faster)"

    geo_tree.links.new(
        bool_node.outputs["Boolean"], point_cloud_switch.inputs["Switch"]
    )
    geo_tree.links.new(
        point_cloud_switch.outputs["Output"], set_material.inputs["Geometry"]
    )
    geo_tree.links.new(set_material.outputs["Geometry"], output_node.inputs["Geometry"])

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
    geo_tree.links.new(ico_sphere.outputs["Mesh"], set_shade_smooth.inputs["Geometry"])

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
    geo_tree.links.new(scale_mult_x2.outputs["Value"], scale_mult_inst.inputs["Scale"])

    # Instance on Points
    instance_node = geo_tree.nodes.new("GeometryNodeInstanceOnPoints")
    instance_node.location = (600, 150)
    geo_tree.links.new(
        mesh_to_points_inst.outputs["Points"], instance_node.inputs["Points"]
    )
    geo_tree.links.new(
        set_shade_smooth.outputs["Geometry"], instance_node.inputs["Instance"]
    )
    geo_tree.links.new(rot_attr.outputs["Attribute"], instance_node.inputs["Rotation"])
    geo_tree.links.new(scale_mult_inst.outputs["Vector"], instance_node.inputs["Scale"])

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
    geo_tree.links.new(scale_mult_pc.outputs["Value"], mesh_to_points.inputs["Radius"])

    # Connect to switch (True = point cloud mode)
    geo_tree.links.new(
        mesh_to_points.outputs["Points"], point_cloud_switch.inputs["True"]
    )
