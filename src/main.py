import cmd_manager
import cmd_volume
import gfx_mpr
import gfx_manager
import gfx_shaders
import gfx_utils
import image_dicom
import image_manager
import image_utils
from tool_common import *
import tool_manager

import glfw
import OpenGL.GL as gl
import numpy as np
import glm
import imgui
from imgui.integrations.glfw import GlfwRenderer
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox

import os
import copy
import time


class Settings:
    def __init__(self):
        self.bg_color1 = [0.9, 0.9, 0.9]
        self.bg_color2 = [0.1, 0.1, 0.1]
        self.show_mask = True
        self.fov_degrees = 45.0
        self.projection_mode = 1  # 0=orthographic; 1=perspective
        self.show_stats = False
        self.show_navigator = True
        self.show_input_guide = False
        self.dark_mode = True
        self.basepath = ""


class Context:
    def __init__(self):
        self.settings = Settings()

        # Managers for resources and undo stack
        self.gfx = gfx_manager.GfxManager()
        self.images = image_manager.ImageManager()
        self.tools = tool_manager.ToolManager()
        self.cmds = cmd_manager.CmdManager()

        # Other utilities for interaction, etc.
        self.trackball = gfx_utils.Trackball()
        self.panning = gfx_utils.Panning()
        self.mpr = gfx_mpr.MPR()


def do_initialize(ctx):
    """Initialize the application state"""
    ctx.images.labels = ["Label 255", "Label 0 (Clear)"]  # TODO
    ctx.images.load_volume_fromfile("")  # Create an empty volume and mask
    ctx.mpr.update_minmax_range_from_volume(ctx.images.volume)

    # TODO Placeholder images for XYZ views shown in the navigator
    axial_view = np.array([0, 0, 0], dtype=np.uint8).reshape((1, 1, 3))
    coronal_view = np.array([0, 0, 0], dtype=np.uint8).reshape((1, 1, 3))
    sagital_view = np.array([0, 0, 0], dtype=np.uint8).reshape((1, 1, 3))

    ctx.gfx.programs["raycast"] = gfx_utils.create_program(
        (gfx_shaders.raycast_vs, gl.GL_VERTEX_SHADER),
        (gfx_shaders.raycast_fs, gl.GL_FRAGMENT_SHADER),
    )
    ctx.gfx.programs["polygon"] = gfx_utils.create_program(
        (gfx_shaders.polygon_vs, gl.GL_VERTEX_SHADER),
        (gfx_shaders.polygon_fs, gl.GL_FRAGMENT_SHADER),
    )
    ctx.gfx.programs["background"] = gfx_utils.create_program(
        (gfx_shaders.background_vs, gl.GL_VERTEX_SHADER),
        (gfx_shaders.background_fs, gl.GL_FRAGMENT_SHADER),
    )

    ctx.gfx.vaos["empty"] = gl.glGenVertexArrays(1)
    ctx.gfx.buffers["polygon"] = gfx_utils.create_mesh_buffer([])
    ctx.gfx.vaos["polygon"] = gfx_utils.create_mesh_vao([], ctx.gfx.buffers["polygon"])
    ctx.gfx.buffers["markers"] = gfx_utils.create_mesh_buffer([])
    ctx.gfx.vaos["markers"] = gfx_utils.create_mesh_vao([], ctx.gfx.buffers["markers"])

    ctx.gfx.textures["volume"] = gfx_utils.create_texture_3d(ctx.images.volume)
    ctx.gfx.textures["mask"] = gfx_utils.create_texture_3d(ctx.images.mask)
    ctx.gfx.textures["axial"] = gfx_utils.create_texture_2d(axial_view)
    ctx.gfx.textures["sagital"] = gfx_utils.create_texture_2d(sagital_view)
    ctx.gfx.textures["coronal"] = gfx_utils.create_texture_2d(coronal_view)


def do_rendering(ctx):
    """Do rendering"""
    tools = ctx.tools
    tool_op = TOOL_OP_ADD if ctx.images.active_label == 0 else TOOL_OP_SUBTRACT

    mpr_planes = ctx.mpr.get_snapped_planes(ctx.images.volume)
    level_range = ctx.mpr.level_range
    level_range_scaled = level_range  # Adjusted for normalized texture formats
    if ctx.images.volume.dtype == np.uint8:
        level_range_scaled = [v / 255.0 for v in level_range]  # Scale to range [0,1]
    if ctx.images.volume.dtype == np.int16:
        level_range_scaled = [v / 32767.0 for v in level_range]  # Scale to range [-1,1]
    filter_mode = gl.GL_NEAREST if ctx.mpr.show_voxels else gl.GL_LINEAR

    aspect = float(ctx.gfx.width) / ctx.gfx.height
    proj = glm.perspective(glm.radians(ctx.settings.fov_degrees), aspect, 0.1, 10.0)
    if ctx.settings.projection_mode:
        k = glm.radians(ctx.settings.fov_degrees)
        proj = glm.ortho(-k * aspect, k * aspect, -k, k, 0.1, 10.0)
    view = (
        glm.translate(glm.mat4(1.0), glm.vec3(0, 0, -2))
        * glm.translate(glm.mat4(1.0), -ctx.panning.position)
        * glm.mat4_cast(ctx.trackball.quat)
    )
    header = ctx.images.header
    spacing = glm.vec3(header["spacing"])
    extent = glm.vec3(header["spacing"]) * glm.vec3(header["dimensions"])
    model = glm.scale(glm.mat4(1.0), extent / glm.max(extent.x, glm.max(extent.y, extent.z)))
    mv = view * model
    mvp = proj * view * model

    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    program = ctx.gfx.programs["background"]
    gl.glUseProgram(program)
    gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glDepthMask(gl.GL_FALSE)
    gl.glViewport(0, 0, ctx.gfx.width + ctx.gfx.sidebar_width, ctx.gfx.height)
    gl.glUniform3f(gl.glGetUniformLocation(program, "u_bg_color1"), *ctx.settings.bg_color1)
    gl.glUniform3f(gl.glGetUniformLocation(program, "u_bg_color2"), *ctx.settings.bg_color2)
    gl.glBindVertexArray(ctx.gfx.vaos["empty"])
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
    gl.glViewport(0, 0, ctx.gfx.width, ctx.gfx.height)
    gl.glDepthMask(gl.GL_TRUE)

    if max(ctx.images.volume.shape) > 1:
        program = ctx.gfx.programs["raycast"]
        gl.glUseProgram(program)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_3D, ctx.gfx.textures["mask"])
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MIN_FILTER, filter_mode)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAG_FILTER, filter_mode)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_3D, ctx.gfx.textures["volume"])
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MIN_FILTER, filter_mode)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAG_FILTER, filter_mode)
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(program, "u_mvp"), 1, False, glm.value_ptr(mvp)
        )
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(program, "u_mv"), 1, False, glm.value_ptr(mv))
        gl.glUniform1i(gl.glGetUniformLocation(program, "u_label"), 0)
        gl.glUniform1i(gl.glGetUniformLocation(program, "u_show_mask"), ctx.settings.show_mask)
        gl.glUniform1i(
            gl.glGetUniformLocation(program, "u_projection_mode"), ctx.settings.projection_mode
        )
        gl.glUniform1i(gl.glGetUniformLocation(program, "u_show_mpr"), ctx.mpr.enabled)
        gl.glUniform3f(gl.glGetUniformLocation(program, "u_mpr_planes"), *mpr_planes)
        gl.glUniform2f(gl.glGetUniformLocation(program, "u_level_range"), *level_range_scaled)
        gl.glUniform3f(gl.glGetUniformLocation(program, "u_extent"), *extent)
        gl.glUniform4f(gl.glGetUniformLocation(program, "u_brush"), *tools.brush.position)
        if tools.smartbrush.enabled:
            gl.glUniform4f(gl.glGetUniformLocation(program, "u_brush"), *tools.smartbrush.position)
        gl.glUniform1i(gl.glGetUniformLocation(program, "u_volume"), 0)
        gl.glUniform1i(gl.glGetUniformLocation(program, "u_mask"), 1)
        gl.glBindVertexArray(ctx.gfx.vaos["empty"])
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 14)
        gl.glBindVertexArray(0)

    if tools.polygon.enabled:
        program = ctx.gfx.programs["polygon"]
        gl.glUseProgram(program)
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(program, "u_mvp"), 1, False, glm.value_ptr(mvp)
        )
        gl.glUniform1i(gl.glGetUniformLocation(program, "u_label"), 0)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glPointSize(5.0)
        gl.glBindVertexArray(ctx.gfx.vaos["polygon"])
        gl.glDrawArrays(gl.GL_LINE_STRIP, 0, len(tools.polygon.points) // 3)
        gl.glBindVertexArray(ctx.gfx.vaos["markers"])
        gl.glDrawArrays(gl.GL_POINTS, 0, len(tools.polygon.points) // 3)
        gl.glBindVertexArray(0)
        gl.glPointSize(1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)

    if tools.livewire.enabled:
        program = ctx.gfx.programs["polygon"]
        gl.glUseProgram(program)
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(program, "u_mvp"), 1, False, glm.value_ptr(mvp)
        )
        gl.glUniform1i(gl.glGetUniformLocation(program, "u_label"), 0)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glPointSize(5.0)
        gl.glBindVertexArray(ctx.gfx.vaos["polygon"])
        gl.glDrawArrays(gl.GL_LINE_STRIP, 0, len(tools.livewire.points) // 3)
        gl.glBindVertexArray(ctx.gfx.vaos["markers"])
        gl.glDrawArrays(gl.GL_POINTS, 0, len(tools.livewire.markers) // 3)
        gl.glBindVertexArray(0)
        gl.glEnable(gl.GL_DEPTH_TEST)

    if ctx.mpr.enabled:
        x, y = glfw.get_cursor_pos(ctx.gfx.window)
        w, h = ctx.gfx.width, ctx.gfx.height
        depth = gl.glReadPixels(x, (h - 1) - y, 1, 1, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
        ndc_pos = glm.vec3(0.0)
        ndc_pos.x = (float(x) / w) * 2.0 - 1.0
        ndc_pos.y = -((float(y) / h) * 2.0 - 1.0)
        ndc_pos.z = depth[0][0] * 2.0 - 1.0
        view_pos = gfx_utils.reconstruct_view_pos(ndc_pos, proj)
        texcoord = glm.vec3(glm.inverse(mv) * glm.vec4(view_pos, 1.0)) + 0.5001

        tools.brush.position = glm.vec4(texcoord - 0.5, tools.brush.position.w)
        tools.smartbrush.position = glm.vec4(texcoord - 0.5, tools.smartbrush.position.w)

        if depth != 1.0 and tools.brush.painting and tools.brush.enabled:
            brush = tools.brush
            if brush.frame_count == 0:
                # Store full copy of current segmentation for undo
                mask_copy = np.copy(ctx.images.mask)
                cmd = cmd_volume.UpdateVolumeCmd(
                    ctx.images.mask, mask_copy, (0, 0, 0), ctx.gfx.textures["mask"]
                )
                ctx.cmds.push_apply(cmd)
            brush.frame_count += 1
            result = brush.apply(ctx.images.mask, texcoord, spacing, tool_op)
            if result:
                gfx_utils.update_subtexture_3d(ctx.gfx.textures["mask"], result[0], result[1])

        if depth != 1.0 and tools.smartbrush.painting and tools.smartbrush.enabled:
            smartbrush = tools.smartbrush
            if smartbrush.frame_count == 0:
                # Store full copy of current segmentation for undo
                mask_copy = np.copy(ctx.images.mask)
                cmd = cmd_volume.UpdateVolumeCmd(
                    ctx.images.mask, mask_copy, (0, 0, 0), ctx.gfx.textures["mask"]
                )
                ctx.cmds.push_apply(cmd)
            if smartbrush.xy[0] != x or smartbrush.xy[1] != y:
                smartbrush.frame_count += 1
                smartbrush.momentum = min(5, smartbrush.momentum + 2)
                smartbrush.xy = (x, y)
            if smartbrush.momentum > 0:
                result = smartbrush.apply(
                    ctx.images.mask, ctx.images.volume, texcoord, spacing, level_range, tool_op
                )
                if result:
                    gfx_utils.update_subtexture_3d(ctx.gfx.textures["mask"], result[0], result[1])
                smartbrush.momentum = max(0, smartbrush.momentum - 1)

        if depth != 1.0 and tools.polygon.enabled:
            polygon = tools.polygon
            if polygon.clicking:
                # First, check if an existing polygon vertex is close to cursor
                radius = 3.5e-4 * ctx.settings.fov_degrees  # Search radius
                closest = polygon.find_closest(texcoord - 0.5, radius)
                if len(polygon.points) and closest >= 0:
                    # If an existing vertex was found, select it for manipulation
                    polygon.selected = closest
                else:
                    # Otherwise, add a new polygon vertex at the cursor location
                    polygon.selected = len(polygon.points)
                    polygon.points.extend(texcoord - 0.5)
                polygon.clicking = False
            if polygon.selected >= 0:
                # Add a few frames delay to the manipulation
                polygon.frame_count += 1
                if polygon.frame_count > 8:
                    # Move selected polygon vertex to cursor location
                    offset = polygon.selected
                    polygon.points[offset : offset + 3] = (texcoord - 0.5).to_tuple()
                gfx_utils.update_mesh_buffer(ctx.gfx.buffers["polygon"], polygon.points)
                gfx_utils.update_mesh_buffer(ctx.gfx.buffers["markers"], polygon.points)

        if depth != 1.0 and tools.livewire.enabled:
            livewire = tools.livewire
            clicking = False
            if livewire.clicking:
                livewire.markers.extend(texcoord - 0.5)
                gfx_utils.update_mesh_buffer(ctx.gfx.buffers["markers"], livewire.markers)
                livewire.update_graph(ctx.images.volume, texcoord, level_range)
                livewire.clicking = False
                clicking = True
            if len(livewire.path):
                livewire.update_path(ctx.images.volume, texcoord, level_range, clicking)
                # Also update polyline for preview
                gfx_utils.update_mesh_buffer(ctx.gfx.buffers["polygon"], livewire.points)


def do_update(ctx):
    """Update application state"""
    tools = ctx.tools
    tool_op = TOOL_OP_ADD if ctx.images.active_label == 0 else TOOL_OP_SUBTRACT

    tools.brush.position.w = 0.0 if not tools.brush.enabled else tools.brush.size * 0.5
    tools.smartbrush.position.w = (
        0.0 if not tools.smartbrush.enabled else tools.smartbrush.size * 0.5
    )

    if tools.polygon.rasterise and len(tools.polygon.points):
        polygon = tools.polygon
        # Rasterise polygon into mask image
        result = polygon.apply(ctx.images.mask, tool_op)
        if result:
            cmd = cmd_volume.UpdateVolumeCmd(
                ctx.images.mask, result[0], result[1], ctx.gfx.textures["mask"]
            )
            ctx.cmds.push_apply(cmd)
        # Clean up for drawing next polygon
        tools.cancel_drawing_all()
        polygon.rasterise = False

    if tools.livewire.rasterise and len(tools.livewire.points):
        livewire = tools.livewire
        # Rasterise livewire into mask image
        result = livewire.apply(ctx.images.mask, tool_op)
        if result:
            cmd = cmd_volume.UpdateVolumeCmd(
                ctx.images.mask, result[0], result[1], ctx.gfx.textures["mask"]
            )
            ctx.cmds.push_apply(cmd)
        # Clean up for drawing next livewire
        tools.cancel_drawing_all()
        livewire.rasterise = False

    show_menubar(ctx)
    show_gui(ctx)


def show_file_selection():
    root = tk.Tk()
    root.withdraw()  # Hide Tk window
    filepath = tk.filedialog.askopenfilename(filetypes=[("Volume file", ".vtk .dcm")])
    return filepath


def show_save_selection():
    root = tk.Tk()
    root.withdraw()  # Hide Tk window
    filepath = tk.filedialog.asksaveasfilename(
        filetypes=[("VTK image", ".vtk")], defaultextension=".vtk"
    )
    return filepath


def show_resample_orientation_dialog():
    root = tk.Tk()
    root.withdraw()  # Hide Tk window
    msg = "Warning: this will change the volume resolution and clear any current segmentation"
    return tk.messagebox.askokcancel("Resample orientation", msg)


def show_quit_dialog():
    root = tk.Tk()
    root.withdraw()  # Hide Tk window
    msg = "Warning: this will lose any unsaved changes to the current segmentation"
    return tk.messagebox.askokcancel("Quit", msg)


def show_menubar(ctx):
    """Show ImGui menu bar"""
    imgui.begin_main_menu_bar()
    if imgui.begin_menu("File"):
        if imgui.menu_item("Open volume...")[0]:
            filename = show_file_selection()
            if filename:
                ctx.settings.basepath = filename  # FIXME
                ctx.images.load_volume_fromfile(filename)
                ctx.cmds.clear_stack()  # Clear undo history
                ctx.mpr.update_minmax_range_from_volume(ctx.images.volume)
                gfx_utils.update_texture_3d(ctx.gfx.textures["volume"], ctx.images.volume)
                gfx_utils.update_texture_3d(ctx.gfx.textures["mask"], ctx.images.mask)
                ctx.tools.cancel_drawing_all()
        if imgui.menu_item("Save volume...")[0]:
            filename = show_save_selection()
            if filename:
                image_utils.save_vtk(filename, ctx.images.volume, ctx.images.header)
        if imgui.menu_item("Open segmentation...")[0]:
            filename = show_file_selection()
            if filename:
                ctx.images.load_mask_fromfile(filename)
                ctx.cmds.clear_stack()  # Clear undo history
                ctx.mpr.update_minmax_range_from_volume(ctx.images.volume)
                gfx_utils.update_texture_3d(ctx.gfx.textures["volume"], ctx.images.volume)
                gfx_utils.update_texture_3d(ctx.gfx.textures["mask"], ctx.images.mask)
                ctx.tools.cancel_drawing_all()
        if imgui.menu_item("Save segmentation...")[0]:
            filename = show_save_selection()
            if filename:
                mask_header = copy.deepcopy(ctx.images.header)
                mask_header["format"] = "unsigned_char"
                image_utils.save_vtk(filename, ctx.images.mask, mask_header)
        if imgui.menu_item("Quit")[0]:
            if show_quit_dialog():
                glfw.set_window_should_close(ctx.gfx.window, glfw.TRUE)
        imgui.end_menu()
    if imgui.begin_menu("Edit"):
        if imgui.menu_item("Undo (Ctrl+z)")[0] and not ctx.tools.is_active_any():
            ctx.cmds.pop_undo()
        if imgui.menu_item("Resample orientation...")[0]:
            if show_resample_orientation_dialog():
                ctx.images.resample_volume()
                gfx_utils.update_texture_3d(ctx.gfx.textures["volume"], ctx.images.volume)
                gfx_utils.update_texture_3d(ctx.gfx.textures["mask"], ctx.images.mask)
                ctx.tools.cancel_drawing_all()
                ctx.cmds.clear_stack()
        if imgui.menu_item("Clear segmentation")[0]:
            zeros = np.zeros(ctx.images.mask.shape, dtype=ctx.images.mask.dtype)
            cmd = cmd_volume.UpdateVolumeCmd(
                ctx.images.mask, zeros, (0, 0, 0), ctx.gfx.textures["mask"]
            )
            ctx.cmds.push_apply(cmd)
        imgui.end_menu()
    if imgui.begin_menu("Tools"):
        if imgui.menu_item("Show volume info")[0]:
            ctx.settings.show_stats = not ctx.settings.show_stats
        if imgui.menu_item("Show navigator")[0]:
            ctx.settings.show_navigator = not ctx.settings.show_navigator
        imgui.end_menu()
    if imgui.begin_menu("Help"):
        if imgui.menu_item("Quick reference")[0]:
            ctx.settings.show_input_guide = not ctx.settings.show_input_guide
        imgui.end_menu()
    imgui.end_main_menu_bar()


def draw_list_add_mpr_lines(draw_list, x, y, w, h, mpr, axis):
    """Draw outline of the input axis and crosshair for the other axes"""
    planes = mpr.planes
    colors = [(0.5, 0.5, 1, 1), (0, 1, 0, 1), (1, 0, 0, 1)]
    if axis == 0:
        px, py = (planes[0] + 0.5) * w, (planes[1] + 0.5) * h
        draw_list.add_line(
            x + px, y, x + px, y + h, imgui.get_color_u32_rgba(*colors[2]), thickness=1.5
        )
        draw_list.add_line(
            x, y + py, x + w, y + py, imgui.get_color_u32_rgba(*colors[1]), thickness=1.5
        )
    if axis == 1:
        px, py = (planes[0] + 0.5) * w, (planes[2] + 0.5) * h
        draw_list.add_line(
            x + px, y, x + px, y + h, imgui.get_color_u32_rgba(*colors[2]), thickness=1.5
        )
        draw_list.add_line(
            x, y + py, x + w, y + py, imgui.get_color_u32_rgba(*colors[0]), thickness=1.5
        )
    if axis == 2:
        px, py = (planes[1] + 0.5) * w, (planes[2] + 0.5) * h
        draw_list.add_line(
            x + px, y, x + px, y + h, imgui.get_color_u32_rgba(*colors[1]), thickness=1.5
        )
        draw_list.add_line(
            x, y + py, x + w, y + py, imgui.get_color_u32_rgba(*colors[0]), thickness=1.5
        )
    draw_list.add_rect(
        x, y, x + w, y + h, imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.5), thickness=1.0
    )


def show_navigator(ctx):
    """Show a navigator window with axial, coronal, and sagital views"""
    sf = imgui.get_io().font_global_scale
    flags = imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR
    views = ["axial", "coronal", "sagital"]
    planes = [gfx_mpr.MPR_PLANE_Z, gfx_mpr.MPR_PLANE_Y, gfx_mpr.MPR_PLANE_X]
    xy_mapping = [(2, 1), (2, 0), (1, 0)]
    tile_size = ctx.gfx.height / 5
    padding = 8

    imgui.set_next_window_size(tile_size * sf, (tile_size * 3 - padding * 2) * sf)
    imgui.set_next_window_position(0, ctx.gfx.height - (tile_size * 4 - padding) * sf)
    imgui.set_next_window_bg_alpha(0.8)
    _, ctx.settings.show_navigator = imgui.begin("Navigator views", flags=flags)
    for i in range(0, 3):
        imgui.image(ctx.gfx.textures[views[i]], tile_size - padding * 2, tile_size - padding * 2)
        image_pos, image_size = imgui.get_item_rect_min(), imgui.get_item_rect_size()
        if i < 2:
            imgui.spacing()

        hovered = imgui.is_item_hovered()
        delta = imgui.get_io().mouse_delta
        dragging = hovered and imgui.is_mouse_down()
        dblclick = hovered and imgui.is_mouse_double_clicked()
        scrolling = hovered and imgui.get_io().mouse_wheel

        # TODO: This could need a bit of refactoring...
        if views[i] == "axial" and dblclick:
            ctx.trackball.quat = glm.quat(glm.radians(glm.vec3(0, 0, 0)))
            ctx.tools.set_plane_all(planes[i])
            ctx.tools.cancel_drawing_all()
        if views[i] == "coronal" and dblclick:
            ctx.trackball.quat = glm.quat(glm.radians(glm.vec3(-90, 180, 0)))
            ctx.tools.set_plane_all(planes[i])
            ctx.tools.cancel_drawing_all()
        if views[i] == "sagital" and dblclick:
            ctx.trackball.quat = glm.quat(glm.radians(glm.vec3(-90, 90, 0)))
            ctx.tools.set_plane_all(planes[i])
            ctx.tools.cancel_drawing_all()

        if dragging and max(abs(delta.x), abs(delta.y)) < 10:
            # Note: need to check delta's magnitude, since this value can be
            # very large if the window is not in focus
            xindex, yindex = xy_mapping[i]
            steps_x = (float(delta.x) / image_size.x) * ctx.images.volume.shape[xindex]
            steps_y = (float(delta.y) / image_size.y) * ctx.images.volume.shape[yindex]
            ctx.mpr.scroll_by_axis(ctx.images.volume, planes[xindex], steps_x)
            ctx.mpr.scroll_by_axis(ctx.images.volume, planes[yindex], steps_y)
            ctx.tools.cancel_drawing_all()

        if scrolling:
            # Clamp step size so that we always scroll at most 1 voxel
            steps = max(-1.0, min(1.0, scrolling))
            ctx.mpr.scroll_by_axis(ctx.images.volume, planes[i], steps)
            ctx.tools.cancel_drawing_all()

        draw_list = imgui.get_window_draw_list()
        draw_list_add_mpr_lines(draw_list, *image_pos, *image_size, ctx.mpr, i)
    imgui.end()


def show_volume_stats(ctx):
    sf = imgui.get_io().font_global_scale
    imgui.set_next_window_size(250 * sf, 120 * sf)
    imgui.set_next_window_position(ctx.gfx.width - 250 * sf, 18 * sf)
    imgui.set_next_window_bg_alpha(0.8)

    flags = imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_TITLE_BAR
    _, ctx.settings.show_stats = imgui.begin("Volume info", closable=True, flags=flags)
    imgui.text("Volume info")
    imgui.spacing()
    imgui.indent(5)
    imgui.text("Dimensions (voxels): %d %d %d" % tuple(ctx.images.header["dimensions"]))
    imgui.text("Spacing (mm): %.2f %.2f %.2f" % tuple(ctx.images.header["spacing"]))
    imgui.text("Scalar type: %s" % str(ctx.images.volume.dtype))
    imgui.text("Mask type: %s" % str(ctx.images.mask.dtype))
    imgui.text("Segmented volume (ml): %.2f" % (ctx.images.measured_volume_ml))
    imgui.unindent(5)
    if ctx.cmds.dirty and not ctx.tools.is_painting_any():
        print("Recalculating volume measurement...")
        tic = time.time()
        ctx.images.update_measured_volume()
        print("Done (elapsed time: %.2f s)" % (time.time() - tic))
        ctx.cmds.dirty = False
    imgui.end()


def show_input_guide(ctx):
    sf = imgui.get_io().font_global_scale
    imgui.set_next_window_size(250 * sf, 260 * sf)
    if ctx.settings.show_stats:
        imgui.set_next_window_position(ctx.gfx.width - 250 * sf, (18 + 120) * sf)
    else:
        imgui.set_next_window_position(ctx.gfx.width - 250 * sf, 18 * sf)
    imgui.set_next_window_bg_alpha(0.8)

    flags = imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_TITLE_BAR
    _, ctx.settings.show_input_guide = imgui.begin("Quick reference", closable=True, flags=flags)
    imgui.text("Quick reference")
    imgui.spacing()
    imgui.indent(5)
    imgui.text("Left button: Paint/draw/grab")
    imgui.text("Middle button: Rotate view")
    imgui.text("Right button: Pan view")
    imgui.text("Scroll: Zoom view")
    imgui.text("Shift+Scroll: Scroll slices")
    imgui.text("Key 1: Show axial view")
    imgui.text("Key 2: Show sagital view")
    imgui.text("Key 3: Show coronal view")
    imgui.text("Page up/down: Change label")
    imgui.text("Enter: Close polygon")
    imgui.text("Escape: Cancel polygon")
    imgui.text("Space: Hide segmentation")
    imgui.text("Ctrl+z: Undo")
    imgui.unindent(5)
    imgui.end()


def show_volume_settings(ctx):
    """Show widgets for volume settings"""
    clicked, ctx.mpr.level_preset = imgui.combo(
        "Level preset", ctx.mpr.level_preset, gfx_mpr.MPR_PRESET_NAMES
    )
    if clicked:
        ctx.mpr.update_level_range()
    clicked, ctx.mpr.level_range = imgui.drag_int2("Level range", *ctx.mpr.level_range, 10)
    if clicked:
        ctx.mpr.level_preset = gfx_mpr.MPR_PRESET_NAMES.index("Custom")


def show_segmentation_settings(ctx):
    """Show widgets for segmentation settings"""
    labels = ctx.images.labels  # TODO
    _, ctx.images.active_label = imgui.combo("Label", ctx.images.active_label, labels)
    _, ctx.settings.show_mask = imgui.checkbox("Show segmentation", ctx.settings.show_mask)


def show_tools_settings(ctx):
    """Show widgets for tools settings"""
    tools = ctx.tools

    # Polygon tool settings
    polygon = tools.polygon
    clicked, polygon.enabled = imgui.checkbox("Polygon tool", polygon.enabled)
    if clicked and polygon.enabled:
        tools.disable_all_except(polygon)
    if polygon.enabled:
        imgui.indent(5)
        _, polygon.antialiasing = imgui.checkbox("Antialiasing", polygon.antialiasing)
        imgui.unindent(5)

    # Brush tool settings
    brush = tools.brush
    clicked, brush.enabled = imgui.checkbox("Brush tool", brush.enabled)
    if clicked and brush.enabled:
        tools.disable_all_except(brush)
    if brush.enabled:
        imgui.indent(5)
        _, brush.mode = imgui.combo("Mode", brush.mode, ["2D", "3D"])
        _, brush.size = imgui.slider_int("Brush size", brush.size, 1, 80)
        _, brush.antialiasing = imgui.checkbox("Antialiasing", brush.antialiasing)
        imgui.unindent(5)

    # Livewire tool settings
    livewire = tools.livewire
    clicked, livewire.enabled = imgui.checkbox("Livewire tool", livewire.enabled)
    if clicked and livewire.enabled:
        tools.disable_all_except(livewire)
    if livewire.enabled:
        imgui.indent(5)
        _, livewire.smoothing = imgui.checkbox("Smoothing enabled", livewire.smoothing)
        imgui.unindent(5)

    # Smart brush tool settings
    smartbrush = tools.smartbrush
    clicked, smartbrush.enabled = imgui.checkbox("Smartbrush tool", smartbrush.enabled)
    if clicked and smartbrush.enabled:
        tools.disable_all_except(smartbrush)
    if smartbrush.enabled:
        imgui.indent(5)
        _, smartbrush.mode = imgui.combo("Mode", smartbrush.mode, ["2D", "3D"])
        _, smartbrush.size = imgui.slider_int("Brush size", smartbrush.size, 1, 80)
        _, smartbrush.sensitivity = imgui.slider_float(
            "Sensitivity", smartbrush.sensitivity, 0.0, 10.0
        )
        _, smartbrush.delta_scaling = imgui.slider_float(
            "Delta scale", smartbrush.delta_scaling, 1.0, 5.0
        )
        imgui.unindent(5)

    # Seed painting tool settings
    seedpaint = tools.seedpaint
    clicked, seedpaint.enabled = imgui.checkbox("Seed paint tool", seedpaint.enabled)
    if clicked and seedpaint.enabled:
        tools.disable_all_except(seedpaint)
    if seedpaint.enabled:
        imgui.indent(5)
        imgui.text("Not implemented yet (TODO)")
        imgui.unindent(5)


def show_viewing_settings(ctx):
    """Show widgets for viewing settings"""
    mpr = ctx.mpr
    _, mpr.enabled = imgui.checkbox("Show MPR", mpr.enabled)
    _, mpr.show_voxels = imgui.checkbox("Show voxels", mpr.show_voxels)


def show_misc_settings(ctx):
    """Show widgets for miscelaneous settings"""
    _, ctx.settings.bg_color1 = imgui.color_edit3("BG color 1", *ctx.settings.bg_color1)
    _, ctx.settings.bg_color2 = imgui.color_edit3("BG color 2", *ctx.settings.bg_color2)
    clicked, ctx.settings.dark_mode = imgui.checkbox("Dark mode", ctx.settings.dark_mode)
    if clicked:
        set_gui_style(ctx.settings.dark_mode)


def show_gui(ctx):
    """Show ImGui windows"""
    sf = imgui.get_io().font_global_scale
    imgui.set_next_window_size(ctx.gfx.sidebar_width, ctx.gfx.height - 18 * sf)
    imgui.set_next_window_position(ctx.gfx.width, 18 * sf)
    flags = imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_TITLE_BAR

    imgui.begin("Segmentation", flags=flags)
    if imgui.collapsing_header("Volume", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        show_volume_settings(ctx)
        imgui.text("")  # Add some spacing to next settings group
    if imgui.collapsing_header("Segmentation", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        show_segmentation_settings(ctx)
        imgui.text("")  # Add some spacing to next settings group
    if imgui.collapsing_header("Tools", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        show_tools_settings(ctx)
        imgui.text("")  # Add some spacing to next settings group
    if imgui.collapsing_header("Viewing")[0]:
        show_viewing_settings(ctx)
        imgui.text("")  # Add some spacing to next settings group
    if imgui.collapsing_header("Misc")[0]:
        show_misc_settings(ctx)
    imgui.end()

    if ctx.settings.show_stats:
        show_volume_stats(ctx)
    if ctx.settings.show_input_guide:
        show_input_guide(ctx)
    if ctx.settings.show_navigator and max(ctx.images.volume.shape) > 1:
        show_navigator(ctx)


def set_style_color_rgb(idx, r, g, b):
    """Update global ImGui style color without changing alpha"""
    style = imgui.get_style()
    tmp = style.colors[idx]
    style.colors[idx] = imgui.Vec4(r, g, b, tmp.w)


def set_gui_style(dark_mode=False):
    """Apply global ImGui style settings for light or dark theme"""
    if dark_mode:
        imgui.style_colors_dark()
    else:
        imgui.style_colors_light()
    style = imgui.get_style()
    style.window_rounding = 0.0
    set_style_color_rgb(imgui.COLOR_HEADER, 0.5, 0.5, 0.5)
    set_style_color_rgb(imgui.COLOR_HEADER_ACTIVE, 0.5, 0.5, 0.5)
    set_style_color_rgb(imgui.COLOR_HEADER_HOVERED, 0.5, 0.5, 0.5)
    set_style_color_rgb(imgui.COLOR_BUTTON, 0.5, 0.5, 0.5)
    set_style_color_rgb(imgui.COLOR_BUTTON_ACTIVE, 0.5, 0.5, 0.5)
    set_style_color_rgb(imgui.COLOR_BUTTON_HOVERED, 0.5, 0.5, 0.5)
    set_style_color_rgb(imgui.COLOR_FRAME_BACKGROUND_ACTIVE, 0.5, 0.5, 0.5)
    set_style_color_rgb(imgui.COLOR_FRAME_BACKGROUND_HOVERED, 0.5, 0.5, 0.5)
    if dark_mode:
        set_style_color_rgb(imgui.COLOR_WINDOW_BACKGROUND, 0.18, 0.18, 0.18)
        set_style_color_rgb(imgui.COLOR_FRAME_BACKGROUND, 0.5, 0.5, 0.5)


def char_callback(window, char):
    ctx = glfw.get_window_user_pointer(window)
    if imgui.get_io().want_text_input:
        ctx.imgui_glfw.char_callback(window, char)
        return


def key_callback(window, key, scancode, action, mods):
    ctx = glfw.get_window_user_pointer(window)
    if imgui.get_io().want_capture_keyboard:
        ctx.imgui_glfw.keyboard_callback(window, key, scancode, action, mods)
        return

    if key == glfw.KEY_LEFT_SHIFT:
        # Note: some keyboards will repeat action keys, so must check for that
        # case as well
        ctx.mpr.scrolling = action == glfw.PRESS or action == glfw.REPEAT
    if key == glfw.KEY_1:  # Show top-view
        ctx.trackball.quat = glm.quat(glm.radians(glm.vec3(0, 0, 0)))
        ctx.tools.set_plane_all(gfx_mpr.MPR_PLANE_Z)
    if key == glfw.KEY_2:  # Show side-view
        ctx.trackball.quat = glm.quat(glm.radians(glm.vec3(-90, 90, 0)))
        ctx.tools.set_plane_all(gfx_mpr.MPR_PLANE_X)
    if key == glfw.KEY_3:  # Show front-view
        ctx.trackball.quat = glm.quat(glm.radians(glm.vec3(-90, 180, 0)))
        ctx.tools.set_plane_all(gfx_mpr.MPR_PLANE_Y)
    if key == glfw.KEY_PAGE_UP and action == glfw.PRESS:
        ctx.images.active_label = max(ctx.images.active_label - 1, 0)
    if key == glfw.KEY_PAGE_DOWN and action == glfw.PRESS:
        max_label = len(ctx.images.labels) - 1
        ctx.images.active_label = min(ctx.images.active_label + 1, max_label)
    if key == glfw.KEY_SPACE and action == glfw.PRESS:
        ctx.settings.show_mask = False
    if key == glfw.KEY_SPACE and action == glfw.RELEASE:
        ctx.settings.show_mask = True
    if key == glfw.KEY_ENTER:  # Rasterise polygon or livewire
        ctx.tools.polygon.rasterise = action == glfw.PRESS
        ctx.tools.livewire.rasterise = action == glfw.PRESS
    if key == glfw.KEY_ESCAPE:  # Cancel polygon or livewire
        ctx.tools.cancel_drawing_all()
    if key == glfw.KEY_Z and (mods & glfw.MOD_CONTROL):
        if action == glfw.PRESS and not ctx.tools.is_active_any():
            ctx.cmds.pop_undo()


def mouse_button_callback(window, button, action, mods):
    ctx = glfw.get_window_user_pointer(window)
    if imgui.get_io().want_capture_mouse:
        ctx.imgui_glfw.mouse_callback(window, button, action, mods)
        return

    x, y = glfw.get_cursor_pos(window)
    if button == glfw.MOUSE_BUTTON_MIDDLE:
        ctx.trackball.center = glm.vec2(x, y)
        # ctx.trackball.tracking = (action == glfw.PRESS)
    if button == glfw.MOUSE_BUTTON_RIGHT:
        ctx.panning.center = glm.vec2(x, y)
        ctx.panning.panning = action == glfw.PRESS
    if button == glfw.MOUSE_BUTTON_LEFT:
        if ctx.tools.brush.enabled:
            ctx.tools.brush.painting = action == glfw.PRESS
            ctx.tools.brush.frame_count = 0
        if ctx.tools.smartbrush.enabled:
            ctx.tools.smartbrush.painting = action == glfw.PRESS
            ctx.tools.smartbrush.frame_count = 0
        if ctx.tools.polygon.enabled:
            ctx.tools.polygon.clicking = action == glfw.PRESS
            ctx.tools.polygon.selected = -1
            ctx.tools.polygon.frame_count = 0
        if ctx.tools.livewire.enabled:
            ctx.tools.livewire.clicking = action == glfw.PRESS


def cursor_pos_callback(window, x, y):
    ctx = glfw.get_window_user_pointer(window)
    if imgui.get_io().want_capture_mouse:
        return

    ctx.trackball.move(x, y)
    ctx.panning.move(x, y)


def scroll_callback(window, xoffset, yoffset):
    ctx = glfw.get_window_user_pointer(window)
    if imgui.get_io().want_capture_mouse:
        ctx.imgui_glfw.scroll_callback(window, xoffset, yoffset)
        return

    if ctx.mpr.scrolling:
        ray_dir = glm.vec3(glm.inverse(glm.mat4_cast(ctx.trackball.quat))[2])
        ctx.mpr.scroll_by_ray(ctx.images.volume, ray_dir, yoffset)
        # Cancel all drawing in case the user was drawing a polygon or livewire
        # on the MPR plane while scrolling
        ctx.tools.cancel_drawing_all()
    else:
        ctx.settings.fov_degrees += 2.0 * yoffset
        ctx.settings.fov_degrees = max(5.0, min(90.0, ctx.settings.fov_degrees))


def resize_callback(window, w, h):
    ctx = glfw.get_window_user_pointer(window)
    ctx.gfx.width = max(1, w - ctx.gfx.sidebar_width)
    ctx.gfx.height = max(1, h)
    gl.glViewport(0, 0, ctx.gfx.width, ctx.gfx.height)


def main():
    # Create variable for passing around application state
    ctx = Context()

    # Create GLFW window with OpenGL context
    if not glfw.init():
        return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
    ctx.gfx.window = glfw.create_window(ctx.gfx.width, ctx.gfx.height, "ichseg", None, None)
    glfw.set_window_user_pointer(ctx.gfx.window, ctx)
    glfw.set_window_size_callback(ctx.gfx.window, resize_callback)
    glfw.set_char_callback(ctx.gfx.window, char_callback)
    glfw.set_key_callback(ctx.gfx.window, key_callback)
    glfw.set_mouse_button_callback(ctx.gfx.window, mouse_button_callback)
    glfw.set_cursor_pos_callback(ctx.gfx.window, cursor_pos_callback)
    glfw.set_scroll_callback(ctx.gfx.window, scroll_callback)
    glfw.make_context_current(ctx.gfx.window)
    print("GL version: %s" % gl.glGetString(gl.GL_VERSION).decode())

    # Initialize ImGui
    imgui.create_context()
    set_gui_style(ctx.settings.dark_mode)
    ctx.imgui_glfw = GlfwRenderer(ctx.gfx.window, False)

    # This should fix GUI-scaling for high-DPI screens:
    primary = glfw.get_primary_monitor()
    xscale, yscale = glfw.get_monitor_content_scale(primary)
    if xscale > 1.25:
        imgui.get_io().font_global_scale = xscale
        ctx.gfx.sidebar_width = int(ctx.gfx.sidebar_width * xscale)

    # Initialize application
    do_initialize(ctx)
    resize_callback(ctx.gfx.window, ctx.gfx.width, ctx.gfx.height)

    # Start main loop
    while not glfw.window_should_close(ctx.gfx.window):
        glfw.poll_events()
        ctx.imgui_glfw.process_inputs()
        imgui.new_frame()
        do_update(ctx)
        do_rendering(ctx)
        imgui.render()
        ctx.imgui_glfw.render(imgui.get_draw_data())
        glfw.swap_buffers(ctx.gfx.window)
    glfw.terminate()


if __name__ == "__main__":
    main()
