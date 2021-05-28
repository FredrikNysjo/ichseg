import cmd_manager
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
        self.show_navigator = False
        self.show_input_guide = False
        self.dark_mode = True
        self.basepath = ""


class Context:
    def __init__(self):
        self.window = None
        self.width = 1000
        self.height = 700
        self.sidebar_width = 290
        self.programs = {}
        self.vaos = {}
        self.buffers = {}
        self.textures = {}

        self.settings = Settings()
        self.trackball = gfx_utils.Trackball()
        self.panning = gfx_utils.Panning()
        self.mpr = gfx_mpr.MPR()
        self.tools = tool_manager.ToolManager()
        self.cmds = cmd_manager.CmdManager()
        self.segmented_volume_ml = 0.0


class UpdateVolumeCmd:
    def __init__(self, volume_, subimage_, offset_, texture_=0):
        self.volume = volume_
        self.subimage = subimage_
        self.offset = offset_
        self.texture = texture_
        self._prev_subimage = None

    def apply(self):
        x, y, z = self.offset
        d, h, w = self.subimage.shape
        self._prev_subimage = np.copy(self.volume[z : z + d, y : y + h, x : x + w])
        self.volume[z : z + d, y : y + h, x : x + w] = self.subimage
        if self.texture:
            gfx_utils.update_subtexture_3d(self.texture, self.subimage, self.offset)
        return self

    def undo(self):
        x, y, z = self.offset
        d, h, w = self.subimage.shape
        self.volume[z : z + d, y : y + h, x : x + w] = self._prev_subimage
        if self.texture:
            gfx_utils.update_subtexture_3d(self.texture, self._prev_subimage, self.offset)
        return self


def load_segmentation_mask(ct_volume, dirname):
    """Load segmentation masks from JPG-files and into a mask volume of
    same size as the input CT volume
    """
    mask = np.zeros(ct_volume.shape, dtype=np.uint8)
    if os.path.exists(dirname):
        for filename in os.listdir(dirname):
            if not filename.lower().count(".jpg"):
                continue
            image = image_utils.load_image(os.path.join(dirname, filename)).reshape(512, 512)
            image = image[::-1, :]  # Flip along Y-axis
            slice_pos = int(filename.split("_")[0]) - 1  # Indices in filenames seem to start at 1
            mask[slice_pos, :, :] = np.maximum(mask[slice_pos, :, :], image)
    return mask


def create_dummy_dataset():
    volume = np.zeros([1, 1, 1], dtype=np.int16)
    header = {"dimensions": (1, 1, 1), "spacing": (1, 1, 1)}
    return volume, header


def load_dataset_fromfile(ctx, filename):
    ctx.datasets = []
    ctx.current, ctx.label = 0, 0
    if ".vtk" in filename:
        ctx.volume, ctx.header = image_utils.load_vtk(filename)
    elif filename:  # Assume format is DICOM
        ctx.volume, ctx.header = image_dicom.load_dicom(filename)
    else:
        ctx.volume, ctx.header = create_dummy_dataset()
    ctx.mask = np.zeros(ctx.volume.shape, dtype=np.uint8)
    ctx.mpr.minmax_range[0] = np.min(ctx.volume)
    ctx.mpr.minmax_range[1] = np.max(ctx.volume)
    ctx.mpr.update_level_range()
    ctx.cmds.clear_stack()


def load_mask_fromfile(ctx, filename):
    """Load a segmentation mask from file

    If the grayscale volume we are segmenting does not match the size
    of the loaded mask, a new empty volume will be created

    Currently, only 8-bit masks in VTK format are supported
    """
    if ".vtk" not in filename:
        return
    mask, header = image_utils.load_vtk(filename)
    if mask.dtype == np.uint8:
        if mask.shape != ctx.volume.shape:
            ctx.header = header
            ctx.volume = np.zeros(mask.shape, dtype=np.uint8)
        ctx.mask = mask
        ctx.cmds.clear_stack()


def do_initialize(ctx):
    """Initialize the application state"""
    tools = ctx.tools

    ctx.classes = ["Label 255", "Label 0 (Clear)"]
    load_dataset_fromfile(ctx, "")

    # Placeholders for XYZ views shown in the navigator
    axial_view = np.array([0, 0, 255], dtype=np.uint8).reshape((1, 1, 3))
    coronal_view = np.array([0, 255, 0], dtype=np.uint8).reshape((1, 1, 3))
    sagital_view = np.array([255, 0, 0], dtype=np.uint8).reshape((1, 1, 3))

    ctx.programs["raycast"] = gfx_utils.create_program(
        (gfx_shaders.raycast_vs, gl.GL_VERTEX_SHADER),
        (gfx_shaders.raycast_fs, gl.GL_FRAGMENT_SHADER),
    )
    ctx.programs["polygon"] = gfx_utils.create_program(
        (gfx_shaders.polygon_vs, gl.GL_VERTEX_SHADER),
        (gfx_shaders.polygon_fs, gl.GL_FRAGMENT_SHADER),
    )
    ctx.programs["background"] = gfx_utils.create_program(
        (gfx_shaders.background_vs, gl.GL_VERTEX_SHADER),
        (gfx_shaders.background_fs, gl.GL_FRAGMENT_SHADER),
    )

    ctx.vaos["empty"] = gl.glGenVertexArrays(1)
    ctx.buffers["polygon"] = gfx_utils.create_mesh_buffer(tools.polygon.points)
    ctx.vaos["polygon"] = gfx_utils.create_mesh_vao(tools.polygon.points, ctx.buffers["polygon"])
    ctx.buffers["markers"] = gfx_utils.create_mesh_buffer(tools.polygon.points)
    ctx.vaos["markers"] = gfx_utils.create_mesh_vao(tools.polygon.points, ctx.buffers["markers"])

    ctx.textures["volume"] = gfx_utils.create_texture_3d(ctx.volume)
    ctx.textures["mask"] = gfx_utils.create_texture_3d(ctx.mask)
    ctx.textures["axial"] = gfx_utils.create_texture_2d(axial_view)
    ctx.textures["sagital"] = gfx_utils.create_texture_2d(sagital_view)
    ctx.textures["coronal"] = gfx_utils.create_texture_2d(coronal_view)


def do_rendering(ctx):
    """Do rendering"""
    tools = ctx.tools
    tool_op = TOOL_OP_ADD if ctx.label == 0 else TOOL_OP_SUBTRACT

    mpr_planes = ctx.mpr.get_snapped_planes(ctx.volume)
    level_range = ctx.mpr.level_range
    level_range_scaled = level_range  # Adjusted for normalized texture formats
    if ctx.volume.dtype == np.uint8:
        level_range_scaled = [v / 255.0 for v in level_range]  # Scale to range [0,1]
    if ctx.volume.dtype == np.int16:
        level_range_scaled = [v / 32767.0 for v in level_range]  # Scale to range [-1,1]
    filter_mode = gl.GL_NEAREST if ctx.mpr.show_voxels else gl.GL_LINEAR

    aspect = float(ctx.width) / ctx.height
    proj = glm.perspective(glm.radians(ctx.settings.fov_degrees), aspect, 0.1, 10.0)
    if ctx.settings.projection_mode:
        k = glm.radians(ctx.settings.fov_degrees)
        proj = glm.ortho(-k * aspect, k * aspect, -k, k, 0.1, 10.0)
    view = (
        glm.translate(glm.mat4(1.0), glm.vec3(0, 0, -2))
        * glm.translate(glm.mat4(1.0), -ctx.panning.position)
        * glm.mat4_cast(ctx.trackball.quat)
    )
    spacing = glm.vec3(ctx.header["spacing"])
    extent = glm.vec3(ctx.header["spacing"]) * glm.vec3(ctx.header["dimensions"])
    model = glm.scale(glm.mat4(1.0), extent / glm.max(extent.x, glm.max(extent.y, extent.z)))
    mv = view * model
    mvp = proj * view * model

    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    program = ctx.programs["background"]
    gl.glUseProgram(program)
    gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glDepthMask(gl.GL_FALSE)
    gl.glViewport(0, 0, ctx.width + ctx.sidebar_width, ctx.height)
    gl.glUniform3f(gl.glGetUniformLocation(program, "u_bg_color1"), *ctx.settings.bg_color1)
    gl.glUniform3f(gl.glGetUniformLocation(program, "u_bg_color2"), *ctx.settings.bg_color2)
    gl.glBindVertexArray(ctx.vaos["empty"])
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
    gl.glViewport(0, 0, ctx.width, ctx.height)
    gl.glDepthMask(gl.GL_TRUE)

    program = ctx.programs["raycast"]
    gl.glUseProgram(program)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_CULL_FACE)
    gl.glActiveTexture(gl.GL_TEXTURE1)
    gl.glBindTexture(gl.GL_TEXTURE_3D, ctx.textures["mask"])
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MIN_FILTER, filter_mode)
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAG_FILTER, filter_mode)
    gl.glActiveTexture(gl.GL_TEXTURE0)
    gl.glBindTexture(gl.GL_TEXTURE_3D, ctx.textures["volume"])
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MIN_FILTER, filter_mode)
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAG_FILTER, filter_mode)
    gl.glUniformMatrix4fv(gl.glGetUniformLocation(program, "u_mvp"), 1, False, glm.value_ptr(mvp))
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
    gl.glBindVertexArray(ctx.vaos["empty"])
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 14)
    gl.glBindVertexArray(0)

    if tools.polygon.enabled:
        program = ctx.programs["polygon"]
        gl.glUseProgram(program)
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(program, "u_mvp"), 1, False, glm.value_ptr(mvp)
        )
        gl.glUniform1i(gl.glGetUniformLocation(program, "u_label"), 0)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glPointSize(5.0)
        gl.glBindVertexArray(ctx.vaos["polygon"])
        gl.glDrawArrays(gl.GL_LINE_STRIP, 0, len(tools.polygon.points) // 3)
        gl.glBindVertexArray(ctx.vaos["markers"])
        gl.glDrawArrays(gl.GL_POINTS, 0, len(tools.polygon.points) // 3)
        gl.glBindVertexArray(0)
        gl.glPointSize(1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)

    if tools.livewire.enabled:
        program = ctx.programs["polygon"]
        gl.glUseProgram(program)
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(program, "u_mvp"), 1, False, glm.value_ptr(mvp)
        )
        gl.glUniform1i(gl.glGetUniformLocation(program, "u_label"), 0)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glPointSize(5.0)
        gl.glBindVertexArray(ctx.vaos["polygon"])
        gl.glDrawArrays(gl.GL_LINE_STRIP, 0, len(tools.livewire.points) // 3)
        gl.glBindVertexArray(ctx.vaos["markers"])
        gl.glDrawArrays(gl.GL_POINTS, 0, len(tools.livewire.markers) // 3)
        gl.glBindVertexArray(0)
        gl.glEnable(gl.GL_DEPTH_TEST)

    if ctx.mpr.enabled:
        x, y = glfw.get_cursor_pos(ctx.window)
        depth = gl.glReadPixels(x, (ctx.height - 1) - y, 1, 1, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
        ndc_pos = glm.vec3(0.0)
        ndc_pos.x = (float(x) / ctx.width) * 2.0 - 1.0
        ndc_pos.y = -((float(y) / ctx.height) * 2.0 - 1.0)
        ndc_pos.z = depth[0][0] * 2.0 - 1.0
        view_pos = gfx_utils.reconstruct_view_pos(ndc_pos, proj)
        texcoord = glm.vec3(glm.inverse(mv) * glm.vec4(view_pos, 1.0)) + 0.5001

        tools.brush.position = glm.vec4(texcoord - 0.5, tools.brush.position.w)
        tools.smartbrush.position = glm.vec4(texcoord - 0.5, tools.smartbrush.position.w)

        if depth != 1.0 and tools.brush.painting and tools.brush.enabled:
            brush = tools.brush
            if brush.frame_count == 0:
                # Store full copy of current segmentation for undo
                cmd = UpdateVolumeCmd(ctx.mask, np.copy(ctx.mask), (0, 0, 0), ctx.textures["mask"])
                ctx.cmds.push_apply(cmd)
            brush.frame_count += 1
            result = brush.apply(ctx.mask, texcoord, spacing, tool_op)
            if result:
                gfx_utils.update_subtexture_3d(ctx.textures["mask"], result[0], result[1])

        if depth != 1.0 and tools.smartbrush.painting and tools.smartbrush.enabled:
            smartbrush = tools.smartbrush
            if smartbrush.frame_count == 0:
                # Store full copy of current segmentation for undo
                cmd = UpdateVolumeCmd(ctx.mask, np.copy(ctx.mask), (0, 0, 0), ctx.textures["mask"])
                ctx.cmds.push_apply(cmd)
            if smartbrush.xy[0] != x or smartbrush.xy[1] != y:
                smartbrush.frame_count += 1
                smartbrush.momentum = min(5, smartbrush.momentum + 2)
                smartbrush.xy = (x, y)
            if smartbrush.momentum > 0:
                result = smartbrush.apply(
                    ctx.mask, ctx.volume, texcoord, spacing, level_range, tool_op
                )
                if result:
                    gfx_utils.update_subtexture_3d(ctx.textures["mask"], result[0], result[1])
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
                gfx_utils.update_mesh_buffer(ctx.buffers["polygon"], polygon.points)
                gfx_utils.update_mesh_buffer(ctx.buffers["markers"], polygon.points)

        if depth != 1.0 and tools.livewire.enabled:
            livewire = tools.livewire
            clicking = False
            if livewire.clicking:
                livewire.markers.extend(texcoord - 0.5)
                gfx_utils.update_mesh_buffer(ctx.buffers["markers"], livewire.markers)
                livewire.update_graph(ctx.volume, texcoord, level_range)
                livewire.clicking = False
                clicking = True
            if len(livewire.path):
                livewire.update_path(ctx.volume, texcoord, level_range, clicking)
                # Also update polyline for preview
                gfx_utils.update_mesh_buffer(ctx.buffers["polygon"], livewire.points)


def do_update(ctx):
    """Update application state"""
    tools = ctx.tools
    tool_op = TOOL_OP_ADD if ctx.label == 0 else TOOL_OP_SUBTRACT

    tools.brush.position.w = 0.0 if not tools.brush.enabled else tools.brush.size * 0.5
    tools.smartbrush.position.w = (
        0.0 if not tools.smartbrush.enabled else tools.smartbrush.size * 0.5
    )

    if tools.polygon.rasterise and len(tools.polygon.points):
        polygon = tools.polygon
        # Rasterise polygon into mask image
        result = polygon.apply(ctx.mask, tool_op)
        if result:
            cmd = UpdateVolumeCmd(ctx.mask, result[0], result[1], ctx.textures["mask"])
            ctx.cmds.push_apply(cmd)
        # Clean up for drawing next polygon
        tools.cancel_drawing_all()
        polygon.rasterise = False

    if tools.livewire.rasterise and len(tools.livewire.points):
        livewire = tools.livewire
        # Rasterise livewire into mask image
        result = livewire.apply(ctx.mask, tool_op)
        if result:
            cmd = UpdateVolumeCmd(ctx.mask, result[0], result[1], ctx.textures["mask"])
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
                ctx.settings.basepath = filename
                load_dataset_fromfile(ctx, filename)
                gfx_utils.update_texture_3d(ctx.textures["volume"], ctx.volume)
                gfx_utils.update_texture_3d(ctx.textures["mask"], ctx.mask)
                ctx.tools.cancel_drawing_all()
        if imgui.menu_item("Save volume...")[0]:
            filename = show_save_selection()
            if filename:
                image_utils.save_vtk(filename, ctx.volume, ctx.header)
        if imgui.menu_item("Open segmentation...")[0]:
            filename = show_file_selection()
            if filename:
                load_mask_fromfile(ctx, filename)
                gfx_utils.update_texture_3d(ctx.textures["volume"], ctx.volume)
                gfx_utils.update_texture_3d(ctx.textures["mask"], ctx.mask)
                ctx.tools.cancel_drawing_all()
        if imgui.menu_item("Save segmentation...")[0]:
            filename = show_save_selection()
            if filename:
                mask_header = copy.deepcopy(ctx.header)
                mask_header["format"] = "unsigned_char"
                image_utils.save_vtk(filename, ctx.mask, mask_header)
        if imgui.menu_item("Quit")[0]:
            if show_quit_dialog():
                glfw.set_window_should_close(ctx.window, glfw.TRUE)
        imgui.end_menu()
    if imgui.begin_menu("Edit"):
        if imgui.menu_item("Undo (Ctrl+z)")[0]:
            ctx.cmds.pop_undo()
        if imgui.menu_item("Resample orientation...")[0]:
            if show_resample_orientation_dialog():
                ctx.volume, ctx.header = image_dicom.resample_volume(ctx.volume, ctx.header)
                ctx.mask = np.zeros(ctx.volume.shape, dtype=np.uint8)
                gfx_utils.update_texture_3d(ctx.textures["volume"], ctx.volume)
                gfx_utils.update_texture_3d(ctx.textures["mask"], ctx.mask)
                ctx.tools.cancel_drawing_all()
                ctx.cmds.clear_stack()
        if imgui.menu_item("Clear segmentation")[0]:
            zeros = np.zeros(ctx.mask.shape, dtype=ctx.mask.dtype)
            cmd = UpdateVolumeCmd(ctx.mask, zeros, (0, 0, 0), ctx.textures["mask"])
            ctx.cmds.push_apply(cmd)
        imgui.end_menu()
    if imgui.begin_menu("Tools"):
        if imgui.menu_item("Volume statistics")[0]:
            ctx.settings.show_stats = not ctx.settings.show_stats
        if imgui.menu_item("Navigator views")[0]:
            ctx.settings.show_navigator = not ctx.settings.show_navigator
        imgui.end_menu()
    if imgui.begin_menu("Help"):
        if imgui.menu_item("Quick reference")[0]:
            ctx.settings.show_input_guide = not ctx.settings.show_input_guide
        imgui.end_menu()
    imgui.end_main_menu_bar()


def show_navigator(ctx):
    """Show a navigator window with axial, coronal, and sagital views"""
    sf = imgui.get_io().font_global_scale
    flags = imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR
    views = ["axial", "coronal", "sagital"]
    planes = [gfx_mpr.MPR_PLANE_Z, gfx_mpr.MPR_PLANE_Y, gfx_mpr.MPR_PLANE_X]
    wh = ctx.height / 5.0
    padding = 8

    imgui.set_next_window_size(wh * sf, (wh * 3 - padding * 2) * sf)
    imgui.set_next_window_position(0, ctx.height - (wh * 4 - padding) * sf)
    imgui.set_next_window_bg_alpha(0.8)
    _, ctx.settings.show_navigator = imgui.begin("Navigator views", flags=flags)
    for i in range(0, 3):
        imgui.image(ctx.textures[views[i]], wh - padding * 2, wh - padding * 2)
        clicked = imgui.is_item_clicked()
        hovered = imgui.is_item_hovered()
        scrolling = imgui.get_io().mouse_wheel
        if views[i] == "axial" and clicked:
            ctx.trackball.quat = glm.quat(glm.radians(glm.vec3(0, 0, 0)))
            ctx.tools.set_plane_all(planes[i])
        if views[i] == "coronal" and clicked:
            ctx.trackball.quat = glm.quat(glm.radians(glm.vec3(-90, 180, 0)))
            ctx.tools.set_plane_all(planes[i])
        if views[i] == "sagital" and clicked:
            ctx.trackball.quat = glm.quat(glm.radians(glm.vec3(-90, 90, 0)))
            ctx.tools.set_plane_all(planes[i])
        if hovered and scrolling:
            # This implements a form of quick-scroll when the user scrolls
            # over the miniature view in the navigator
            steps = max(1.0, ctx.volume.shape[i] / 25.0)
            ctx.mpr.scroll_by_axis(ctx.volume, planes[i], scrolling * steps)
        if i < 2:
            imgui.spacing()
    imgui.end()


def show_volume_stats(ctx):
    sf = imgui.get_io().font_global_scale
    imgui.set_next_window_size(250 * sf, 120 * sf)
    imgui.set_next_window_position(ctx.width - 250 * sf, 18 * sf)
    imgui.set_next_window_bg_alpha(0.8)

    flags = imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_TITLE_BAR
    _, ctx.settings.show_stats = imgui.begin("Volume statistics", closable=True, flags=flags)
    imgui.text("Volume statistics")
    imgui.spacing()
    imgui.indent(5)
    imgui.text("Dimensions (voxels): %d %d %d" % tuple(ctx.header["dimensions"]))
    imgui.text("Spacing (mm): %.2f %.2f %.2f" % tuple(ctx.header["spacing"]))
    imgui.text("Scalar type: %s" % str(ctx.volume.dtype))
    imgui.text("Mask type: %s" % str(ctx.mask.dtype))
    imgui.text("Segmented volume (ml): %.2f" % (ctx.segmented_volume_ml))
    imgui.unindent(5)
    if ctx.cmds.dirty and not ctx.tools.is_painting_any():
        print("Recalculating volume...")
        tic = time.time()
        ctx.segmented_volume_ml = np.sum(ctx.mask > 127) * np.prod(ctx.header["spacing"]) * 1e-3
        print("Done (elapsed time: %.2f s)" % (time.time() - tic))
        ctx.cmds.dirty = False
    imgui.end()


def show_input_guide(ctx):
    sf = imgui.get_io().font_global_scale
    imgui.set_next_window_size(250 * sf, 240 * sf)
    if ctx.settings.show_stats:
        imgui.set_next_window_position(ctx.width - 250 * sf, (18 + 120) * sf)
    else:
        imgui.set_next_window_position(ctx.width - 250 * sf, 18 * sf)
    imgui.set_next_window_bg_alpha(0.8)

    flags = imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_TITLE_BAR
    _, ctx.settings.show_input_guide = imgui.begin("Quick reference", closable=True, flags=flags)
    imgui.text("Quick reference")
    imgui.spacing()
    imgui.indent(5)
    imgui.text("Left mouse: Paint/draw/grab")
    imgui.text("Right mouse: Rotate view")
    imgui.text("Middle mouse: Pan view")
    imgui.text("Scroll: Zoom view")
    imgui.text("Shift+Scroll: Scroll slices")
    imgui.text("Key 1: Show axial view")
    imgui.text("Key 2: Show sagital view")
    imgui.text("Key 3: Show coronal view")
    imgui.text("Page up/down: Change label")
    imgui.text("Enter: Close polygon")
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
    _, ctx.label = imgui.combo("Label", ctx.label, ctx.classes)
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
    imgui.set_next_window_size(ctx.sidebar_width, ctx.height - 18 * sf)
    imgui.set_next_window_position(ctx.width, 18 * sf)
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
    if ctx.settings.show_navigator:
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
        ctx.label = max(ctx.label - 1, 0)
    if key == glfw.KEY_PAGE_DOWN and action == glfw.PRESS:
        ctx.label = min(ctx.label + 1, len(ctx.classes) - 1)
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
        if action == glfw.PRESS:
            ctx.cmds.pop_undo()


def mouse_button_callback(window, button, action, mods):
    ctx = glfw.get_window_user_pointer(window)
    if imgui.get_io().want_capture_mouse:
        ctx.imgui_glfw.mouse_callback(window, button, action, mods)
        return

    x, y = glfw.get_cursor_pos(window)
    if button == glfw.MOUSE_BUTTON_RIGHT:
        ctx.trackball.center = glm.vec2(x, y)
        # ctx.trackball.tracking = (action == glfw.PRESS)
    if button == glfw.MOUSE_BUTTON_MIDDLE:
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
        ctx.mpr.scroll_by_ray(ctx.volume, ray_dir, yoffset)
        # Cancel all drawing in case the user was drawing a polygon or livewire
        # on the MPR plane while scrolling
        ctx.tools.cancel_drawing_all()
    else:
        ctx.settings.fov_degrees += 2.0 * yoffset
        ctx.settings.fov_degrees = max(5.0, min(90.0, ctx.settings.fov_degrees))


def resize_callback(window, w, h):
    ctx = glfw.get_window_user_pointer(window)
    ctx.width = max(1, w - ctx.sidebar_width)
    ctx.height = max(1, h)
    gl.glViewport(0, 0, ctx.width, ctx.height)


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
    ctx.window = glfw.create_window(ctx.width, ctx.height, "ichseg", None, None)
    glfw.set_window_user_pointer(ctx.window, ctx)
    glfw.set_window_size_callback(ctx.window, resize_callback)
    glfw.set_char_callback(ctx.window, char_callback)
    glfw.set_key_callback(ctx.window, key_callback)
    glfw.set_mouse_button_callback(ctx.window, mouse_button_callback)
    glfw.set_cursor_pos_callback(ctx.window, cursor_pos_callback)
    glfw.set_scroll_callback(ctx.window, scroll_callback)
    glfw.make_context_current(ctx.window)
    print("GL version: %s" % gl.glGetString(gl.GL_VERSION).decode())

    # Initialize ImGui
    imgui.create_context()
    set_gui_style(ctx.settings.dark_mode)
    ctx.imgui_glfw = GlfwRenderer(ctx.window, False)

    # This should fix GUI-scaling for high-DPI screens:
    primary = glfw.get_primary_monitor()
    xscale, yscale = glfw.get_monitor_content_scale(primary)
    if xscale > 1.25:
        imgui.get_io().font_global_scale = xscale
        ctx.sidebar_width = int(ctx.sidebar_width * xscale)

    # Initialize application
    do_initialize(ctx)
    resize_callback(ctx.window, ctx.width, ctx.height)

    # Start main loop
    while not glfw.window_should_close(ctx.window):
        glfw.poll_events()
        ctx.imgui_glfw.process_inputs()
        imgui.new_frame()
        do_update(ctx)
        do_rendering(ctx)
        imgui.render()
        ctx.imgui_glfw.render(imgui.get_draw_data())
        glfw.swap_buffers(ctx.window)
    glfw.terminate()


if __name__ == "__main__":
    main()
