from gfx_shaders import *
from gfx_utils import *
from gfx_visualization import *
from image_segmentation import *
from image_utils import *
from image_dicom import *

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
import subprocess


class Settings:
    def __init__(self):
        self.bg_color1 = [0.9, 0.9, 0.9]
        self.bg_color2 = [0.0, 0.0, 0.0]
        self.show_mask = True
        self.fov_degrees = 45.0
        self.projection_mode = 1  # 0=orthographic; 1=perspective
        self.show_stats = False
        self.show_input_guide = False
        self.basepath = "/home/fredrik/Desktop/ct-ich-raw/Raw_ct_scans"


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
        self._prev_subimage = np.copy(self.volume[z:z+d,y:y+h,x:x+w])
        self.volume[z:z+d,y:y+h,x:x+w] = self.subimage
        if self.texture:
            update_subtexture_3d(self.texture, self.subimage, self.offset)
        return self

    def undo(self):
        x, y, z = self.offset
        d, h, w = self.subimage.shape
        self.volume[z:z+d,y:y+h,x:x+w] = self._prev_subimage
        if self.texture:
            update_subtexture_3d(self.texture, self._prev_subimage, self.offset)
        return self


class CommandManager:
    def __init__(self):
        self.stack = []
        self.max_undo_length = 8


class Context:
    def __init__(self):
        self.window = None
        self.width = 1000
        self.height = 700
        self.aspect = 1000.0 / 700.0
        self.sidebar_width = 290
        self.programs = {}
        self.vaos = {}
        self.buffers = {}
        self.textures = {}

        self.settings = Settings()
        self.trackball = Trackball()
        self.panning = Panning()
        self.mpr = MPR()
        self.tools = ToolManager()
        self.segmented_volume_ml = 0.0
        self.cmds = CommandManager()


def cmds_push_apply(cmds, cmd):
    """ Apply command and push it onto the manager's undo stack """
    if len(cmds.stack) >= cmds.max_undo_length:
        cmds.stack.pop(0)  # Make space on the stack
    cmds.stack.append(cmd.apply())


def cmds_pop_undo(cmds):
    """ Pop command from the manager's undo stack and undo it """
    if len(cmds.stack):
        cmds.stack.pop().undo()


def cmds_clear_stack(cmds):
    """ Clear all commands from the manager's undo stack """
    cmds.stack = []


def load_segmentation_mask(ct_volume, dirname):
    """ Load segmentation masks from JPG-files and into a mask volume of
    same size as the input CT volume
    """
    mask = np.zeros(ct_volume.shape, dtype=np.uint8)
    if os.path.exists(dirname):
        for filename in os.listdir(dirname):
            if not filename.lower().count(".jpg"):
                continue
            image = load_image(os.path.join(dirname, filename)).reshape(512, 512)
            image = image[::-1,:]  # Flip along Y-axis
            slice_pos = int(filename.split("_")[0]) - 1  # Indices in filenames seem to start at 1
            mask[slice_pos,:,:] = np.maximum(mask[slice_pos,:,:], image)
    return mask


def create_dummy_dataset():
    volume = np.zeros([1,1,1], dtype=np.int16)
    header = {"dimensions": (1, 1, 1), "spacing": (1, 1, 1)}
    return volume, header


def load_dataset_fromfile(ctx, filename) -> None:
    ctx.datasets = []
    ctx.current, ctx.label = 0, 0
    if ".vtk" in filename:
        ctx.volume, ctx.header = load_vtk(filename, True)
    elif filename:  # Assume format is DICOM
        ctx.volume, ctx.header = load_dicom(filename)
    else:
        ctx.volume, ctx.header = create_dummy_dataset()
    ctx.mask = np.zeros(ctx.volume.shape, dtype=np.uint8)
    cmds_clear_stack(ctx.cmds)


def load_mask_fromfile(ctx, filename) -> None:
    """ Load a segmentation mask from file

    If the grayscale volume we are segmenting does not match the size
    of the loaded mask, a new empty volume will be created

    Currently, only 8-bit masks in VTK format are supported
    """
    if ".vtk" not in filename:
        return
    mask, header = load_vtk(filename)
    if mask.dtype == np.uint8:
        if mask.shape != ctx.volume.shape:
            ctx.header = header
            ctx.volume = np.zeros(mask.shape, dtype=np.uint8)
        ctx.mask = mask
        cmds_clear_stack(ctx.cmds)


def do_initialize(ctx) -> None:
    """ Initialize the application state """
    tools = ctx.tools

    ctx.classes = ["Label 255", "Label 0 (Clear)"]

    load_dataset_fromfile(ctx, "")

    ctx.textures["volume"] = create_texture_3d(ctx.volume, filter_mode=gl.GL_LINEAR)
    ctx.textures["mask"] = create_texture_3d(ctx.mask, filter_mode=gl.GL_LINEAR)
    ctx.vaos["default"] = gl.glGenVertexArrays(1)
    ctx.programs["raycast"] = create_program((raycast_vs, gl.GL_VERTEX_SHADER), (raycast_fs, gl.GL_FRAGMENT_SHADER))
    ctx.programs["polygon"] = create_program((polygon_vs, gl.GL_VERTEX_SHADER), (polygon_fs, gl.GL_FRAGMENT_SHADER))
    ctx.programs["background"] = create_program((background_vs, gl.GL_VERTEX_SHADER), (background_fs, gl.GL_FRAGMENT_SHADER))
    ctx.vaos["polygon"], ctx.buffers["polygon"] = create_mesh_buffer(tools.polygon.points)
    ctx.vaos["markers"], ctx.buffers["markers"] = create_mesh_buffer(tools.polygon.points)
    ctx.vaos["empty"] = gl.glGenVertexArrays(1)


def do_rendering(ctx) -> None:
    """ Do rendering """
    tools = ctx.tools
    tool_op = TOOL_OP_ADD if ctx.label == 0 else TOOL_OP_SUBTRACT

    mpr_planes_snapped = snap_mpr_to_grid(ctx.volume, ctx.mpr.planes)
    level_range = ctx.mpr.level_range
    level_range_scaled = level_range  # Adjusted for normalized texture formats
    if ctx.volume.dtype == np.uint8:
        level_range_scaled = [v / 255.0 for v in level_range]  # Scale to range [0,1]
    if ctx.volume.dtype == np.int16:
        level_range_scaled = [v / 32767.0 for v in level_range]  # Scale to range [-1,1]
    filter_mode = gl.GL_NEAREST if ctx.mpr.show_voxels else gl.GL_LINEAR;

    proj = glm.perspective(glm.radians(ctx.settings.fov_degrees), ctx.aspect, 0.1, 10.0)
    if ctx.settings.projection_mode:
        k = glm.radians(ctx.settings.fov_degrees)
        proj = glm.ortho(-k * ctx.aspect, k * ctx.aspect, -k, k, 0.1, 10.0)
    view = glm.translate(glm.mat4(1.0), glm.vec3(0, 0, -2)) * glm.translate(glm.mat4(1.0), -ctx.panning.position) * glm.mat4_cast(ctx.trackball.quat)
    spacing = glm.vec3(ctx.header["spacing"])
    extent = glm.vec3(ctx.header["spacing"]) * glm.vec3(ctx.header["dimensions"])
    model = glm.scale(glm.mat4(1.0), extent / glm.max(extent.x, glm.max(extent.y, extent.z)))
    mv = view * model;
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
    gl.glUniform1i(gl.glGetUniformLocation(program, "u_projection_mode"), ctx.settings.projection_mode)
    gl.glUniform1i(gl.glGetUniformLocation(program, "u_show_mpr"), ctx.mpr.enabled)
    gl.glUniform3f(gl.glGetUniformLocation(program, "u_mpr_planes"), *mpr_planes_snapped)
    gl.glUniform2f(gl.glGetUniformLocation(program, "u_level_range"), *level_range_scaled)
    gl.glUniform3f(gl.glGetUniformLocation(program, "u_extent"), *extent)
    gl.glUniform4f(gl.glGetUniformLocation(program, "u_brush"), *tools.brush.position)
    if tools.smartbrush.enabled:
        gl.glUniform4f(gl.glGetUniformLocation(program, "u_brush"), *tools.smartbrush.position)
    gl.glUniform1i(gl.glGetUniformLocation(program, "u_volume"), 0)
    gl.glUniform1i(gl.glGetUniformLocation(program, "u_mask"), 1)
    gl.glBindVertexArray(ctx.vaos["default"])
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 14)
    gl.glBindVertexArray(0)

    if tools.polygon.enabled:
        program = ctx.programs["polygon"]
        gl.glUseProgram(program)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(program, "u_mvp"), 1, False, glm.value_ptr(mvp))
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
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(program, "u_mvp"), 1, False, glm.value_ptr(mvp))
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
        view_pos = reconstruct_view_pos(ndc_pos, proj)
        texcoord = glm.vec3(glm.inverse(mv) * glm.vec4(view_pos, 1.0)) + 0.5001
        tools.brush.position = glm.vec4(texcoord - 0.5, tools.brush.position.w)
        tools.smartbrush.position = glm.vec4(texcoord - 0.5, tools.smartbrush.position.w)

        if depth != 1.0 and tools.brush.painting and tools.brush.enabled:
            if tools.brush.count == 0:
                # Add full copy of current mask to command buffer to allow undo
                cmd = UpdateVolumeCmd(ctx.mask, np.copy(ctx.mask), (0, 0, 0), ctx.textures["mask"])
                cmds_push_apply(ctx.cmds, cmd)
            tools.brush.count += 1
            result = brush_tool_apply(tools.brush, ctx.mask, texcoord, spacing, tool_op)
            if result:
                update_subtexture_3d(ctx.textures["mask"], result[0], result[1])

        if depth != 1.0 and tools.smartbrush.painting and tools.smartbrush.enabled:
            if tools.smartbrush.count == 0:
                cmd = UpdateVolumeCmd(ctx.mask, np.copy(ctx.mask), (0, 0, 0), ctx.textures["mask"])
                cmds_push_apply(ctx.cmds, cmd)
            if tools.smartbrush.xy[0] != x or tools.smartbrush.xy[1] != y:
                tools.smartbrush.count += 1
                tools.smartbrush.momentum = min(5, tools.smartbrush.momentum + 2);
                tools.smartbrush.xy = (x, y)
            if tools.smartbrush.momentum > 0:
                result = smartbrush_tool_apply(
                    tools.smartbrush, ctx.mask, ctx.volume, texcoord, spacing, level_range, tool_op)
                if result:
                    update_subtexture_3d(ctx.textures["mask"], result[0], result[1])
                tools.smartbrush.momentum = max(0, tools.smartbrush.momentum - 1)

        if depth != 1.0 and tools.polygon.clicking and tools.polygon.enabled:
            tools.polygon.points.extend((texcoord.x - 0.5, texcoord.y - 0.5, texcoord.z - 0.5))
            update_mesh_buffer(ctx.buffers["polygon"], tools.polygon.points)
            update_mesh_buffer(ctx.buffers["markers"], tools.polygon.points)
            tools.polygon.clicking = False

        if depth != 1.0 and tools.livewire.enabled:
            clicking = False
            if tools.livewire.clicking:
                tools.livewire.markers.extend((texcoord.x - 0.5, texcoord.y - 0.5, texcoord.z - 0.5))
                update_mesh_buffer(ctx.buffers["markers"], tools.livewire.markers)
                livewire_tool_update_graph(tools.livewire, ctx.volume, texcoord, level_range)
                tools.livewire.clicking = False
                clicking = True
            if len(tools.livewire.path):
                livewire_tool_update_path(
                    tools.livewire, ctx.volume, texcoord, level_range, clicking)
                # Update polyline for preview
                update_mesh_buffer(ctx.buffers["polygon"], tools.livewire.points)


def do_update(ctx) -> None:
    """ Update application state """
    tools = ctx.tools
    tool_op = TOOL_OP_ADD if ctx.label == 0 else TOOL_OP_SUBTRACT

    tools.brush.position.w = 0.0 if not tools.brush.enabled else tools.brush.size * 0.5
    tools.smartbrush.position.w = 0.0 if not tools.smartbrush.enabled else tools.smartbrush.size * 0.5

    if tools.polygon.rasterise and len(tools.polygon.points):
        # Rasterise polygon into mask image
        result = polygon_tool_apply(tools.polygon, ctx.mask, tool_op)
        if result:
            cmd = UpdateVolumeCmd(ctx.mask, result[0], result[1], ctx.textures["mask"])
            cmds_push_apply(ctx.cmds, cmd)
        # Clean up for drawing next polygon
        tools_cancel_drawing_all(tools)
        tools.polygon.rasterise = False

    if tools.livewire.rasterise and len(tools.livewire.points):
        # Rasterise livewire into mask image
        result = livewire_tool_apply(tools.livewire, ctx.mask, tool_op)
        if result:
            cmd = UpdateVolumeCmd(ctx.mask, result[0], result[1], ctx.textures["mask"])
            cmds_push_apply(ctx.cmds, cmd)
        # Clean up for drawing next livewire
        tools_cancel_drawing_all(tools)
        tools.livewire.rasterise = False

    show_menubar(ctx)
    show_gui(ctx)


def show_file_selection() -> str:
    root = tk.Tk()
    root.withdraw()  # Hide Tk window
    filepath = tk.filedialog.askopenfilename(filetypes=[("Volume file", ".vtk .dcm")])
    return filepath


def show_save_selection() -> str:
    root = tk.Tk()
    root.withdraw()  # Hide Tk window
    filepath = tk.filedialog.asksaveasfilename(filetypes=[("VTK image", ".vtk")], defaultextension=".vtk")
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


def show_menubar(ctx) -> None:
    """ Show ImGui menu bar """
    imgui.begin_main_menu_bar()
    if imgui.begin_menu("File"):
        if imgui.menu_item("Open volume...")[0]:
            filename = show_file_selection()
            if filename:
                ctx.settings.basepath = filename
                load_dataset_fromfile(ctx, filename)
                update_texture_3d(ctx.textures["volume"], ctx.volume)
                update_texture_3d(ctx.textures["mask"], ctx.mask)
                tools_cancel_drawing_all(ctx.tools)
        if imgui.menu_item("Save volume...")[0]:
            filename = show_save_selection()
            if filename:
                save_vtk(filename, ctx.volume, ctx.header)
        if imgui.menu_item("Open segmentation...")[0]:
            filename = show_file_selection()
            if filename:
                load_mask_fromfile(ctx, filename)
                update_texture_3d(ctx.textures["volume"], ctx.volume)
                update_texture_3d(ctx.textures["mask"], ctx.mask)
                tools_cancel_drawing_all(ctx.tools)
        if imgui.menu_item("Save segmentation...")[0]:
            filename = show_save_selection()
            if filename:
                mask_header = copy.deepcopy(ctx.header)
                mask_header["format"] = "unsigned_char"
                save_vtk(filename, ctx.mask, mask_header)
        if imgui.menu_item("Quit")[0]:
            if show_quit_dialog():
                glfw.set_window_should_close(ctx.window, glfw.TRUE)
        imgui.end_menu()
    if imgui.begin_menu("Edit"):
        if imgui.menu_item("Undo (Ctrl+z)")[0]:
            cmds_pop_undo(ctx.cmds)
        if imgui.menu_item("Resample orientation...")[0]:
            if show_resample_orientation_dialog():
                ctx.volume, ctx.header = resample_volume(ctx.volume, ctx.header)
                ctx.mask = np.zeros(ctx.volume.shape, dtype=np.uint8)
                update_texture_3d(ctx.textures["volume"], ctx.volume)
                update_texture_3d(ctx.textures["mask"], ctx.mask)
                tools_cancel_drawing_all(ctx.tools)
                cmds_clear_stack(ctx.cmds)
        if imgui.menu_item("Clear segmentation")[0]:
            zeros = np.zeros(ctx.mask.shape, dtype=ctx.mask.dtype)
            cmd = UpdateVolumeCmd(ctx.mask, zeros, (0, 0, 0), ctx.textures["mask"])
            cmds_push_apply(ctx.cmds, cmd)
        imgui.end_menu()
    if imgui.begin_menu("Tools"):
        if imgui.menu_item("Volume statistics")[0]:
            ctx.settings.show_stats = not ctx.settings.show_stats
        imgui.end_menu()
    if imgui.begin_menu("Help"):
        if imgui.menu_item("Quick reference")[0]:
            ctx.settings.show_input_guide = not ctx.settings.show_input_guide
        imgui.end_menu()
    imgui.end_main_menu_bar()


def show_volume_stats(ctx) -> None:
    sf = imgui.get_io().font_global_scale
    imgui.set_next_window_size(250 * sf, 140 * sf)
    imgui.set_next_window_position(ctx.width - 260 * sf, 18 * sf)

    flags = (imgui.WINDOW_NO_RESIZE|imgui.WINDOW_NO_COLLAPSE)
    _, ctx.settings.show_stats = imgui.begin("Volume statistics", closable=True, flags=flags)
    imgui.text("Dimensions (voxels): %d %d %d" % tuple(ctx.header["dimensions"]))
    imgui.text("Spacing (mm): %.2f %.2f %.2f" % tuple(ctx.header["spacing"]))
    imgui.text("Scalar type: %s" % str(ctx.volume.dtype))
    imgui.text("Mask type: %s" % str(ctx.mask.dtype))
    imgui.text("Segmented volume (ml): %.2f" % (ctx.segmented_volume_ml))
    if imgui.button("Update"):
        ctx.segmented_volume_ml = np.sum(ctx.mask > 127) * np.prod(ctx.header["spacing"]) * 1e-3
    imgui.end()


def show_input_guide(ctx) -> None:
    sf = imgui.get_io().font_global_scale
    imgui.set_next_window_size(250 * sf, 240 * sf)
    imgui.set_next_window_position(10 * sf, 18 * sf)

    flags = (imgui.WINDOW_NO_RESIZE|imgui.WINDOW_NO_COLLAPSE)
    _, ctx.settings.show_input_guide = imgui.begin("Quick reference", closable=True, flags=flags)
    imgui.text("Left mouse: Paint/draw")
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
    imgui.end()


def show_gui(ctx) -> None:
    """ Show ImGui windows """
    tools = ctx.tools

    sf = imgui.get_io().font_global_scale
    imgui.set_next_window_size(ctx.sidebar_width, ctx.height - 18 * sf)
    imgui.set_next_window_position(ctx.width, 18 * sf)

    imgui.begin("Segmentation", flags=(imgui.WINDOW_NO_RESIZE|imgui.WINDOW_NO_COLLAPSE))
    _, ctx.label = imgui.combo("Label", ctx.label, ctx.classes)
    _, ctx.settings.show_mask = imgui.checkbox("Show segmentation", ctx.settings.show_mask)
    _, ctx.mpr.level_range = imgui.drag_int2("Level range", *ctx.mpr.level_range, 10, -1000, 3000)
    if imgui.collapsing_header("Tools", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        # Polygon tool settings
        clicked, tools.polygon.enabled = imgui.checkbox("Polygon tool", tools.polygon.enabled)
        if clicked and tools.polygon.enabled:
            tools_disable_all_except(tools, tools.polygon)
        if tools.polygon.enabled:
            imgui.indent(5)
            _, tools.polygon.antialiasing = imgui.checkbox("Antialiasing", tools.polygon.antialiasing)
            imgui.unindent(5)
        # Brush tool settings
        clicked, tools.brush.enabled = imgui.checkbox("Brush tool", tools.brush.enabled)
        if clicked and tools.brush.enabled:
            tools_disable_all_except(tools, tools.brush)
        if tools.brush.enabled:
            imgui.indent(5)
            _, tools.brush.mode = imgui.combo("Mode", tools.brush.mode, ["2D", "3D"])
            _, tools.brush.size = imgui.slider_int("Brush size", tools.brush.size, 1, 80)
            _, tools.brush.antialiasing = imgui.checkbox("Antialiasing", tools.brush.antialiasing)
            imgui.unindent(5)
        # Livewire tool settings
        clicked, tools.livewire.enabled = imgui.checkbox("Livewire tool", tools.livewire.enabled)
        if clicked and tools.livewire.enabled:
            tools_disable_all_except(tools, tools.livewire)
        if tools.livewire.enabled:
            imgui.indent(5)
            _, tools.livewire.smoothing = imgui.checkbox("Smoothing enabled", tools.livewire.smoothing)
            imgui.unindent(5)
        # Smart brush tool settings
        clicked, tools.smartbrush.enabled = imgui.checkbox("Smartbrush tool", tools.smartbrush.enabled)
        if clicked and tools.smartbrush.enabled:
            tools_disable_all_except(tools, tools.smartbrush)
        if tools.smartbrush.enabled:
            imgui.indent(5)
            _, tools.smartbrush.mode = imgui.combo("Mode", tools.smartbrush.mode, ["2D", "3D"])
            _, tools.smartbrush.size = imgui.slider_int("Brush size", tools.smartbrush.size, 1, 80)
            _, tools.smartbrush.sensitivity = imgui.slider_float(
                "Sensitivity", tools.smartbrush.sensitivity, 0.0, 10.0)
            _, tools.smartbrush.delta_scaling = imgui.slider_float(
                "Delta scale", tools.smartbrush.delta_scaling, 1.0, 5.0)
            imgui.unindent(5)
        # Seed painting tool settings
        clicked, tools.seedpaint.enabled = imgui.checkbox("Seed paint tool", tools.seedpaint.enabled)
        if clicked and tools.seedpaint.enabled:
            tools_disable_all_except(tools, tools.seedpaint)
        if tools.seedpaint.enabled:
            imgui.indent(5)
            imgui.text("Not implemented yet (TODO)")
            imgui.unindent(5)
    if imgui.collapsing_header("Misc")[0]:
        _, ctx.settings.bg_color1 = imgui.color_edit3("BG color 1", *ctx.settings.bg_color1)
        _, ctx.settings.bg_color2 = imgui.color_edit3("BG color 2", *ctx.settings.bg_color2)
        _, ctx.mpr.enabled = imgui.checkbox("Show MPR", ctx.mpr.enabled)
        _, ctx.mpr.show_voxels = imgui.checkbox("Show voxels", ctx.mpr.show_voxels)
    imgui.end()
    if ctx.settings.show_stats:
        show_volume_stats(ctx)
    if ctx.settings.show_input_guide:
        show_input_guide(ctx)


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

    tools = ctx.tools

    if key == glfw.KEY_LEFT_SHIFT:
        # Note: some keyboards will repeat action keys, so must check for that
        # case as well
        ctx.mpr.scrolling = (action == glfw.PRESS or action == glfw.REPEAT)
    if key == glfw.KEY_1:  # Show top-view
        ctx.trackball.quat = glm.quat(glm.radians(glm.vec3(0, 0, 0)))
        tools_set_plane_all(tools, MPR_PLANE_Z)
    if key == glfw.KEY_2:  # Show side-view
        ctx.trackball.quat = glm.quat(glm.radians(glm.vec3(-90, 90, 0)))
        tools_set_plane_all(tools, MPR_PLANE_X)
    if key == glfw.KEY_3:  # Show front-view
        ctx.trackball.quat = glm.quat(glm.radians(glm.vec3(-90, 180, 0)))
        tools_set_plane_all(tools, MPR_PLANE_Y)
    if key == glfw.KEY_PAGE_UP and action == glfw.PRESS:
        ctx.label = max(ctx.label - 1, 0)
    if key == glfw.KEY_PAGE_DOWN and action == glfw.PRESS:
        ctx.label = min(ctx.label + 1, len(ctx.classes) - 1)
    if key == glfw.KEY_SPACE and action == glfw.PRESS:
        ctx.settings.show_mask = False
    if key == glfw.KEY_SPACE and action == glfw.RELEASE:
        ctx.settings.show_mask = True
    if key == glfw.KEY_ENTER:  # Rasterise polygon or livewire
        tools.polygon.rasterise = (action == glfw.PRESS)
        tools.livewire.rasterise = (action == glfw.PRESS)
    if key == glfw.KEY_ESCAPE:  # Cancel polygon or livewire
        tools_cancel_drawing_all(tools)
    if key == glfw.KEY_Z and (mods & glfw.MOD_CONTROL):
        if action == glfw.PRESS:
            cmds_pop_undo(ctx.cmds)


def mouse_button_callback(window, button, action, mods):
    ctx = glfw.get_window_user_pointer(window)
    if imgui.get_io().want_capture_mouse:
        ctx.imgui_glfw.mouse_callback(window, button, action, mods)
        return

    tools = ctx.tools

    x, y = glfw.get_cursor_pos(window)
    if button == glfw.MOUSE_BUTTON_RIGHT:
        ctx.trackball.center = glm.vec2(x, y)
        #ctx.trackball.tracking = (action == glfw.PRESS)
    if button == glfw.MOUSE_BUTTON_MIDDLE:
        ctx.panning.center = glm.vec2(x, y)
        ctx.panning.panning = (action == glfw.PRESS)
    if button == glfw.MOUSE_BUTTON_LEFT:
        tools.brush.painting = (action == glfw.PRESS)
        tools.brush.count = 0
        tools.smartbrush.painting = (action == glfw.PRESS)
        tools.smartbrush.count = 0
        tools.polygon.clicking = (action == glfw.PRESS)
        tools.livewire.clicking = (action == glfw.PRESS)


def cursor_pos_callback(window, x, y):
    ctx = glfw.get_window_user_pointer(window)
    if imgui.get_io().want_capture_mouse:
        return

    ctx.trackball.move(x, y)
    ctx.panning.move(x, y)


def scroll_callback(window, x, y):
    ctx = glfw.get_window_user_pointer(window)
    if imgui.get_io().want_capture_mouse:
        return

    if ctx.mpr.scrolling:
        view_dir = glm.vec3(glm.inverse(glm.mat4_cast(ctx.trackball.quat))[2])
        for i in range(0, 3):
            if abs(view_dir[i]) == max(abs(view_dir.x), max(abs(view_dir.y), abs(view_dir.z))):
                # Scroll by axis and step size calculated from volume size
                delta = 1.0 / ctx.volume.shape[2 - i]
                step = glm.sign(view_dir[i]) * delta
                ctx.mpr.planes[i] = max(-0.4999, min(0.4999, ctx.mpr.planes[i] + step * y))
        # Prevent user from continuing, e.g., a polygon on another plane
        tools_cancel_drawing_all(ctx.tools)
    else:
        ctx.settings.fov_degrees = max(5.0, min(90.0, ctx.settings.fov_degrees + 2.0 * y))


def resize_callback(window, w, h):
    ctx = glfw.get_window_user_pointer(window)
    ctx.width = w - ctx.sidebar_width
    ctx.height = h
    ctx.aspect = float(ctx.width) / ctx.height
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
    ctx.window = glfw.create_window(ctx.width, ctx.height, "ICH segmentation", None, None)
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
    imgui.style_colors_light();  # Comment out for default dark theme
    ctx.imgui_glfw = GlfwRenderer(ctx.window, False)

    # This should fix GUI-scaling for high-DPI screens:
    primary = glfw.get_primary_monitor()
    xscale, yscale = glfw.get_monitor_content_scale(primary);
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
