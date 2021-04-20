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

import os
import subprocess


class Settings:
    def __init__(self):
        self.bg_color1 = [0.9, 0.9, 0.9]
        self.bg_color2 = [0.0, 0.0, 0.0]
        self.show_mask = True
        self.fov_degrees = 45.0
        self.projection_mode = 1  # 0=orthographic; 1=perspective
        self.show_stats = False
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
        d, w, h = self.subimage.shape
        self._prev_subimage = np.copy(self.volume[z:z+d,y:y+h,x:x+w])
        self.volume[z:z+d,y:y+h,x:x+w] = self.subimage
        if self.texture:
            update_subtexture_3d(self.texture, self.subimage, self.offset)
        return self

    def undo(self):
        x, y, z = self.offset
        d, w, h = self.subimage.shape
        self.volume[z:z+d,y:y+h,x:x+w] = self._prev_subimage
        if self.texture:
            update_subtexture_3d(self.texture, self._prev_subimage, self.offset)
        return self


class Context:
    def __init__(self):
        self.window = None
        self.width = 1000
        self.height = 700
        self.aspect = 1000.0 / 700.0
        self.sidebar_width = 270
        self.programs = {}
        self.vaos = {}
        self.buffers = {}
        self.textures = {}

        self.settings = Settings()
        self.trackball = Trackball()
        self.panning = Panning()
        self.mpr = MPR()
        self.brush = BrushTool()
        self.polygon = PolygonTool()
        self.livewire = LivewireTool()
        self.smartbrush = SmartBrushTool()
        self.segmented_volume_ml = 0.0
        self.cmds = []


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


def load_datasets_from_dir(dirname):
    """ Extract list of datasets (VTK-images) from directory """
    datasets = []
    if os.path.exists(dirname):
        for filename in os.listdir(dirname):
            if filename.lower().count(".vtk"):
                datasets.append(filename)
    datasets.sort()
    return datasets


def create_dummy_dataset():
    volume = np.zeros([1,1,1], dtype=np.int16)
    header = {"dimensions": (1, 1, 1), "spacing": (1, 1, 1)}
    mask = np.zeros(volume.shape, dtype=np.uint8)
    return volume, header, mask


def load_dataset(basepath, vtk_filenames, current):
    if not os.path.exists(basepath) or not vtk_filenames:
        return create_dummy_dataset()
    vtk_filename = vtk_filenames[current]
    volume, header = load_vtk(os.path.join(basepath, "ct_scans/" + vtk_filename), normalize_scalars=True)
    mask = load_segmentation_mask(volume, os.path.join(basepath, "masks/" + vtk_filename.split(".")[0]))
    return volume, header, mask


def update_datasets(ctx) -> None:
    """ Update list of dataset names """
    ctx.datasets = load_datasets_from_dir(os.path.join(ctx.settings.basepath, "ct_scans"))
    ctx.current, ctx.label = 0, 0
    update_current_dataset(ctx)


def update_current_dataset(ctx) -> None:
    """ Update volume and segmentation mask for current dataset """
    ctx.volume, ctx.header, ctx.mask = load_dataset(ctx.settings.basepath, ctx.datasets, ctx.current)
    ctx.cmds = []  # Clear command buffer since it will be invalid for new volume


def load_dataset_fromfile(ctx, filename) -> None:
    ctx.datasets = []
    ctx.current, ctx.label = 0, 0
    if ".vtk" in filename:
        ctx.volume, ctx.header = load_vtk(filename, True)
    elif filename:  # Assume it is DICOM
        ctx.volume, ctx.header = load_dicom(filename, True)
    else:
        ctx.volume, ctx.header, ctx.mask = create_dummy_dataset()
    ctx.mask = np.zeros(ctx.volume.shape, dtype=np.uint8) 
    ctx.cmds = []


def do_initialize(ctx) -> None:
    """ Initialize the application state """
    ctx.classes = ["Any", "Intraventricular", "Intraparenchymal", "Subarachnoid", "Epidural", "Subdural"]

    load_dataset_fromfile(ctx, "")

    ctx.textures["volume"] = create_texture_3d(ctx.volume, filter_mode=gl.GL_LINEAR)
    ctx.textures["mask"] = create_texture_3d(ctx.mask, filter_mode=gl.GL_LINEAR)
    ctx.vaos["default"] = gl.glGenVertexArrays(1)
    ctx.programs["raycast"] = create_program((raycast_vs, gl.GL_VERTEX_SHADER), (raycast_fs, gl.GL_FRAGMENT_SHADER))
    ctx.programs["polygon"] = create_program((polygon_vs, gl.GL_VERTEX_SHADER), (polygon_fs, gl.GL_FRAGMENT_SHADER))
    ctx.programs["background"] = create_program((background_vs, gl.GL_VERTEX_SHADER), (background_fs, gl.GL_FRAGMENT_SHADER))
    ctx.vaos["polygon"], ctx.buffers["polygon"] = create_mesh_buffer(ctx.polygon.points)
    ctx.vaos["empty"] = gl.glGenVertexArrays(1)


def do_rendering(ctx) -> None:
    """ Do rendering """
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
    gl.glActiveTexture(gl.GL_TEXTURE0)
    gl.glBindTexture(gl.GL_TEXTURE_3D, ctx.textures["volume"])
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MIN_FILTER, filter_mode)
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAG_FILTER, filter_mode)
    gl.glUniformMatrix4fv(gl.glGetUniformLocation(program, "u_mvp"), 1, False, glm.value_ptr(mvp))
    gl.glUniformMatrix4fv(gl.glGetUniformLocation(program, "u_mv"), 1, False, glm.value_ptr(mv))
    gl.glUniform1i(gl.glGetUniformLocation(program, "u_label"), ctx.label)
    gl.glUniform1i(gl.glGetUniformLocation(program, "u_show_mask"), ctx.settings.show_mask)
    gl.glUniform1i(gl.glGetUniformLocation(program, "u_projection_mode"), ctx.settings.projection_mode)
    gl.glUniform1i(gl.glGetUniformLocation(program, "u_show_mpr"), ctx.mpr.enabled)
    gl.glUniform3f(gl.glGetUniformLocation(program, "u_mpr_planes"), *mpr_planes_snapped)
    gl.glUniform2f(gl.glGetUniformLocation(program, "u_level_range"), *level_range_scaled)
    gl.glUniform3f(gl.glGetUniformLocation(program, "u_extent"), *extent)
    gl.glUniform4f(gl.glGetUniformLocation(program, "u_brush"), *ctx.brush.position)
    if ctx.smartbrush.enabled:
        gl.glUniform4f(gl.glGetUniformLocation(program, "u_brush"), *ctx.smartbrush.position)
    gl.glUniform1i(gl.glGetUniformLocation(program, "u_volume"), 0)
    gl.glUniform1i(gl.glGetUniformLocation(program, "u_mask"), 1)
    gl.glBindVertexArray(ctx.vaos["default"])
    gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 14)
    gl.glBindVertexArray(0)

    if ctx.polygon.enabled:
        program = ctx.programs["polygon"]
        gl.glUseProgram(program)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(program, "u_mvp"), 1, False, glm.value_ptr(mvp))
        gl.glBindVertexArray(ctx.vaos["polygon"])
        gl.glDrawArrays(gl.GL_LINE_STRIP, 0, len(ctx.polygon.points) // 3)
        gl.glBindVertexArray(0)

    if ctx.livewire.enabled:
        program = ctx.programs["polygon"]
        gl.glUseProgram(program)
        gl.glUniformMatrix4fv(gl.glGetUniformLocation(program, "u_mvp"), 1, False, glm.value_ptr(mvp))
        gl.glBindVertexArray(ctx.vaos["polygon"])
        gl.glDrawArrays(gl.GL_LINE_STRIP, 0, len(ctx.livewire.points) // 3)
        gl.glBindVertexArray(0)

    if ctx.mpr.enabled:
        x, y = glfw.get_cursor_pos(ctx.window)
        depth = gl.glReadPixels(x, (ctx.height - 1) - y, 1, 1, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
        ndc_pos = glm.vec3(0.0)
        ndc_pos.x = (float(x) / ctx.width) * 2.0 - 1.0
        ndc_pos.y = -((float(y) / ctx.height) * 2.0 - 1.0)
        ndc_pos.z = depth[0][0] * 2.0 - 1.0
        view_pos = reconstruct_view_pos(ndc_pos, proj)
        texcoord = glm.vec3(glm.inverse(mv) * glm.vec4(view_pos, 1.0)) + 0.5001
        ctx.brush.position = glm.vec4(texcoord - 0.5, ctx.brush.position.w)
        ctx.smartbrush.position = glm.vec4(texcoord - 0.5, ctx.smartbrush.position.w)

        if depth != 1.0 and ctx.brush.painting and ctx.brush.enabled:
            if ctx.brush.count == 0:
                # Add full copy of current mask to command buffer to allow undo
                cmd = UpdateVolumeCmd(ctx.mask, np.copy(ctx.mask), (0, 0, 0), ctx.textures["mask"])
                ctx.cmds.append(cmd.apply())
            ctx.brush.count += 1
            result = apply_brush(ctx.mask, texcoord, ctx.brush, spacing)
            if result:
                update_subtexture_3d(ctx.textures["mask"], result[0], result[1])

        if depth != 1.0 and ctx.smartbrush.painting and ctx.smartbrush.enabled:
            if ctx.smartbrush.count == 0:
                cmd = UpdateVolumeCmd(ctx.mask, np.copy(ctx.mask), (0, 0, 0), ctx.textures["mask"])
                ctx.cmds.append(cmd.apply())
            if ctx.smartbrush.xy[0] != x or ctx.smartbrush.xy[1] != y:
                ctx.smartbrush.count += 1
                ctx.smartbrush.momentum = min(5, ctx.smartbrush.momentum + 2);
                ctx.smartbrush.xy = (x, y)
            if ctx.smartbrush.momentum > 0:
                result = apply_smartbrush(ctx.mask, ctx.volume, texcoord, ctx.smartbrush, spacing, level_range)
                if result:
                    update_subtexture_3d(ctx.textures["mask"], result[0], result[1])
                ctx.smartbrush.momentum = max(0, ctx.smartbrush.momentum - 1)

        if depth != 1.0 and ctx.polygon.clicking and ctx.polygon.enabled:
            ctx.polygon.points.extend((texcoord.x - 0.5, texcoord.y - 0.5, texcoord.z - 0.5))
            update_mesh_buffer(ctx.buffers["polygon"], ctx.polygon.points)
            ctx.polygon.clicking = False

        if depth != 1.0 and ctx.livewire.enabled:
            d, h, w = ctx.volume.shape
            seed = int(texcoord.y * h) * w + int(texcoord.x * w)
            clicking = False
            if ctx.livewire.clicking:
                if not len(ctx.livewire.path):
                    shift = level_range[0]
                    scale = 1.0 / max(1e-9, level_range[1] - level_range[0])
                    slice_ = ctx.volume[int(texcoord.z * d),:,:].astype(np.float32)
                    slice_normalized = np.maximum(0.0, np.minimum(1.0, (slice_ - shift) * scale))

                    ctx.livewire.graph = create_graph_from_image(slice_normalized)
                    update_edge_weights(ctx.livewire.graph, slice_normalized, 0.0, 1.0)
                    ctx.livewire.dist, ctx.livewire.pred = compute_dijkstra(ctx.livewire.graph, seed)
                    ctx.livewire.path.append(seed)
                ctx.livewire.clicking = False
                clicking = True
            if len(ctx.livewire.path):
                path = compute_shortest_path(ctx.livewire.pred, ctx.livewire.path[-1], seed)
                update_livewire(ctx.livewire, path, texcoord.z - 0.5, ctx.volume)
                if ctx.livewire.smoothing:
                    smooth_livewire(ctx.livewire)
                update_mesh_buffer(ctx.buffers["polygon"], ctx.livewire.points)
                if clicking:
                    ctx.livewire.dist, ctx.livewire.pred = compute_dijkstra(ctx.livewire.graph, seed)
                    ctx.livewire.path.extend(path)
                    ctx.livewire.path.append(seed)


def do_update(ctx) -> None:
    """ Update application state """
    ctx.brush.position.w = 0.0 if not ctx.brush.enabled else ctx.brush.size * 0.5
    ctx.smartbrush.position.w = 0.0 if not ctx.smartbrush.enabled else ctx.smartbrush.size * 0.5

    if ctx.polygon.rasterise and len(ctx.polygon.points):
        npoints = len(ctx.polygon.points) // 3
        polygon = np.zeros((npoints + 1, 2))
        for i in range(0, npoints):
            polygon[i, 0] = ctx.polygon.points[3 * i + 0] + 0.5
            polygon[i, 1] = ctx.polygon.points[3 * i + 1] + 0.5
        polygon[npoints, 0] = ctx.polygon.points[0] + 0.5
        polygon[npoints, 1] = ctx.polygon.points[1] + 0.5
        zoffset = int((ctx.polygon.points[2] + 0.5) * ctx.mask.shape[0])
        image = rasterise_polygon(polygon, ctx.mask, zoffset)
        image |= ctx.mask[zoffset,:,:]
        cmd = UpdateVolumeCmd(ctx.mask, image, (0, 0, zoffset), ctx.textures["mask"])
        ctx.cmds.append(cmd.apply())
        ctx.polygon.points = []
        ctx.polygon.rasterise = False

    if ctx.livewire.rasterise and len(ctx.livewire.points):
        npoints = len(ctx.livewire.points) // 3
        polygon = np.zeros((npoints + 1, 2))
        for i in range(0, npoints):
            polygon[i, 0] = ctx.livewire.points[3 * i + 0] + 0.5
            polygon[i, 1] = ctx.livewire.points[3 * i + 1] + 0.5
        polygon[npoints, 0] = ctx.livewire.points[0] + 0.5
        polygon[npoints, 1] = ctx.livewire.points[1] + 0.5
        zoffset = int((ctx.livewire.points[2] + 0.5) * ctx.mask.shape[0])
        image = rasterise_polygon(polygon, ctx.mask, zoffset)
        image |= ctx.mask[zoffset,:,:]
        cmd = UpdateVolumeCmd(ctx.mask, image, (0, 0, zoffset), ctx.textures["mask"])
        ctx.cmds.append(cmd.apply())
        ctx.livewire.path = []
        ctx.livewire.points = []
        ctx.livewire.rasterise = False

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


def show_menubar(ctx) -> None:
    """ Show ImGui menu bar """
    imgui.begin_main_menu_bar()
    if imgui.begin_menu("File"):
        if imgui.menu_item("Open volume file...")[0]:
            filename = show_file_selection()
            if filename:
                ctx.settings.basepath = filename
                load_dataset_fromfile(ctx, filename)
                update_texture_3d(ctx.textures["volume"], ctx.volume)
                update_texture_3d(ctx.textures["mask"], ctx.mask)
        if imgui.menu_item("Save segmentation...")[0]:
            filename = show_save_selection()
            if filename:
                save_vtk(filename, ctx.mask, ctx.header)
        if imgui.menu_item("Quit")[0]:
            glfw.set_window_should_close(ctx.window, glfw.TRUE)
        imgui.end_menu()
    if imgui.begin_menu("Edit"):
        if imgui.menu_item("Undo (Ctrl+z)")[0] and len(ctx.cmds):
            ctx.cmds.pop().undo()
        if imgui.menu_item("Clear segmentation")[0]:
            zeros = np.zeros(ctx.mask.shape, dtype=ctx.mask.dtype)
            cmd = UpdateVolumeCmd(ctx.mask, zeros, (0, 0, 0), ctx.textures["mask"])
            ctx.cmds.append(cmd.apply())
        imgui.end_menu()
    if imgui.begin_menu("Tools"):
        if imgui.menu_item("Volume statistics")[0]:
            ctx.settings.show_stats = not ctx.settings.show_stats
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


def disable_tools(ctx, selected_tool) -> None:
    """ Disable all tools except the selected one """
    ctx.polygon.enabled = False
    ctx.brush.enabled = False
    ctx.livewire.enabled = False
    ctx.smartbrush.enabled = False
    selected_tool.enabled = True


def show_gui(ctx) -> None:
    """ Show ImGui windows """
    sf = imgui.get_io().font_global_scale
    imgui.set_next_window_size(ctx.sidebar_width, ctx.height - 18 * sf)
    imgui.set_next_window_position(ctx.width, 18 * sf)

    imgui.begin("Segmentation", flags=(imgui.WINDOW_NO_RESIZE|imgui.WINDOW_NO_COLLAPSE))
    _, ctx.label = imgui.combo("Class", ctx.label, ctx.classes)
    _, ctx.settings.show_mask = imgui.checkbox("Show segmentation", ctx.settings.show_mask)
    _, ctx.mpr.show_voxels = imgui.checkbox("Show voxels", ctx.mpr.show_voxels)
    _, ctx.mpr.level_range = imgui.drag_int2("Level range", *ctx.mpr.level_range, 10, -1000, 3000)
    if imgui.collapsing_header("Tools", flags=imgui.TREE_NODE_DEFAULT_OPEN)[0]:
        clicked, ctx.polygon.enabled = imgui.checkbox("Polygon tool", ctx.polygon.enabled)
        if clicked and ctx.polygon.enabled:
            disable_tools(ctx, ctx.polygon)
        clicked, ctx.brush.enabled = imgui.checkbox("Brush tool", ctx.brush.enabled)
        if clicked and ctx.brush.enabled:
            disable_tools(ctx, ctx.brush)
        if ctx.brush.enabled:
            _, ctx.brush.size = imgui.slider_int("Brush size", ctx.brush.size, 1, 80)
        clicked, ctx.livewire.enabled = imgui.checkbox("Livewire tool", ctx.livewire.enabled)
        if clicked and ctx.livewire.enabled:
            disable_tools(ctx, ctx.livewire)
        clicked, ctx.smartbrush.enabled = imgui.checkbox("Smartbrush tool", ctx.smartbrush.enabled)
        if clicked and ctx.smartbrush.enabled:
            disable_tools(ctx, ctx.smartbrush)
        if ctx.smartbrush.enabled:
            _, ctx.smartbrush.size = imgui.slider_int("Brush size", ctx.smartbrush.size, 1, 80)
            _, ctx.smartbrush.sensitivity = imgui.slider_float("Sensitivity", ctx.smartbrush.sensitivity, 0.0, 10.0)
            _, ctx.smartbrush.delta_scaling = imgui.slider_float("Delta scaling", ctx.smartbrush.delta_scaling, 1.0, 5.0)
    if imgui.collapsing_header("Misc")[0]:
        _, ctx.settings.bg_color1 = imgui.color_edit3("BG color 1", *ctx.settings.bg_color1)
        _, ctx.settings.bg_color2 = imgui.color_edit3("BG color 2", *ctx.settings.bg_color2)
        _, ctx.mpr.enabled = imgui.checkbox("Show MPR", ctx.mpr.enabled)
    imgui.end()
    if ctx.settings.show_stats:
        show_volume_stats(ctx)


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
        ctx.mpr.scrolling = (action == glfw.PRESS or action == glfw.REPEAT)
    if key == glfw.KEY_1:  # Show top-view
        ctx.trackball.quat = glm.quat(glm.vec3(0, 0, 0))
    if key == glfw.KEY_2:  # Show side-view
        ctx.trackball.quat = glm.quat(glm.vec3(glm.radians(-90), glm.radians(90), 0))
    if key == glfw.KEY_3:  # Show front-view
        ctx.trackball.quat = glm.quat(glm.vec3(glm.radians(-90), glm.radians(180), 0))
    if key == glfw.KEY_S and (action == glfw.PRESS):
        ctx.livewire.smoothing = not ctx.livewire.smoothing
    if key == glfw.KEY_SPACE and action == glfw.PRESS:
        ctx.settings.show_mask = False
    if key == glfw.KEY_SPACE and action == glfw.RELEASE:
        ctx.settings.show_mask = True
    if key == glfw.KEY_ENTER:  # Rasterise polygon or livewire
        ctx.polygon.rasterise = (action == glfw.PRESS)
        ctx.livewire.rasterise = (action == glfw.PRESS)
    if key == glfw.KEY_ESCAPE:  # Cancel polygon or livewire
        ctx.polygon.points = []
        ctx.livewire.path = []
        ctx.livewire.points = []
    if key == glfw.KEY_Z and (mods & glfw.MOD_CONTROL):
        if action == glfw.PRESS and len(ctx.cmds):
            ctx.cmds.pop().undo()


def mouse_button_callback(window, button, action, mods):
    ctx = glfw.get_window_user_pointer(window)
    if imgui.get_io().want_capture_mouse:
        ctx.imgui_glfw.mouse_callback(window, button, action, mods)
        return

    x, y = glfw.get_cursor_pos(window)
    if button == glfw.MOUSE_BUTTON_RIGHT:
        ctx.trackball.center = glm.vec2(x, y)
        #ctx.trackball.tracking = (action == glfw.PRESS)
    if button == glfw.MOUSE_BUTTON_MIDDLE:
        ctx.panning.center = glm.vec2(x, y)
        ctx.panning.panning = (action == glfw.PRESS)
    if button == glfw.MOUSE_BUTTON_LEFT:
        ctx.brush.painting = (action == glfw.PRESS)
        ctx.brush.count = 0
        ctx.smartbrush.painting = (action == glfw.PRESS)
        ctx.smartbrush.count = 0
        ctx.polygon.clicking = (action == glfw.PRESS)
        ctx.livewire.clicking = (action == glfw.PRESS)


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
                step = glm.sign(view_dir[i]) * 1e-2;
                ctx.mpr.planes[i] = max(-0.4999, min(0.4999, ctx.mpr.planes[i] + step * y))
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
