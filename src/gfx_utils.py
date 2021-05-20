import numpy as np
import OpenGL.GL as gl
import glm

import struct
from typing import List, Tuple, Dict


class Trackball:
    """Class for trackball interaction"""

    def __init__(self):
        self.center = glm.vec2(0.0)
        self.quat = glm.quat()
        self.speed = 0.004
        self.clamp = 100.0
        self.tracking = False

    def move(self, x, y) -> None:
        """Update trackball state from 2D mouse input"""
        if not self.tracking:
            return

        motion = glm.vec2(x, y) - self.center
        if abs(motion.x) < 1.0 and abs(motion.y) < 1.0:
            return
        theta = self.speed * glm.clamp(motion, -self.clamp, self.clamp)
        delta_x = glm.angleAxis(theta.x, glm.vec3(0.0, 1.0, 0.0))
        delta_y = glm.angleAxis(theta.y, glm.vec3(1.0, 0.0, 0.0))
        q = delta_y * delta_x * self.quat
        q = glm.normalize(q) if glm.length(q) > 0.0 else glm.quat()

        # Final quaternion should be in positive hemisphere
        self.quat = q if q.w >= 0.0 else -q
        self.center = glm.vec2(x, y)


class Panning:
    """Class for panning interaction"""

    def __init__(self):
        self.center = glm.vec2(0.0)
        self.position = glm.vec3(0.0)
        self.panning = False

    def move(self, x, y) -> None:
        """Update panning state from 2D mouse input"""
        if not self.panning:
            return

        self.position.x -= (x - self.center.x) * 0.001
        self.position.y += (y - self.center.y) * 0.001
        self.center = glm.vec2(x, y)


def create_shader(source, stage) -> int:
    """Compile GLSL shader from source string"""
    shader = gl.glCreateShader(stage)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)
    if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
        print(gl.glGetShaderInfoLog(shader))
    return shader


def create_program(*shaders) -> int:
    """Compile and link GLSL program from shader sources"""
    program = gl.glCreateProgram()
    for source, stage in shaders:
        shader = create_shader(source, stage)
        gl.glAttachShader(program, shader)
        gl.glDeleteShader(shader)  # Flag shader for deletion with program
    gl.glLinkProgram(program)
    if gl.glGetProgramiv(program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
        print(gl.glGetProgramInfoLog(program))
    return program


def create_mesh_buffer(mesh):
    """Create a vertex buffer + VAO for a mesh. This will also upload
    the mesh's vertex data to the buffer.
    """
    vao = gl.glGenVertexArrays(1)
    vbo = gl.glGenBuffers(1)
    gl.glBindVertexArray(vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, np.array(mesh, dtype=np.float32), gl.GL_STATIC_DRAW)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    gl.glBindVertexArray(0)
    return vao, vbo


def update_mesh_buffer(vbo, mesh):
    """Update vertex buffer data from mesh"""
    gl.glBindBuffer(gl.GL_COPY_WRITE_BUFFER, vbo)
    gl.glBufferData(gl.GL_COPY_WRITE_BUFFER, np.array(mesh, dtype=np.float32), gl.GL_STATIC_DRAW)


def create_texture_3d(image, filter_mode=gl.GL_LINEAR):
    """Create a 3D texture for a volume image. This will also upload
    the voxel data to the 3D texture.
    """
    d, h, w = image.shape[0:3]
    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_3D, texture)
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_BORDER)
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MIN_FILTER, filter_mode)
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAG_FILTER, filter_mode)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    if image.dtype == np.uint8:
        gl.glTexImage3D(
            gl.GL_TEXTURE_3D,
            0,
            gl.GL_R8,
            w,
            h,
            d,
            0,
            gl.GL_RED,
            gl.GL_UNSIGNED_BYTE,
            image,
        )
    elif image.dtype == np.int16:
        gl.glTexImage3D(
            gl.GL_TEXTURE_3D,
            0,
            gl.GL_R16_SNORM,
            w,
            h,
            d,
            0,
            gl.GL_RED,
            gl.GL_SHORT,
            image,
        )
    elif image.dtype == np.float32:
        gl.glTexImage3D(gl.GL_TEXTURE_3D, 0, gl.GL_R32F, w, h, d, 0, gl.GL_RED, gl.GL_FLOAT, image)
    else:
        assert False, "Image type not supported: " + image.dtype.name
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
    gl.glBindTexture(gl.GL_TEXTURE_3D, 0)
    return texture


def update_texture_3d(texture, image):
    """Update 3D texture data from volume image"""
    d, h, w = image.shape[0:3]
    gl.glBindTexture(gl.GL_TEXTURE_3D, texture)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    if image.dtype == np.uint8:
        gl.glTexImage3D(
            gl.GL_TEXTURE_3D,
            0,
            gl.GL_R8,
            w,
            h,
            d,
            0,
            gl.GL_RED,
            gl.GL_UNSIGNED_BYTE,
            image,
        )
    elif image.dtype == np.int16:
        gl.glTexImage3D(
            gl.GL_TEXTURE_3D,
            0,
            gl.GL_R16_SNORM,
            w,
            h,
            d,
            0,
            gl.GL_RED,
            gl.GL_SHORT,
            image,
        )
    elif image.dtype == np.float32:
        gl.glTexImage3D(gl.GL_TEXTURE_3D, 0, gl.GL_R32F, w, h, d, 0, gl.GL_RED, gl.GL_FLOAT, image)
    else:
        assert False, "Image type not supported: " + image.dtype.name
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
    gl.glBindTexture(gl.GL_TEXTURE_3D, 0)


def update_subtexture_3d(texture, subimage, offset):
    """Update 3D texture data from volume image, but only inside
    bounding box defined by subimage size and offset
    """
    x, y, z = offset
    d, h, w = subimage.shape
    gl.glBindTexture(gl.GL_TEXTURE_3D, texture)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    if subimage.dtype == np.uint8:
        gl.glTexSubImage3D(
            gl.GL_TEXTURE_3D,
            0,
            x,
            y,
            z,
            w,
            h,
            d,
            gl.GL_RED,
            gl.GL_UNSIGNED_BYTE,
            subimage,
        )
    elif subimage.dtype == np.int16:
        gl.glTexSubImage3D(gl.GL_TEXTURE_3D, 0, x, y, z, w, h, d, gl.GL_RED, gl.GL_SHORT, subimage)
    elif subimage.dtype == np.float32:
        gl.glTexSubImage3D(gl.GL_TEXTURE_3D, 0, x, y, z, w, h, d, gl.GL_RED, gl.GL_FLOAT, subimage)
    else:
        assert False, "Image type not supported: " + image.dtype.name
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
    gl.glBindTexture(gl.GL_TEXTURE_3D, 0)


def load_stl_binary(filename) -> List[float]:
    """Load mesh from binary format STL file"""
    mesh = []
    with open(filename, "rb") as stream:
        header = stream.read(80)
        (ntriangles,) = struct.unpack("i", stream.read(4))
        for i in range(ntriangles):
            mesh.extend(struct.unpack_from("fffffffff", stream.read(50), 12))
    return mesh


def reconstruct_view_pos(ndc_pos, proj) -> glm.vec3:
    """Reconstruct view-space position from NDC position and projection matrix"""
    view_pos = glm.inverse(proj) * glm.vec4(ndc_pos, 1.0)
    return glm.vec3(view_pos / view_pos.w)
