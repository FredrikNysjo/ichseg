"""
.. module:: gfx_manager
   :platform: Linux, Windows
   :synopsis: Manager for handling OpenGL resources and other graphics-related things

.. moduleauthor:: Fredrik Nysjo
"""

import glm


class GfxManager:
    def __init__(self):
        self.window = None
        self.width = 1000
        self.height = 700
        self.sidebar_width = 290

        self.programs = {}
        self.vaos = {}
        self.buffers = {}
        self.textures = {}

        self.proj_from_view = glm.mat4(1.0)
        self.view_from_world = glm.mat4(1.0)
        self.world_from_local = glm.mat4(1.0)
