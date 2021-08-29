"""
.. module:: tool_seedpaint
   :platform: Linux, Windows
   :synopsis: Tool for seeded segmentation (not implemented yet)

.. moduleauthor:: Fredrik Nysjo
"""

from tool_common import *

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.csgraph
import scipy.ndimage
import glm


class SeedPaintTool:
    def __init__(self):
        self.enabled = False
        self.plane = TOOL_PLANE_Z
