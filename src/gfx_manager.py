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
