from tool_common import *
from tool_brush import BrushTool
from tool_polygon import PolygonTool
from tool_livewire import LivewireTool
from tool_smartbrush import SmartBrushTool
from tool_seedpaint import SeedPaintTool


class ToolManager:
    def __init__(self):
        self.brush = BrushTool()
        self.polygon = PolygonTool()
        self.livewire = LivewireTool()
        self.smartbrush = SmartBrushTool()
        self.seedpaint = SeedPaintTool()

    def disable_all_except(self, selected):
        """Disable all tools except the selected one (provided as reference)"""
        self.polygon.enabled = False
        self.brush.enabled = False
        self.livewire.enabled = False
        self.smartbrush.enabled = False
        self.seedpaint.enabled = False
        selected.enabled = True
        self.cancel_drawing_all()

    def cancel_drawing_all(self):
        """Cancel drawing for all tools"""
        self.polygon.points = []
        self.livewire.path = []
        self.livewire.points = []
        self.livewire.markers = []

    def set_plane_all(self, axis):
        """Set the active drawing plane for all tools

        This also cancels all drawing, to prevent the user from
        continue a polygon or livewire on another plane.
        """
        self.polygon.plane = axis
        self.brush.plane = axis
        self.livewire.plane = axis
        self.smartbrush.plane = axis
        self.seedpaint.plane = axis
        self.cancel_drawing_all()

    def is_painting_any(self):
        """Check if any of the brush tools are painting"""
        return self.brush.painting or self.smartbrush.painting
