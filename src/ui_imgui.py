"""
.. module:: ui_imgui
   :platform: Linux, Windows
   :synopsis: Helper functions for ImGui styling, etc.

.. moduleauthor:: Fredrik Nysjo
"""

import imgui


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
