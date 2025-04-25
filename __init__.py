"""
@author: Timon
@title: Stereo Tools (CUDA)
@nickname: StereoTools
@description: Provides CUDA GPU accelerated nodes for creating 3D images (Anaglyph, Cross-Eye, Stereogram).
"""

# Import the node classes and their specific mappings
from .Anaglyphtool import AnaglyphTool # Assumes class name is AnaglyphTool
from .CrossEyeTool import NODE_CLASS_MAPPINGS as CrossEye_MAPPINGS
from .CrossEyeTool import NODE_DISPLAY_NAME_MAPPINGS as CrossEye_DISPLAY_MAPPINGS
# Import the new StereogramTool
from .StereogramTool import StereogramTool # Assumes class name is StereogramTool from StereogramTool.py


print(f"-- Loading Stereo Tools Plugin (Anaglyph, CrossEye, Stereogram) --")

# Combine class mappings
NODE_CLASS_MAPPINGS = {
    "AnaglyphTool": AnaglyphTool,           # From Anaglyphtool.py
    **CrossEye_MAPPINGS,                   # From CrossEyeTool.py
    "StereogramTool": StereogramTool        # From StereogramTool.py
}

# Combine display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnaglyphTool": "Anaglyph Tool (CUDA)", # From Anaglyphtool.py
    **CrossEye_DISPLAY_MAPPINGS,           # From CrossEyeTool.py
    "StereogramTool": "Stereogram Tool (CUDA)" # From StereogramTool.py
}

# Export all mappings for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']