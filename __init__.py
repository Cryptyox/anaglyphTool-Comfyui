"""
@author: Timon
@title: Anaglyph Tool (CUDA)
@nickname: Angly
@description: Provides CUDA GPU accelerated nodes for creating 3D images.
"""

# Import the node classes and their specific mappings
from .Anaglyphtool import AnaglyphTool # Assumes class name is AnaglyphTool
from .CrossEyeTool import NODE_CLASS_MAPPINGS as CrossEye_MAPPINGS
from .CrossEyeTool import NODE_DISPLAY_NAME_MAPPINGS as CrossEye_DISPLAY_MAPPINGS

print(f"-- Loading Stereo Tools Plugin --") 

# Combine class mappings
NODE_CLASS_MAPPINGS = {
    "AnaglyphTool": AnaglyphTool,
    **CrossEye_MAPPINGS
}

# Combine display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnaglyphTool": "Anaglyph Tool (CUDA)",
    **CrossEye_DISPLAY_MAPPINGS
}

# Export all mappings for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']