"""
@author: Timon
@title: AnaglyphTool    
@nickname: Angly
@description: The comfyui plugin provides CUDA GPU accelerated creation of anagyph images from a color and depth image input.
"""

from .Anaglyphtool import AnaglyphTool

print(f"-- AnaglyphTool Plugin Loaded --")

node_list = [
    "Anaglyphtool",
]

NODE_CLASS_MAPPINGS = {
    "Anaglyphtool" : AnaglyphTool,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "Anaglyphtool" : "Anaglyphtool",
    }
