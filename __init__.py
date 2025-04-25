"""
@author: Timon
@title: AnaglyphTool    
@nickname: Angly
@description: The comfyui plugin provides CUDA GPU accelerated creation of anagyph images from a color and depth image input.
"""

from .Anaglyphtool import Anaglyphtool

print(f"-- AnaglyphTool Plugin Loaded --")

node_list = [
    "Anaglyphtool",
]

NODE_CLASS_MAPPINGS = {
    "Anaglyphtool" : Anaglyphtool,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "Anaglyphtool" : "Anaglyphtool",
    }
