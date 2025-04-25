import torch
import torch.nn.functional as F
import time

class AnaglyphTool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),        # Expecting a batch of images [B, H, W, C]
                "depthmap": ("IMAGE",),     # Expecting a batch of depth maps [B, H, W, C] or [B, H, W]
                "invert_depthmap": ("BOOLEAN", {"default": True}),
                "divergence": ("FLOAT", {"default": 2.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "zero_parallax_depth": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "create_anaglyph_batch"
    CATEGORY = "ImageProcessing/GPU"

    def warp_image_batch(self, img_bchw, shift_map_bhw, target_device): # Changed 'device' to 'target_device' for clarity
        B, C, H, W = img_bchw.shape
        # Create base grid on the target_device
        yy, xx = torch.meshgrid(torch.arange(H, device=target_device, dtype=torch.float32),
                                torch.arange(W, device=target_device, dtype=torch.float32),
                                indexing='ij')
        # Add batch-specific shift
        new_x = xx.unsqueeze(0) + shift_map_bhw # Shape becomes (B, H, W)
        yy_b = yy.unsqueeze(0).expand_as(new_x) # Shape becomes (B, H, W)

        # Normalize coordinates
        norm_x = (2.0 * new_x / (W - 1)) - 1.0 if W > 1 else torch.zeros_like(new_x)
        norm_y = (2.0 * yy_b / (H - 1)) - 1.0 if H > 1 else torch.zeros_like(yy_b)

        # Create grid (B, H, W, 2)
        grid = torch.stack((norm_x, norm_y), dim=-1).to(dtype=torch.float32)

        # Perform batch warping (inputs img_bchw and grid are already on target_device)
        warped_bchw = F.grid_sample(
            img_bchw.to(dtype=torch.float32), grid, mode='bilinear', padding_mode='zeros', align_corners=False
        )
        return warped_bchw

    # Fully vectorized batch processing function
    def create_anaglyph_batch(self, image: torch.Tensor, depthmap: torch.Tensor, invert_depthmap, divergence, zero_parallax_depth):
        start_time = time.time()

        # Force Target Device to CUDA if available
        if torch.cuda.is_available():
            # Explicitly set the target device to the default CUDA device - torch defaults to CPU from most depth nodes 
            target_device = torch.device("cuda")
        else:
            # Fallback to CPU if CUDA is not available - had issues previously - currently not under developement
            print("Warning: CUDA not available, falling back to CPU.")
            target_device = torch.device("cpu")

        print(f"[AnaglyphTool] Target device forced to: {target_device}")
        print(f"[AnaglyphTool Debug] Input image device BEFORE move: {image.device}")
        print(f"[AnaglyphTool Debug] Input depthmap device BEFORE move: {depthmap.device}")

        # Move input tensors to the target device
        try:
            image = image.to(target_device)
            depthmap = depthmap.to(target_device)
            print(f"[AnaglyphTool Debug] Input image device AFTER move: {image.device}")
            print(f"[AnaglyphTool Debug] Input depthmap device AFTER move: {depthmap.device}")
        except Exception as e:
            print(f"Error moving inputs to {target_device}: {e}")
            # Fallback: try to process on the original device if move failed - no handling for out of VRAM, I expect the user to set the Batch size manually according to their VRAM
            target_device = image.device # Revert target_device to original input device
            print(f"Warning: Failed to move tensors to CUDA. Processing on original device: {target_device}") # This happens if out of VRAM or copy error
            # No need to move depthmap again if image.device was the original target

        # Basic Input Validation (on target_device)
        if image.shape[0] != depthmap.shape[0]:
             print(f"Error: Image batch size ({image.shape[0]}) and Depthmap batch size ({depthmap.shape[0]}) do not match.")
             # Return original image batch (now potentially on target_device)
             return (image.permute(0, 2, 3, 1).contiguous(),) # Ensure BHWC output if permuted earlier

        # Use float32 for internal processing
        img_dtype = image.dtype

        # Ensure correct shapes and types (already on target_device)
        img_bchw = image.permute(0, 3, 1, 2).to(dtype=torch.float32)
        B, C, H, W = img_bchw.shape

        depthmap = depthmap.to(dtype=torch.float32)
        if depthmap.ndim == 4: # BHWC
            if depthmap.shape[3] == 1: # Grayscale (B, H, W, 1) -> (B, H, W)
                 depth_bhw = depthmap.squeeze(-1)
            elif depthmap.shape[3] >= 3: # Color (B, H, W, C) -> Average RGB -> (B, H, W)
                 depth_bhw = torch.mean(depthmap[..., :3], dim=3)
            else:
                 print(f"Error: Unexpected depth channel count: {depthmap.shape[3]}")
                 return (image.permute(0, 2, 3, 1).contiguous(),)
        elif depthmap.ndim == 3: # BHW
            depth_bhw = depthmap
        else:
             print(f"Error: Unexpected depth map dimensions: {depthmap.shape}")
             return (image.permute(0, 2, 3, 1).contiguous(),)

        if depth_bhw.shape[1:] != (H, W):
             print(f"Error: Image ({H}x{W}) and Depthmap ({depth_bhw.shape[1:]}) dimensions mismatch.") # Some Video inputs have wierd scaling issues, has been solved by a change I did but don't really know why, shouldn't really occur anymore
             return (image.permute(0, 2, 3, 1).contiguous(),)

        # Batch Depth Processing (on target_device)
        processed_depth = 1.0 - depth_bhw if invert_depthmap else depth_bhw
        depth_min = torch.amin(processed_depth, dim=(1, 2), keepdim=True)
        depth_max = torch.amax(processed_depth, dim=(1, 2), keepdim=True)
        depth_range = depth_max - depth_min
        epsilon = 1e-6
        depth_range_safe = torch.where(depth_range < epsilon, torch.ones_like(depth_range), depth_range)
        depth_normalized = (processed_depth - depth_min) / depth_range_safe
        depth_normalized = torch.where(depth_range < epsilon, torch.full_like(depth_normalized, 0.5), depth_normalized)

        # Parameter Conversion
        divergence_val = float(divergence)
        zero_parallax_depth_val = float(zero_parallax_depth)
        max_shift_pixels = (divergence_val / 100.0) * W / 2.0

        # Generate Left/Right Shift Maps (on target_device)
        relative_depth_shift = depth_normalized - zero_parallax_depth_val
        shift_left = -relative_depth_shift * max_shift_pixels
        shift_right = relative_depth_shift * max_shift_pixels

        # Warp Image Batch (on target_device)
        # Pass the explicitly defined target_device to the warp function - had it fall back to CPU in the past if not explicit
        left_eye_bchw = self.warp_image_batch(img_bchw, shift_left, target_device)
        right_eye_bchw = self.warp_image_batch(img_bchw, shift_right, target_device)

        # Combine Views (on target_device)
        anaglyph_bchw = torch.zeros_like(img_bchw, dtype=torch.float32)
        anaglyph_bchw[:, 0, :, :] = right_eye_bchw[:, 0, :, :]
        anaglyph_bchw[:, 1, :, :] = left_eye_bchw[:, 1, :, :]
        anaglyph_bchw[:, 2, :, :] = left_eye_bchw[:, 2, :, :]
        anaglyph_bchw = anaglyph_bchw.clamp(0, 1)

        # Convert final batch back to HWC
        # Output tensor will be on target_device (GPU)
        output_batch_bhwc = anaglyph_bchw.permute(0, 2, 3, 1) # .to(dtype=img_dtype)

        print(f"[AnaglyphTool] Processed batch of {B} frames on {target_device}. Total time: {time.time() - start_time:.3f}s")
        return (output_batch_bhwc,)

# --- MAPPINGS ---
NODE_CLASS_MAPPINGS = {
    "AnaglyphTool": AnaglyphTool
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnaglyphTool": "Anaglyph Tool (CUDA)"
}