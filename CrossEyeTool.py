# Filename: CrossEyeTool.py
import torch
import torch.nn.functional as F
import time

class CrossEyeTool:
    """
    Generates a side-by-side 3D image suitable for cross-eye viewing
    from an input image and depth map, utilizing GPU acceleration.
    The output image places the right-eye view on the left half
    and the left-eye view on the right half.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),        # Expecting a batch of images [B, H, W, C]
                "depthmap": ("IMAGE",),     # Expecting a batch of depth maps [B, H, W, C] or [B, H, W]
                "invert_depthmap": ("BOOLEAN", {"default": True}),
                "divergence": ("FLOAT", {"default": 2.0, "min": -10.0, "max": 10.0, "step": 0.1}), # Controls the separation distance
                "zero_parallax_depth": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}), # Depth value (0-1) that should have zero shift
            },
        }

    RETURN_TYPES = ("IMAGE",) # Output is a single, wider image

    FUNCTION = "create_crosseye_batch"
    CATEGORY = "ImageProcessing/GPU/Stereo"

    def warp_image_batch(self, img_bchw, shift_map_bhw, target_device):
        B, C, H, W = img_bchw.shape
        # Create base grid on the target_device
        yy, xx = torch.meshgrid(torch.arange(H, device=target_device, dtype=torch.float32),
                                torch.arange(W, device=target_device, dtype=torch.float32),
                                indexing='ij')
        # Add batch-specific shift
        new_x = xx.unsqueeze(0) + shift_map_bhw # Shape becomes (B, H, W)
        yy_b = yy.unsqueeze(0).expand_as(new_x) # Shape becomes (B, H, W)

        # Normalize coordinates for grid_sample: range [-1, 1]
        norm_x = (2.0 * new_x / (W - 1)) - 1.0 if W > 1 else torch.zeros_like(new_x)
        norm_y = (2.0 * yy_b / (H - 1)) - 1.0 if H > 1 else torch.zeros_like(yy_b)

        # Create sampling grid (B, H, W, 2) with (x, y) coordinates
        grid = torch.stack((norm_x, norm_y), dim=-1).to(dtype=torch.float32)

        # Perform batch warping using bilinear interpolation
        warped_bchw = F.grid_sample(
            img_bchw.to(dtype=torch.float32), grid, mode='bilinear', padding_mode='zeros', align_corners=False
        )
        return warped_bchw

    def create_crosseye_batch(self, image: torch.Tensor, depthmap: torch.Tensor, invert_depthmap, divergence, zero_parallax_depth):
        start_time = time.time()

        # --- GPU/Device Handling ---
        if torch.cuda.is_available():
            target_device = torch.device("cuda")
        else:
            print("Warning: CUDA not available, falling back to CPU.")
            target_device = torch.device("cpu")
        print(f"[CrossEyeTool] Target device forced to: {target_device}")

        # Move input tensors to the target device
        try:
            image = image.to(target_device)
            depthmap = depthmap.to(target_device)
        except Exception as e:
            print(f"Error moving inputs to {target_device}: {e}")
            target_device = image.device # Fallback to original device if move fails
            print(f"Warning: Failed to move tensors to CUDA. Processing on original device: {target_device}")

        # --- Input Validation and Preparation ---
        if image.shape[0] != depthmap.shape[0]:
             print(f"Error: Image batch size ({image.shape[0]}) and Depthmap batch size ({depthmap.shape[0]}) do not match.")
             # Return original image batch (potentially moved to target_device)
             return (image.permute(0, 2, 3, 1).contiguous(),) # Ensure BHWC output

        img_dtype = image.dtype # Store original dtype if needed later
        # Permute image to BCHW format for PyTorch operations, convert to float32 for processing
        img_bchw = image.permute(0, 3, 1, 2).to(dtype=torch.float32)
        B, C, H, W = img_bchw.shape

        # Process depth map: Ensure it's [B, H, W] and float32
        depthmap = depthmap.to(dtype=torch.float32, device=target_device) # Ensure on target device
        if depthmap.ndim == 4: # Input is BHWC
            if depthmap.shape[3] == 1: # Grayscale depth map (B, H, W, 1) -> (B, H, W)
                 depth_bhw = depthmap.squeeze(-1)
            elif depthmap.shape[3] >= 3: # Color depth map (B, H, W, C) -> Average RGB -> (B, H, W)
                 depth_bhw = torch.mean(depthmap[..., :3], dim=3)
            else:
                 print(f"Error: Unexpected depth channel count: {depthmap.shape[3]}")
                 return (image.permute(0, 2, 3, 1).contiguous(),)
        elif depthmap.ndim == 3: # Input is BHW
            depth_bhw = depthmap
        else:
             print(f"Error: Unexpected depth map dimensions: {depthmap.shape}")
             return (image.permute(0, 2, 3, 1).contiguous(),)

        # Validate dimensions match between image and processed depth map
        if depth_bhw.shape[1:] != (H, W):
             print(f"Error: Image ({H}x{W}) and Depthmap ({depth_bhw.shape[1:]}) dimensions mismatch after processing.")
             return (image.permute(0, 2, 3, 1).contiguous(),)

        # --- Depth Map Normalization ---
        # Invert if requested (typically needed as depth maps often store near=0, far=1)
        processed_depth = 1.0 - depth_bhw if invert_depthmap else depth_bhw
        # Normalize depth map per image in the batch to range [0, 1]
        depth_min = torch.amin(processed_depth, dim=(1, 2), keepdim=True)
        depth_max = torch.amax(processed_depth, dim=(1, 2), keepdim=True)
        depth_range = depth_max - depth_min
        epsilon = 1e-6 # Avoid division by zero if depth range is tiny
        # Use safe division: if range is near zero, set normalized depth to 0.5 to avoid NaNs
        depth_range_safe = torch.where(depth_range < epsilon, torch.ones_like(depth_range), depth_range)
        depth_normalized = (processed_depth - depth_min) / depth_range_safe
        depth_normalized = torch.where(depth_range < epsilon, torch.full_like(depth_normalized, 0.5), depth_normalized)

        # --- Shift Calculation ---
        divergence_val = float(divergence)
        zero_parallax_depth_val = float(zero_parallax_depth)
        # Calculate maximum pixel shift based on divergence percentage and image width
        max_shift_pixels = (divergence_val / 100.0) * W / 2.0

        # Calculate shift relative to the zero parallax plane
        relative_depth_shift = depth_normalized - zero_parallax_depth_val
        # Left eye shifts left for objects behind zero plane, right for objects in front
        shift_left = -relative_depth_shift * max_shift_pixels
        # Right eye shifts right for objects behind zero plane, left for objects in front
        shift_right = relative_depth_shift * max_shift_pixels

        # --- Image Warping ---
        # Warp the original image batch to create left and right eye views
        left_eye_bchw = self.warp_image_batch(img_bchw, shift_left, target_device)
        right_eye_bchw = self.warp_image_batch(img_bchw, shift_right, target_device)

        # --- Combine Views for Cross-Eye Side-by-Side ---
        # Concatenate along the width dimension (dim=3 in BCHW format)
        # Right eye view on the left, Left eye view on the right
        crosseye_bchw = torch.cat((right_eye_bchw, left_eye_bchw), dim=3)
        # Clamp pixel values to the valid [0, 1] range
        crosseye_bchw = crosseye_bchw.clamp(0, 1)

        # --- Final Output Preparation ---
        # Permute back to BHWC format for ComfyUI compatibility
        # The width dimension (now dim=2 after permute) is 2*W
        output_batch_bhwc = crosseye_bchw.permute(0, 2, 3, 1).contiguous()

        # Optionally convert back to original image dtype if necessary
        # output_batch_bhwc = output_batch_bhwc.to(dtype=img_dtype)

        print(f"[CrossEyeTool] Processed batch of {B} frames on {target_device}. Output shape: {output_batch_bhwc.shape}. Total time: {time.time() - start_time:.3f}s")
        # Return the batch as a tuple (ComfyUI standard)
        return (output_batch_bhwc,)

# --- MAPPINGS for ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "CrossEyeTool": CrossEyeTool
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CrossEyeTool": "Cross-Eye 3D Tool (CUDA)"
}