import torch
import torch.nn.functional as F
import time

class StereogramTool:
    """
    Generates an autostereogram (SIS - Single Image Stereogram)
    from a depth map and a pattern image using GPU acceleration.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depthmap": ("IMAGE",),         # Expecting a batch of depth maps [B, H, W, C] or [B, H, W]
                "pattern_image": ("IMAGE",),    # Expecting a batch of pattern images [B, H, W, C]
                "invert_depthmap": ("BOOLEAN", {"default": True}),
                "divergence_factor": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05}), # Controls perceived depth strength
                "eye_separation_percent": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 25.0, "step": 0.5}), # Simulated distance between eyes as % of width
                "zero_parallax_depth": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}), # Depth value (0-1, after normalization) that aligns directly with the pattern
                "pattern_tile_factor": ("INT", {"default": 1, "min": 1, "max": 16}), # How many times to tile the pattern horizontally before use
            },
             "optional": {
                 # Optional color image input - if provided, colors the pattern based on the image
                 "color_image": ("IMAGE", {"default": None}),
             }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_stereogram_batch"
    CATEGORY = "ImageProcessing/GPU/Stereo"

    def normalize_depth_batch(self, depth_bhw, invert_depthmap):
        """Normalizes depth map per image in the batch to [0, 1]."""
        processed_depth = 1.0 - depth_bhw if invert_depthmap else depth_bhw
        depth_min = torch.amin(processed_depth, dim=(1, 2), keepdim=True)
        depth_max = torch.amax(processed_depth, dim=(1, 2), keepdim=True)
        depth_range = depth_max - depth_min
        epsilon = 1e-6
        depth_range_safe = torch.where(depth_range < epsilon, torch.ones_like(depth_range), depth_range)
        depth_normalized = (processed_depth - depth_min) / depth_range_safe
        # Handle flat depth maps: set normalized depth to 0.5
        depth_normalized = torch.where(depth_range < epsilon, torch.full_like(depth_normalized, 0.5), depth_normalized)
        return depth_normalized # Shape (B, H, W)

    def create_stereogram_batch(self, depthmap: torch.Tensor, pattern_image: torch.Tensor, invert_depthmap: bool, divergence_factor: float, eye_separation_percent: float, zero_parallax_depth: float, pattern_tile_factor: int, color_image: torch.Tensor | None = None):
        start_time = time.time()

        # --- GPU/Device Handling ---
        if torch.cuda.is_available():
            target_device = torch.device("cuda")
        else:
            print("Warning: CUDA not available, falling back to CPU.")
            target_device = torch.device("cpu")
        print(f"[StereogramTool] Target device: {target_device}")

        # --- Input Preparation ---
        try:
            depthmap = depthmap.to(target_device)
            pattern_image = pattern_image.to(target_device)
            if color_image is not None:
                color_image = color_image.to(target_device)
        except Exception as e:
            print(f"Error moving inputs to {target_device}: {e}")
            # Fallback is difficult here as we need consistent device placement.
            # If initial move fails, we probably can't proceed reliably.
            return (torch.zeros_like(depthmap),) # Return empty/black image on error

        # -- Depth Map Processing --
        # Convert depthmap to BHW float32
        depthmap = depthmap.to(dtype=torch.float32)
        if depthmap.ndim == 4: # BHWC
            if depthmap.shape[3] == 1:
                 depth_bhw = depthmap.squeeze(-1)
            elif depthmap.shape[3] >= 3:
                 depth_bhw = torch.mean(depthmap[..., :3], dim=3) # Average color channels if needed
            else:
                 print(f"Error: Unexpected depth channel count: {depthmap.shape[3]}")
                 return (torch.zeros_like(depthmap.permute(0, 2, 3, 1)),) # Return black BHWC
        elif depthmap.ndim == 3: # BHW
            depth_bhw = depthmap
        else:
             print(f"Error: Unexpected depth map dimensions: {depthmap.shape}")
             return (torch.zeros_like(depthmap.permute(0, 2, 3, 1)),) # Return black BHWC

        B, H, W = depth_bhw.shape

        # Normalize depth map per image in the batch
        depth_normalized = self.normalize_depth_batch(depth_bhw, invert_depthmap) # Shape (B, H, W)

        # -- Pattern Image Processing --
        # Ensure pattern is BCHW float32
        pattern_image_bhwc = pattern_image
        pattern_image_bchw = pattern_image_bhwc.permute(0, 3, 1, 2).to(dtype=torch.float32)
        Bp, Cp, Hp, Wp = pattern_image_bchw.shape

        if Bp != B and Bp != 1 :
             print(f"Error: Batch size mismatch between depth ({B}) and pattern ({Bp})")
             # Fallback: try broadcasting pattern if Bp=1
             if Bp == 1:
                 pattern_image_bchw = pattern_image_bchw.expand(B, -1, -1, -1)
                 print("Warning: Broadcasting pattern image batch to match depth map batch size.")
             else:
                 return (torch.zeros(B, H, W, Cp, device=target_device, dtype=torch.float32),) # Return black BHWC

        # Resize pattern height to match output height if necessary
        if Hp != H:
            print(f"Warning: Pattern height ({Hp}) differs from depth map height ({H}). Resizing pattern.")
            pattern_image_bchw = F.interpolate(pattern_image_bchw, size=(H, Wp), mode='bilinear', align_corners=False)

        # Tile pattern horizontally
        if pattern_tile_factor > 1:
            pattern_image_bchw = pattern_image_bchw.repeat(1, 1, 1, pattern_tile_factor)
            Wp *= pattern_tile_factor

        # -- Color Image Processing (Optional) --
        color_src_bchw = None
        if color_image is not None:
            color_image_bhwc = color_image
            if color_image_bhwc.shape[0] != B or color_image_bhwc.shape[1:3] != (H, W):
                 print(f"Warning: Color image shape mismatch ({color_image_bhwc.shape}). Resizing.")
                 try:
                      color_src_bchw = color_image_bhwc.permute(0, 3, 1, 2).to(dtype=torch.float32)
                      # Resize if needed - check batch size compatibility first
                      if color_src_bchw.shape[0] == 1 and B > 1:
                          color_src_bchw = color_src_bchw.expand(B, -1, -1, -1)
                      elif color_src_bchw.shape[0] != B:
                           print(f"Error: Cannot broadcast color image batch size {color_src_bchw.shape[0]} to {B}")
                           color_src_bchw = None # Disable coloring
                      if color_src_bchw is not None and (color_src_bchw.shape[2] != H or color_src_bchw.shape[3] != W):
                           color_src_bchw = F.interpolate(color_src_bchw, size=(H, W), mode='bilinear', align_corners=False)
                 except Exception as e:
                     print(f"Error processing color image: {e}. Disabling color.")
                     color_src_bchw = None
            else:
                 color_src_bchw = color_image_bhwc.permute(0, 3, 1, 2).to(dtype=torch.float32)


        # --- Stereogram Calculation (SIS Algorithm) ---
        # Output image, initialized potentially with tiled pattern (or zeros)
        stereogram_bchw = torch.zeros(B, Cp, H, W, dtype=torch.float32, device=target_device)

        # Calculate separation based on depth
        # separation = eye_sep * (1 - depth * divergence) # Original formula idea
        # depth_normalized: 0=far, 1=near (after inversion)
        # zero_parallax_depth: depth value where separation is exactly eye_sep
        # divergence_factor: scales the depth effect
        eye_sep_pixels = (eye_separation_percent / 100.0) * W
        # Shift relative to zero parallax plane:
        # Depth=zero_parallax -> relative_depth=0 -> sep=eye_sep
        # Depth > zero_parallax (closer) -> relative_depth > 0 -> sep < eye_sep (converge)
        # Depth < zero_parallax (further) -> relative_depth < 0 -> sep > eye_sep (diverge)
        relative_depth = depth_normalized - zero_parallax_depth
        separation = eye_sep_pixels * (1 - relative_depth * divergence_factor)
        separation = torch.round(separation).int() # Separation in pixels (integer)

        # Ensure minimum separation (e.g., 1 pixel) to avoid issues
        separation = torch.clamp(separation, min=1, max=W-1) # Clamp within valid image width range

        # Generate stereogram pixel by pixel (vectorized across batch and height)
        # links: stores the x-coordinate of the linked pixel to the left
        links = torch.zeros_like(depth_normalized, dtype=torch.int, device=target_device)

        for x in range(W):
            # Calculate separation for the current column x for all batches/rows
            sep_x = separation[:, :, x] # Shape (B, H)

            # Calculate the x-coordinate of the left pixel (xl)
            xl = x - sep_x // 2 # Shape (B, H)
            xr = xl + sep_x     # x-coordinate of the right pixel (should be == x)

            # Boundary check: Pixels near the left edge get pattern directly
            # Pixels where xl < 0 are "visible" only to the right eye initially
            left_boundary_mask = (xl < 0) # Shape (B, H)

            # --- Determine pixel color ---
            # 1. Pixels visible only to the right eye (near left edge)
            # Get color from the pattern image, tiled appropriately
            pattern_x_indices = x % Wp # Corresponding x in the potentially tiled pattern
            pattern_color = pattern_image_bchw[:, :, :, pattern_x_indices] # Shape (B, C, H) - Slice pattern

            # Apply pattern color where xl < 0
            # Use advanced indexing/masking - expand mask to match channel dimension
            left_boundary_mask_bch = left_boundary_mask.unsqueeze(1).expand(-1, Cp, -1) # Shape (B, C, H)
            stereogram_bchw[:, :, :, x] = torch.where(left_boundary_mask_bch, pattern_color, stereogram_bchw[:, :, :, x])

            # 2. Pixels visible to both eyes (xl >= 0)
            # These pixels should inherit color from their linked left pixel (xl)
            # Need to gather colors from stereogram at column xl for valid xl
            valid_xl_mask = ~left_boundary_mask # Shape (B, H)

            if torch.any(valid_xl_mask):
                # Create indices for gathering: Batch indices, Channel indices, Height indices, Width indices (xl)
                B_idx, H_idx = torch.meshgrid(torch.arange(B, device=target_device),
                                              torch.arange(H, device=target_device),
                                              indexing='ij')
                B_idx_flat = B_idx[valid_xl_mask]
                H_idx_flat = H_idx[valid_xl_mask]
                xl_flat = xl[valid_xl_mask] # Valid xl coordinates

                # Clamp xl_flat just in case (shouldn't be necessary with separation clamp)
                xl_flat_clamped = torch.clamp(xl_flat, 0, x - 1 if x > 0 else 0) # Ensure xl < x

                # Gather colors from already computed part of the stereogram
                # Need to handle channels implicitly or explicitly
                # Gather from stereogram_bchw using the flat indices
                # Indices shape: (num_valid_pixels)
                # stereogram_bchw shape: (B, C, H, W)
                gathered_colors = stereogram_bchw[B_idx_flat, :, H_idx_flat, xl_flat_clamped] # Shape (num_valid_pixels, C)

                # Scatter these colors back into the current column x where xl was valid
                # Use boolean mask assignment (more robust)
                valid_mask_bch = valid_xl_mask.unsqueeze(1).expand(-1, Cp, -1) # Shape (B, C, H)
                # stereogram_bchw[:, :, :, x][valid_mask_bch] = gathered_colors # This assignment shape mismatch
                # Need to reshape gathered_colors or assign carefully
                # Alternative: loop through channels or use put_ along the width dimension (complex)

                # Simpler approach: Use torch.where again
                left_pixel_color = torch.zeros_like(pattern_color) # Default color if gather failed
                # This gather needs careful indexing if using flat indices
                # Let's try direct indexing for a small example:
                # stereogram[b, c, h, x] = stereogram[b, c, h, xl[b,h]] IF xl[b,h] >= 0

                # Using torch.gather for pixel lookup (might be slow but more direct)
                # Create index tensor for gather along width dim (dim=3)
                # xl shape (B,H) -> Need (B, C, H, 1) for gather index
                xl_gather_idx = xl.unsqueeze(1).unsqueeze(-1).expand(-1, Cp, -1, -1) # Shape (B, C, H, 1)
                # Clamp index before gathering
                xl_gather_idx_clamped = torch.clamp(xl_gather_idx, 0, W-1) # Clamp to valid width range
                # We want to gather from potentially incomplete stereogram up to column x-1
                # This iterative dependency is tricky to fully vectorize with gather on the *same* tensor
                # Reverting to a slightly less efficient but more understandable masked assignment

                # Create a tensor of colors from the left pixel (xl) position
                # Initialize with zeros, then fill valid values
                left_color_source = torch.zeros_like(pattern_color)
                # Create full coordinate indices for valid pixels
                coords_b = B_idx_flat
                coords_h = H_idx_flat
                coords_xl = xl_flat_clamped
                # Extract colors channel by channel or use advanced indexing
                left_color_source_flat = stereogram_bchw[coords_b, :, coords_h, coords_xl] # Shape (num_valid, C)

                # Place the gathered colors into the source tensor based on the mask
                left_color_source = left_color_source.permute(0,2,1) # BH C
                left_color_source[valid_xl_mask] = left_color_source_flat
                left_color_source = left_color_source.permute(0,2,1) # B C H

                # Combine pattern color (left boundary) and inherited color (valid xl)
                stereogram_bchw[:, :, :, x] = torch.where(left_boundary_mask_bch, pattern_color, left_color_source)


            # --- Constraint satisfaction (optional but improves quality) ---
            # Check if the right pixel (xr == x) is consistent with the left pixel (xl)
            # If xr is within bounds and links[xr] is set, it should link back to xl.
            # This part is complex to implement efficiently in batch/GPU and often omitted in basic SIS.


        # --- Apply Optional Coloring ---
        if color_src_bchw is not None:
             # Basic coloring: Multiply stereogram intensity by color image color
             # Assuming stereogram_bchw is grayscale-like (can adapt if pattern has color)
             if Cp == 1: # If pattern produced grayscale
                 stereogram_intensity = stereogram_bchw.repeat(1, 3, 1, 1) # Repeat mono channel to RGB
             elif Cp == 3:
                 stereogram_intensity = stereogram_bchw
             elif Cp >= 3: # Handle RGBA etc. by taking first 3
                 stereogram_intensity = stereogram_bchw[:, :3, :, :]
             else: # Should not happen based on pattern input check
                 stereogram_intensity = stereogram_bchw # Fallback

             # Match color source channels (expecting 3)
             if color_src_bchw.shape[1] == 1:
                 color_rgb = color_src_bchw.repeat(1, 3, 1, 1)
             elif color_src_bchw.shape[1] >= 3:
                 color_rgb = color_src_bchw[:, :3, :, :]
             else:
                 print("Warning: Cannot use color image with <3 channels.")
                 color_rgb = torch.ones_like(stereogram_intensity) # Fallback to white

             # Modulate coloring
             colored_stereogram = stereogram_intensity * color_rgb
             # Clamp result
             final_stereogram_bchw = colored_stereogram.clamp(0, 1)

             # Ensure correct number of output channels (match color source)
             if color_src_bchw.shape[1] == 4 and final_stereogram_bchw.shape[1] == 3:
                 # Add alpha channel (e.g., solid alpha)
                 alpha_channel = torch.ones(B, 1, H, W, device=target_device, dtype=torch.float32)
                 final_stereogram_bchw = torch.cat((final_stereogram_bchw, alpha_channel), dim=1)
             elif final_stereogram_bchw.shape[1] > color_src_bchw.shape[1]:
                  final_stereogram_bchw = final_stereogram_bchw[:, :color_src_bchw.shape[1], :, :]


        else:
             # If no color image, output the raw stereogram pattern
             final_stereogram_bchw = stereogram_bchw.clamp(0, 1)


        # --- Final Output Preparation ---
        # Permute back to BHWC format for ComfyUI
        output_batch_bhwc = final_stereogram_bchw.permute(0, 2, 3, 1).contiguous()

        print(f"[StereogramTool] Processed batch of {B} frames on {target_device}. Output shape: {output_batch_bhwc.shape}. Total time: {time.time() - start_time:.3f}s")
        return (output_batch_bhwc,)


# --- MAPPINGS for ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "StereogramTool": StereogramTool
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StereogramTool": "Stereogram Tool (CUDA)"
}

# Example Usage Notes (if running standalone):
# B, H, W, C = 1, 256, 512, 3
# dummy_depth = torch.rand(B, H, W, 1)
# dummy_pattern = torch.rand(B, H, W//4, C) # Pattern usually narrower than output
# tool = StereogramTool()
# result_tuple = tool.create_stereogram_batch(dummy_depth, dummy_pattern, True, 0.5, 10.0, 1.0, 4)
# output_image = result_tuple[0]
# print(output_image.shape)