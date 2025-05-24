# Robust imports with fallbacks for standalone or missing dependencies

import torch
import numpy as np
from PIL import Image
import cv2
from comfy.utils import ProgressBar
import os
import subprocess
try:
    import folder_paths
except ImportError:
    class FolderPaths:
        def get_input_directory(self):
            return "input"
        def get_output_directory(self):
            return "output"
        def get_temp_directory(self):
            return "temp"
    folder_paths = FolderPaths()
# Check for ffmpeg availability
def get_ffmpeg_path():
    """Get the path to ffmpeg executable"""
    # Try to find ffmpeg in the system path
    try:
        if os.name == 'nt':  # Windows
            ffmpeg_command = "where ffmpeg"
        else:  # Unix/Linux/Mac
            ffmpeg_command = "which ffmpeg"

        result = subprocess.run(ffmpeg_command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass

    # Check if ffmpeg is in the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.name == 'nt':  # Windows
        ffmpeg_path = os.path.join(current_dir, "ffmpeg.exe")
    else:  # Unix/Linux/Mac
        ffmpeg_path = os.path.join(current_dir, "ffmpeg")

    if os.path.exists(ffmpeg_path):
        return ffmpeg_path

    # Try to use imageio's ffmpeg
    try:
        import imageio
        return imageio.plugins.ffmpeg.get_exe()
    except:
        pass

    return None

ffmpeg_path = get_ffmpeg_path()

class SSBeforeAndAfterVideoWithDepthMap:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "before_image": ("IMAGE",),
                "after_image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "transition_type": (["back_to_front", "front_to_back", "middle_out"], {"default": "back_to_front"}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120, "step": 1}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "output_path": ("STRING", {"default": "/output/"}),
                "filename_prefix": ("STRING", {"default": "before_after_depth_transition"}),
                "transition_duration": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "hold_duration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "feather": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "loop_transition": ("BOOLEAN", {"default": False}),
                "easing_method": (["none", "ease-in", "ease-out", "both"], {"default": "none"}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "preview_frame")
    OUTPUT_NODE = True
    FUNCTION = "create_transition_video"
    CATEGORY = "👀 SamSeen"
    DESCRIPTION = "Create before and after transition videos using a depth map for depth-based transitions."

    def tensor_to_pil(self, tensor):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        np_image = tensor.cpu().numpy()
        np_image = (np_image * 255).astype(np.uint8)
        return Image.fromarray(np_image)

    def resize_images(self, before_img, after_img, depth_img, width, height):
        before_pil = self.tensor_to_pil(before_img)
        after_pil = self.tensor_to_pil(after_img)
        depth_pil = self.tensor_to_pil(depth_img)
        before_resized = before_pil.resize((width, height), Image.LANCZOS)
        after_resized = after_pil.resize((width, height), Image.LANCZOS)
        depth_resized = depth_pil.resize((width, height), Image.LANCZOS)
        return before_resized, after_resized, depth_resized

    def create_depth_transition(self, before_img, after_img, depth_img, progress, transition_type, feather):
        before_np = np.array(before_img, dtype=np.float32)
        after_np = np.array(after_img, dtype=np.float32)
        depth_np = np.array(depth_img, dtype=np.float32)
        if depth_np.ndim == 3:
            depth_np = depth_np[..., 0]
        depth_norm = (depth_np - depth_np.min()) / (depth_np.ptp() + 1e-8)
        if transition_type == "back_to_front":
            threshold = 1.0 - progress
            mask = (depth_norm >= threshold).astype(np.float32)
        elif transition_type == "front_to_back":
            threshold = progress
            mask = (depth_norm <= threshold).astype(np.float32)
        elif transition_type == "middle_out":
            center = 0.5
            spread = progress * 0.5
            mask = ((depth_norm >= center - spread) & (depth_norm <= center + spread)).astype(np.float32)
        else:
            mask = np.zeros_like(depth_norm, dtype=np.float32)
        feather_px = max(1, int(feather * 50))
        if feather_px > 1:
            mask = cv2.GaussianBlur(mask, (feather_px | 1, feather_px | 1), 0)
            mask = np.clip(mask, 0, 1)
        mask_3d = np.stack([mask] * 3, axis=2)
        result = before_np * (1 - mask_3d) + after_np * mask_3d
        return Image.fromarray(result.astype(np.uint8))

    def generate_transition_frames(self, before_img, after_img, depth_img, transition_type, fps, transition_duration, hold_duration, feather, loop_transition=False, easing_method="none"):
        frames = []
        hold_frames = int(fps * hold_duration)
        transition_frames = int(fps * transition_duration)
        total_frames = hold_frames + transition_frames + (hold_frames if loop_transition else hold_frames) + (transition_frames if loop_transition else 0)
        pbar = ProgressBar(total_frames)
        # Hold on before image
        for i in range(hold_frames):
            frames.append(before_img.copy())
            pbar.update(1)
        # Forward transition
        for i in range(transition_frames):
            progress = i / (transition_frames - 1) if transition_frames > 1 else 1.0
            frame = self.create_depth_transition(before_img, after_img, depth_img, progress, transition_type, feather)
            frames.append(frame)
            pbar.update(1)
        if loop_transition:
            # Hold on after image before looping back
            for i in range(hold_frames):
                frames.append(after_img.copy())
                pbar.update(1)
            # Reverse transition
            for i in range(transition_frames):
                progress = i / (transition_frames - 1) if transition_frames > 1 else 1.0
                frame = self.create_depth_transition(after_img, before_img, depth_img, progress, transition_type, feather)
                frames.append(frame)
                pbar.update(1)
            # Hold on before image again (optional, can be omitted if not needed)
            for i in range(hold_frames):
                frames.append(before_img.copy())
                pbar.update(1)
        else:
            # Hold on after image
            for i in range(hold_frames):
                frames.append(after_img.copy())
                pbar.update(1)
        return frames

    def create_transition_video(self, before_image, after_image, depth_map, transition_type, fps, width, height, output_path, filename_prefix, transition_duration, hold_duration, feather, loop_transition=False, easing_method="none"):
        before_img, after_img, depth_img = self.resize_images(before_image, after_image, depth_map, width, height)
        frames = self.generate_transition_frames(before_img, after_img, depth_img, transition_type, fps, transition_duration, hold_duration, feather, loop_transition, easing_method)
        if output_path.startswith("/output/"):
            output_dir = folder_paths.get_output_directory()
            relative_path = output_path[8:]
            if relative_path:
                output_dir = os.path.join(output_dir, relative_path)
        else:
            output_dir = output_path
        os.makedirs(output_dir, exist_ok=True)
        counter = 1
        while True:
            filename = f"{filename_prefix}_{counter:05d}.mp4"
            video_path = os.path.join(output_dir, filename)
            if not os.path.exists(video_path):
                break
            counter += 1
        temp_dir = os.path.join(output_dir, "temp_depth_frames")
        os.makedirs(temp_dir, exist_ok=True)
        for idx, frame in enumerate(frames):
            frame.save(os.path.join(temp_dir, f"frame_{idx:04d}.png"))
        if ffmpeg_path is None:
            raise RuntimeError("FFmpeg executable not found. Please ensure FFmpeg is installed and accessible.")
        ffmpeg_cmd = [ffmpeg_path, "-y", "-framerate", str(fps), "-i", os.path.join(temp_dir, "frame_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path]
        subprocess.run(ffmpeg_cmd, check=True)
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
        preview_frame = frames[len(frames) // 2]
        preview_tensor = torch.from_numpy(np.array(preview_frame).astype(np.float32) / 255.0).unsqueeze(0)
        return (video_path, preview_tensor)