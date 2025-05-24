# Robust imports with fallbacks for standalone or missing dependencies
try:
    import torch
except ImportError:
    raise ImportError('Please install torch to use this module.')
try:
    import numpy as np
except ImportError:
    raise ImportError('Please install numpy to use this module.')
try:
    from PIL import Image
except ImportError:
    raise ImportError('Please install pillow to use this module.')
import cv2
try:
    from comfy.utils import ProgressBar
except ImportError:
    raise ImportError('Please ensure comfy.utils is available in your Python path.')
try:
    import folder_paths
except ImportError:
    raise ImportError('Please ensure folder_paths is available in your Python path.')
try:
    import imageio
except ImportError:
    raise ImportError('Please install imageio to use this module.')
import os
import subprocess
import time
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
    except ImportError:
        raise ImportError('Please install imageio to use this module.')
    except:
        pass

    return None

ffmpeg_path = get_ffmpeg_path()

class SSBeforeAndAfterVideo:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "before_image": ("IMAGE",),
                "after_image": ("IMAGE",),
                "transition_type": (["fade", "wipe_from_left", "wipe_from_top", "wipe_from_right", "wipe_from_bottom"], {"default": "fade"}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120, "step": 1}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "output_path": ("STRING", {"default": "/output/"}),
                "filename_prefix": ("STRING", {"default": "before_after_transition"}),
                "transition_duration": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "hold_duration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "feather": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "loop_transition": ("BOOLEAN", {"default": False}),
                "loop_transition": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "preview_frame")
    OUTPUT_NODE = True
    FUNCTION = "create_transition_video"
    CATEGORY = "👀 SamSeen"
    DESCRIPTION = "Create stunning before and after transition videos with various wipe and fade effects. Perfect for showcasing transformations and comparisons!"

    def tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Convert from tensor to numpy
        np_image = tensor.cpu().numpy()
        
        # Scale to 0-255 and convert to uint8
        np_image = (np_image * 255).astype(np.uint8)
        
        return Image.fromarray(np_image)

    def pil_to_tensor(self, pil_image):
        """Convert PIL Image to tensor"""
        np_image = np.array(pil_image).astype(np.float32) / 255.0
        return torch.from_numpy(np_image).unsqueeze(0)

    def resize_images(self, before_img, after_img, width, height):
        """Resize both images to the specified dimensions"""
        before_pil = self.tensor_to_pil(before_img)
        after_pil = self.tensor_to_pil(after_img)
        
        before_resized = before_pil.resize((width, height), Image.LANCZOS)
        after_resized = after_pil.resize((width, height), Image.LANCZOS)
        
        return before_resized, after_resized

    def create_fade_transition(self, before_img, after_img, progress):
        """Create a fade transition between two images"""
        # Convert PIL images to numpy arrays
        before_np = np.array(before_img, dtype=np.float32)
        after_np = np.array(after_img, dtype=np.float32)
        
        # Blend the images based on progress
        blended = before_np * (1 - progress) + after_np * progress
        
        return Image.fromarray(blended.astype(np.uint8))

    def create_wipe_transition(self, before_img, after_img, progress, direction, feather):
        """Create a wipe transition between two images with feathering (blurred edge)"""
        width, height = before_img.size
        mask = np.zeros((height, width), dtype=np.float32)
        feather_pixels_x = max(1, int(width * feather))
        feather_pixels_y = max(1, int(height * feather))

        if direction == "wipe_from_left":
            wipe_pos = int(progress * width)
            left = max(0, wipe_pos - feather_pixels_x // 2)
            right = min(width, wipe_pos + feather_pixels_x // 2)
            mask[:, :left] = 1.0
            mask[:, right:] = 0.0
            if right > left:
                feather = np.linspace(1.0, 0.0, right - left)
                mask[:, left:right] = feather
        elif direction == "wipe_from_top":
            wipe_pos = int(progress * height)
            top = max(0, wipe_pos - feather_pixels_y // 2)
            bottom = min(height, wipe_pos + feather_pixels_y // 2)
            mask[:top, :] = 1.0
            mask[bottom:, :] = 0.0
            if bottom > top:
                feather = np.linspace(1.0, 0.0, bottom - top)
                mask[top:bottom, :] = feather[:, None]
        elif direction == "wipe_from_right":
            wipe_pos = int((1.0 - progress) * width)
            left = max(0, wipe_pos - feather_pixels_x // 2)
            right = min(width, wipe_pos + feather_pixels_x // 2)
            mask[:, :left] = 0.0
            mask[:, right:] = 1.0
            if right > left:
                feather = np.linspace(0.0, 1.0, right - left)
                mask[:, left:right] = feather
        elif direction == "wipe_from_bottom":
            wipe_pos = int((1.0 - progress) * height)
            top = max(0, wipe_pos - feather_pixels_y // 2)
            bottom = min(height, wipe_pos + feather_pixels_y // 2)
            mask[:top, :] = 0.0
            mask[bottom:, :] = 1.0
            if bottom > top:
                feather = np.linspace(0.0, 1.0, bottom - top)
                mask[top:bottom, :] = feather[:, None]
        elif direction == "wipe_from_bottom":
            wipe_pos = int(height * (1 - progress))
            top = max(0, wipe_pos - feather_pixels_y // 2)
            bottom = min(height, wipe_pos + feather_pixels_y // 2)
            mask[:top, :] = 0.0
            mask[bottom:, :] = 1.0
            if bottom > top:
                alpha = np.linspace(1, 0, bottom - top)
                mask[top:bottom, :] = alpha[:, None]

        # Blur the mask for a feathered (soft) edge
        blur_size = max(1, int(max(feather_pixels_x, feather_pixels_y) // 2) * 2 + 1)
        if blur_size > 1:
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
            mask = np.clip(mask, 0, 1)

        before_np = np.array(before_img, dtype=np.float32)
        after_np = np.array(after_img, dtype=np.float32)
        mask_3d = np.stack([mask] * 3, axis=2)
        result = before_np * (1 - mask_3d) + after_np * mask_3d
        return Image.fromarray(result.astype(np.uint8))

    def generate_transition_frames(self, before_img, after_img, transition_type, fps, transition_duration, hold_duration, feather, loop_transition):
        frames = []
        total_transition_frames = int(fps * transition_duration)
        total_hold_frames = int(fps * hold_duration)
        # Hold the before image
        for _ in range(total_hold_frames):
            frames.append(before_img.copy())
        # Transition frames
        for i in range(total_transition_frames):
            progress = i / max(1, total_transition_frames - 1)
            if transition_type == "fade":
                frame = self.create_fade_transition(before_img, after_img, progress)
            elif transition_type.startswith("wipe"): 
                frame = self.create_wipe_transition(before_img, after_img, progress, transition_type, feather)
            else:
                frame = before_img.copy()
            frames.append(frame)
        # Hold the after image
        for _ in range(total_hold_frames):
            frames.append(after_img.copy())
        # Optionally loop back
        if loop_transition:
            for i in range(total_transition_frames):
                progress = 1 - (i / max(1, total_transition_frames - 1))
                if transition_type == "fade":
                    frame = self.create_fade_transition(before_img, after_img, progress)
                elif transition_type.startswith("wipe"): 
                    frame = self.create_wipe_transition(before_img, after_img, progress, transition_type, feather)
                else:
                    frame = after_img.copy()
                frames.append(frame)
            for _ in range(total_hold_frames):
                frames.append(before_img.copy())
        return frames

    def create_transition_video(self, before_image, after_image, transition_type, fps, width, height, output_path, filename_prefix, transition_duration, hold_duration, feather, loop_transition=False):
        if ffmpeg_path is None:
            raise RuntimeError("ffmpeg not found. Please install ffmpeg to use video functionality.")
        before_img, after_img = self.resize_images(before_image, after_image, width, height)
        # Use the correct frame generation method for transitions
        frames = self.generate_transition_frames(
            before_img, after_img, transition_type, fps, transition_duration, hold_duration, feather, loop_transition
        )
        # Prepare output directory
        if output_path.startswith("/output/"):
            output_dir = folder_paths.get_output_directory()
            relative_path = output_path[8:]
            if relative_path:
                output_dir = os.path.join(output_dir, relative_path)
        else:
            output_dir = output_path
        os.makedirs(output_dir, exist_ok=True)
        # Generate unique filename
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
        ffmpeg_cmd = [ffmpeg_path, "-y", "-framerate", str(fps), "-i", os.path.join(temp_dir, "frame_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path]
        subprocess.run(ffmpeg_cmd, check=True)
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
        preview_frame = frames[len(frames) // 2]
        preview_tensor = torch.from_numpy(np.array(preview_frame).astype(np.float32) / 255.0).unsqueeze(0)
        return (video_path, preview_tensor)
