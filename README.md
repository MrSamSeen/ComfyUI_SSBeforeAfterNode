[![SamSeen Logo](https://img.shields.io/badge/ComfyUI SS_Before_After_Transition-by_SamSeen_Solutions-orange?style=for-the-badge)](https://github.com/MrSamSeen)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

# ComfyUI SS Before After Node

This repository provides two powerful custom nodes for ComfyUI to create stunning before-and-after transition videos. These nodes are designed for visual comparisons, transformations, and creative effects, supporting both standard and depth map-based transitions.

## Features

- **SSBeforeAndAfterVideo**: Create videos with fade and wipe transitions between two images.
- **SSBeforeAndAfterVideoWithDepthMap**: Create depth-aware transitions using a depth map for more immersive effects.
- High-quality output with customizable resolution, frame rate, feathering, and looping.
- Automatic handling of output directories and filenames.
- Progress bar for frame generation.
- Robust ffmpeg integration for video encoding.

---

## Node 1: SSBeforeAndAfterVideo

### Description
Creates before-and-after transition videos using classic fade and wipe effects. Ideal for showcasing transformations, comparisons, or visual storytelling.

### Supported Transitions
- **Fade**: Smoothly blends from the before image to the after image.
- **Wipe**: Animated reveal from left, right, top, or bottom, with optional feathered (blurred) edge.

### Parameters
- **before_image**: The starting image (tensor).
- **after_image**: The ending image (tensor).
- **transition_type**: Choose from `fade`, `wipe_from_left`, `wipe_from_top`, `wipe_from_right`, `wipe_from_bottom`.
- **fps**: Frames per second (1-120).
- **width/height**: Output video resolution (64-4096 px).
- **output_path**: Output directory (default `/output/`).
- **filename_prefix**: Prefix for output video files.
- **transition_duration**: Duration of the transition (0.5-10.0 seconds).
- **hold_duration**: Time to hold before and after images (0.0-5.0 seconds).
- **feather**: Feathering amount for wipe transitions (0.0-1.0).
- **loop_transition**: If enabled, the video will loop back to the start.

### How It Works
1. **Resizes** both images to the specified resolution.
2. **Generates frames** for the hold, transition, and (optionally) loop-back phases.
3. **Applies the selected transition** (fade or wipe) with feathering.
4. **Saves frames** to a temporary directory.
5. **Encodes video** using ffmpeg.
6. **Cleans up** temporary files and returns the video path and a preview frame.

---

## Node 2: SSBeforeAndAfterVideoWithDepthMap

### Description
Creates before-and-after transition videos using a depth map to control the transition. This enables advanced effects like revealing the after image from back-to-front, front-to-back, or from the middle outwards, based on scene depth.

### Supported Transitions
- **back_to_front**: Reveals the after image starting from the farthest (background) regions.
- **front_to_back**: Reveals the after image starting from the closest (foreground) regions.
- **middle_out**: Reveals the after image from the center depth outward.

### Parameters
- **before_image**: The starting image (tensor).
- **after_image**: The ending image (tensor).
- **depth_map**: Depth map image (tensor, same size as input images).
- **transition_type**: Choose from `back_to_front`, `front_to_back`, `middle_out`.
- **fps**: Frames per second (1-120).
- **width/height**: Output video resolution (64-4096 px).
- **output_path**: Output directory (default `/output/`).
- **filename_prefix**: Prefix for output video files.
- **transition_duration**: Duration of the transition (0.5-10.0 seconds).
- **hold_duration**: Time to hold before and after images (0.0-5.0 seconds).
- **feather**: Feathering amount for the depth mask (0.0-1.0).
- **loop_transition**: If enabled, the video will loop back to the start.
- **easing_method**: Transition easing ("none", "ease-in", "ease-out", "both").

### How It Works
1. **Resizes** all images to the specified resolution.
2. **Normalizes the depth map** and creates a mask based on the selected transition type and progress.
3. **Applies feathering** to the mask for smooth transitions.
4. **Blends images** according to the mask for each frame.
5. **Handles looping** and hold frames as configured.
6. **Encodes video** using ffmpeg and cleans up temporary files.

---

## Installation

1. Place this folder in your `ComfyUI/custom_nodes` directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure ffmpeg is installed and available in your system PATH.

---

## Usage

1. Restart ComfyUI after installing the node.
2. Add the desired node (`SSBeforeAndAfterVideo` or `SSBeforeAndAfterVideoWithDepthMap`) to your workflow.
3. Configure the parameters as needed.
4. Run the workflow to generate your before-and-after transition video.
5. The output video and a preview frame will be saved to your specified output directory.

---

## Troubleshooting
- **ffmpeg not found**: Make sure ffmpeg is installed and accessible from your system PATH.
- **Missing dependencies**: Install all required Python packages listed in `requirements.txt`.
- **Output not appearing**: Check the output directory path and permissions.

---

## License
MIT License

---

## Credits
Developed by SamSeen. Inspired by creative visual storytelling needs in ComfyUI.

---

## Contribution
Pull requests and suggestions are welcome!