print("Initializing SS Before and After node for transition view creation...")

from .ss_before_after_video import SSBeforeAndAfterVideo
from .ss_before_after_video_with_depth import SSBeforeAndAfterVideoWithDepthMap

NODE_CLASS_MAPPINGS = {
    "SSBeforeAndAfterVideo": SSBeforeAndAfterVideo,
    "SSBeforeAndAfterVideoWithDepthMap": SSBeforeAndAfterVideoWithDepthMap
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SSBeforeAndAfterVideo": "SS Before After Video",
    "SSBeforeAndAfterVideoWithDepthMap": "SS Before After Video (Depth Map)"
}