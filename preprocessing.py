"""Frame preprocessing interface owned by Rifqi.

During the synthetic KNN demo, frames are passed through unchanged; the
baseline classifier extracts scaled BGR+HSV features per parking ROI.
"""

import numpy as np


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Accept one BGR uint8 frame and return BGR frame of SAME dimensions.

    Rifqi may adjust this function after coordinating with Fachriza, but
    must not resize/crop the frame because ROI coordinates refer to its
    original resolution. Any transformation must remain consistent with
    the classification model's training inputs.
    """
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("preprocess_frame expects a BGR image (H, W, 3)")
    return frame
