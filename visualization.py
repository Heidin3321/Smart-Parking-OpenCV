"""Draw ROIs and occupancy totals. Fachriza may improve visual design."""
import cv2
import numpy as np

COLORS = {"empty": (0, 198, 69), "occupied": (32, 44, 245)}  # BGR


def draw_detection(frame: np.ndarray, results: list[dict], frame_index: int,
                   dataset_label: str = "SYNTHETIC DEMO ONLY") -> np.ndarray:
    """Return annotated BGR copy. Raise when status labels violate contract."""
    rendered = frame.copy()
    for item in results:
        state = str(item["status"])
        if state not in COLORS:
            raise ValueError(f"Unsupported status: {state}")
        x, y, w, h = (int(item[key]) for key in ("x", "y", "w", "h"))
        color = COLORS[state]
        cv2.rectangle(rendered, (x, y), (x+w, y+h), color, 3)
        label = f'{item["id"]} {state.upper()}'
        # Solid label strip INSIDE ROI for legibility, unlike white text on cars.
        strip_h = min(24, h)
        cv2.rectangle(rendered, (x+1, y+1), (x+w-1, y+strip_h), color, -1)
        cv2.putText(rendered, label, (x+4, y+min(17, h-2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.41, (255, 255, 255), 1, cv2.LINE_AA)

    empty = sum(item["status"] == "empty" for item in results)
    occupied = len(results) - empty
    frame_w = frame.shape[1]
    cv2.rectangle(rendered, (0, 0), (frame_w, 85), (20, 35, 50), -1)
    top_text = f"SMART PARKING | EMPTY {empty} | OCCUPIED {occupied} | TOTAL {len(results)}"
    top_scale = min(0.75, max(0.38, frame_w / 1700))
    cv2.putText(rendered, top_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                top_scale, (246, 250, 252), 2, cv2.LINE_AA)
    bottom_text = f"KNN prototype | frame {frame_index} | {dataset_label}"
    cv2.putText(rendered, bottom_text, (15, 66), cv2.FONT_HERSHEY_SIMPLEX,
                min(0.55, max(0.33, frame_w / 2300)), (51, 233, 242), 1, cv2.LINE_AA)
    return rendered


def draw_input_test(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, "SMART PARKING - INPUT TEST (NO DETECTION)", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 235, 60), 2, cv2.LINE_AA)
    return out
