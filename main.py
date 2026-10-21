
import cv2
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Smart Parking OpenCV"
    )

    parser.add_argument(
        "--source",
        default="dataset/parking_sample.mp4",
        help="Path video atau 0 untuk webcam"
    )

    args = parser.parse_args()

    is_webcam = args.source == "0"

    if not is_webcam and not Path(args.source).is_file():
        print(f"Video tidak ditemukan: {args.source}")
        print("Gunakan --source 0 untuk mencoba webcam.")
        return

    source = 0 if is_webcam else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Gagal membuka sumber video.")
        return

    print("SMART PARKING - VIDEO INPUT TEST")
    print("Tekan Q untuk keluar")
    print("Tekan S untuk menyimpan screenshot")

    Path("results").mkdir(exist_ok=True)

    while True:
        success, frame = cap.read()

        if not success:
            break

        cv2.putText(
            frame,
            "SMART PARKING - INPUT TEST",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Smart Parking OpenCV", frame)

        key = cv2.waitKey(1 if is_webcam else 30) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            cv2.imwrite("results/input_test.png", frame)
            print("Screenshot tersimpan di results/input_test.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
