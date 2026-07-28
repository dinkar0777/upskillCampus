# ===========================================
# Crop and Weed Detection using YOLO
# main.py
# ===========================================

from ultralytics import YOLO
import cv2
import os
import argparse

# -----------------------------
# Load YOLO Model
# -----------------------------
MODEL_PATH = "best.pt"   # Place your trained model in the project folder

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = YOLO(MODEL_PATH)


# -----------------------------
# Detection Function
# -----------------------------
def detect(image_path):

    if not os.path.exists(image_path):
        print("Image not found!")
        return

    image = cv2.imread(image_path)

    # Run Detection
    results = model(image)

    result = results[0]

    annotated_image = result.plot()

    crop_count = 0
    weed_count = 0

    print("\nDetection Results")
    print("--------------------------")

    for box in result.boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        label = model.names[cls]

        print(f"{label:<10} Confidence : {conf:.2f}")

        if label.lower() == "crop":
            crop_count += 1
        elif label.lower() == "weed":
            weed_count += 1

    print("--------------------------")
    print(f"Total Crops : {crop_count}")
    print(f"Total Weeds : {weed_count}")

    os.makedirs("output", exist_ok=True)

    output_path = os.path.join("output", "result.jpg")

    cv2.imwrite(output_path, annotated_image)

    print(f"\nOutput saved to : {output_path}")

    cv2.imshow("Crop & Weed Detection", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path of input image"
    )

    args = parser.parse_args()

    detect(args.image)
