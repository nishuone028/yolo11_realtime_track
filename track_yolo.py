from ultralytics import YOLO

# Load a pre-trained YOLO model
model = YOLO("yolo11n.pt")

# Start tracking objects in a video
# You can also use live video streams or webcam input
results = model.track(source="E:\\yolo_track\\video_\\1.mp4", stream=True, tracker="track_yaml/bytetrack.yaml", conf=0.5, iou=0.6, show=True)

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    result.save(filename="result.mp4")  # save to disk

