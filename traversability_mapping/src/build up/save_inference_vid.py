# Run live yolov8 inference on video
import cv2
import ultralytics
from ultralytics import YOLO

model = YOLO('/home/cam/ros_ws/src/traversability_mapping/src/2000imgs_weights.pt')     # replace with weights
print(ultralytics.__file__)
file_path = '/home/cam/for_segmentation.mp4'    # replace with your video
output_path = 'segmented_video.mp4'  # Output video file path

cap = cv2.VideoCapture(file_path)

#### cv2.namedWindow("YOLOv8 inference", cv2.WINDOW_NORMAL)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame =cap.read()
    # break if no frames are loaded
    if not ret:
        break

    # perform inference on the frame
    results = model(frame, stream=True)
    for result in results:
        # annotate each frame
        annotated_frame = result.plot(boxes=False)

        # display the annotated frame in the window
        ## cv2.imshow("YOLOv8 inference", annotated_frame)

        # write the annotated frame to the video file
        out.write(annotated_frame)


print(f"video annotated and saved to {output_path}")
out.release()
# Release the video capture object
cap.release()
# Close all opencv windows
cv2.destroyAllWindows()