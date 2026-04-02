# from ultralytics import YOLO
# import cv2
# import numpy
# model = YOLO("yolov8n.pt")
# img = cv2.imread("cars_road.jpg")
# resized = cv2.resize(img, (640, 480))

# if resized is None:
#     print("Image not loaded")
# else:
#     results = model(resized)

#     for box in results[0].boxes:
#         cls_id = int(box.cls[0])
#         conf = float(box.conf[0])
#         class_name = model.names[cls_id]

#         if class_name in ["car", "bus", "truck", "motorcycle"]:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 250, 0), 1)
#             cv2.putText(
#                 resized,
#                 f"{class_name} {conf:.2f}",
#                 (x1, y1 - 5),
#                 cv2.FONT_HERSHEY_TRIPLEX,
#                 0.5,
#                 (0, 250, 0),
#                 1
#             )

#     cv2.imshow("Vehicle Detection", resized)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     cv2.imwrite("vehicle_output.jpg", resized)
#     print("Vehicle detection completed")




















from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("sampleVideo.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        if class_name in ["car", "bus", "truck", "motorcycle"]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 250, 0), 2)
            cv2.putText(
                frame,
                f"{class_name} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.6,
                (0, 250, 0),
                2
            )

    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()