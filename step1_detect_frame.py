# step1_detect_frame.py
import cv2
from ultralytics import YOLO
import sys

def main():
    # 1️⃣ Load the video file
    video_path = "data/raw_videos/sample.mkv"  # Adjust path as needed
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file: {video_path}")
        return

    # 2️⃣ Read the first frame
    
    #cap.set(cv2.CAP_PROP_POS_FRAMES, 50)


    
    # cap.read()
    # cap.read()
    # cap.read()
    # cap.read()
    # cap.read()
    # cap.read()
    # cap.read()
    # cap.read()
    # cap.read()
    # cap.read()
    # cap.read()
    # cap.read()
    # cap.read()
    # cap.read()
    # cap.read()
    ret, frame = cap.read()

    # frame_i = 0
    # while True:
    #     ret, frame = cap.read()
    #     frame_i += 1
    #     if ret:
    #         if frame_i == 10:
    #             ret, frame = cap.read()
    #     else:
    #         break





    if not ret:
        print("Error: Unable to read the first frame.")
        return
    cap.release()

    # 3️⃣ Save the frame to disk
    cv2.imwrite("outputs/frame1.jpg", frame)
    print("First frame saved as outputs/frame1.jpg")

    # 4️⃣ Load YOLOv8 model (make sure model weights are downloaded)
    model = YOLO("yolov8n.pt")  # or yolov8s/b/m depending on performance




    # 5️⃣ Run inference
    results = model(frame)

    #print(results)
    #sys.exit()


    # 6️⃣ Draw bounding boxes and centroids on frame
    annotated = results[0].plot()  # returns an image with boxes

    #print(annotated)

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        #cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
        #cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        
        text_x = int(x1)
        text_y = max(int(y1) - 10, 10)  # ensure y is not negative

        cv2.putText(
            annotated,
            f"x2: {int(x2)}, y2: {int(y2)}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )





    # 7️⃣ Display annotated frame
    cv2.imshow("YOLO Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 8️⃣ Save annotated result
    cv2.imwrite("outputs/frame1_detected.jpg", annotated)
    print("Detection results saved as outputs/frame1_detected.jpg")

if __name__ == "__main__":
    main()
