import os
import cv2
import time

class YoloV4DNN:

    def __init__(self, nms_threshold=0.4, conf_threshold=0.5,
                 class_labels=None, video_path=None,
                 yolo_path=None, path_to_yolo_weights=None,
                 output_path=None):
        self.nms_threshold = nms_threshold
        self.conf_threshold = conf_threshold
        self.video_path = video_path
        self.yolo_path = yolo_path
        self.path_to_yolo_weights = path_to_yolo_weights
        self.output_path = output_path

        # Load class labels
        if class_labels is None:
            raise ValueError("class_labels path must be provided")
        with open(class_labels, 'r') as f:
            self.class_labels = [line.strip() for line in f.readlines()]

        # Process the video
        self.process_video()

    def inference_dnn(self, image):
        network = cv2.dnn.readNet(self.yolo_path, self.path_to_yolo_weights)

        # Uncomment below to use GPU (if available)
        # network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        model = cv2.dnn_DetectionModel(network)
        model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True, crop=False)

        classes, scores, boxes = model.detect(
            image,
            confThreshold=self.conf_threshold,
            nmsThreshold=self.nms_threshold
        )
        return classes, scores, boxes

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {self.video_path}")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            resized_frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
            processed_frame = self.inference_run(resized_frame, frame)
            # Resize back to original size for output
            output_frame = cv2.resize(processed_frame, (width, height), interpolation=cv2.INTER_AREA)
            out.write(output_frame)
            if frame_count % 30 == 0:
                print(f"[INFO] Processed {frame_count} frames...")

        cap.release()
        out.release()
        print(f"[INFO] Video saved to: {self.output_path}")

    def inference_run(self, image, original_frame):
        start_time = time.time()
        class_ids, confidences, boxes = self.inference_dnn(image)
        end_time = time.time()

        frame_time = (end_time - start_time) * 1000  # ms
        fps = 1000 / frame_time if frame_time > 0 else 0

        for (classId, score, box) in zip(class_ids, confidences, boxes):
            x, y, w, h = box
            # Scale box to original frame size if needed
            x = int(x * original_frame.shape[1] / 640)
            y = int(y * original_frame.shape[0] / 640)
            w = int(w * original_frame.shape[1] / 640)
            h = int(h * original_frame.shape[0] / 640)
            # Draw bounding box
            cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = "Time: %.2fms | FPS: %.2f | ID: %s | Score: %.2f" % (
                frame_time, fps, self.class_labels[classId], score
            )
            cv2.putText(original_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        return original_frame


def main():
    path_to_classes = '/root/Onnx_YoloV4/coco-classes.txt'
    video_path = '/root/Onnx_YoloV4/video/1.mp4'
    path_to_cfg = '/root/Onnx_YoloV4/models/yolov4-tiny.cfg'
    path_to_weights = '/root/Onnx_YoloV4/models/yolov4-tiny.weights'
    output_path = '/root/Onnx_YoloV4/output.avi'

    YoloV4DNN(
        nms_threshold=0.4,
        conf_threshold=0.5,
        class_labels=path_to_classes,
        video_path=video_path,
        yolo_path=path_to_cfg,
        path_to_yolo_weights=path_to_weights,
        output_path=output_path
    )


if __name__ == "__main__":
    main()
