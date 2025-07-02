import os
import cv2
import time

class YoloV4DNN:

    def __init__(self, nms_threshold=0.4, conf_threshold=0.5,
                 class_labels=None, image_path=None,
                 yolo_path=None, path_to_yolo_weights=None):
        
        self.nms_threshold = nms_threshold
        self.conf_threshold = conf_threshold
        self.image_path = image_path
        self.yolo_path = yolo_path
        self.path_to_yolo_weights = path_to_yolo_weights

        # Load class labels
        if class_labels is None:
            raise ValueError("class_labels path must be provided")
        with open(class_labels, 'r') as f:
            self.class_labels = [line.strip() for line in f.readlines()]

        # Load all image paths
        self.frames = self.load_images(self.image_path)

        # Process each image
        for frame_path in self.frames:
            image = cv2.imread(frame_path)
            resized_image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)

            # Run inference
            self.inference_run(resized_image, frame_path)

    def load_images(self, image_path):
        images_list = []
        for img_original in os.listdir(image_path):
            if img_original.endswith(('.jpg', '.png', '.jpeg')):
                img_full_path = os.path.join(image_path, img_original)
                images_list.append(img_full_path)
        return images_list

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

    def inference_run(self, image, frame_path):
        start_time = time.time()
        class_ids, confidences, boxes = self.inference_dnn(image)
        end_time = time.time()

        frame_time = (end_time - start_time) * 1000  # ms
        fps = 1000 / frame_time if frame_time > 0 else 0

        for (classId, score, box) in zip(class_ids, confidences, boxes):
            x, y, w, h = box

            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            label = "Time: %.2fms | FPS: %.2f | ID: %s | Score: %.2f" % (
                frame_time, fps, self.class_labels[classId], score
            )
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        # Save result
        filename = os.path.basename(frame_path)
        output_path = os.path.join("/root/Onnx_YoloV4", f"output_{filename}")
        cv2.imwrite(output_path, image)
        print(f"[INFO] Saved: {output_path}")


def main():
    path_to_classes = '/root/Onnx_YoloV4/coco-classes.txt'
    image_path = '/root/Onnx_YoloV4/images/'
    path_to_cfg = '/root/Onnx_YoloV4/models/yolov4-tiny.cfg'
    path_to_weights = '/root/Onnx_YoloV4/models/yolov4-tiny.weights'

    YoloV4DNN(
        nms_threshold=0.4,
        conf_threshold=0.5,
        class_labels=path_to_classes,
        image_path=image_path,
        yolo_path=path_to_cfg,
        path_to_yolo_weights=path_to_weights
    )


if __name__ == "__main__":
    main()
