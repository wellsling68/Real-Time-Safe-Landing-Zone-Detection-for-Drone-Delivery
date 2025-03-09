import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import threading
import time
import mpu6050
import math

# Initialize MPU6050 sensor
mpu = mpu6050.mpu6050(0x68)

# Define color mapping
ID2COLOR = {
    0: (0, 0, 128),    # Building
    1: (0, 128, 0),    # Tree
    2: (0, 128, 128),  # Low Vegetation
    3: (192, 0, 192),  # Static Car
    4: (128, 64, 128), # Road
    5: (0, 64, 64),    # Human
    6: (0, 0, 0),      # Background Clutter
}


class TFLiteModel:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = (1, 256, 256, 3)
        self.lock = threading.Lock()
        self.latest_result = None
        self.latest_frame = None
        self.running = True
        self.inference_counter = 0
        self.thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.thread.start()

    def preprocess(self, frame):
        frame_resized = cv2.resize(frame, (256, 256))
        normalized = frame_resized.astype(np.float32) / 255.0
        normalized = np.expand_dims(normalized, axis=0)

        if self.input_details[0]['dtype'] == np.int8:
            scale, zero_point = self.input_details[0]['quantization']
            if scale == 0:
                raise ValueError("Quantization scale is 0. Check your model conversion.")
            quantized = normalized / scale + zero_point
            quantized = np.clip(quantized, -128, 127).astype(np.int8)
            return quantized
        else:
            return normalized

    def inference_loop(self):
        while self.running:
            if self.latest_frame is not None:
                input_data = self.preprocess(self.latest_frame)
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                result = np.argmax(output_data[0], axis=-1).astype(np.uint8)

                with self.lock:
                    self.latest_result = result
                    self.inference_counter += 1

            time.sleep(0.01)

    def update_frame(self, frame):
        self.latest_frame = frame

    def get_latest_result(self):
        with self.lock:
            if self.latest_result is not None:
                self.latest_result = np.clip(self.latest_result, 0, 6)
            return self.latest_result

    def stop(self):
        self.running = False
        self.thread.join()


def read_camera_angle():
    accel_data = mpu.get_accel_data()
    ax, ay, az = accel_data['x'], accel_data['y'], accel_data['z']
    pitch = math.atan2(ax, math.sqrt(ay**2 + az**2)) * 180 / math.pi
    return pitch


def find_largest_ellipse(mask, target_class, camera_angle):
    binary_mask = np.zeros_like(mask, dtype=np.uint8)
    binary_mask[mask == target_class] = 255

    if np.count_nonzero(binary_mask) == 0:
        return None, None, None

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip background
    largest_component_mask = (labels == largest_label).astype(np.uint8)

    dist_transform = cv2.distanceTransform(largest_component_mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)

    center = max_loc
    radius = int(max_val)
    major_axis = radius
    minor_axis = int(radius * np.cos(np.radians(camera_angle)))

    while major_axis > 0 and minor_axis > 0:
        ellipse_mask = np.zeros_like(largest_component_mask, dtype=np.uint8)
        cv2.ellipse(ellipse_mask, center, (major_axis, minor_axis), 0, 0, 360, 255, -1)

        intersection = cv2.bitwise_and(ellipse_mask, binary_mask)

        if np.count_nonzero(intersection) == np.count_nonzero(ellipse_mask):
            break

        major_axis -= 1
        minor_axis = int(major_axis * np.cos(np.radians(camera_angle)))

    if major_axis <= 0 or minor_axis <= 0:
        return None, None, None

    return center, major_axis, minor_axis


def apply_custom_colormap(mask):
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in ID2COLOR.items():
        color_mask[mask == class_id] = color

    return color_mask


def refine_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask_refined = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, kernel)
    return mask_refined


def median_filter_smoothing(mask):
    return cv2.medianBlur(mask, ksize=5)

def main():
    model_path = "model.tflite"
    model = TFLiteModel(model_path)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    latest_mask = None
    frame_counter = 0
    inference_counter_last = 0
    last_frame_time = time.time()
    last_inference_time = time.time()
    fps = 0.0
    inference_fps = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        current_time = time.time()
        if current_time - last_frame_time >= 1:
            fps = frame_counter / (current_time - last_frame_time)
            frame_counter = 0
            last_frame_time = current_time

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model.update_frame(frame_rgb)

        new_mask = model.get_latest_result()
        if new_mask is not None:
            latest_mask = new_mask

        if latest_mask is not None:
            output_mask_resized = cv2.resize(latest_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            output_mask_refined = refine_mask(output_mask_resized)
            output_mask_smoothed = median_filter_smoothing(output_mask_refined)
            output_mask_colored = apply_custom_colormap(output_mask_smoothed)

            overlay = cv2.addWeighted(frame, 0.5, output_mask_colored, 0.5, 0)

            camera_angle = read_camera_angle()
            center, major_axis, minor_axis = find_largest_ellipse(output_mask_smoothed, 4, camera_angle)

            if center is not None:
                cv2.ellipse(overlay, center, (major_axis, minor_axis), 0, 0, 360, (255, 255, 255), 2, cv2.LINE_AA)

            display_width = 432
            display_height = 288

            # Resize each window to the desired display size
            frame_resized = cv2.resize(frame, (display_width, display_height))
            overlay_resized = cv2.resize(overlay, (display_width, display_height))
            mask_resized = cv2.resize(output_mask_colored, (display_width, display_height))

            # Combine the three views
            combined_output = np.hstack((frame_resized, overlay_resized, mask_resized))

            # Add text overlay for FPS and inference rate
            current_inference_time = time.time()
            if current_inference_time - last_inference_time >= 1:
                inference_fps = model.inference_counter / (current_inference_time - last_inference_time)  # Inference FPS calculation
                model.inference_counter = 0  # Reset the counter for next calculation
                last_inference_time = current_inference_time

            cv2.putText(combined_output, f'Frame Rate: {fps:.2f} FPS', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined_output, f'Inference Rate: {inference_fps:.2f} FPS', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show the output
            cv2.imshow("Inference", combined_output)

        else:
            cv2.imshow("Inference", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    model.stop()
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
