import os
import cv2
import numpy as np
import pandas as pd
import random
import argparse

def generate_single_pseudo_trajectory(num_frames, frame_width, frame_height):
    trajectory = []

    center_x = frame_width // 2
    center_y = frame_height // 2

    start_x = random.randint(center_x - 100, center_x + 100)
    start_y = random.randint(center_y - 100, center_y + 100)

    while abs(start_x - center_x) < 50:
        start_x = random.randint(center_x - 100, center_x + 100)

    while abs(start_y - center_y) < 50:
        start_y = random.randint(center_y - 100, center_y + 100)

    x_step = random.uniform(-1, 1)  # Random linear step for x-axis
    y_step = random.uniform(-1, 1)  # Random linear step for y-axis

    # Reduce the range of random width and height to create smaller bounding boxes
    w = random.randint(int(frame_width * 0.05), int(frame_width * 0.15))
    h = random.randint(int(frame_height * 0.05), int(frame_height * 0.15))

    for i in range(num_frames):
        x = start_x + i * x_step
        y = start_y + i * y_step

        # Keep bounding box within frame boundaries
        x = max(0, min(x, frame_width - w))
        y = max(0, min(y, frame_height - h))

        trajectory.append((i, x, y, w, h))

    return trajectory




def interpolate_trajectory(trajectory):
    x_interp = np.interp(range(trajectory[-1][0] + 1), [t[0] for t in trajectory], [t[1] for t in trajectory])
    y_interp = np.interp(range(trajectory[-1][0] + 1), [t[0] for t in trajectory], [t[2] for t in trajectory])
    w_interp = np.interp(range(trajectory[-1][0] + 1), [t[0] for t in trajectory], [t[3] for t in trajectory])
    h_interp = np.interp(range(trajectory[-1][0] + 1), [t[0] for t in trajectory], [t[4] for t in trajectory])
    return list(zip(x_interp, y_interp, w_interp, h_interp))

def overlay_patch(frame, patch, bbox):
    x, y, w, h = bbox
    resized_patch = cv2.resize(patch, (int(w), int(h)))

    x = min(x, frame.shape[1] - w)  # Ensure x is within frame width
    y = min(y, frame.shape[0] - h)  # Ensure y is within frame height

    frame[y:y+int(h), x:x+int(w)] = resized_patch
    return frame

def process_video(video_path, output_csv, output_video):
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    trajectory = generate_single_pseudo_trajectory(num_frames, frame_width, frame_height)
    interpolated_trajectory = interpolate_trajectory(trajectory)

    records = []

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Select the patch from the first bounding box in the trajectory
    ret, first_frame = cap.read()
    first_bbox = tuple(map(int, interpolated_trajectory[0]))
    patch = first_frame[first_bbox[1]:first_bbox[1]+first_bbox[3], first_bbox[0]:first_bbox[0]+first_bbox[2]]

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame {i}")
            break

        if i < len(interpolated_trajectory):
            bbox = tuple(map(int, interpolated_trajectory[i]))
            frame = overlay_patch(frame, patch, bbox)
            color = (0, 255, 0)  # Green color
            thickness = 2
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + int(bbox[2]), bbox[1] + int(bbox[3])), color, thickness)

        out.write(frame)
        print(f"Processed frame {i}")

        record = {"video": video_name, "frame": f"{video_name}_{i:04d}", "x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3]}
        records.append(record)

    cap.release()
    out.release()  # Release the video writer
    print("Video writer released")

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)



def main(input_folder, output_folder, output_video):
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".avi"):
                video_path = os.path.join(root, filename)
                output_csv = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_bboxes.csv")
                output_path = os.path.join(output_video, filename)
                process_video(video_path, output_csv, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process videos to generate pseudo trajectories and save output.')
    parser.add_argument('input_folder', type=str, help='Path to the input videos folder')
    parser.add_argument('output_folder', type=str, help='Path to the folder where CSV annotations will be saved')
    parser.add_argument('output_video', type=str, help='Path to the folder where processed videos will be saved')
    
    args = parser.parse_args()
    main(args.input_folder, args.output_folder, args.output_video)
