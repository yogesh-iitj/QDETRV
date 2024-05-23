import os
import cv2
import torch
import numpy as np
from model import QDETRv
import utils.utils as utils
import config

def load_model(checkpoint_path, num_classes):
    model = QDETRv(num_classes)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def split_video_into_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def draw_boxes_on_frame(frame, boxes, labels):
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return frame

def process_frames(frames, model, device, transform):
    processed_frames = []
    for frame in frames:
        input_tensor = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
        pred_boxes = outputs[0]['boxes'].cpu().numpy().astype(int)
        pred_labels = outputs[0]['labels'].cpu().numpy()
        frame_with_boxes = draw_boxes_on_frame(frame, pred_boxes, pred_labels)
        processed_frames.append(frame_with_boxes)
    return processed_frames

def combine_frames_into_video(frames, output_video_path, fps=30):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in frames:
        video.write(frame)
    video.release()

def main(query_image_path, target_video_path, output_video_path, checkpoint_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, num_classes).to(device)
    transform = utils.get_val_transforms()

    frames = split_video_into_frames(target_video_path)
    processed_frames = process_frames(frames, model, device, transform)
    combine_frames_into_video(processed_frames, output_video_path)

if __name__ == "__main__":
    query_image_path = "path/to/query/image.jpg"
    target_video_path = "path/to/target/video.mp4"
    output_video_path = "path/to/output/video.mp4"
    checkpoint_path = "path/to/checkpoint.pth"
    num_classes = config.num_classes

    main(query_image_path, target_video_path, output_video_path, checkpoint_path, num_classes)
