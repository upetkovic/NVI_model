import cv2
import os
from facenet_pytorch import MTCNN
import torch
from utils.helpers import compute_mask_box_iou
import numpy as np

# Paths
tracking_path = "data/outputs/tracking/00317_teacher"
video_path = "data/outputs/resampled_videos/00317.avi"
faces_path = "data/outputs/teacher_faces/00317"
#model
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=15, device=device)

def detect_face(frame):
    bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
    if bounding_boxes is None:
        return None
    bounding_boxes=bounding_boxes[probs>0]
    return bounding_boxes

def extract_teacher_faces(tracking_path, video_path, faces_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    os.makedirs(faces_path, exist_ok=True)

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        bboxes = detect_face(frame)  # list of [x1, y1, x2, y2]

        # Load teacher mask
        mask_path = os.path.join(tracking_path, f"{frame_idx:05d}.png")
        teacher_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        teacher_mask = (teacher_mask > 0).astype(np.uint8)  # ensure binary

        # Find best matching face
        best_iou = 0
        best_box = None
        if bboxes is None or len(bboxes) == 0:
            bboxes = []

        for box in bboxes:
            iou = compute_mask_box_iou(teacher_mask, box)
            if iou > best_iou:
                best_iou = iou
                best_box = box

        # Optional: crop and save the best-matching face
        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box)
            face_img = frame[y1:y2, x1:x2]
            save_path = os.path.join(faces_path, f"{frame_idx:05d}.png")
            cv2.imwrite(save_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))

        frame_idx += 1

    cap.release()
    print("Done.")

if __name__ == "__main__":
    """
    extract_teacher_faces(
        tracking_path="data/outputs/tracking/00281_teacher",
        video_path="data/outputs/resampled_videos/00281.avi",
        faces_path="data/outputs/teacher_faces/00281"
    )
    """

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking_path", required=True, help="Full path to the tracking folder of a teacher(including subfolder)")
    parser.add_argument("--video_path", required=True, help="Full path to the video file of a teacher")
    parser.add_argument("--faces_path", required=True, help="Full path to the output faces folder of teacher(including subfolder)")
    args = parser.parse_args()

    #extract_teacher_faces(tracking_path, video_path, faces_path)
    extract_teacher_faces(
        tracking_path=args.tracking_path,
        video_path=args.video_path,
        faces_path=args.faces_path
    )
