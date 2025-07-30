# Organize my own data into the format and structure required by the model

import os
import cv2
import csv
import numpy as np
import sys

def generate_csv(video_folder_path,
                 save_csv_path):
    video_data = []

    # Loop through all files in the folder
    for filename in os.listdir(video_folder_path):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(video_folder_path, filename)

            # Open video to get fps and frame count
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            video_data.append({
                'path': filename,
                'fps': int(fps),
                'start_frame': 1,
                'end_frame': total_frames,
                'mask_id': 1,
                'caption': None
            })

    # Write to CSV
    with open(save_csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['path', 'fps', 'start_frame', 'end_frame', 'mask_id', 'caption']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for entry in video_data:
            writer.writerow(entry)

    print(f"CSV saved to: {save_csv_path}")




def process_mask_video(mask_video_path, save_npz_path):
    cap = cv2.VideoCapture(mask_video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {mask_video_path}")

    frames = []

    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(np.unique(np.array(frame), return_counts=True))

        frames.append(frame)
        i += 1

        # Convert to grayscale if not already
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        print(np.unique(np.array(gray_frame), return_counts=True))

        frames.append(gray_frame)
        sys.exit()

        # cv2.imwrite("/home/ubuntu/jin/data/video_painter/" + str(i) + "_test_mask.png", frame)
        # i += 1

    cap.release()

    if not frames:
        raise ValueError("No frames read from video.")

    # Stack into a 3D numpy array of shape (num_frames, height, width)
    mask_array = np.stack(frames, axis=0)

    # Save to npz file with key 'arr_0'
    np.savez_compressed(save_npz_path, arr_0=mask_array)
    print(f"Saved {len(frames)} frames to {save_npz_path} with shape {mask_array.shape}")


if __name__ == '__main__':
    # generate_csv(video_folder_path="/home/ubuntu/jin/data/video_painter/mt_test",
    #              save_csv_path="/home/ubuntu/jin/data/video_painter/mt_test.csv")

    process_mask_video(mask_video_path="/home/ubuntu/jin/data/test_03_and_04/test_03_mask_trimmed.mp4",
                       save_npz_path="/home/ubuntu/jin/data/video_painter/mt_test_mask/test_03/all_masks.npz")
    #
    # process_mask_video(mask_video_path="/home/ubuntu/jin/data/test_03_and_04/test_04_mask.mp4",
    #                    save_npz_path="/home/ubuntu/jin/data/video_painter/mt_test_mask/test_04/all_masks.npz")
