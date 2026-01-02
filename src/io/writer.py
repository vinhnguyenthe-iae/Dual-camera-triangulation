import csv
from pathlib import Path

def save_triangulated_keypoints(output_path, triangulated_data):
    """Save triangulated keypoints to a CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ['frame']
        for i in range(21):
            header.extend([f'kp{i}_x', f'kp{i}_y', f'kp{i}_z'])
        writer.writerow(header)
        
        # Data
        for data in triangulated_data:
            row = [data['frame']]
            for point in data['keypoints']:
                if point is not None:
                    row.extend([f"{point[0]:.6f}", f"{point[1]:.6f}", f"{point[2]:.6f}"])
                else:
                    row.extend(['', '', ''])
            writer.writerow(row)

def save_video(output_path, video_frames):
    """Save video frames to a video file."""
    if not video_frames:
        return
    
    height, width, _ = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, 30, (width, height))
    
    for frame in video_frames:
        video_writer.write(frame)
    
    video_writer.release()