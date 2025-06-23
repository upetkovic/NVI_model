import cv2
import os


def get_video_frame_count(video_path):
    """
    Returns the number of frames in the input video.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        int: Total number of frames in the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count



def get_video_length(video_path):
    """
    Returns the length of the video in seconds.

    Args:
        video_path (str): Path to the video file.

    Returns:
        float: Length of the video in seconds, or None if unable to retrieve.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps > 0:
        return int(frame_count / fps)
    else:
        print("Error: Unable to retrieve FPS.")
        return None

def save_first_frame(video_path, output_path=None):
    """
    Extracts and saves the first frame of a video.

    Parameters:
    - video_path (str): Path to the input video file.
    - output_path (str, optional): Where to save the image. If None, saves as <video_name>_first_frame.jpg in the same folder.

    Returns:
    - str: Path to the saved image file, or None if extraction failed.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ Failed to read the first frame from: {video_path}")
        return None

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(os.path.dirname(video_path), f"{base_name}_first_frame.jpg")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if cv2.imwrite(output_path, frame):
        print(f"✅ First frame saved to: {output_path}")
    else:
        print(f"Failed to save the first frame of {os.path.basename(video_path)}")
    return output_path

import cv2

def resample_video(input_path, output_path, frame_step=5):
    # Open the input video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {input_path}")
        return 0

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID', etc.
    codec = cv2.VideoWriter_fourcc(*'X264')  # if your system supports it


    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Open output video writer
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    frame_idx = 0
    written_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            out.write(frame)
            written_frames += 1

        frame_idx += 1

    cap.release()
    out.release()
    print(f"Resampled video saved to: {output_path}")
    return written_frames


import subprocess

import subprocess

def resample_with_ffmpeg(input_path, output_path, frame_step=5):
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-i', input_path,
        '-vf', f'select=not(mod(n\\,{frame_step}))',
        '-vsync', 'vfr',
        '-c:v', 'libx264',
        '-crf', '14',
        '-preset', 'fast',
        output_path
    ]

    try:
        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print("FFmpeg completed successfully.")
        print(result.stderr)

    except subprocess.CalledProcessError as e:
        print("FFmpeg failed:")
        print(e.stderr)
        return None

    # Now use ffprobe to get the number of frames
    ffprobe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-count_frames',
        '-show_entries', 'stream=nb_read_frames',
        '-print_format', 'default=nokey=1:noprint_wrappers=1',
        output_path
    ]

    try:
        result = subprocess.run(
            ffprobe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        frame_count = int(result.stdout.strip())
        return frame_count

    except subprocess.CalledProcessError as e:
        print("FFprobe failed:")
        print(e.stderr)
        return None







def get_video_paths(folder_path, extensions=None):
    """
    Returns a list of full paths to video files in the given folder.

    Args:
        folder_path (str): Path to the folder.
        extensions (set or list, optional): Video file extensions to include. 
            Defaults to common video formats.

    Returns:
        list: List of full paths to video files.
    """
    if extensions is None:
        extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

    video_paths = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                video_paths.append(os.path.join(folder_path, filename))
    return video_paths

def get_images_paths(folder_path, extensions=None):
    """
    Returns a list of full paths to image files in the given folder.

    Args:
        folder_path (str): Path to the folder.
        extensions (set or list, optional): Image file extensions to include. 
            Defaults to common image formats.

    Returns:
        list: List of full paths to image files.
    """
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    image_paths = []
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                image_paths.append(full_path)
    return image_paths






if __name__ == "__main__":

    video_path = "data/inputs/videos/00281.avi"
    #output_path = "data/inputs/videos/00152r.avi"
    # Example usage
    #resample_video(video_path, output_path, frame_step=7)

    #video_path = "/home/uros/Documents/project31/programming/NI_clean/data/videos/00152.avi"
    output_path = "data/inputs/first_frames/00281.png"
    save_first_frame(video_path, output_path=output_path)