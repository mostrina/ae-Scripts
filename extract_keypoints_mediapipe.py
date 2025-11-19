# Previous content goes here
# Import necessary libraries
import cv2
import mediapipe as mp

# Function to extract keypoints from video with frame skipping and progress tracking
def extract_keypoints(video_path, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a dictionary to store keypoints
    keypoints_data = []

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_count % frame_skip == 0:
            # Convert the image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            if results.pose_landmarks:
                keypoints = results.pose_landmarks.landmark
                keypoints_data.append([kp for kp in keypoints])

            # Progress tracking
            progress = (frame_count / total_frames) * 100
            print(f'Processing frame: {frame_count}/{total_frames} - Progress: {progress:.2f}%')

        frame_count += 1

    cap.release()
    return keypoints_data

# Example Usage
# keypoints = extract_keypoints('path_to_video.mp4', frame_skip=5)