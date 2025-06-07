import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def get_landmarks_with_skeleton(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results is None or results.pose_landmarks is None:
        return None, None

    landmarks = []
    skeleton_frame = frame.copy()

    for lm in results.pose_landmarks.landmark:
        landmarks.append((lm.x, lm.y, lm.z, lm.visibility))

    mp_drawing.draw_landmarks(
        image=skeleton_frame,
        landmark_list=results.pose_landmarks,
        connections=mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
    )

    return results.pose_landmarks, skeleton_frame
