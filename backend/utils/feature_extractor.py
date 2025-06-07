import numpy as np

def get_landmark_vector(lm, idx):
    return np.array([lm[idx].x, lm[idx].y, lm[idx].z])

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# 1. 跨步角度
def calc_stride_angle(lm):
    return calculate_angle(get_landmark_vector(lm, 24), get_landmark_vector(lm, 26), get_landmark_vector(lm, 23))

# 2. 投擲角度
def calc_throwing_angle(lm):
    return calculate_angle(get_landmark_vector(lm, 12), get_landmark_vector(lm, 14), get_landmark_vector(lm, 16))

# 3. 雙手對稱性
def calc_arm_symmetry(lm):
    return 1 - abs(lm[15].y - lm[16].y)

# 4. 髖部旋轉角度
def calc_hip_rotation(lm):
    return abs(lm[23].z - lm[24].z)

# 5. 右手手肘的高度
def calc_elbow_height(lm):
    return lm[14].y

# 6. 右腳踝高度
def calc_ankle_height(lm):
    return lm[28].y

# 7. 肩膀旋轉角度（z軸差異）
def calc_shoulder_rotation(lm):
    return abs(lm[11].z - lm[12].z)

# 8. 軀幹傾斜角度
def calc_torso_tilt_angle(lm):
    return calculate_angle(get_landmark_vector(lm, 11), get_landmark_vector(lm, 23), get_landmark_vector(lm, 24))

# 9. 投擲距離（右手手腕到肩膀）
def calc_release_distance(lm):
    return np.linalg.norm(get_landmark_vector(lm, 16) - get_landmark_vector(lm, 12))

# 10. 肩膀與髖部橫向距離
def calc_shoulder_to_hip(lm):
    return abs(lm[12].x - lm[24].x)

# 將所有特徵組合成一個向量
def extract_all_features(landmarks):
    features = {
        "stride_angle": float(calc_stride_angle(landmarks)),
        "throwing_angle": float(calc_throwing_angle(landmarks)),
        "arm_symmetry": float(calc_arm_symmetry(landmarks)),
        "hip_rotation": float(calc_hip_rotation(landmarks)),
        "elbow_height": float(calc_elbow_height(landmarks)),
        "ankle_height": float(calc_ankle_height(landmarks)),
        "shoulder_rotation": float(calc_shoulder_rotation(landmarks)),
        "torso_tilt_angle": float(calc_torso_tilt_angle(landmarks)),
        "release_distance": float(calc_release_distance(landmarks)),
        "shoulder_to_hip": float(calc_shoulder_to_hip(landmarks)),
    }
    return features