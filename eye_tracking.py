import cv2
import mediapipe as mp

CAP_WIDTH = 1280 # カメラ画像の横幅
CAP_HEIGHT = 720 # カメラ画像の縦幅

# カメラ画像の解像度に特徴点の座標を合わせる
def resize_lm2camera(x, y):
    return (int(x * CAP_WIDTH), int(y * CAP_HEIGHT))

# 顔の大きさに対する相対的な特徴点の位置を計算
def relative_lm2face(pt, rect):
    x, y = list(pt)
    x1, y1, x2, y2 = rect
    fwidth = abs(x2 - x1)
    fheight = abs(y2 - y1)
    tmp_x = (x - x1) / fwidth
    tmp_y = (y - y1) / fheight
    return (tmp_x, tmp_y)

# 第二引数を基準点としたベクトルの計算
def relative_lm2refpoint(pt1, pt2):
    x = pt1[0] - pt2[0]
    y = pt1[1] - pt2[1]
    return (x, y)

drawing = mp.solutions.drawing_utils

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.8)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

while True:
    
    if cv2.waitKey(1) != -1:
        cv2.destroyAllWindows()
        break
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        for face_landmark in results.multi_face_landmarks:
            min_x = int(min(list(i.x for i in face_landmark.landmark)) * CAP_WIDTH)
            min_y = int(min(list(i.y for i in face_landmark.landmark)) * CAP_HEIGHT)
            max_x = int(max(list(i.x for i in face_landmark.landmark)) * CAP_WIDTH)
            max_y = int(max(list(i.y for i in face_landmark.landmark)) * CAP_HEIGHT)
            face_rect = [min_x, min_y, max_x, max_y]
            eye_L = resize_lm2camera(face_landmark.landmark[473].x, face_landmark.landmark[473].y)
            eye_R = resize_lm2camera(face_landmark.landmark[468].x, face_landmark.landmark[468].y)
            eye_head_L = resize_lm2camera(face_landmark.landmark[243].x, face_landmark.landmark[243].y)
            eye_head_R = resize_lm2camera(face_landmark.landmark[463].x, face_landmark.landmark[463].y)
            eye_end_L = resize_lm2camera(face_landmark.landmark[263].x, face_landmark.landmark[263].y)
            eye_end_R = resize_lm2camera(face_landmark.landmark[33].x, face_landmark.landmark[33].y)
            rel_point_L = relative_lm2refpoint(relative_lm2face(eye_L, rect=face_rect), relative_lm2face(eye_head_L, rect=face_rect))
            rel_point_R = relative_lm2refpoint(relative_lm2face(eye_R, rect=face_rect), relative_lm2face(eye_head_R, rect=face_rect))
            #print('relative_eye_L:{}, relative_eye_R:{}'.format(relative_lm2face(eye_L, rect=face_rect), relative_lm2face(eye_R, rect=face_rect)))
            #print("x:{}, y:{}".format((min_x, max_x), (min_x, min_y)))
            print('Left', rel_point_L)
            print(' Right', rel_point_R)
            print()
            cv2.putText(frame_rgb, str(rel_point_L), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame_rgb, str(rel_point_R), (650, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            
            cv2.rectangle(frame_rgb, (min_x, max_y), (max_x, min_y), color=(0, 255,0), thickness=1)
            cv2.circle(frame_rgb, eye_L, color=(255, 255, 255), radius=5, thickness=-1)
            cv2.circle(frame_rgb, eye_R, color=(255, 255, 255), radius=5, thickness=-1)
            
            cv2.circle(frame_rgb, eye_head_L, color=(255, 255, 255), radius=3, thickness=-1)
            cv2.circle(frame_rgb, eye_head_R, color=(255, 255, 255), radius=3, thickness=-1)
            
            cv2.circle(frame_rgb, eye_end_L, color=(255, 255, 255), radius=3, thickness=-1)
            cv2.circle(frame_rgb, eye_end_R, color=(255, 255, 255), radius=3, thickness=-1)
            
            """
            drawing.draw_landmarks(
                image=frame_rgb,
                landmark_list=face_landmark,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                connection_drawing_spec=drawing.DrawingSpec(thickness=1, circle_radius=1))
            """
    else:
        continue

    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow('mediapipe_face_mesh', frame)
    

# 終了処理 
cap.release()
cv2.destroyAllWindows()
