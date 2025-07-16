import cv2

def draw_boxes(frame, tracks):
    for track in tracks:
        x1, y1, x2, y2 = map(int, track['bbox'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)
        cv2.putText(frame, f'ID: {track["id"]}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
