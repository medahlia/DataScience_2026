"""
Підрахувати кількість об’єктів на обраному цифровому зображенні.
"""

import cv2


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

images = [
    ('face.jpg', 'example 1'),
    ('friends.jpg', 'example 2'),
    ('test.jpg', 'example 3')
]

scaling_factor = 0.5

for file_name, window_name in images:

    frame = cv2.imread(file_name)

    if frame is None:
        print(f"Не вдалося відкрити {file_name}")
        continue

    frame = cv2.resize(
        frame,
        None,
        fx=scaling_factor,
        fy=scaling_factor,
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_rects = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    print(f"{file_name}: Found {len(face_rects)} faces")

    for (x, y, w, h) in face_rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv2.imshow(window_name, frame)

cv2.waitKey(0)

cv2.destroyAllWindows()