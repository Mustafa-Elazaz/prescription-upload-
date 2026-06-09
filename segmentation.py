import cv2

def segment_medicines(original, thresh):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (40,5)
    )

    dilated = cv2.dilate(
        thresh,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for cnt in contours:

        x,y,w,h = cv2.boundingRect(cnt)

        if w > 100 and h > 20:

            boxes.append((x,y,w,h))

    boxes = sorted(
        boxes,
        key=lambda b: b[1]
    )

    crops = []

    for x,y,w,h in boxes:

        crop = original[
            y:y+h,
            x:x+w
        ]

        crops.append(crop)

    return crops