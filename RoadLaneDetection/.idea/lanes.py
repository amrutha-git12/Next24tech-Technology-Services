import cv2
import numpy as np
import os
from moviepy.editor import VideoFileClip

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def draw_lines(img, lines):
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 10)


def process_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Define region of interest
    height, width = edges.shape
    vertices = np.array([[(0, height), (width // 2, height // 2), (width, height)]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)

    # Hough transform to detect lines
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, maxLineGap=50, minLineLength=100)

    # Draw lines on the frame
    line_img = np.zeros((height, width, 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    # Overlay the lines on the original frame
    result = cv2.addWeighted(frame, 0.8, line_img, 1, 0)

    return result


def main(input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)

        out.write(processed_frame)

        frame_count += 1
        #print(f"Processed frame {frame_count}")

    cap.release()
    out.release()
    print("All frames processed. Check output_video.avi.")
    os.startfile('output_video.avi')
    #cv2.destroyAllWindows()


if __name__ == "__main__":
    input_video = 'test2.mp4'  # Replace with your input video file
    output_video = 'output_video.avi'  # Output video file
    main(input_video, output_video)
