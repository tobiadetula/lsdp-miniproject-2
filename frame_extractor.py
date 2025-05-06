import numpy as np
import cv2
import os

main_dir = "../"
script_dir = os.path.dirname(__file__)
output_rel = "../output"   
first_1200 = "../output/first_1200"
leftover = "../output/leftover"
output_path = os.path.join(script_dir, output_rel)
first_1200_path = os.path.join(script_dir, first_1200)
leftover_path = os.path.join(script_dir, leftover)

cap = cv2.VideoCapture(os.path.join(main_dir, "DJI_0199.MOV"))
image = None
frame_number_to_save = 1  # specify the frame number you want to save

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Save a specified frame as an image
    print(f"Current frame number: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}")
    current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if current_frame_number == frame_number_to_save:
        row, col, channel = frame.shape
        image = frame
        frame_number_to_save = current_frame_number + 25
        print(f"Frame shape: {row}x{col}x{channel}")
        print(f"Saving frame {frame_number_to_save} to {output_path}")
        # cv2.imwrite(os.path.join(output_path, f"frame{current_frame_num ber:06d}.png"), frame)
        if current_frame_number < 1200:
            cv2.imwrite(os.path.join(first_1200_path, f"frame{current_frame_number:06d}.png"), frame)
        else:
            cv2.imwrite(os.path.join(leftover_path, f"frame{current_frame_number:06d}.png"), frame)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Get the number of frames in the video
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the duration of the video in seconds
video_length = num_frames / fps
print(f"Video length in seconds: {video_length}")
print(f"Number of frames: {num_frames}")


cap.release()
cv2.destroyAllWindows()
