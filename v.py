# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:05:50 2024

@author: jishn
"""

import cv2

def main():
    # Open the video file
    video_path = "video1.mp4"
    cap = cv2.VideoCapture(video_path)

    # Get the video's frames per second (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Main loop to display and control the video
    play = True
    while cap.isOpened():
        # If 'play' is True, read and display the frame
        if play:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Video Player', frame)

        # Wait for a key press
        key = cv2.waitKey(30)

        # If 'q' is pressed, exit the loop and close the video
        if key == ord('q'):
            break

        # If 'f' is pressed, move forward 10 seconds
        elif key == ord('f'):
            forward_frames = int(fps * 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + forward_frames)

        # If 'b' is pressed, move backward 10 seconds
        elif key == ord('b'):
            backward_frames = int(fps * 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - backward_frames)

        # If 'p' is pressed, toggle the play state
        elif key == ord('p'):
            play = not play

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

