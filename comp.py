import cv2
import time
from simpleHRNet.SimpleHRNet import SimpleHRNet

# Open the video file with cv2

cap = cv2.VideoCapture('sample.mp4')
fps = cap.get(cv2.cv2.CAP_PROP_FPS)
frame_count = 0
#Initialize the HRNet Model 384X288
model = SimpleHRNet(48, 17, "simpleHRNet/weights/pose_hrnet_w48_384x288.pth")

start_time = time.time()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
        break
    else:
        frame_count+=1
        res = f'{frame.shape[0]}X{frame.shape[1]}'
    # Call skeleton frameworks
    joints = model.predict(frame)
    #Save the frame and the skeletons in the list
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# Caclulate performance stats
proc_time = end_time-start_time

print (f'{fps},{res},{frame_count},{proc_time},{frame_count/proc_time},{proc_time/frame_count}')

# Save the images with the skeletons in the results folder
