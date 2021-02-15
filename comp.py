import cv2
import time
from simpleHRNet.SimpleHRNet import SimpleHRNet
import torch

# Open the video file with cv2

cap = cv2.VideoCapture('sample_multi_person.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
#Initialize the HRNet Model 384X288
model = SimpleHRNet(48, 17, "simpleHRNet/weights/pose_hrnet_w48_384x288.pth", resolution=(384,288), multiperson=True, 
return_bounding_boxes=True,
yolo_model_def='simpleHRNet/models/detectors/yolo/config/yolov3.cfg',
yolo_class_path='simpleHRNet/models/detectors/yolo/data/coco.names', 
yolo_weights_path='simpleHRNet/models/detectors/yolo/weights/yolov3.weights', 
device=torch.device('cuda'),)

start_time = time.time()

def draw_bounding_box(frame, skeletons):
    for bbox in skeletons[0]:
        mod_frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),2)
    return mod_frame

def draw_skeletons(frame, skeletons):
    for skeleton in skeletons[1]:
        for joint in skeleton:
            mod_frame = cv2.circle(frame, (joint[1],joint[0]), 2, (0,0,255),2)
    return mod_frame

Img_list = []


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
        break
    else:
        frame_count+=1
        res = f'{frame.shape[0]}X{frame.shape[1]}'
    # Call skeleton frameworks
    skeletons = model.predict(frame)
    print (skeletons[0], skeletons[1])
    #Save the frame and the skeletons in the list
    Img_list.append((frame, skeletons))
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
for i,f in enumerate(Img_list):
    bbox_frame = draw_bounding_box(f[0],f[1])
    cv2.imwrite('results/'+str(i)+'.jpg',draw_skeletons(bbox_frame,f[1]))
