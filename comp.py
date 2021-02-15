import cv2
import time
from simpleHRNet.SimpleHRNet import SimpleHRNet
import torch
import numpy as np
from PIL import Image

COLORS = [
    (31,119,180),
    (174,199,232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (188, 189, 34),
    (219, 219, 141),
    (23, 190, 207),
    (158, 218, 229)
]

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
device=torch.device('cuda'))

#Initialize the Openpippaf model
import openpifpaf
net_cpu, _ = openpifpaf.network.factory(checkpoint='shufflenetv2k16w', download_progress=False)
net = net_cpu.to(torch.device('cuda'))
openpifpaf.decoder.CifSeeds.threshold = 0.5
openpifpaf.decoder.nms.Keypoints.keypoint_threshold = 0.2
openpifpaf.decoder.nms.Keypoints.instance_threshold = 0.2
processor = openpifpaf.decoder.factory_decode(net.head_nets, basenet_stride=net.base_net.stride)

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
    # Call HRNet skeleton frameworks
    # skeletons = model.predict(frame)
    # Call Openpifpaf
    # torch_frame = (torch.tensor(np.ascontiguousarray(np.flip(frame, 2)).transpose(2,0,1))).unsqueeze(0).float()
    pil_im = Image.fromarray(frame)
    data = openpifpaf.datasets.PilImageList([pil_im], preprocess=None)
    loader = torch.utils.data.DataLoader(data, batch_size=1, pin_memory=True, collate_fn=openpifpaf.datasets.collate_images_anns_meta)
    for images_batch, _, __ in loader:
        predictions = processor.batch(net, images_batch, device=torch.device('cuda'))[0]

    # #For visualizations
    # for i, ann in enumerate(predictions):
    #     kps = ann.data
    #     assert kps.shape[1] == 3
    #     x = kps[:, 0] * 1
    #     y = kps[:, 1] * 1
    #     v = kps[:, 2]
    #     skeleton = ann.skeleton
    #     if not np.any(v > 0):
    #         break
    #     lines, line_colors, line_styles = [], [], []
    #     for ci, (j1i, j2i) in enumerate(np.array(skeleton) - 1):
    #         if v[j1i] > 0 and v[j2i] > 0:
    #             lines.append([(x[j1i], y[j1i]), (x[j2i], y[j2i])])
    #             line_colors.append(COLORS[ci][::-1] )
    #             if v[j1i] > 0.5 and v[j2i] > 0.5:
    #                 line_styles.append('solid')
    #             else:
    #                 line_styles.append('dashed')
    #     for i, (line,line_color) in enumerate(zip(lines,line_colors)):
    #         start_point = line[0]
    #         end_point = line[1]
    #         frame = cv2.line(frame, start_point, end_point,line_color , 2)

    # print(skeletons[0], skeletons[1])
    #Save the frame and the skeletons in the list
    # Img_list.append((frame, skeletons))
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
# for i,f in enumerate(Img_list):
#     bbox_frame = draw_bounding_box(f[0],f[1])
#     cv2.imwrite('results/'+str(i)+'.jpg',draw_skeletons(bbox_frame,f[1]))
