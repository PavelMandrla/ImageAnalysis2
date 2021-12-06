import glob
import cv2
import torch
import torchvision.models.detection
from torchvision import transforms
from PIL import Image
from utils import load_parking_map

coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
              'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
              'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
              'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A',
              'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
              'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def get_ROI(one_c, x_factor=1, y_factor=1):
    pts = [((float(one_c[0])), float(one_c[1])),
           ((float(one_c[2])), float(one_c[3])),
           ((float(one_c[4])), float(one_c[5])),
           ((float(one_c[6])), float(one_c[7]))]

    min_x = int(min([x for x, y in pts]))
    min_y = int(min([y for x, y in pts]))
    max_x = int(max([x for x, y in pts]))
    max_y = int(max([y for x, y in pts]))

    if x_factor != 1:
        dx = (max_x - min_x)
        difference = dx * x_factor - dx
        min_x -= int(difference / 2)
        max_x += int(difference / 2)

    if y_factor != 1:
        dy = (max_y - min_y)
        difference = dy * y_factor - dy
        min_y -= int(difference / 2)
        max_y += int(difference / 2)


    return min_x, min_y, max_x, max_y


cv2.namedWindow("detection", 0)
print("main")
test_images = [img for img in glob.glob('test_images/*.jpg')]
test_images.sort()

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.eval().to(device)

transform_RCNN = transforms.Compose([
    transforms.ToTensor()
])

pkm_coordinates = load_parking_map()



for img in test_images:

    one_img = cv2.imread(img)
    one_img_paint = one_img.copy()

    one_img_rgb = cv2.cvtColor(one_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(one_img_rgb)
    imageRCNN = transform_RCNN(img_pil).to(device)
    imageRCNN = imageRCNN.unsqueeze(0)

    for one_c in pkm_coordinates:
        #one_img_paint = one_img.copy()

        min_x, min_y, max_x, max_y = get_ROI(one_c, x_factor=0.7, y_factor=0.7)

        #roi = one_img_rgb[min_y:max_y, min_x:max_x]
        '''
        img_pil = Image.fromarray(roi)
        imageRCNN = transform_RCNN(img_pil).to(device)
        imageRCNN = imageRCNN.unsqueeze(0)
        outputsRCNN = model(imageRCNN)
        '''
        #print(min_x, min_y, max_x, max_y)
        #outputsRCNN = model(imageRCNN[:,:,min_x:max_x,min_y:max_y])
        outputsRCNN = model(imageRCNN[:, :, min_y:max_y, min_x:max_x])
        pred_classes = [coco_names[i] for i in outputsRCNN[0]['labels'].cpu().numpy() if coco_names[i] in ['car', 'truck', 'bus']]

        if len(pred_classes) > 0:
            pred_scores = outputsRCNN[0]['scores'].detach().cpu().numpy()
            if max(pred_scores) > 0:
                #print(max(pred_scores))
                cv2.line(one_img_paint, (min_x, min_y), (min_x, max_y), (255, 0, 0), 2)
                cv2.line(one_img_paint, (min_x, min_y), (max_x, min_y), (255, 0, 0), 2)
                cv2.line(one_img_paint, (max_x, max_y), (min_x, max_y), (255, 0, 0), 2)
                cv2.line(one_img_paint, (max_x, max_y), (max_x, min_y), (255, 0, 0), 2)

        #cv2.imshow("detection", one_img_paint)
        #cv2.waitKey(50)

    pred_classes = [coco_names[i] for i in outputsRCNN[0]['labels'].cpu().numpy()]
    pred_scores = outputsRCNN[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputsRCNN[0]['boxes'].detach().cpu().numpy()

    for i, box in enumerate(pred_bboxes):
        if pred_classes[i] not in ['car', 'truck', 'bus']:
            continue

        pts = [((int(box[0])), int(box[1])),
               ((int(box[2])), int(box[1])),
               ((int(box[0])), int(box[3])),
               ((int(box[2])), int(box[3]))]

        cv2.line(one_img_paint, pts[0], pts[1], (0, 255, 0), 2)
        cv2.line(one_img_paint, pts[1], pts[3], (0, 255, 0), 5)
        cv2.line(one_img_paint, pts[3], pts[2], (0, 255, 0), 5)
        cv2.line(one_img_paint, pts[2], pts[0], (0, 255, 0), 5)

    cv2.imshow("detection", one_img_paint)
    cv2.waitKey(30)