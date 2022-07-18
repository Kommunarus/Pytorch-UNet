import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='checkpoints/checkpoint_epoch200.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', help='Filename of input image', default='video/photo10.jpg')
    parser.add_argument('--output', '-o', help='Filename of output image', default='1')
    parser.add_argument('--x-y', '-x', help='x y', default='0.85, 0.5')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.3,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()




def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


def add_draw(img, x_y):
    arr = x_y.split(',')
    w = img.size[0]
    h = img.size[1]
    if len(arr) % 2 == 0:
        arr = [float(x.strip()) for x in arr]
        arr2 = [(int(arr[i]), int(arr[i+1])) for i in range(0, len(arr), 2)]
        draw = ImageDraw.Draw(img)
        r = 20
        for x, y in arr2:
            a1 = x - r
            a2 = x
            b1 = y - r
            b2 = y
            draw.ellipse((a1, b1, a2, b2), fill='red', outline='red')
            a1 = x
            a2 = x + r
            b1 = y
            b2 = y + r
            draw.ellipse((a1, b1, a2, b2), fill='green', outline='red')
            a1 = x - r
            a2 = x
            b1 = y
            b2 = y + r
            draw.ellipse((a1, b1, a2, b2), fill='blue', outline='red')
            a1 = x
            a2 = x + r
            b1 = y - r
            b2 = y
            draw.ellipse((a1, b1, a2, b2), fill='yellow', outline='red')
    return img

def get_points(img):
    template = cv2.imread('Крестик.jpg', 0)
    w, h = template.shape[::-1]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    res = (255 * (res + 1)/2).astype(np.uint8)
    threshold = 190
    # loc = np.where(res >= threshold)
    ret, thresh = cv2.threshold(res, threshold, 255, 0)
    сontours, hierarchy = cv2.findContours(thresh, 1, 2)

    # x = np.zeros_like(img)
    # cv2.drawContours(x, сontours, -1, (0, 255, 0), 3)
    # plt.imshow(thresh)
    # plt.show()
    # print(len(сontours))

    x_y = ''
    for cnt in сontours:
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'] + w/2)
        cy = int(M['m01'] / M['m00'] + h/2)
        x_y += str(cx)+', ' + str(cy) + ', '
    # for pt in zip(*loc[::-1]):
    #     x_y += str((pt[0] + w)/2)+', ' + str((pt[1] + h)/2) + ', '
    if len(x_y) > 0:
        x_y = x_y[: -2]
    return x_y

def run_net(inp, id_file, inp_cross):
    net = UNet(n_channels=3, n_classes=2, bilinear=False)

    device = torch.device('cpu')
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    net.to(device=device)
    net.load_state_dict(torch.load('/home/neptun/PycharmProjects/Pytorch-UNet/checkpoints/checkpoint_epoch200.pth', map_location=device))

    out_name = '/home/neptun/PycharmProjects/Pytorch-UNet/out/' + id_file + '.jpg'

    frame = cv2.imread(inp)
    frame_cross = cv2.imread(inp_cross)
    img_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inp_cross_raw = cv2.cvtColor(frame_cross, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_raw)
    x_y = get_points(inp_cross_raw)
    img = add_draw(img, x_y)
    # img_raw = add_draw(img_raw, x_y)

    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=0.3,
                       out_threshold=0.5,
                       device=device)


    ndx = np.where(mask[1] == 1)

    arr = x_y.split(',')
    w = img.size[0]
    h = img.size[1]
    if len(arr) % 2 == 0:
        arr = [float(x.strip()) for x in arr]
        arr2 = [(int(arr[i]), int(arr[i+1])) for i in range(0, len(arr), 2)]
        for x, y in arr2:
            cv2.circle(img_raw, (x, y), 7, (255, 0, 0), 3, 1)



    img_raw[ndx[0], ndx[1],:] = (0.3*img_raw[ndx[0], ndx[1],:] + 0.7*np.array([96, 96, 196])).astype(np.uint8)
    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_name, img_raw)
    os.remove(inp)
    os.remove(inp_cross)


if __name__ == '__main__':
    # ts = time.time()
    args = get_args()
    in_files = 'local/9472d0ff-51ee-496e-af7f-9126cb9064a9.jpg'
    out_files = args.output
    in_files2 = 'local/d19620d4-1103-4d78-815d-5f09be22d45c.jpg'
    run_net(in_files, out_files, in_files2)
    # print(time.time() - ts)