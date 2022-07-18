from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

# from pycocotools.coco import COCO
from lvis import LVIS, LVISResults, LVISEval

from PIL import Image, ImageDraw
import os

import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import albumentations as A

class CocoDetection(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile):
        self.root = root
        self.coco = LVIS(annFile)
        self.ids = list(self.coco.imgs.keys())
        transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.ToTensor()
        ])

        self.transform = transform
        # self.target_transform = target_transform
        self.al_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(256, 256),
        ])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.get_ann_ids(img_ids=img_id)
        target = coco.load_anns(ann_ids)

        targ2 = [x for x in target if x['area'] > 1000]

        # print(len(target), len(targ2))

        if len(targ2) > 0:
            target = copy.deepcopy(targ2)

        path = coco.load_imgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if len(target) > 0:
            maski = []
            non_maski = []

            if random.random() < 0.95:

                n_rands = list(set(random.choices(range(len(target)), k=random.randint(1, len(target)))))
                draw = ImageDraw.Draw(img)

                for n_rand in n_rands:
                    bbox = [int(x) for x in target[n_rand]['bbox']]
                    x1 = bbox[0]
                    y1 = bbox[1]
                    x2 = bbox[0] + bbox[2]
                    y2 = bbox[1] + bbox[3]
                    max_r = 0.2 * min(x2 - x1, y2 - y1)
                    if max_r > 15:
                        r = random.randint(15, int(max_r))
                    else:
                        r = 15

                    self.draw_negativ_points(x1, x2, y1, y2, r, draw)

                for i in range(len(target)):
                    if i in n_rands:
                        non_maski.append(coco.ann_to_mask(target[i]))
                    else:
                        maski.append(coco.ann_to_mask(target[i]))
            else:
                for i in range(len(target)):
                    maski.append(coco.ann_to_mask(target[i]))


            if len(maski) == 0:
                # img = random.randint(0, 255) * np.ones((256, 256, 3), dtype=np.uint8)
                target = np.zeros((256, 256), dtype=np.uint8)
            else:
                mask = maski[0]
                for i in range(1, len(maski)):
                    mask += maski[i]

                for i in range(len(non_maski)):
                    indx = np.where(non_maski[i] == 1)
                    mask[indx] = 0

                mask = np.clip(mask, 0, 1)

                target = mask

        else:
            # img = random.randint(0, 255) * np.ones((256, 256, 3), dtype=np.uint8)
            target = np.zeros((256, 256), dtype=np.uint8)


        transformed = self.al_transform(image=np.asarray(img), mask=target)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']

        if self.transform is not None:
            transformed_image = self.transform(transformed_image)

        # if self.target_transform is not None:
        #     transformed_mask = self.target_transform(transformed_mask)

        return {'image': transformed_image, 'mask': transformed_mask}

    def draw_negativ_points(self, x1, x2, y1, y2, r, draw):

        a1 = int((x1 + x2) / 2 - r)
        a2 = int((x1 + x2) / 2)
        b1 = int((y1 + y2) / 2 - r)
        b2 = int((y1 + y2) / 2)
        draw.ellipse((a1, b1, a2, b2), fill='red', outline='red')
        a1 = int((x1 + x2) / 2)
        a2 = int((x1 + x2) / 2 + r)
        b1 = int((y1 + y2) / 2)
        b2 = int((y1 + y2) / 2 + r)
        draw.ellipse((a1, b1, a2, b2), fill='green', outline='red')
        a1 = int((x1 + x2) / 2 - r)
        a2 = int((x1 + x2) / 2)
        b1 = int((y1 + y2) / 2)
        b2 = int((y1 + y2) / 2 + r)
        draw.ellipse((a1, b1, a2, b2), fill='blue', outline='red')
        a1 = int((x1 + x2) / 2)
        a2 = int((x1 + x2) / 2 + r)
        b1 = int((y1 + y2) / 2 - r)
        b2 = int((y1 + y2) / 2)
        draw.ellipse((a1, b1, a2, b2), fill='yellow', outline='red')


    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        # fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

if __name__ == '__main__':
    # transform = transforms.Compose([
    #     # you can add other transformations in this list
    #     transforms.ToTensor()
    # ])
    dataset = CocoDetection(root='/media/alex/DAtA2/Datasets/coco/train2017',
                            annFile='/media/alex/DAtA2/Datasets/coco/annotations_trainval2017/annotations/instances_train2017.json',
                            )
    train_loader = DataLoader(dataset, shuffle=True)

    i = 0
    for batch in train_loader:
        img = batch['image']
        targ = batch['mask']
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.moveaxis(img.numpy()[0], 0, 2))
        ax[1].imshow(targ.numpy()[0])
        plt.show()
        i += 1
        if i == 10:
            break
