from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from pycocotools.coco import COCO
from lvis import LVIS
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

    def __init__(self, root, annFile_coco, annFile_lvis, is_train):
        self.is_train = is_train
        self.root = root
        self.lvis = LVIS(annFile_lvis)
        self.coco = COCO(annFile_coco)

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
        lvis = self.lvis
        img_id = self.ids[index]
        # ann_ids = coco.getAnnIds(imgIds=img_id)
        ann_ids_lvis = lvis.get_ann_ids(img_ids=[img_id])
        target_lvis = lvis.load_anns(ann_ids_lvis)
        ann_ids_coco = coco.getAnnIds(imgIds=img_id)
        target_coco = coco.loadAnns(ann_ids_coco)

        path = os.path.basename(coco.loadImgs(img_id)[0]['coco_url'])
        # path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        S = img.width*img.height
        targ_base = target_coco #[x for x in target_coco if S*0.1 < x['area'] < S*0.8]
        targ_green = [x for x in target_coco if S*0.02 < x['area'] < S*0.1]
        targ_red = [x for x in target_lvis if S*0.02 < x['area'] < S*0.1]

        draw = ImageDraw.Draw(img)

        if len(targ_base) + len(targ_green) > 0:
            maski = []
            non_maski = []
            if self.is_train:
                for i in range(len(targ_base)):
                    maski.append(coco.annToMask(targ_base[i]))

                for i in range(len(targ_green)):
                    if random.random() < 0.5:
                        # maski.append(coco.annToMask(targ_green[i]))
                        self.draw_points(draw, targ_green, i, colore='green')
                    else:
                        non_maski.append(coco.annToMask(targ_green[i]))
                        self.draw_points(draw, targ_green, i, colore='red')

                for i in range(len(targ_red)):
                    if random.random() < 0.5:
                        non_maski.append(lvis.ann_to_mask(targ_red[i]))
                        self.draw_points(draw, targ_red, i, colore='red')
                    else:
                        maski.append(lvis.ann_to_mask(targ_red[i]))
                        self.draw_points(draw, targ_red, i, colore='green')


            else:
                for i in range(len(targ_base)):
                    maski.append(coco.annToMask(targ_base[i]))
                    self.draw_points(draw, targ_base, i, colore='green')

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

    def draw_points(self, draw, target, n_obj, colore='red'):
        bbox = [int(x) for x in target[n_obj]['bbox']]
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[0] + bbox[2]
        y2 = bbox[1] + bbox[3]
        max_r = 0.2 * min(x2 - x1, y2 - y1)
        if max_r > 15:
            r = random.randint(15, int(max_r))
        else:
            r = 15

        a1 = int((x1 + x2) / 2 - r // 2)
        a2 = int((x1 + x2) / 2 + r // 2)
        b1 = int((y1 + y2) / 2 - r // 2)
        b2 = int((y1 + y2) / 2 + r // 2)
        draw.ellipse((a1, b1, a2, b2), fill=colore, outline=colore)


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
                            annFile_coco='/media/alex/DAtA2/Datasets/coco/lvic_annot/lvis_v1_train.json',
                            annFile_lvis='/media/alex/DAtA2/Datasets/coco/annotations_trainval2017/annotations/instances_train2017.json',
                            is_train=True)
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
