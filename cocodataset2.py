from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from pycocotools.coco import COCO
from PIL import Image
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
        self.coco = COCO(annFile)
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
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        targ2 = [x for x in target if x['area'] > 1000]

        # print(len(target), len(targ2))

        if len(targ2) > 0:
            target = copy.deepcopy(targ2)

        if len(target) > 0:

            n_rand = random.choice(range(len(target)))

            mask = coco.annToMask(target[n_rand])

            path = coco.loadImgs(img_id)[0]['file_name']

            img = Image.open(os.path.join(self.root, path)).convert('RGB')
            bbox = [int(x) for x in target[n_rand]['bbox']]
            x1 = max(bbox[0] - random.randint(0, 15), 0)
            y1 = max(bbox[1] - random.randint(0, 15), 0)
            x2 = min(bbox[0] + bbox[2] + random.randint(0, 15), img.size[0])
            y2 = min(bbox[1] + bbox[3] + random.randint(0, 15), img.size[1])
            img = img.crop((x1, y1, x2, y2))
            target = mask[y1:y2, x1:x2]

        else:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            target = np.zeros((256, 256), dtype=np.uint8)


        transformed = self.al_transform(image=np.asarray(img), mask=target)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']

        if self.transform is not None:
            transformed_image = self.transform(transformed_image)

        # if self.target_transform is not None:
        #     transformed_mask = self.target_transform(transformed_mask)

        return {'image': transformed_image, 'mask': transformed_mask}


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
    dataset = CocoDetection(root='/home/neptun/PycharmProjects/coco_dataset/img/train2017',
                            annFile='/home/neptun/PycharmProjects/coco_dataset/annotations/instances_train2017.json',
                            )
    train_loader = DataLoader(dataset, shuffle=True)

    i = 0
    for img, targ in train_loader:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(np.moveaxis(img.numpy()[0], 0, 2))
        ax[1].imshow(targ.numpy()[0])
        plt.show()
        i += 1
        if i == 10:
            break
