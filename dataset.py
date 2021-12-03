#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np

# 关于lmdb数据库使用, 当时对接Python 2.x，所以使用Bytestrings，而不是unicode，
# 所以在Python 3.x中要显示encode，decode。
# https://lmdb.readthedocs.io/en/release/
# uses bytestring to mean either the Python<=2.7 str() type, or the Python>=3.0 bytes() type, d
# Always explicitly encode and decode any Unicode values before passing them to LMDB.

class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()).decode())
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode()).decode()

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        output_images = []
        for image in images:
            if self.keep_ratio:
                w, h = image.size
                ratio = w / float(h)
                imgW = int(np.floor(ratio * imgH))
                imgW = min(imgH * self.min_ratio, imgW)  # assure image.w <= imgW
            # resize to the same imgH
            transform = resizeNormalize((imgW, imgH))
            output_images.append(transform(image))
        # padding
        # image.shape i.e. (1, 32, 100)
        max_image_width = max([image.shape[2] for image in output_images])
        max_label_length = max([len(label) for label in labels])
        batch_size = len(output_images)
        channel_size = 1
        inputs = np.zeros((batch_size, channel_size, imgH, max_image_width), dtype='float32')
        # '_' for blank label
        output_labels =[['_'] * max_label_length for _ in range(batch_size)]
        for x in range(batch_size):
            image = output_images[x]
            width = image.shape[2]
            inputs[x, :, :, :width] = image
            output_labels[x][:len(labels[x])] = labels[x]

        # list to str
        output_labels = [''.join(x) for x in output_labels]
        images = torch.cat([torch.from_numpy(t).unsqueeze(0) for t in inputs], 0)

        return images, output_labels
