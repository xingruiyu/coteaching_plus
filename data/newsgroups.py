from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import pickle
from .utils import noisify

def regroup_dataset(labels):
    """
    categories = dataset.target_names
    labels = [(dataset.target_names.index(cat), cat) for cat in categories]
    [(0, 'alt.atheism'), (1, 'comp.graphics'), (2, 'comp.os.ms-windows.misc'), (3, 'comp.sys.ibm.pc.hardware'), (4, 'comp.sys.mac.hardware'), (5, 'comp.windows.x'), (6, 'misc.forsale'), (7, 'rec.autos'), (8, 'rec.motorcycles'), (9, 'rec.sport.baseball'), (10, 'rec.sport.hockey'), (11, 'sci.crypt'), (12, 'sci.electronics'), (13, 'sci.med'), (14, 'sci.space'), (15, 'soc.religion.christian'), (16, 'talk.politics.guns'), (17, 'talk.politics.mideast'), (18, 'talk.politics.misc'), (19, 'talk.religion.misc')]
    """
    batch_y = labels.copy()
    for i, label in enumerate(labels):
        if label in [0]:
            batch_y[i]=0
        if label in [1, 2, 3, 4, 5,]:
            batch_y[i]=1
        if label in [6]:
            batch_y[i]=2
        if label in [7,8,9,10]:
            batch_y[i]=3
        if label in [11,12,13,14]:
            batch_y[i]=4
        if label in [15]:
            batch_y[i]=5
        if label in [16,17,18,19]:
            batch_y[i]=6
            
    print('regrouped label', batch_y.shape)
    return batch_y
  

class NewsGroups(data.TensorDataset):
    """

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,
                 noise_type=None, noise_rate=0.2, random_state=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.noise_type=noise_type
        self.dataset='news'
        self.weights_matrix, data, labels=pickle.load(open(os.path.join(self.root, "news.pkl"), "rb"), encoding='iso-8859-1')
        labels = regroup_dataset(labels)
        
        length=labels.shape[0]
        self.num_classes = len(set(labels))

        if self.train:
            self.train_data = torch.from_numpy(data[:int(length*0.70)])
            self.train_labels = torch.from_numpy(labels[:int(length*0.70)]).long()

            # noisify train data 
            print(self.train_labels)
            if noise_type is not None:
                self.train_labels=np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                self.train_noisy_labels, self.actual_noise_rate = noisify(dataset=self.dataset, nb_classes=self.num_classes, train_labels=self.train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state)
                self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
                _train_labels=[i[0] for i in self.train_labels]
                self.noise_or_not = np.transpose(self.train_noisy_labels)==np.transpose(_train_labels)
                print('label precision:', 1- self.actual_noise_rate)
        else:
            self.test_data = torch.from_numpy(data[int(length*0.70):])
            self.test_labels = torch.from_numpy(labels[int(length*0.70):])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            if self.noise_type is not None:
                img, target = self.train_data[index], self.train_noisy_labels[index]
            else:
                img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target, index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

