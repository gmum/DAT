from torchvision import datasets
import numpy as np
import torch


class ConditionalDataset(datasets.ImageFolder):
    def __init__(self, *args, n_classes: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes

    def __getitem__(self, *args, **kwargs):
        image, label = super().__getitem__(*args, **kwargs)

        # class will be positive == 'is given class' or 'not given class'
        cls = np.random.randint(2)

        if cls == 0:  # negative sample
            query_vector = torch.zeros(self.n_classes, device=image.device, dtype=image.dtype)
            query_vector[np.random.randint(self.n_classes)] = 1
            while query_vector[label] == 1:
                query_vector = torch.zeros(self.n_classes, device=image.device, dtype=image.dtype)
                query_vector[np.random.randint(self.n_classes)] = 1
            label = 0

        else:  # positive sample
            query_vector = torch.zeros(self.n_classes, device=image.device, dtype=image.dtype)
            query_vector[label] = 1
            label = 1

        return [image, query_vector], label
