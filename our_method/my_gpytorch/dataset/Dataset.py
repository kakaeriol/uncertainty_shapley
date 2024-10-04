import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data import Dataset
from sklearn import preprocessing
import ot
class My_Single_Dataset(Dataset):
    """
    This is class to measuring
    """
    def __init__(self,x,y):
        self.data = x
        self.targets = y
        self.unique_label = torch.unique(y)
        self.num_class = len(self.unique_label)
        if self.unique_label.is_cuda:
            self.unique_label = self.unique_label.detach().cpu().numpy()
        else:
            self.unique_label = self.unique_label.numpy()
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, item):
        return self.data[item],self.targets[item]
    def __delitem__(self, key):
        self.data = np.delete(self.data, key, axis=0)
        self.targets = np.delete(self.targets, key, axis=0)

class UDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.Y, self.data = self.__compute_union_of_ds(datasets)
        self.clf = preprocessing.LabelEncoder()
        self.targets = torch.tensor(self.clf.fit_transform(self.Y), dtype=int)
        self.num_class = len(torch.unique(self.Y))
        self.unique_label = torch.unique(self.Y)
        if self.unique_label.is_cuda:
            self.unique_label = self.unique_label.detach().cpu().numpy()
        else:
            self.unique_label = self.unique_label.numpy()

    def __compute_union_of_ds(self, datasets):
        targets_set = [dataset.targets for dataset in datasets if hasattr(dataset, 'targets')]
        data_set = [dataset.data for dataset in datasets if hasattr(dataset, 'data')]
        return torch.cat(targets_set), torch.cat(data_set)

    def __len__(self):
        return len(self.targets)
    def __getitem__(self, item):
        return self.data[item],self.targets[item]
    def __delitem__(self, key):
        self.data = np.delete(self.data, key, axis=0)
        self.targets = np.delete(self.targets, key, axis=0)

def load_dataset(indicates, datasets, method='dsconcat', batch_size=8):
    """
        There is fours methods "dsconcat", "dlconcat", "concat", "dl" 
    """
    # From indicates to index:
    # print('ds', indicates)
    lds = [datasets[i] for i in indicates]
    if method == 'dsconcat':
        return UDataset(lds)
    elif method == "dlconcat":
        return DataLoader(UDataset(lds), 
        batch_size=batch_size, 
        pin_memory=True, 
        shuffle=True) # should add it into yml file
    elif method == 'concat':
        return torch.utils.data.ConcatDataset(lds)
    elif method == 'dl':
        return torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(lds), 
        batch_size=batch_size,
        pin_memory=True, 
        shuffle=True) #not shuffle to have the same result

