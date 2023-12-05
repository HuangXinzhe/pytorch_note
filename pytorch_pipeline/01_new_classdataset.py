import torch.utils.data as data
import torchvision.transforms as transforms

class MyDataset(data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = pd.read_csv(data_path)
        self.data = self.data.values

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)