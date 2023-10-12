import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyTrainDataset(Dataset):
    def __init__(self):
        with open('./training/mimic_cxr/annotation.json', 'rt') as jsonfile:
            f = json.load(jsonfile)
            self.data = f['train']
            # for i in range(len(f['train'])):
            #     self.report = f['train'][i]['report']
            #     self.img_path = f['train'][i]['image_path'][0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['image_path'][0]
        target_filename = item['image_path'][0]
        prompt = item['report']

        source = cv2.imread('./training/mimic_cxr/images/' + source_filename)
        target = cv2.imread('./training/mimic_cxr/images/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
    
class MyValDataset(Dataset):
    def __init__(self):
        with open('./training/mimic_cxr/annotation.json', 'rt') as jsonfile:
            f = json.load(jsonfile)
            self.data = f['val']
            # for i in range(len(f['train'])):
            #     self.report = f['train'][i]['report']
            #     self.img_path = f['train'][i]['image_path'][0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['image_path'][0]
        target_filename = item['image_path'][0]
        prompt = item['report']

        source = cv2.imread('./training/mimic_cxr/images/' + source_filename)
        target = cv2.imread('./training/mimic_cxr/images/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
    
class MyTestDataset(Dataset):
    def __init__(self):
        with open('./training/mimic_cxr/annotation.json', 'rt') as jsonfile:
            f = json.load(jsonfile)
            self.data = f['test']
            # for i in range(len(f['train'])):
            #     self.report = f['train'][i]['report']
            #     self.img_path = f['train'][i]['image_path'][0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['image_path'][0]
        target_filename = item['image_path'][0]
        prompt = item['report']

        source = cv2.imread('./training/mimic_cxr/images/' + source_filename)
        target = cv2.imread('./training/mimic_cxr/images/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
    
class MyNIH(Dataset):
    def __init__(self):
        self.data = []
        with open('/home/leo/NIHjson/train/Atelectasis.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['source']
        prompt = item['prompt']

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
    
class MyInference(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        prompt = 'cardiomegaly'

        source = cv2.imread('/mnt/2TBHDD/leo/ControlNet/1.png')
        target = cv2.imread('/mnt/2TBHDD/leo/ControlNet/1.png')
        
        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
    
if __name__== "__main__" :
    dataset = MyTrainDataset()
    print(len(dataset))

    item = dataset[1234]
    jpg = item['jpg']
    txt = item['txt']
    hint = item['hint']
    print(txt)
    print(jpg.shape)
    print(hint.shape)