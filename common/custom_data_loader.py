# ImageNet
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
import torchvision.transforms as transforms
import os

class UnifiedImageFolder(Dataset):
    def __init__(self, root_dirs, transform=None, loader=default_loader, class_to_idx=None):
        self.root_dirs = root_dirs if isinstance(root_dirs, list) else [root_dirs]
        self.transform = transform
        self.loader = loader
        
        self.classes, self.class_to_idx = self._find_classes(self.root_dirs, class_to_idx)
        self.samples = self._make_dataset(self.root_dirs, self.class_to_idx)
        
        print(f"[UnifiedLoader] Found {len(self.classes)} classes and {len(self.samples)} images across {len(self.root_dirs)} folders.")

    def _find_classes(self, dirs, provided_mapping=None):
        if provided_mapping is not None:
            return list(provided_mapping.keys()), provided_mapping
            
        classes = set()
        for d in dirs:
            if not os.path.isdir(d): continue
            for root, subdirs, files in os.walk(d, followlinks=True):
                for subdir in subdirs:
                    classes.add(subdir)
                break 
        
        classes = sorted(list(classes))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, dirs, class_to_idx):
        instances = []
        for d in dirs:
            d = os.path.expanduser(d)
            if not os.path.isdir(d): continue
            for target_class in sorted(class_to_idx.keys()):
                if target_class not in class_to_idx: continue
                
                class_index = class_to_idx[target_class]
                target_dir = os.path.join(d, target_class)
                if not os.path.isdir(target_dir): continue
                
                for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                    for fname in fnames:
                        if self._is_valid_file(fname):
                            path = os.path.join(root, fname)
                            instances.append((path, class_index))
        return instances

    def _is_valid_file(self, filename):
        return filename.lower().endswith(IMG_EXTENSIONS)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except Exception as e:
            print(f"Error loading {path}: {e}, skipping...")
            return self.__getitem__((index + 1) % len(self.samples))
            
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

def get_data_loaders(config):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Train Transform
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Validation Transform
    val_transform = transforms.Compose([
        transforms.Resize(int(config['image_size'] * 1.14)), # 약 256
        transforms.CenterCrop(config['image_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    base_path = config['data_path']
    # train.X1 ~ train.X4 폴더 자동 탐색
    train_dirs = [os.path.join(base_path, f'train.X{i}') for i in range(1, 5)]
    val_dir = os.path.join(base_path, 'val.X')
    
    # 1. 훈련 데이터셋 생성
    train_dataset = UnifiedImageFolder(train_dirs, transform=train_transform)
    
    # 2. 검증 데이터셋 생성 (훈련 데이터와 동일한 클래스 매핑 적용)
    val_dataset = UnifiedImageFolder([val_dir], transform=val_transform, class_to_idx=train_dataset.class_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                              num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, 
                            num_workers=config['num_workers'], pin_memory=True)
    
    return train_loader, val_loader, val_loader # Test loader 생략 (Validation으로 대체)