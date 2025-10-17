import os
from enum import auto, Enum
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config.structure import get_data_sources
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


class DatasetTypes(Enum):
    train = auto()
    val = auto()


class StanfordCarsDatasetUltra(Dataset):
    def __init__(self, data_directory, annotations, image_size, dataset_type: DatasetTypes, in_channels):
        self.data_directory = data_directory
        self.annotations = annotations
        self.image_size = image_size
        self.dataset_type = dataset_type
        self.greyscale_conversion = in_channels == 1

        # Ultra-optimized transforms - minimal augmentation for speed
        if dataset_type == DatasetTypes.train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # Filter annotations for this dataset type
        is_test = dataset_type == DatasetTypes.val
        self.filtered_annotations = self.annotations[self.annotations['test'] == is_test]
        
        # Get image file names
        self.image_file_names = self.filtered_annotations['relative_im_path']
        
        # Debug: print dataset info
        print(f"Ultra Dataset type: {dataset_type}")
        print(f"Total annotations: {len(annotations)}")
        print(f"Test flag: {is_test}")
        print(f"Filtered images: {len(self.image_file_names)}")
        if len(self.image_file_names) > 0:
            print(f"First image: {self.image_file_names.iloc[0]}")
        else:
            print("No images found for this dataset type!")

    def load_transform(self, image_file_name):
        image_path = self.data_directory / image_file_name
        image = Image.open(image_path)
        
        if self.greyscale_conversion:
            image = image.convert('L')
            image = image.convert('RGB')
        
        return self.transform(image)

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, index):
        try:
            file_name = self.image_file_names.iloc[index]
            image = self.load_transform(image_file_name=file_name)
            class_id = self.annotations[self.annotations['relative_im_path'] == file_name]['class'].values[0]
            return image, torch.as_tensor(class_id, dtype=torch.long)
        except Exception as e:
            log.error(f"Error loading image {index}: {e}")
            raise


class StanfordCarsDataModuleUltra(LightningDataModule):
    """Ultra-optimized data module for maximum speed."""

    def __init__(self, image_size, batch_size, in_channels, root_path=Path('.')):
        super().__init__()
        self.dataset_info = get_data_sources(root_path)['stanford']
        self.annotations = pd.read_csv(self.dataset_info['annotations']['csv_file_path'])
        self.in_channels = in_channels
        self.image_size = image_size
        self.batch_size = batch_size
        
        # Ultra-optimized settings
        self.num_workers = min(16, os.cpu_count() * 2)  # Maximum workers
        self.pin_memory = True
        self.persistent_workers = True
        self.prefetch_factor = 4  # More prefetching

    def setup(self, stage=None):
        log.info(f"Loading ultra train data from: {self.dataset_info['data_dir']}; image size: {self.image_size}")
        log.info(f"Using {self.num_workers} workers for ultra data loading")
        
        self.train_data = StanfordCarsDatasetUltra(
            self.dataset_info['data_dir'], 
            self.annotations, 
            self.image_size,
            DatasetTypes.train, 
            self.in_channels
        )
        self.val_data = StanfordCarsDatasetUltra(
            self.dataset_info['data_dir'], 
            self.annotations, 
            self.image_size,
            DatasetTypes.val, 
            self.in_channels
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=self.pin_memory, 
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
            multiprocessing_context='spawn',  # Use spawn for better performance
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, 
            batch_size=self.batch_size, 
            pin_memory=self.pin_memory, 
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            multiprocessing_context='spawn',
        )
