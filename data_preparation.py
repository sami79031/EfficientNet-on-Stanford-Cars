from pathlib import Path

import pandas as pd
from mat4py import loadmat


def prepare_raw_data(data_root=Path('data/input/stanford')):
    devkit = data_root / 'devkit'
    
    # Convert Path objects to strings for loadmat
    cars_meta_path = str(devkit / 'cars_meta.mat')
    cars_annos_path = str(devkit / 'cars_annos.mat')
    
    if Path(cars_meta_path).exists():
        cars_meta = pd.DataFrame(loadmat(cars_meta_path))
        cars_meta.to_csv(devkit / 'cars_meta.csv', index=False)
    else:
        print(f"Warning: {cars_meta_path} not found, skipping cars_meta processing")

    if Path(cars_annos_path).exists():
        cars_annos = pd.DataFrame(loadmat(cars_annos_path)['annotations'])
        cars_annos['class'] -= 1
        cars_annos.to_csv(devkit / 'cars_annos.csv', index=False)
    else:
        print(f"Warning: {cars_annos_path} not found, skipping cars_annos processing")


if __name__ == '__main__':
    prepare_raw_data()