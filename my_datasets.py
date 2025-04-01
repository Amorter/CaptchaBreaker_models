from modelscope import MsDataset
import os
datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets')

def get_dataset(dataset_name: str):
    if dataset_name == 'captcha_chinese_click_1':
        return MsDataset.load('Amorter/captcha_chinese_click_1', subset_name='default', split='train', cache_dir=datasets_dir)