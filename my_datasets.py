from modelscope import MsDataset
from modelscope.utils.constant import Hubs
import os


datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets')
repo_configs = {
    'modelscope_config':{
        'name': 'Amorter',
        'hub': Hubs.modelscope
    },
    'huggingface_config':{
        'name': 'Amort',
        'hub': Hubs.huggingface
    }
}

# 配置仓库
repo_config = repo_configs['modelscope_config']

def get_dataset(dataset_name: str):
    os.makedirs(datasets_dir, exist_ok=True)
    if dataset_name == 'captcha_chinese_click_1':
        return MsDataset.load(repo_config['name'] + '/captcha_chinese_click_1', subset_name='default', split='train', cache_dir=os.path.join(datasets_dir, "captcha_chinese_click_1"), hub=repo_config['hub'])