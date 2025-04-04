from modelscope import MsDataset
from modelscope.utils.constant import Hubs
import os


datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets')
modelscope_name = "Amorter"
huggingface_name = "Amort"
name = None

hub = Hubs.modelscope
name = modelscope_name if hub == Hubs.modelscope else name
name = huggingface_name if hub == Hubs.huggingface else name

def get_dataset(dataset_name: str):
    os.makedirs(datasets_dir, exist_ok=True)
    if dataset_name == 'captcha_chinese_click_1':
        return MsDataset.load(name + '/captcha_chinese_click_1', subset_name='default', split='train', cache_dir=os.path.join(datasets_dir, "captcha_chinese_click_1"), hub=hub)