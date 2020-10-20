from utils.dataset import Cam2007Dataset
class Args :
    def __init__(self):
        self.data_dir = './data/Cam2007_n'

args=Args()
dataset = Cam2007Dataset(args)
dataset.get_pairs()
dataset.statistic_dataset()