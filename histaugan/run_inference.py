from histaugan.datasets import dataset_multi
from model import MD_multi
from options import TrainOptions
from saver import Saver
parser = TrainOptions()
opts = parser.parse()
saver = Saver(opts)
model = MD_multi(opts)
dataset = dataset_multi(opts)
saver.run_infrence(-1, model, dataset.images)