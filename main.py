import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import dataset_generator.dataset_generator as dsg
import cnn_classifier.model as cl

#dsg.DatasetGenerator(save_path='datasets/gitignored', generate_immediately=True)
cl.Classifier().train()
