import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import dataset_generator.dataset_generator as dsg
import cnn_classifier.model as cl
import tensorflow as tf

dsg.DatasetGenerator(save_path='datasets/gitignored', generate_immediately=True)
model = cl.Classifier("datasets/gitignored")
model.build()
model.train()
model.plot_confusion_matrix()

model.save('model.keras')