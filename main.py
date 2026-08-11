import dataset_generator.dataset_generator as dsg

d = dsg.DatasetGenerator(dsg.ShapeLabels.SPHERE, save_path='datasets/train')

d.generate_samples(100)