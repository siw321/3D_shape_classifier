import dataset_generator.dataset_generator as dsg

d = dsg.DatasetGenerator(dsg.ShapeLabels.TETRAHEDRON, save_path='datasets/gitignored')
d.generate_samples(5)
d.set_shape(dsg.ShapeLabels.SPHERE)
d.generate_samples(5)
d.set_shape(dsg.ShapeLabels.CUBE)
d.generate_samples(5)