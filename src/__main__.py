from src.utils.data.process.DataSetPreparation import AugmentImages
from src.utils.data.importer.DataImporter import import_training_data,import_test_data
from src.utils.data.process.DataProcessor import preprocess_training_images,preprocess_test_images
from src.train.Trainer import perform_train
import sys
from ast import literal_eval

from src.test.Tester import perform_test


def main():
    #import_training_data()
    #import_test_data()
    #preprocess_training_images()
    #preprocess_test_images()
    #perform_train()
    #perform_test()
    AugmentImages(sys.argv[1], sys.argv[2], literal_eval(sys.argv[3]), int(sys.argv[4]))

if __name__ == '__main__':
    main()
