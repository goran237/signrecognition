from src.utils.data.importer.DataImporter import import_data
from src.utils.data.process.DataHelper import DataHelper
from src.utils.data.process.DataProcessor import preprocess_images
from src.utils.data.process.DataSetPreparation import DataSetPreparation
from src.utils.data.process.DataSetPreparation import AugmentImages
import sys


def main():
    import_data()
    preprocess_images()
    dh = DataHelper()
    dh.set_up_images()


if __name__ == '__main__':
    AugmentImages('utils/data/GTSRB/Final_Training/sorted')
    # DataSetPreparation('utils/data/GTSRB/Final_Training/sorted')
    sys.exit(0)
    #main()
    #sys.exit(main() or 0)
