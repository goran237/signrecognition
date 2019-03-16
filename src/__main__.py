from src.utils.data.importer.DataImporter import import_data
from src.utils.data.process.DataHelper import DataHelper
from src.utils.data.process.DataProcessor import preprocess_images
from src.utils.data.process.DataSetPreparation import DataSetPreparation
from src.utils.data.process.DataSetPreparation import AugmentImages
import sys
from ast import literal_eval

def main():
    import_data()
    preprocess_images()
    dh = DataHelper()
    dh.set_up_images()


if __name__ == '__main__':
    AugmentImages(sys.argv[1], sys.argv[2], literal_eval(sys.argv[3]), int(sys.argv[4]))
    # DataSetPreparation('utils/data/GTSRB/Final_Training/sorted')
    sys.exit(0)
    #main()
    #sys.exit(main() or 0)
