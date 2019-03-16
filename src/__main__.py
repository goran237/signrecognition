from src.utils.data.process.DataSetPreparation import AugmentImages
import sys
from ast import literal_eval

def main():
    AugmentImages(sys.argv[1], sys.argv[2], literal_eval(sys.argv[3]), int(sys.argv[4]))

if __name__ == '__main__':
    main()
