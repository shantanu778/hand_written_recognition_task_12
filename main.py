import argparse
import os
from character_segement import *
from char_classifier import *
from map_character import *
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type = str, help='Path to input image.', default='test_image')
    parser.add_argument('--classifier_path', type = str, help='Path to Pretrained Model to Classify Character.', default='cnn.sav')
    parser.add_argument('--verbose', action= 'store_true', help='use this argumnent save intermediary images')
    parser.add_argument('--flag', type = int, help='Model type 0 to 3.', default=0)
    args = parser.parse_args()

    if args.flag == 0:
        IMG_WIDTH = 60
        IMG_HEIGHT = 40
    else:
        IMG_WIDTH = 100
        IMG_HEIGHT = 100
    #image_path = cv.imread(cv.samples.findFile(args.input))
    dir_path = f"{args.image_path}"
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    total_files = []
    image_names = []
    total_spaces = []
    for file in files:
        imageName = file.split("/")[-1].split(".")[0]
        print(imageName)
        image_names.append(imageName)
        line_chars, total_space = segment_lines(file, args.verbose)
        total_files.append(line_chars)
        total_spaces.append(total_space) 

    # print(total_files)
    # print(image_names)

    os.makedirs("results/", exist_ok=True)
    folder = "results/"
    
    classifier = args.classifier_path
    loaded_model = pickle.load(open(classifier, 'rb'))

    for idx, file in enumerate(total_files):
        file1 = open(f"{folder}/{image_names[idx]}_characters.txt","a")#append mode
        for id, characters in enumerate(file):
            if len(characters)>0:
                letters, values = classify(characters, loaded_model, IMG_WIDTH, IMG_HEIGHT)
                values.reverse()
                
                ascii = map_characters(values)
                ascii = [c for c in ascii if c is not None]
                for s in total_spaces[idx][id]:
                    if s == -100:
                        ascii.insert(s, "@")
                    else:
                        ascii.insert(s, "#")

                s = str("".join(ascii))
                s = s.replace('#',' ')
                s = s.replace('@','\n')
                s ="".join(s)
                # print(s)
                file1.write(s)
                file1.write('\n')
            else:
                file1.write('\n')
                
        print("Classify Finished", imageName)
        print("complete writing", imageName)
        file1.close()


if __name__ == '__main__':
    main()
