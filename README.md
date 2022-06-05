# hand_written_recognition_project

## Task 1

First, ensure to have Python version 3.8.
Then, install the required packages using the following command:
```
pip install -r requirements.txt
```

## Test Image
Keep your images in one folder. Our model is only designed for binary image.

You can pass the folder path of images by using `--image_path "Path/to/Folder"`


## Character segmentation using prtrained model
We have 4 different models. The pretrained models can be found [here.](https://drive.google.com/drive/folders/1NqDdevh42zpoUPsWMux6de33plXEdUpB). 
We recommend to download the InceptionResNetV2, since it yielded the highest accuracy on the test data. 
Then, pass the path of the classifier using `--classifier_path "Path/of/model"

Use the corresponding *flag* from the following list based on the model you are using for results:

1. scratch implemented CNN (`flag = 0`)
2. InceptionV3 (`flag = 1`)
3. ResNet50 (`flag = 2`)
4. InceptionResNetV2 (`flag = 3`)<br/>

Example:
```
python main.py --image_path "Path/to/Folder" --classifier_path "Path/to/model" --flag 3
```


## Training Classifier

This part is optional and is not required for evaluating the pipeline, as the classifier is already trained.
If you wish to retrain the classifier you can run:

The script includes 4 different models, which can be chosen by adjusting the `flag` 
1. scratch implemented CNN (`flag = 0`)
2. InceptionV3 (`flag = 1`)
3. ResNet50 (`flag = 2`)
4. InceptionResNetV2 (`flag = 3`)<br/>

Furthermore, the number of augmented pictures n can be modified by selecting `number_of_augmentations_per_class = n`. 
The current parameters are the parameters which were used to train our classifier (InceptionResNetV2, no augmentation). 
After running the script, the model is automatically saved in *Model_dir*. It is also possible to pass model directory as argument.

Example:

```
python classifier_pipeline.py --flag 0 --model_dir "models" --epochs 50 --aug_no 0
```







