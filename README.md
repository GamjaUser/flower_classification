# flower_classification
## Summary
102 종류의 꽃 데이터를 학습하여 꽃을 판별해줍니다.
(데이터는 [https://www.robots.ox.ac.uk/~vgg/data/flowers/102/] 이곳에서)
___
## Installation
Python 3.11 with the following installed :
```
git clone https://github.com/GamjaUser/flower_classification.git
cd flower_classification
pip install -r requirements.txt
```
___
## Dataset
* link : [https://www.robots.ox.ac.uk/~vgg/data/flowers/102/]
* categories : [https://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html]
* test_img
  * 인터넷에 떠도는 꽃 이미지
___
## Model
* EfficientNetB0
___
## Steps
1. download data
2. Run ```python oxford_flowers102_dataset_builder.py```to train the model
3. Run ```python train.py```to train the model
4. Run ```python test.py```to train the model
___
## Results
거의 90% 이상의 정확도를 보여줌.
___
## Reference
[1] 102 Category Flower Dataset [https://www.robots.ox.ac.uk/~vgg/data/flowers/102/]

[2] oxford_flowers102_dataset_builder [https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/oxford_flowers102/oxford_flowers102_dataset_builder.py]

[3] Apache License, Version 2.0 [https://www.apache.org/licenses/LICENSE-2.0]
