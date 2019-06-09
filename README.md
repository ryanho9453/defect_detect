# Image Defect detection 
Image classification for defect detection, using Sparse Representation Classification[1] and other models
(Reference :  J. Wright, A. Yang, A. Ganesh, S. Sastry, and Y. Ma:
Robust face recognition via sparse representation, PAMI (2009))


# Overview

Deal with defect detection problem with images of the product
Using models :
  -  SRC (Sparse Representation-based classification)[1]
  - SVM
  -  Random Forest  
 

Preprocess image data with :
  -   Resize 
  -   Max-pooling
  -   t-SNE ( optional ) 


# Setting
All setting of parameter could be done in config.json
### 1. Preprocessing
  -   process_list : select the process of preprocessing ex :["resize", "max_pooling", "tsne"]
  - data_path : the path of the data folder ( with 2 folder with 'OK' and 'NG' name )
  - train_size : training data size of each class
  - resize : the shape of target size
  - norm : 1/ 0 , normalize the image or not
  - sne_n_comps : n_components in t-SNE
  - mp_filter_size : filter size of the max pooling

### 2. Model
- src alpha - the penalty used in Lasso L1-norm

# Demo

Set your 'OK' and 'NG' folder in "data_path" in config.son
               

```sh
$ python main.py 
```

# Structure 
### 1. Preprocessing
  -   img_processor.py : 
  1. choose train/ test mode to take train/test data
  2. read image files 
  3. preprocess the images
  4. output  shape : ( data_size , feature_size)

### 2. Model
- model_picker.py
1. choose SVM/ Random Forest / SRC 
2. output model 

### 3. Evaluate
- evaluator.py 
evaluate with F1-score and Confusion Matrix

# Reference
[1] J. Wright, A. Yang, A. Ganesh, S. Sastry, and Y. Ma: Robust face recognition via sparse representation, PAMI (2009)

