# text_classfication
text_classfication with CHI and TF-IDF

## 1. word segmentation:  
  - place `train/val/test` dataset into `data/` dir  
  - set `PATH` in `word_segmentation.py`  
  - run `python word segmentation.py`  
  
## 2. extract keyword using CHI value:  
  - all keyword should be extract only using training set  
  - run `python chi.py`  
  - keywords are stored in `data/train_chi.py`
  
## 3. extract text feature using TF-IDF:  
  - set `DATAPATH` & `MATRAIXFILE` in `tf_idf.py`  
  - **DO NOT modify FEATUREPATH**, keywords should always be extract by training set  
  - text feature are stored in `data/train.txt`, `data/val.txt`, `data/test.txt`  

## 4. shuffle samples:  
  - set input and output txt files in `shuffle.txt`  
  - run `python shuffle.py`  
  
## 5. train xgboost model:  
  - run `python xgb.py`  
  - model are stored as `xtrain.model`  
  - test result are stored as `result.txt`
  
## 6. post precess:  
  - run `python post_process.py`  
  - test dataset are divided by it's prediction result into `output/test_result/` directory
