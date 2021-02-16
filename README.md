## Marmara University - Computer Engineering Term Project

Developed by 

- **Ayşenur YILMAZ** [@Aysenuryilmazz](https://github.com/Aysenuryilmazz)
- **Mahmut AKTAŞ** [@mahmutaktas](https://github.com/mahmutaktas)
- **Mustafa Abdullah HAKKOZ** [@mustafahakkoz](https://github.com/mustafahakkoz)

# Driver_Drowsiness_Detection

Driver drowsiness is one of the causes of traffic accidents. According to the statistics; highway road crashes hold 11.09% of the total number of accidents. There are several reasons of drowsy driving such as: a lack of quality of sleep, may be overnight driving or having sleep disorders e.g. sleep apnea. However; all people should know that: People can not fight against to sleep. 

Using Image Processing and both classical and brand-new Machine Learning techniques such as SVM, k-NN, XGBoost, and also LSTM; we are trying to predict beforehand the driver's drowsiness and warning him/her with an alert before any crash happened. 

### Aims of the Project:

<img src="https://github.com/Aysenuryilmazz/Driver_Drowsiness_Detection/blob/hakkoz/images/aims.png" height="150" />  

### Real-time Application:

<img src="https://github.com/Aysenuryilmazz/Driver_Drowsiness_Detection/blob/hakkoz/images/app.png" height="200" />  

### Pipeline:

<img src="https://github.com/Aysenuryilmazz/Driver_Drowsiness_Detection/blob/hakkoz/images/methodology.png" height="400" />  

### Hand-made Features:

<img src="https://github.com/Aysenuryilmazz/Driver_Drowsiness_Detection/blob/hakkoz/images/features.png" height="500" />  

### Frame Based Models Classification Results:

<img src="https://github.com/Aysenuryilmazz/Driver_Drowsiness_Detection/blob/hakkoz/images/framebasedresults.jpeg" height="300" />  

### Sequential Models Regression and Classification Results:

<img src="https://github.com/Aysenuryilmazz/Driver_Drowsiness_Detection/blob/hakkoz/images/sequentialresults.png" height="200" />

### Technologies that We Used:

<img src="https://github.com/Aysenuryilmazz/Driver_Drowsiness_Detection/blob/hakkoz/images/technologies.png" height="150" />

### Online links of notebooks and input/output files:

1.construct-df

2.feature importances
>- [feature-importances-wrf.ipynb](https://www.kaggle.com/hakkoz/feature-importances-wrf?scriptVersionId=34271902)

3.normalization  
>- [final_step2_scaled.pkl](https://drive.google.com/file/d/1C2LJzimJQrjW0x_dymkIGsxlxycL1Wxe/view?usp=sharing)

4.ML classification

5.Process RLDD  
>- [merged_normalized_scaled.pkl](https://drive.google.com/file/d/1kuchEW2wRLup1veVM8M01ilE5R9LbjTC/view?usp=sharing)
>- [rldd_normalized_scaled.pkl](https://drive.google.com/file/d/11H8Duy34HDfgpTX6RuGhD86eBVbv4wAq/view?usp=sharing)

6.experiments
>frame-based models:
>>- [adaboost_RLDD.ipynb](https://www.kaggle.com/aysenur95/ddd-1-adaboost?scriptVersionId=50447882)
>>- [adaboost_MERGED.ipynb](https://www.kaggle.com/ayenurylmaz/ddd-1-merged-adaboost?scriptVersionId=50448464)
>>- [bagging_MERGED.ipynb](https://www.kaggle.com/mahmutaktas/bagging?scriptVersionId=50660571)
>>- [bagging_RLDD.ipynb](https://www.kaggle.com/mahmutaktas/bagging?scriptVersionId=50663317)
>>- [cat_MERGED.ipynb](https://www.kaggle.com/areukolateamleader/ddd-cat-merged?scriptVersionId=54162298)
>>- [cat_NTHU.ipynb](https://www.kaggle.com/hakkoz/ddd-cat?scriptVersionId=46086747)
>>- [cat_RLDD.ipynb](https://www.kaggle.com/areukolateamleader/ddd-cat-rldd?scriptVersionId=50466489)
>>- [cat_MERGED.ipynb](https://www.kaggle.com/hakkoz/ddd-cat-merged?scriptVersionId=51105945)
>>- [dt_MERGED.ipynb](https://www.kaggle.com/hakkoz/ddd-dt-merged?scriptVersionId=50861654)
>>- [dt_RLDD.ipynb](https://www.kaggle.com/areukolateamleader/ddd-dt-rldd?scriptVersionId=50861827)
>>- [extratrees_MERGED.ipynb](https://www.kaggle.com/mahmutaktas/extratrees-merged?scriptVersionId=51219411)
>>- [extratrees_NTHU.ipynb](https://www.kaggle.com/mahmutaktas/extratrees-merged?scriptVersionId=46607446)
>>- [extratrees_RLDD.ipynb](https://www.kaggle.com/mahmutaktas/extratrees-merged?scriptVersionId=51197949)
>>- [gradientboost_RLDD.ipynb](https://www.kaggle.com/aysenur95/ddd-3-gradientboost?scriptVersionId=50668573)
>>- [gradientboost_MERGED.ipynb](https://www.kaggle.com/ayenurylmaz/ddd-3-merged-gradientboost?scriptVersionId=50668190)
>>- [knn_MERGED.ipynb](https://www.kaggle.com/hakkoz/ddd-knn-merged?scriptVersionId=51032671)
>>- [knn_RLDD.ipynb](https://www.kaggle.com/areukolateamleader/ddd-knn-rldd?scriptVersionId=51032718)
>>- [lgbm_MERGED.ipynb](https://www.kaggle.com/hakkoz/ddd-lgbm-merged?scriptVersionId=50646572)
>>- [lgbm_NTHU.ipynb](https://www.kaggle.com/hakkoz/ddd-lgbm?scriptVersionId=46157187)
>>- [lgbm_RLDD.ipynb](https://www.kaggle.com/hakkoz/ddd-lgbm-rldd?scriptVersionId=50571554)
>>- [logisticregression_NTHU.ipynb](https://www.kaggle.com/mahmutaktas/logistic-regression?scriptVersionId=46107245)
>>- [logisticregression_MERGED_RLLD.ipynb](https://www.kaggle.com/mahmutaktas/logistic-regression?scriptVersionId=50580265)
>>- [nb_MERGED.ipynb](https://www.kaggle.com/hakkoz/ddd-nb-merged?scriptVersionId=50895396)
>>- [nb_RLDD.ipynb](https://www.kaggle.com/areukolateamleader/ddd-nb-rldd?scriptVersionId=50895404)
>>- [rf_MERGED.ipynb](https://www.kaggle.com/hakkoz/ddd-rf-merged?scriptVersionId=50484514)
>>- [rf_RLDD.ipynb](https://www.kaggle.com/hakkoz/ddd-rf-rldd?scriptVersionId=50470032)
>>- [simple-thresholding_NTHU_RLDD_MERGED.ipynb](https://www.kaggle.com/hakkoz/ddd-simple-thresholding?scriptVersionId=54076183)
>>- [svm_MERGED.ipynb](https://www.kaggle.com/hakkoz/ddd-svm-merged?scriptVersionId=50777370)
>>- [svm_NTHU.ipynb](https://www.kaggle.com/hakkoz/ddd-svm?scriptVersionId=46708013)
>>- [svm_RLDD.ipynb](https://www.kaggle.com/areukolateamleader/ddd-svm-rldd?scriptVersionId=50777434)
>>- [xgboost_RLDD.ipynb](https://www.kaggle.com/aysenur95/ddd-2-xgboost?scriptVersionId=50448417)
>>- [xgboost_MERGED.ipynb](https://www.kaggle.com/ayenurylmaz/ddd-2-merged-xgboost?scriptVersionId=50461390)  

>sequential models:
>>- [data-preparation_NTHU.ipynb](https://www.kaggle.com/hakkoz/ddd-data-preparation-nthu?scriptVersionId=51800323)
>>- [data-preparation_RLDD.ipynb](https://www.kaggle.com/hakkoz/ddd-data-preparation-rldd?scriptVersionId=54229330)
>>- [lstm-vanilla_NTHU_classification.ipynb](https://www.kaggle.com/mahmutaktas/ddd-lstm-vanilla?scriptVersionId=52379983)
>>- [lstm-vanilla_NTHU_regression.ipynb](https://www.kaggle.com/mahmutaktas/ddd-lstm-vanilla?scriptVersionId=52176732)
>>- [lstm-stacked_NTHU_classification.ipynb](https://www.kaggle.com/hakkoz/ddd-lstm-stacked-v2?scriptVersionId=54162148)
>>- [lstm-stacked_NTHU_regression.ipynb](https://www.kaggle.com/hakkoz/ddd-lstm-stacked?scriptVersionId=54227714)
>>- [bi-lstm_NTHU_classification.ipynb](https://www.kaggle.com/ayenurylmaz/new-bi-lstm?scriptVersionId=52379921)
>>- [bi-lstm_NTHU_regression.ipynb](https://www.kaggle.com/ayenurylmaz/bidirectional-lstm?scriptVersionId=52178775)
>>- [cnn-lstm_NTHU_regression.ipynb](https://www.kaggle.com/areukolateamleader/ddd-lstm-cnn?scriptVersionId=54314304)
>>- [cnn-lstm_NTHU_classification.ipynb](https://www.kaggle.com/areukolateamleader/ddd-lstm-cnn-v2?scriptVersionId=54319854)
>>- [conv-lstm_NTHU_regression.ipynb](https://www.kaggle.com/hakkoz/ddd-lstm-conv?scriptVersionId=54306451)
>>- [conv-lstm_NTHU_classification.ipynb](https://www.kaggle.com/hakkoz/ddd-lstm-conv-v2)

7.demo app
>backend
>>input
>>>- [extra_best_estimator_compressed.pkl](https://drive.google.com/file/d/10sHLRFmnrN2DGokRhbxq6HIi5eV9dP-F/view?usp=sharing)
>>>- [lstm_vanilla.h5](https://drive.google.com/file/d/1q8_6beWeaJTrjRK70bmVGWB98C0KWUp9/view?usp=sharing)
>>>- [lstm_vanilla_v2.h5](https://drive.google.com/file/d/1R2aQ_3uHQ6CmPLtiWpxQ73zwlW93qAUG/view?usp=sharing)
>>>- [scaler.sav](https://drive.google.com/file/d/1vVQ74ceLrpgHEoe6KDKbtBV_OWZOiTGq/view?usp=sharing)
>>>- [shape_predictor_68_face_landmarks.dat](https://drive.google.com/file/d/1nMDw4F7V-8JD1OKjVymd1xYx3cTRjEw9/view?usp=sharing)

8.alternatives
>I3D
>>- [i3D.ipynb](https://www.kaggle.com/hakkoz/i3d-mustafa?scriptVersionId=49546861)

>blink detection
>>- [eye-blink-detection-1-simple-model.ipynb](https://www.kaggle.com/hakkoz/eye-blink-detection-1-simple-model)
>>- [eye-blink-detection-2-adaptive-model.ipynb](https://www.kaggle.com/hakkoz/eye-blink-detection-2-adaptive-model-v2)
>>- [eye-blink-detection-3-ml-model-part1.ipynb](https://www.kaggle.com/hakkoz/eye-blink-detection-3-ml-model-part1)
>>- [eye-blink-detection-3-ml-model-part2.ipynb](https://www.kaggle.com/hakkoz/eye-blink-detection-3-ml-model-part2)
>>- [eye-blink-detection-4-comparison.ipynb](https://www.kaggle.com/hakkoz/eye-blink-detection-4-comparison)

documents
  
input  
