# PlayGroundSeries
## Kaggle playground prediction series notebooks
---
## [S4 E02: Multi-Class Prediction of Obesity Risk](https://github.com/andrewbremner3/PlayGroundSeries/blob/main/weightrisk-s4e2-pytorch.ipynb)

[Kaggle Competition Link](https://www.kaggle.com/competitions/playground-series-s4e2/overview)

**Goal: To use various factors to predict obesity risk in individuals, which is related to cardiovascular disease.**

This is a multi catagorical prediction with 7 options of strings as outputs.
### EDA
* Check on balance of the data set and deal with catagorical fields for dummy variables.
* Add a BMI field as well (has the highest correlation to the target) - 0.961441 correlation coefficient
### First: use Sklearn models for multi-catagorical training and testing
* RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier and XGBClassifier tested and XGBClassifier has the best results.
* **Final accuracy Evaluation of the XGBClassifier - 0.90823**
### Second: use a Neural Network build with Pytorch
* Various forms of the NN with a few types of data input.
* Use all the data as numerical and categorical as well as convert the categories to dummy variables.
* Test NN with many layer options and use best [1024,512] (2 layers of 1024 then 512 nodes)
* * **Final accuracy Evaluation of the NN - 0.86343**

---
## [S4 E01: Binary Classification with a Bank Churn Dataset](https://github.com/andrewbremner3/PlayGroundSeries/blob/main/bankchurn-s4e1.ipynb)

[Kaggle Competition Link](https://www.kaggle.com/competitions/playground-series-s4e1/overview)

**Goal: To predict whether a customer continues with their account or closes it (e.g., churns), probablitity of exiting.**

### First: use Sklearn models for catagorical training and testing
* Exploritory data analysis (EDA) to check on balance of the data set and to deal with catagorical fields.
* Test sklearn models on split training data set for training time and accuracy (roc_auc_score used)
* Retrain with the model on the entire data set with the best accuracy score for final model, then test and sbmit with probabilities of exiting.
* **Final Evaluation (area under roc curve) = 0.88159**
### Second: use a Neural Network build with Pytorch
* Convert the data to the corrct format (catagories vs continuous fields) then tensors for the NN
* Train the NN with a split data set with few different layer setups to find a good NN setup (2 hidden layers with [101,21] nodes)
* Retrain the NN with the entire data set for the final model, then test and submit with probabilities of exiting.
* **Final Evaluation (area under roc curve) = 0.85983**
---
## [S3 E26: Multi-Class Prediction of Cirrhosis Outcomes](https://github.com/andrewbremner3/PlayGroundSeries/blob/main/predict-cirrhosis-s3e26-pytorch-flexible-model.ipynb)

[Kaggle Competition Link](https://www.kaggle.com/competitions/playground-series-s3e26/overview)

**Goal: To use a multi-class approach to predict the the outcomes of patients with cirrhosis.**

### Create a flexible Neural Network for various continuous and catagorical fields to produce a set of predicted probabilities for each of the three possible outcomes
* Convert catagorical data to dummy variables, use continuous data as it, then convert all to tensor for training.
* Create NN model:
  * Hidden layers: Linear, Batch Norm 1D, ReLU (rectified linear unit), Dropout
  * Final layers: Linear, Softmax (for probability outputs)
* Train on a split data set, check for time and accuracy
* Format the output data set for final submission
* **Final Evaluation (logloss) = 0.66980**
