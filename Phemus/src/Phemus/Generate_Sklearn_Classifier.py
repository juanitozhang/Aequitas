# source: https://www.kaggle.com/yemishin/eda-and-employee-future-prediction
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from .Dataset import Dataset

def generate_sklearn_classifier(dataset: Dataset, output_pkl_dir):
    """Train a machine learning model on a dataset.

    Takes a input dataset characterised by a dataset object and train a machine learning
    model from the dataset in .pkl format and then store the model locally.
    
    Args:
        dataset -- A dataset object that contains the meta information about the dataset used.
        output_pkl_dir -- The directory where the output model in .pkl format will be saved.
    
    Returns:
        This function does not have a return value.

    Raises:
        ValueError: An error occurs when a not supported model type is selected.
    """
    input_csv_dir = dataset.dataset_dir
    col_to_be_predicted = dataset.col_to_be_predicted
    model_type = dataset.model_type

    df=pd.read_csv(input_csv_dir)
    cat_feature = list(df.columns)
    
    for col in cat_feature:
        df[col]=le.fit_transform(df[col])

    X=df.drop([col_to_be_predicted],axis=1)
    y=df[col_to_be_predicted]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=12)
    
    if model_type == "DecisionTree":
        model = DecisionTreeClassifier(random_state=42, criterion='entropy', splitter='random') 
        model_name = 'DecisionTreeClassifier'
    elif model_type == "MLPC":
        model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(7, 5), random_state=1)
        model_name = 'MLPClassifier'
    elif model_type == "SVM":
        model = SVC(gamma=0.0025)
        model_name = 'SVC'
    elif model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators = 10)
        model_name = 'RandomForestClassifier'
    else:
        error_message = 'The chosen types of model is not supported yet. Please choose from one of the following: \
                            DecisionTree, MLPC, SVM and RandomForest'
        raise ValueError(error_message)

    model.fit(X_train,y_train)
    pred=model.predict(X_test)

    scores=[]
    scores.append({
            'model': model_name,
            'score': model.score(X_test,y_test),
            'f1_score' : f1_score(y_test,pred)
        })

    model.score(X_test, y_test)

    joblib.dump(model, output_pkl_dir)
