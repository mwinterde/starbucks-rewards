import argparse
import os
import pandas as pd
import numpy as np

import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

# Define hyperparameter space
HYPERPARAMETER_GRID = {
    'imputer__strategy': ['mean', 'median'],
    'rf__bootstrap': [True, False],
    'rf__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'rf__max_features': ['auto', 'sqrt'],
    'rf__min_samples_leaf': [1, 2, 4],
    'rf__min_samples_split': [2, 5, 10],
    'rf__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]    
}


def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


def predict_fn(input_data, model):
    """Predict probabilities of belonging to the positive class"""
    
    classes = model.classes_
    pred_prob = model.predict_proba(input_data)
    
    return pred_prob[:,np.argwhere(classes==1)].squeeze()


if __name__ == '__main__':
    """This is the code that will be executed for model training. We train a sklearn
    classifier and dump it to S3."""
    
    parser = argparse.ArgumentParser()

    # Set SageMaker parameters
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Set model parameters
    parser.add_argument('--n_iter', type=int, default=10) # number of cv runs
    
    # Create args that holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Split dataframe into feature matrix and target vector
    X_train = train_data.iloc[:,1:] # Labels are in the first column
    y_train = train_data.iloc[:,0]

    # Create pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=0))
    ])
    
    # Cross validate model
    cv = RandomizedSearchCV(
        estimator=pipeline, 
        param_distributions=HYPERPARAMETER_GRID, 
        n_iter=args.n_iter,
        random_state=0,
        n_jobs=-1
    )
    
    cv.fit(X_train, y_train)
    
    print("Cross validation finished")
    print("Best params: ", cv.best_params_)

    # Save the trained model
    joblib.dump(cv.best_estimator_, os.path.join(args.model_dir, "model.joblib"))