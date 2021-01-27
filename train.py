from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
import pandas as pd
import numpy as np

url = "https://github.com/eaamankwah/foree/raw/main/diabetes.csv"

ds = pd.read_csv(url, sep=',', header=0, encoding= 'unicode_escape')

def clean_data(ds):
    assert isinstance(ds, pd.DataFrame), "df needs to be a pd.DataFrame"
    ds.dropna(inplace=True)
    indices_to_keep = ~ds.isin([np.nan, np.inf, -np.inf]).any(1)
    return ds[indices_to_keep].astype(np.float64) 
def clean_data1(ds):
    ds.replace([np.inf, -np.inf], np.nan, inplace=True)
    ds.fillna(-99999, inplace=True)
    x_df = ds.drop(['Outcome'],axis=1)
    
    y_df = ds['Outcome']

    return x_df, y_df

ds = clean_data(ds)
x, y = clean_data1(ds)

# TODO: Split data into train and test sets.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
scaled_data = scaler.transform(x)
scaled_data = pd.DataFrame(scaled_data, columns= x.columns)
x_train, x_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.2, random_state=1)

from azureml.core.run import Run
run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(LogisticRegression, 'outputs/model.joblib')

if __name__ == '__main__':
    main()

