# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import linear_model


def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None
    
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        # TODO: Define the architecture of the model. It should be able to be trained on pretraing data 
        # and then used to extract features from the training and test data.

        self.features = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            )
        
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000,1),
            )

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # TODO: Implement the forward pass of the model, in accordance with the architecture 
        # defined in the constructor.
        x = self.features(x)
        x = self.classifier(x)
        return x
    
def make_feature_extractor(x, y, batch_size=256, eval_size=1000):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
            
    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    # Pretraining data loading
    in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)
    
    # model declaration
    model = Net()
    model.train()
    
    # TODO: Implement the training loop. The model should be trained on the pretraining data. Use validation set 
    # to monitor the loss.
    model.to(device)
    train_dataset = TensorDataset(x_tr, y_tr)
    valid_dataset = TensorDataset(x_val, y_val)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle=True, pin_memory=True, num_workers=4)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, 
                              shuffle=True, pin_memory=True, num_workers=4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epoches = 20
    for epoch in range(n_epoches):
        trainingLoss = 0
        model.train()
        for [X, y] in train_loader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model.forward(X)
            y = y.unsqueeze(1)
            loss = criterion(y_pred, y)
        
            trainingLoss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        aveTrainingLoss = trainingLoss / len(train_loader)
        print("################################")
        print(f'epoch: {epoch:2}    training loss: {aveTrainingLoss:10.8f}')

        validLoss = 0
        model.eval()
        with torch.no_grad():
            for [X,y] in valid_loader:
                X = X.to(device)
                y = y.to(device)
                y_pred = model.forward(X)
                y = y.unsqueeze(1)
                loss = criterion(y_pred, y)
                validLoss += loss.item()
        aveValidLoss = validLoss / len(valid_loader)
        print(f'epoch: {epoch:2}    validation loss: {aveValidLoss:10.8f}')
    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        model.eval()
        # TODO: Implement the feature extraction, a part of a pretrained model used later in the pipeline.
        # model.classifier = nn.Sequential()
        # x = model.forward(x)
        x = model.features(x)
        return x

    return make_features

def make_pretraining_class(feature_extractors):
    """
    The wrapper function which makes pretraining API compatible with sklearn pipeline
    
    input: feature_extractors: dict, a dictionary of feature extractors

    output: PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode

        def fit(self, X=None, y=None):
            return self

        def transform(self, X):
            assert self.feature_extractor is not None
            X_new = feature_extractors[self.feature_extractor](X)
            return X_new
        
    return PretrainedFeatures

def get_regression_model():
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.

    model = linear_model.Ridge(alpha=0.5)
    return model

# Main function. You don't have to change this
if __name__ == '__main__':
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    print("################processing####################")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")
    # Utilize pretraining data by creating feature extractor which extracts lumo energy 
    # features from available initial features
    feature_extractor =  make_feature_extractor(x_pretrain, y_pretrain)
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    
    # regression model
    regression_model = get_regression_model()

    y_pred = np.zeros(x_test.shape[0])
    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.
    x_train_tensor = torch.tensor(x_train, dtype=torch.float)
    x_test_tensor = torch.tensor(x_test.to_numpy(), dtype=torch.float)
    x_train_tensor = x_train_tensor.to(device)
    x_test_tensor = x_test_tensor.to(device)
    x_train_new = PretrainedFeatureClass.transform(PretrainedFeatureClass(feature_extractor="pretrain"),x_train_tensor)
    x_test_new = PretrainedFeatureClass.transform(PretrainedFeatureClass(feature_extractor="pretrain"),x_test_tensor)
    
    x_train_new = x_train_new.detach().cpu().numpy()
    x_test_new = x_test_new.detach().cpu().numpy()
    
    regression_model.fit(x_train_new, y_train)
    y_pred = regression_model.predict(x_test_new)

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")