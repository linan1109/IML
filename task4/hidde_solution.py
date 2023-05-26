# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import os
from tqdm import tqdm
# from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def load_data(path_dict):
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None

    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv(path_dict["pretrain_features"], index_col="Id", compression='zip').drop("smiles",
                                                                                                     axis=1).to_numpy()
    y_pretrain = pd.read_csv(path_dict["pretrain_labels"], index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv(path_dict["train_features"], index_col="Id", compression='zip').drop("smiles",
                                                                                               axis=1).to_numpy()
    y_train = pd.read_csv(path_dict["train_labels"], index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv(path_dict["test"], index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self, layer_dict, predictor=False):
        """
        The constructor of the model.
        """
        super().__init__()
        if predictor:
            prediction_layers = []
            for i in range(layer_dict["num_pred_layers"]):
                prediction_input_nodes = layer_dict["num_nodes_pred_layer"][i]
                prediction_output_nodes = layer_dict["num_nodes_pred_layer"][i + 1]

                name = 'fc' + str(i)
                prediction_layers.append((name, nn.Linear(prediction_input_nodes, prediction_output_nodes)))
                prediction_layers = add_layers(i, prediction_layers, layer_dict, prediction_output_nodes,
                                               prediction=True)

            prediction_dict = OrderedDict(prediction_layers)

            self.net = nn.Sequential(prediction_dict)

            print(f'Succesfully initialized prediction NeuralNet prediction with {layer_dict["num_pred_layers"]} '
                  f'layers')
            print(f'Using batch norm: {layer_dict["use_pred_batch_norm"]}, \n'
                  f'Using dropout: {layer_dict["use_pred_dropout"]}, with rate: {layer_dict["pred_dropout_rate"]}')

        else:
            # Dynamically instantiate a number of layers.
            layers = []

            # Set up the layers simultaneously for both the encoder and decoder.
            for i in range(layer_dict['num_layers']):
                input_nodes = layer_dict['num_nodes_per_layer'][i]
                output_nodes = layer_dict['num_nodes_per_layer'][i + 1]
                name = 'fc' + str(i)

                layers.append((name, nn.Linear(input_nodes, output_nodes)))
                layers = add_layers(i, layers, layer_dict, output_nodes)

            nn_dict = OrderedDict(layers)
            self.net = nn.Sequential(nn_dict)

            print(f'Succesfully initialized NeuralNet with {layer_dict["num_layers"]} layers')
            print(f'Using batch norm: {self.use_batch_norm}, \n'
                  f'Using dropout: {self.use_dropout}, with rate: {layer_dict["dropout_rate"]}')

    def forward(self, x, prediction=False):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = self.net(x)
        return x


def add_layers(i, temp_layer_lst, layer_dict, output_nodes, prediction=False):
    """
    Helper function in constructing the layers for the AutoEncoder network.
    :param int i: index of where the loop currently is.
    :param list temp_layer_lst: temporary layer dictionary to return.
    :param dict layer_dict: dictionary with the verification variables.
    :return: the inbetween dictionary with batchnorm dropout and function added if necessary.
    """
    # If we are not in the last layer we want to use functions or batchnorms or dropout
    # NOTE order matters!
    if prediction:
        number_of_layers = layer_dict['num_pred_layers']
        batch_norm_string = 'use_pred_batch_norm'
        dropout_string = "use_pred_dropout"
        dropout_rate_string = "pred_dropout_rate"

    else:
        number_of_layers = layer_dict["num_layers"]
        batch_norm_string = 'use_batch_norm'
        dropout_string = "use_dropout"
        dropout_rate_string = "dropout_rate"

    if (i != number_of_layers- 1):
        if layer_dict[batch_norm_string]:
            batch_name = 'bn' + str(i)
            temp_layer_lst.append((batch_name, nn.BatchNorm1d(output_nodes)))

        function_name = 'activate' + str(i)
        temp_layer_lst.append((function_name, layer_dict['layer_function']))

        if layer_dict[dropout_string]:
            assert layer_dict[dropout_rate_string] is not None, 'Use dropout but dropout rate is None'
            dropout_name = 'dropout' + str(i)
            temp_layer_lst.append((dropout_name, nn.Dropout(layer_dict[dropout_rate_string])))

    # No dropout, batchnorm for the final layer, only the sigmoid.
    else:
        if not prediction:
            temp_layer_lst.append(('final_function', layer_dict['final_function']))

        else:
            temp_layer_lst.append(("final_function", layer_dict['layer_function']))

    return temp_layer_lst


class AE(nn.Module):
    def __init__(self, layer_dict):
        super().__init__()

        # Dynamically instantiate a number of layers.
        encoder_layers = []
        decoder_layers = []

        # Set up the layers simultaneously for both the encoder and decoder.
        for i in range(layer_dict['num_layers']):
            encoder_input_nodes = layer_dict['num_nodes_per_layer'][i]
            encoder_output_nodes = layer_dict['num_nodes_per_layer'][i + 1]

            decoder_input_nodes = layer_dict['num_nodes_per_layer'][layer_dict['num_layers'] - i]
            decoder_output_nodes = layer_dict['num_nodes_per_layer'][layer_dict['num_layers'] - (i + 1)]

            name = 'fc' + str(i)
            encoder_layers.append((name, nn.Linear(encoder_input_nodes, encoder_output_nodes)))
            decoder_layers.append((name, nn.Linear(decoder_input_nodes, decoder_output_nodes)))

            encoder_layers = add_layers(i, encoder_layers, layer_dict, encoder_output_nodes)
            decoder_layers = add_layers(i, decoder_layers, layer_dict, decoder_output_nodes)

        encoder_dict = OrderedDict(encoder_layers)
        self.encoder = nn.Sequential(encoder_dict)

        decoder_dict = OrderedDict(decoder_layers)
        self.decoder = nn.Sequential(decoder_dict)

        print(f'Succesfully initialized AutoEncoder with {layer_dict["num_layers"]} layers')
        print(f'Using batch norm: {layer_dict["use_batch_norm"]}, \n'
              f'Using dropout: {layer_dict["use_dropout"]}, with rate: {layer_dict["dropout_rate"]}')

        self.immediate_prediction = layer_dict["immediate_prediction"]
        if layer_dict["immediate_prediction"]:
            prediction_layers = []
            for i in range(layer_dict["num_pred_layers"]):
                prediction_input_nodes = layer_dict["num_nodes_pred_layer"][i]
                prediction_output_nodes = layer_dict["num_nodes_pred_layer"][i + 1]

                name = 'fc' + str(i)
                prediction_layers.append((name, nn.Linear(prediction_input_nodes, prediction_output_nodes)))
                prediction_layers = add_layers(i, prediction_layers, layer_dict, prediction_output_nodes, prediction=True)

            prediction_dict = OrderedDict(prediction_layers)

            self.prediction = nn.Sequential(prediction_dict)

            print(f'Succesfully initialized AutoEncoder prediction with {layer_dict["num_pred_layers"]} layers')
            print(f'Using batch norm: {layer_dict["use_pred_batch_norm"]}, \n'
                  f'Using dropout: {layer_dict["use_pred_dropout"]}, with rate: {layer_dict["pred_dropout_rate"]}')

    def forward(self, x, prediction=False):
        encoded = self.encoder(x)

        if prediction:
            if self.immediate_prediction:
                decoded = self.prediction(encoded)

            else:
                decoded = encoded

        else:
            decoded = self.decoder(encoded)

        return decoded

def train_model(training_data, model, batch_size, device, optimizer, loss_function, scheduler=None, y_train=None):

    # Set model to training mode.
    model.train()

    train_loss = 0

    for index in range(0, len(training_data), batch_size):
        X = training_data[index: index + batch_size]
        X = X.to(device)

        if y_train is None:
            reconstructed_data = model(X)
            original_data = X

        else:
            y = y_train[index: index + batch_size]
            y = y.to(device)

            reconstructed_data = model.forward(X, prediction=True).flatten()
            original_data = y


        # Loss calculation
        loss = loss_function(reconstructed_data, original_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        true_loss = loss.detach()

        train_loss += true_loss

    if scheduler is not None:
        scheduler.step()

    # Compute the total training loss
    train_loss = train_loss / len(training_data)

    return model, optimizer, scheduler, train_loss

def validate_model(validation_data, model, loss_function, batch_size, device, y_val=None):

    # Set model to evaluation mode.
    model.eval()
    with torch.no_grad():
        validation_loss = 0

        for index in range(0, len(validation_data), batch_size):
            X = validation_data[index: index + batch_size]
            X = X.to(device)


            if y_val is not None:
                y = y_val[index: index + batch_size]
                y = y.to(device)
                original_data = y
                prediction = True

            else:
                original_data = X
                prediction = False

            # For the 2 NN it does not matter whether we pass prediction True or False, but if we are using the AE it
            # should not predict with y, if we use two different networks it will pass over the y prediction, because
            # then the predictor was not instantiated.
            reconstructed_data = model.forward(X, prediction=prediction)

            if prediction:
                reconstructed_data = reconstructed_data.flatten()

            val_loss = loss_function(reconstructed_data, original_data)
            validation_loss += val_loss

        validation_loss = validation_loss / len(validation_data)

    return validation_loss

def make_feature_extractor(x, y,
                           device, num_epochs, num_epochs_final, use_all_data,
                           pretrain_model, pretrain_optimizer, pretrain_loss_function, pretrain_scheduler,
                           model=None, optimizer=None, loss_function=None, scheduler=None,
                           batch_size=256, eval_size=1024, seperate_models=False):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
            
    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    assert eval_size % batch_size == 0, "Evaluation loop will crash due to out of bounds indexing with mismatch " \
                                        "eval_size and batch_size."

    if seperate_models:
        assert model is not None, "Seperate models, but no model specified"
        assert optimizer is not None, "Seperate models, but no optimizer specified"
        assert loss_function is not None, "Seperate models, but no loss function specified"

    # Pretraining data loading
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)
    x, y = torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

    # Initialize the minimum validation loss to infinity.
    minimum_val_loss = np.inf
    min_val_state = None

    # Losses
    validation_losses = []
    train_losses = []
    losses_dict = {}

    if seperate_models:
        print(f'Running training over {num_epochs} epochs, extracting the features.')
        print("pretrain_optimizer->", pretrain_optimizer)
        for epoch in tqdm(range(num_epochs)):
            # Training the model
            pretrain_model, pretrain_optimizer, pretrain_scheduler, training_loss = train_model(x_tr, pretrain_model,
                                                                                                batch_size,
                                                                                                device,
                                                                                                pretrain_optimizer,
                                                                                                pretrain_loss_function,
                                                                                                pretrain_scheduler)

            # Validating the model
            validation_loss = validate_model(x_val, pretrain_model, pretrain_loss_function, batch_size, device)

            print(f'Train {epoch} loss: {training_loss}, validation loss: {validation_loss}')

            validation_losses.append(validation_loss)
            train_losses.append(training_loss)

            ### We can use the validation data again to train. For this we need to define the best epoch model.
            if validation_loss < minimum_val_loss:
                minimum_val_loss = validation_loss
                min_val_state = pretrain_model.state_dict()

        losses_dict['validation_losses'] = validation_losses
        losses_dict['train_losses'] = train_losses
        print('extracting feature - pretrain_model', pretrain_model)
        if use_all_data:
            # If the models are the same the data must not be seen.
            if seperate_models:
                # Load the best model.
                print('Training the model with least amount of validation loss on the full dataset.')
                assert min_val_state is not None, "There is no minimal state, no continuation possible."
                pretrain_model.load_state_dict(min_val_state)

                # Training loop.
                final_train_loss = []

                # Final training loop.
                for epoch in tqdm(range(num_epochs_final)):
                    # Training the model
                    pretrain_model, pretrain_optimizer, pretrain_scheduler, training_loss = train_model(x, pretrain_model,
                                                                                                        batch_size,
                                                                                                        device,
                                                                                                        pretrain_optimizer,
                                                                                                        pretrain_loss_function,
                                                                                                        pretrain_scheduler)

                    print(f"Train {epoch} loss: {training_loss}")
                    final_train_loss.append(training_loss)
                losses_dict['all_data_ae_final'] = final_train_loss
            print('extracting feature -use_all_data- pretrain_model', pretrain_model)

    # Prediction loop

    # Setup the losses
    prediction_losses = []
    prediction_validation_losses = []
    minimum_val_loss = np.inf
    min_val_state = None

    # First extract the features from the autoencoder if we are using seperate models.
    if seperate_models:
        pretrain_model.eval()
        with torch.no_grad():
            trained_deviced_data = x_tr.to(device)
            auto_encoded_data = pretrain_model.forward(trained_deviced_data, prediction=True)
            pretrain_data = auto_encoded_data.cpu()

            eval_deviced_data = x_val.to(device)
            auto_encoded_eval_data = pretrain_model.forward(eval_deviced_data, prediction=True)
            pretrain_val_data = auto_encoded_eval_data.cpu()

    else:
        pretrain_data = x_tr
        pretrain_val_data = x_val
        model = pretrain_model
        optimizer = pretrain_optimizer
        loss_function = pretrain_loss_function
        scheduler = pretrain_scheduler

    # Start the loop of the second part of the network.
    print(f'Running the prediction analysis over {num_epochs} epochs using the labels as well.')
    print("optimizer->", optimizer)
    for epoch in tqdm(range(num_epochs)):
        model, optimizer, scheduler, training_loss = train_model(pretrain_data, model, batch_size, device, optimizer,
                                                      loss_function, scheduler, y_tr)

        validation_loss = validate_model(pretrain_val_data, model, loss_function, batch_size, device, y_val)

        print(f'Train {epoch} loss: {training_loss}, validation loss: {validation_loss}')

        prediction_losses.append(training_loss)
        prediction_validation_losses.append(validation_loss)

        ### We can use the validation data again to train. For this we need to define the best epoch model.
        if validation_loss < minimum_val_loss:
            minimum_val_loss = validation_loss
            min_val_state = model.state_dict()

    losses_dict['prediction_validation_losses'] = prediction_validation_losses
    losses_dict['prediction_train_losses'] = prediction_losses
    print("make_feature_extractor-model", model)
    if use_all_data:
        # Load the best model.
        print('Training the model with least amount of validation loss on the full dataset.')
        assert min_val_state is not None, "There is no minimal state, no continuation possible."
        model.load_state_dict(min_val_state)

        # Training loop.
        final_train_loss = []

        # Final training loop.
        for epoch in tqdm(range(num_epochs_final)):
            # Training the model
            model, optimizer, scheduler, training_loss = train_model(x, model, batch_size, device, optimizer,
                                                                     loss_function, scheduler, y)

            print(f"Train {epoch} loss: {training_loss}")
            final_train_loss.append(training_loss)

        losses_dict['final_training_loss'] = final_train_loss
        print("make_feature_extractor-usealldata-model", model)
    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline

        Have to sent models to CPU otherwise the pipeline complains.
        """

        if seperate_models:
            pretrain_model.to("cpu")
            model.to("cpu")
            pretrain_model.eval()
            new_model = torch.nn.Sequential(*(list(model.children())[:-1]))
            print("----------------------------------------------new_model", new_model)
            new_model.eval()
            with torch.no_grad():
                pretrain_features = pretrain_model(x)
                features = new_model(pretrain_features)
                print("features", features.shape)
            features = features.cpu().detach().numpy()

        else:
            # If not seperate models then we are using the AE, which means we need to do the following.
            # Yield a list of the three models, the encoder, the decoder and the prediction, we only want the encoder
            # and the prediction layer.
            lst_of_models = list(model.children())
            lst_of_layers = [lst_of_models[0]] + [lst_of_models[2][:-1]]
            new_model = torch.nn.Sequential(*lst_of_layers)
            new_model.to("cpu")
            new_model.eval()
            with torch.no_grad():
                features = new_model(x)
            features = features.cpu().detach().numpy()

        return features

    return make_features, losses_dict, model, pretrain_model

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

def get_regression_model(regression_dict):
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.
    if regression_dict["linear"]:
        model = LinearRegression()
    elif regression_dict["XGB"]:
        # model = XGBRegressor(**regression_dict["XGB_values"])
        pass
    else:
        raise Exception("There is no model specified")

    return model

# Main function. You don't have to change this
if __name__ == '__main__':
    # the device objectss indexes all available GPU's, since there is only one NVidea GPU in my pc it is cuda:0,
    # or cuda() .
    # use list(range(torch.cuda.device_count())) to find all avaible GPU's.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Used device: {device}')
    # device = "cpu"

    # Define file paths
    pretrain_features = "pretrain_features.csv.zip"
    pretrain_labels = "pretrain_labels.csv.zip"
    train_features = "train_features.csv.zip"
    train_labels = "train_labels.csv.zip"
    test = "test_features.csv.zip"

    # Create dictionary of file paths
    Path = os.path
    dir = Path.join(Path.dirname(__file__))
    path_pretrain_features = Path.join(dir, pretrain_features)
    path_train_features = Path.join(dir, train_features)
    path_pretrain_labels = Path.join(dir, pretrain_labels)
    path_train_labels = Path.join(dir, train_labels)
    path_test = Path.join(dir, test)

    path_dict = {"pretrain_features": path_pretrain_features, "pretrain_labels": path_pretrain_labels,
                 "train_features": path_train_features, "train_labels": path_train_labels, "test": path_test}


    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data(path_dict)
    print("Data loaded!")

    # Setup parameters
    test = True

    pipeline = True
    # Regression formats
    regression = True
    linear_reg = True
    XGB = True
    XGB_dict = {'n_estimators': 10, 'objective': 'reg:squarederror', "random_state": 1, "reg_lambda": 0.001, # L2, if L1 use alpha instead of lambda
                "eval_metric": "logloss", "use_label_encoder": False, "n_jobs": 18, "grow_policy": "lossguide", #depthwise different method losswise
                "max_leaves": 0}

    regression_dict = {'linear': linear_reg, "XGB": XGB, "XGB_values": XGB_dict}

    if regression:
        assert True in regression_dict.values(), "Regression without regressor specified."

    scaler = False

    num_epochs = 15
    num_final_epochs = 15
    batch_size = 256
    eval_size = 4 * batch_size
    num_work = 12
    learning_rate = 0.01
    pretrain_learning_rate = 0.01
    # Scheduler
    use_scheduler = False
    step_size = 2
    gamma = 0.1

    # Only for SGD
    momentum = 0.05
    weight_decay = 0.001

    use_all_data = True

    # Use autoencoder or NN to extract features from the pretraining data.
    use_AE = True
    # If you are using seperate models to train this would work better.
    seperate_models = True

    # Model dictionary
    layer_list = np.array([x_pretrain.shape[-1], 1000, 1000])
    pred_layer_list = np.array([layer_list[-1], 40, 1])
    leaky_rate = 0.01
    layer_function = nn.LeakyReLU(negative_slope=leaky_rate)
    final_function = nn.LeakyReLU(negative_slope=leaky_rate)
    layer_dict = {"num_layers": len(layer_list) - 1,
                  "num_nodes_per_layer": layer_list,
                  "layer_function": layer_function,
                  "final_function": final_function,
                  "use_batch_norm": True,
                  "use_dropout": True,
                  "dropout_rate": 0.6,
                  "num_pred_layers": len(pred_layer_list)-1,
                  "num_nodes_pred_layer": pred_layer_list,
                  "use_pred_batch_norm": True,
                  "use_pred_dropout": True,
                  "pred_dropout_rate": 0.6,
                  "immediate_prediction": not seperate_models}

    if use_AE:
        if seperate_models:
            pretrain_model = AE(layer_dict)
            model = Net(layer_dict, predictor=True)
            model.to(device)

        else:
            pretrain_model = AE(layer_dict)
            model = None

    else:
        assert seperate_models, "Neural net can only work with seperate models due to implementation"
        pretrain_model = Net(layer_dict)
        model = Net(layer_dict, predictor=True)
        model.to(device)

    pretrain_model.to(device)
    # pretrain_optimizer = torch.optim.SGD(pretrain_model.parameters(), lr=pretrain_learning_rate, momentum=momentum,
    #                                      weight_decay=weight_decay) SGD is very bad at training in this case
    pretrain_optimizer = torch.optim.Adam(pretrain_model.parameters(), lr=pretrain_learning_rate)
    pretrain_loss_function = nn.MSELoss()

    if use_scheduler:
        pretrain_scheduler = torch.optim.lr_scheduler.StepLR(pretrain_optimizer, step_size=step_size, gamma=gamma)
    else:
        pretrain_scheduler = None

    if model is not None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()

        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            scheduler = None

    else:
        optimizer, loss_function, scheduler = None, None, None

    feature_extractor, losses_dict, trained_model, pretrain_model = make_feature_extractor(x_pretrain, y_pretrain,
                                                            device,
                                                            num_epochs, num_final_epochs, use_all_data,
                                                            pretrain_model, pretrain_optimizer, pretrain_loss_function,
                                                            pretrain_scheduler,
                                                            model, optimizer, loss_function, scheduler,
                                                            batch_size, eval_size,
                                                            seperate_models)
    print("trained_model", trained_model)
    print("pretrain_model", pretrain_model)
    
    # Create pipeline or use the regular NN. DO NOT sent to device, because then the Pipeline crashes.
    x_train = torch.tensor(x_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)

    pipeline_lst = []
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    PretrainedFeature = PretrainedFeatureClass(feature_extractor="pretrain")
    pipeline_lst.append(("feature_extractor", PretrainedFeature))

    if scaler:
        # Define scaler
        # The scaler acts like a regularization
        scaler = StandardScaler()
        pipeline_lst.append(("scaler", scaler))

    if regression:
        # regression model
        regression_model = get_regression_model(regression_dict)
        pipeline_lst.append(("regression", regression_model))

    if pipeline:
        print(f'Using pipeline with scaler: {scaler}, regression: {regression}')
        pipeline = Pipeline(pipeline_lst)
        print(pipeline)
        # Due to SKLearn we have to sent this to the CPU otherwise it crashes.
        pipeline.fit(x_train, y_train)
        print(regression_model.coef_.shape)
        x_test_to_tensor = torch.tensor(x_test.values, dtype=torch.float)
        y_pred = pipeline.predict(x_test_to_tensor)

    else:
        raise NotImplemented

        if seperate_models:
            pretrain_model.eval()
            trained_model.eval()
            with torch.no_grad():
                pretrained = pretrain_model.forward(x_train)
                data = trained_model.forward(pretrained, prediction=True)
            data = data.cpu().detach().numpy()

    assert y_pred.shape == (x_test.shape[0],)
    if test:
        y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
        y_pred.to_csv("results.csv", index_label="Id")
        print("Predictions saved, all done!")
