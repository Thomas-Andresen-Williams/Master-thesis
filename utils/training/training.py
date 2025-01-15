import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from sklearn.metrics import confusion_matrix
import pandas as pd
from utils.models.models import MLP
from utils.datasets.datasets import MLPDataset
from torch.utils.data import DataLoader
from utils.utils.utils import change_directory
from sklearn.model_selection import KFold,GridSearchCV, RandomizedSearchCV
from tqdm import tqdm
import optuna
def train_val_test_split(X, y, split_percentage):
    """
    Performs a train test validation split on a dataset
    """
    
    val_percentage = 0.10
    assert split_percentage + val_percentage <= 1
    X_train = X[: int(len(X) * split_percentage)]
    X_val = X[
        int(len(X) * split_percentage) : int(
            len(X) * (split_percentage + val_percentage)
        )
    ]
    X_test = X[int(len(X) * split_percentage + val_percentage) :]

    y_train = y[: int(len(y) * split_percentage)]
    y_val = y[
        int(len(y) * split_percentage) : int(
            len(y) * (split_percentage + val_percentage)
        )
    ]
    y_test = y[int(len(y) * split_percentage + val_percentage) :]

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_epoch_regression(model,train_dataloader,val_dataloader,criterion,optimizer):
    """
    Does a regular round of training for a basic regression problem
    """
    train_loss = 0
    val_loss = 0

    model.train()
    for X, y in train_dataloader:
        y_hat = model(X)
        optimizer.zero_grad()
        loss = criterion(y_hat, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    model.eval()
    for X, y in val_dataloader:
       
        y_hat = model(X)
        loss = criterion(y_hat, y)
        val_loss += loss.item()

    return train_loss, val_loss



def train_epoch(model, train_dataloader, val_dataloader, criterion, optimizer):
    train_loss = 0
    val_loss = 0
    correct_predictions = 0

    model.train()
    for X, y in train_dataloader:
        y_hat = model(X)
        optimizer.zero_grad()
        loss = criterion(y_hat, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        y_hat_class = torch.where(
            y_hat > 0.5,
            torch.tensor(1, dtype=torch.int32),
            torch.tensor(0, dtype=torch.int32),
        )
        correct_predictions += torch.sum(y_hat_class == y)
    training_accuracy = correct_predictions / len(train_dataloader.dataset.features)

    correct_predictions = 0
    model.eval()
    for X, y in val_dataloader:
       
        y_hat = model(X)
        loss = criterion(y_hat, y)
        val_loss += loss.item()

        y_hat_class = torch.where(y_hat > 0.5, torch.tensor(1.0), torch.tensor(0.0))
        correct_predictions += torch.sum(y_hat_class == y)
    validation_accuracy = correct_predictions / len(val_dataloader.dataset.features)

    return train_loss, val_loss, training_accuracy, validation_accuracy


def train_model(
    model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, model_name, best_model_path
):
    """
    Trains a classificaton model.
    """
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    best_model_path= best_model_path / model_name 
    best_accuracy = 0
    # Training might go very fast. Will not do any prints until later
    print("Training...")
    for epoch in range(num_epochs):
        train_loss, val_loss, training_accuracy, validation_accuracy = train_epoch(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            criterion=criterion,
            optimizer=optimizer,
        )
        if validation_accuracy > best_accuracy:
            # This code has a bug where the wokring directory has to be changed for it to work properly
            torch.save(model.state_dict(), best_model_path)
            best_accuracy = validation_accuracy
            print(f'Saving best model for epoch {epoch + 1} with accuracy {validation_accuracy:.2f}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(training_accuracy)
        val_accs.append(validation_accuracy)

    return train_losses, val_losses, train_accs, val_accs, best_model_path

def train_model_regression(
    model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, model_name, best_model_path
):
    """
    Trains a regression model
    """
    train_losses = []
    val_losses = []
    best_model_path= best_model_path / model_name 
    best_loss = 0
    # Training might go very fast. Will not do any prints until later
    print("Training...")
    for epoch in range(num_epochs):
        train_loss, val_loss = train_epoch_regression(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            criterion=criterion,
            optimizer=optimizer,
        )
        if val_loss < best_loss or best_loss == 0:
            # This code has a bug where the wokring directory has to be changed for it to work properly
            torch.save(model.state_dict(), best_model_path)
            best_loss = val_loss 
            print(f'Saving best model for epoch {epoch + 1} with loss {val_loss:.7f}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return train_losses, val_losses, best_model_path

def test_model(model, test_dataloader, best_model_path):
    """
    Tests a classification model.
    """
    model.eval()
    model.load_state_dict(torch.load(best_model_path))

    y = test_dataloader.dataset.targets.squeeze()
    y_hat = model(test_dataloader.dataset.features).detach().squeeze()
    y_hat_class = torch.where(
            y_hat > 0.5,
            torch.tensor(1, dtype=torch.int32),
            torch.tensor(0, dtype=torch.int32),
        )
    conf_mat = confusion_matrix(y_hat_class,y)
    
    return conf_mat

def k_fold_cross_validation(df : pd.DataFrame, split_percentage,n_splits, batch_size, num_epochs,criterion,model_path, learning_rate, hidden_size,kfold,target_column, task ='classification', weight_decay = 0.0, dropout = 0.0, z_score = False):
    """
    Function performs k-fold cross validation on the flow loop dataset.
    Results, features and models are saved to file for post-analysis.
    Min max standardization is applied to the data. This process is
    reverted when saving to file.

    """
    X = df.drop(target_column,axis = 1).to_numpy()
    y = df[target_column].to_numpy()
    
    # splitting the shuffled data
    data = {}
    X_k_fold =X[:int(len(df)*split_percentage)]
    y_k_fold =y[:int(len(df)*split_percentage)]

    X_test = X[int(len(df)*split_percentage):]
    y_test = y[int(len(df)*split_percentage):]
    
    best_evaluation_metric = 0
    avg_evaluation_metric =0
    best_path = model_path
    for fold, (train_index, val_index) in enumerate(kfold.split(X_k_fold)):
        
        change_directory()
        X_train, X_val = X_k_fold[train_index], X_k_fold[val_index]
        y_train, y_val = y_k_fold[train_index], y_k_fold[val_index]
        if not z_score:
            xmin = np.min(X_train, axis = 0)
            xmax = np.max(X_train, axis = 0)
            X_train = (X_train - xmin) / (xmax - xmin)
            X_val = (X_val - xmin) / (xmax - xmin)

        if z_score:
            mean = np.mean(X_train, axis = 0)
            std = np.std(X_train, axis = 0)
            X_train = (X_train - mean) / std
            X_val = (X_val - mean) / std
        
        train_data = MLPDataset(X_train, y_train)
        val_data = MLPDataset(X_val, y_val)

        train_dataloader = DataLoader(train_data,batch_size=batch_size, shuffle = True)
        val_dataloader = DataLoader(val_data,batch_size=batch_size, shuffle = False)

        mlp = MLP(in_features=len(X[1]), hidden_size=hidden_size, out_features=1, task =task, dropout=dropout) 
        optimizer = Adam(mlp.parameters(),lr = learning_rate, weight_decay=weight_decay)
        print(f'Training for fold {fold + 1}...')
        if task == 'classification':
            train_losses, val_losses, training_accuracies, validation_accuracies, best_model_path= train_model(model=mlp, 
        
                                                                                                           train_dataloader=train_dataloader,
                                                                                                           val_dataloader=val_dataloader,criterion=criterion, 
                                                                                                           optimizer=optimizer, num_epochs=num_epochs, model_name = f"mlp fold {task} {fold + 1}", 
                                                                                                           best_model_path = model_path)
        else:
            train_losses,val_losses,best_model_path = train_model_regression(model = mlp,
                                                                             train_dataloader=train_dataloader,
                                                                             val_dataloader=val_dataloader,
                                                                             criterion=criterion,
                                                                             optimizer=optimizer,
                                                                             num_epochs = num_epochs,
                                                                             model_name = f"mlp fold {task} {fold + 1}", 
                                                                             best_model_path = model_path) 

        data[f"Fold {fold + 1}"] = {}
        data[f"Fold {fold + 1}"]["Train Losses"] = train_losses 
        data[f"Fold {fold + 1}"]["Val Losses"] = val_losses
        data[f"Fold {fold + 1}"]["Features"] = {"Train features" : train_dataloader.dataset.features * (xmax - xmin) + xmin if not z_score else train_dataloader.dataset.features * std + mean, "Val features" : val_dataloader.dataset.features * (xmax - xmin) + xmin if not z_score else val_dataloader.dataset.features * std + mean}
        data[f"Fold {fold + 1}"]["Targets"] = {"Train targets" : train_dataloader.dataset.targets, "Val targets" : val_dataloader.dataset.targets}

        
        if task == 'classification':

            data[f"Fold {fold + 1}"]["Train accuracies"] = training_accuracies 
            data[f"Fold {fold + 1}"]["Val accuracies"] = validation_accuracies 
            change_directory()
            conf_mat = test_model(model = mlp, test_dataloader = val_dataloader, best_model_path=best_model_path)

            data[f"Fold {fold + 1}"]["Val confmat"] = conf_mat
        
            validation_accuracy = (conf_mat[0,0] + conf_mat[1,1]) / np.sum(conf_mat)

            if validation_accuracy > best_evaluation_metric:
                best_evaluation_metric = validation_accuracy
                best_fold = fold + 1
                best_path = best_model_path
            avg_evaluation_metric += validation_accuracy
        else:
            mlp.eval()
            yhat = mlp(val_dataloader.dataset.features)
            loss = criterion(yhat, val_dataloader.dataset.targets)
            data[f"Fold {fold + 1}"]["MSE"] = loss.item()

            if loss.item() < best_evaluation_metric or best_evaluation_metric == 0:
                best_evaluation_metric = loss.item()
                best_fold=fold +1
                best_path = best_model_path
            avg_evaluation_metric += loss.item()


   
    mlp = MLP(in_features=len(X[1]), hidden_size=hidden_size, out_features=1,task = task,dropout=dropout)
    mlp.load_state_dict(torch.load(best_path))
    X_train = X_k_fold

    if not z_score:
        xmin = np.min(X_train, axis = 0)
        xmax = np.max(X_train, axis = 0)
        X_train = (X_train - xmin) / (xmax - xmin)
        X_test = (X_test - xmin) / (xmax - xmin)

    if z_score:
        mean = np.mean(X_train, axis = 0)
        std = np.std(X_train, axis = 0)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
    train_data = MLPDataset(X_train,y_k_fold)
    train_dataloader = DataLoader(train_data,batch_size=batch_size, shuffle = True)
    test_data = MLPDataset(X_test, y_test)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle = False)
    print("Training final model...")
    # if task == 'classification':
    #     train_losses, val_losses, training_accuracies, validation_accuracies, best_model_path= train_model(model=mlp, 
    #                                                                                                train_dataloader=train_dataloader,
    #                                                                                                        val_dataloader=test_dataloader,criterion=criterion, 
    #                                                                                                        optimizer=optimizer, num_epochs=num_epochs, model_name = f"mlp final {task}", 
    #                                                                                                        best_model_path=model_path)
    
    # else:
    #     train_losses,val_losses,best_model_path = train_model_regression(model = mlp,
    #                                                                      train_dataloader=train_dataloader,
    #                                                                          val_dataloader=val_dataloader,
    #                                                                          criterion=criterion,
    #                                                                          optimizer=optimizer,
    #                                                                          num_epochs = num_epochs,
    #                                                                          model_name = f"mlp final {task}", 
    #                                                                          best_model_path = model_path)
         
                        
    # data["Final training"] = {}
    # data["Final training"]["Train Losses"] = train_losses 
    # data["Final training"]["Val Losses"] = val_losses 
    # data["Final training"]["Features"] = {"Train features" : train_dataloader.dataset.features * (xmax - xmin) + xmin if not z_score else train_dataloader.dataset.features * std + mean}
    # data["Final training"]["Targets"] = {"Train targets" : train_dataloader.dataset.targets}
    
    if task == 'classification':
        change_directory()
        conf_mat_test = test_model(model = mlp, test_dataloader=test_dataloader, best_model_path=best_model_path) 
        data["Test confmat"] = conf_mat_test
        data["Final training"]["Train accuracies"] = training_accuracies 
        data["Final training"]["Val accuracies"] = validation_accuracies 
    else:
        mlp.eval()
        yhat = mlp(test_dataloader.dataset.features)
        loss = criterion(yhat,test_dataloader.dataset.targets)
        data["Test preds"] = yhat.detach()
        data["Test loss"] = loss.item() 
    data["Test features"] = test_dataloader.dataset.features * (xmax - xmin) + xmin if not z_score else test_dataloader.dataset.features * std + mean
    data["Test targets"] = test_dataloader.dataset.targets
     
    return data, best_fold, avg_evaluation_metric / n_splits


def k_fold_cross_validation_sklearn_models(df : pd.DataFrame,split_percentage,model, n_splits, kfold_outer,kfold_inner, param_grid,target_column, task = 'classification', z_score = False):
    """
    Performs k-fold cross validation on a sklearn model with the function.fit
    The function saves the coefficients, and the confusion matrix for every model.
    """
    X = df.drop(target_column,axis = 1).to_numpy()
    y = df[target_column].to_numpy()
    
    # splitting the shuffled data
    data = {}
    X_k_fold =X[:int(len(df)*split_percentage)]
    y_k_fold =y[:int(len(df)*split_percentage)]

    X_test = X[int(len(df)*split_percentage):]
    y_test = y[int(len(df)*split_percentage):]
    
    best_evaluation_metric = 0
    avg_evaluation_metric = 0
    

    for fold, (train_index, val_index) in tqdm(enumerate(kfold_outer.split(X_k_fold))):
        
       
        X_train, X_val = X_k_fold[train_index], X_k_fold[val_index]
        y_train, y_val = y_k_fold[train_index], y_k_fold[val_index]
        if not z_score:
            xmin = np.min(X_train, axis = 0)
            xmax = np.max(X_train, axis = 0)
            X_train = (X_train - xmin) / (xmax - xmin)
            X_val = (X_val - xmin) / (xmax - xmin)

        if z_score:
            mean = np.mean(X_train, axis = 0)
            std = np.std(X_train, axis = 0)
            X_train = (X_train - mean) / std
            X_val = (X_val - mean) / std
        # Min max scaling
        #Performing cross validaton using the inner fold
        fited_grid_search_cv = inner_fold_for_hyperparameter_optimization(model = model, k_fold_inner=kfold_inner, X = X_train,y=y_train, param_grid=param_grid,task = task)
        model_fited = model(**fited_grid_search_cv.best_params_)
        model_fited = model_fited.fit(X_train,y_train)
        yhat_train = model_fited.predict(X_train)
        yhat_val = model_fited.predict(X_val)
        data[f"Fold {fold + 1}"] = {}
        if task == 'classification':
            confmat_val = confusion_matrix(yhat_val,y_val)
            confmat_train = confusion_matrix(yhat_train,y_train)
            evaluation_metric =(confmat_val[0,0] + confmat_val[1,1]) /np.sum(confmat_val)
            avg_evaluation_metric += evaluation_metric
            data[f"Fold {fold + 1}"]["Train confmat"] = confmat_train 
            data[f"Fold {fold + 1}"]["Val confmat"] = confmat_val 
        else:
            mse_train = np.mean(np.power((yhat_train - y_train),2))
            evaluation_metric = np.mean(np.power((yhat_val - y_val),2))
            data[f"Fold {fold + 1}"]["MSE"] = evaluation_metric
            avg_evaluation_metric += evaluation_metric


        data[f"Fold {fold + 1}"]["Features"] = {"Train features" : X_train * (xmax - xmin) + xmin if not z_score else X_train * std + mean, "Val features" : X_val * (xmax - xmin) + xmin if not z_score else X_train *std + mean}
        data[f"Fold {fold + 1}"]["Targets"] = {"Train targets" : y_train, "Val targets" : y_val}
        data[f"Fold {fold + 1}"]["Best parameters"] = fited_grid_search_cv.best_params_
        data[f"Fold {fold + 1}"]["Best score"] = fited_grid_search_cv.best_score_
        if (evaluation_metric > best_evaluation_metric and task =='classification') or (evaluation_metric < best_evaluation_metric and task != 'classification') or (best_evaluation_metric == 0 and task != 'classification'):
            best_fold = fold + 1
            best_evaluation_metric = evaluation_metric
            # Choosing the best parameters over the entire validation process
            best_parameters = fited_grid_search_cv.best_params_
        


    # Testing
    X_train = X_k_fold
    if not z_score:
        xmin = np.min(X_train, axis = 0)
        xmax = np.max(X_train, axis = 0)
        X_train = (X_train - xmin) / (xmax - xmin)
        X_test = (X_test - xmin) / (xmax - xmin)

    if z_score:
        mean = np.mean(X_train, axis = 0)
        std = np.std(X_train, axis = 0)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
    model = model(**best_parameters)
    model_fited = model.fit(X_train, y_k_fold)
    yhat = model_fited.predict(X_test)

    if task == 'classification':
       confmat_test = confusion_matrix(yhat,y_test)
       data["Test confmat"] = confmat_test
       print(f"Best model found for fold: {best_fold} with accuracy of {round(best_evaluation_metric,2)}", "average accuracy is: ", avg_evaluation_metric / n_splits)
    else:
        data["Test mse"] = np.mean(np.power(yhat-y_test,2))
        data["Test preds"] = yhat
        print(f"Best model found for fold: {best_fold} with mse of {round(best_evaluation_metric,7)}", "average mse is: ", avg_evaluation_metric / n_splits)
    data["Final training features"] =  X_train * (xmax - xmin) + xmin if not z_score else X_train * std + mean
    data["Final training targets"] = y_k_fold
    data["Test features"] = X_test * (xmax - xmin) + xmin if not z_score else X_test* std + mean
    data["Test targets"] = y_test


    return data, best_fold, model_fited, avg_evaluation_metric / n_splits

def inner_fold_for_hyperparameter_optimization(model,k_fold_inner, X, y, param_grid,task, n_iter = 100):
    """
    Uses the inner fold of the loop to perform a hyperparameter optimization and evaluates the performance
    on a left out validation set.
    """
    model = model()
    if task =='classification':
        scoring = 'accuracy'

    else:
        # negative due to the higher is better metric of the gridsearchcv function
        scoring ='neg_mean_squared_error' 
    cv = RandomizedSearchCV(estimator = model, param_distributions=param_grid, cv = k_fold_inner, scoring = scoring, n_iter=n_iter)
    fited_cv = cv.fit(X=X,y=y)

    return fited_cv
 
     

    