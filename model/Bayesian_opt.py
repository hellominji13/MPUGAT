

#%%
import os
import argparse
import numpy as np
from datetime import datetime
from torch import nn, optim
from torch.nn import functional as F
import torch
import Utils 
import Multi_model
from bayes_opt import BayesianOptimization,UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import time
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import random

class ModelTrainer(object):
    def __init__(self, params: dict, data: dict, data_container, data_loader,data_generator ):
        self.params = params 
        self.data_loader = data_loader
        self.data_container = data_container
        self.epoch_dict = {}
        self.model_dict = {}
        self.test_loss_dict = {}
        self.data_generator = data_generator
        
    def get_loss(self):
        if self.params['loss'] == 'MSE':
            criterion = nn.MSELoss()
        elif self.params['loss'] == 'MAE':
            criterion = nn.L1Loss()
        else:
            raise NotImplementedError('Invalid loss function.')
        return criterion

    def get_optimizer(self):
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(params=self.model.parameters(),
                                   lr=self.params['learn_rate'],
                                   weight_decay=self.params['weight_decay'])
        else:
            raise NotImplementedError('Invalid optimizer name.')
        return optimizer

    def train(self, lr, hidden_unit,n_head,lstm_layer):
        self.task_level = 1
        self.params["learn_rate"] = lr
        self.params["hidden_unit"] = int(round(hidden_unit))
        self.params["n_head"] = int(round(n_head))
        self.params["layer"] = int(round(lstm_layer))
        self.model = Multi_model.Model(num_feature = self.params['N'], in_len=self.params['obs_len'],out_len=self.params['pred_len'],layers=self.params["layer"],
                                    dropout=0.3, hidden_unit=params["hidden_unit"], model = self.params["model"], n_head=self.params["n_head"]).to(self.params['GPU'])
        self.criterion = self.get_loss()
        self.optimizer = self.get_optimizer()
        val_loss = np.inf
        early_stop = 10
        patience_count = early_stop
        
        starttime = datetime.now()
        check_epoch = 0 
        two_break = False
        for epoch in range(1, 1 + self.params['num_epochs']):
            starttime = datetime.now()
            running_loss = {mode: 0.0 for mode in ["train","validate","test"]}
            for mode in ["train","validate"]:
                if mode == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                step = 0
                for x_node, x_SIR, y_true, matrix in self.data_loader[mode]:
                    #shape of X : [batch, input_time ,feature]
                    #shape of y : [batch, 1 or pred_time]
                    #print(y_true.shape) # batch, pred_len, 16, 4 
                    with torch.set_grad_enabled(mode=(mode == 'train')):

                        beta,y_pred,h = self.model(x_node, x_SIR, matrix)
                    if self.params["model"]=="compartment":
                        loss = self.criterion(y_pred[:,:,:,:], y_true[:,:,:,:])

                    else:
                            if mode == 'train':
                                self.optimizer.zero_grad()
                                loss = self.criterion(y_pred[:,:,:,:], y_true[:,:,:,:])
                                loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                                self.optimizer.step()
                            else:
                                if params["model"] == "deep_learning":
                                    y_pred = self.data_generator.prepro.inverse_transform(y_pred.detach())
                                    y_true = self.data_generator.prepro.inverse_transform(y_true.detach())
                                loss = self.criterion(y_pred[:,:,:,:], y_true[:,:,:,:])
                    running_loss[mode] += loss * y_true.shape[0]  # loss reduction='mean': batchwise average
                    step += y_true.shape[0]
                
                if mode == 'validate':
                    epoch_val_loss = running_loss[mode] / step
                    if epoch_val_loss < val_loss :
                        val_loss = epoch_val_loss
                        patience_count = early_stop
                        print(f'Epoch {epoch}, validation loss drops from {val_loss:.5} to {epoch_val_loss:.5}. '
                              f'Update model checkpoint..', f'used {(datetime.now() - starttime).seconds}s')
                    else:
                        patience_count -= 1
                        if patience_count == 0:
                            two_break = True
                            break
            check_epoch += 1
            if two_break == True:
                break

        self.model_dict[float(val_loss)] = self.model
        self.epoch_dict[float(val_loss)]  = check_epoch
        self.model.eval()
        step=0
        for x_node, x_SIR, y_true, matrix in self.data_loader[mode]:
            b,y_pred,h = self.model(x_node, x_SIR, matrix)
            loss =  self.criterion(y_true[:,:,:,:], y_pred[:,:,:,:]) #self.criterionself.criterion(y_I, pred_I) +
            running_loss["test"] += loss * y_true.shape[0]
            step += y_true.shape[0]  # loss reduction='mean': batchwise average
        self.test_loss_dict[float(val_loss)] = (running_loss["test"] / step).item()
        return  -val_loss.item() if isinstance(val_loss, torch.Tensor) else -val_loss


    def get_epoch(self):
        return self.epoch_dict
    
    def get_model(self):
        return self.model_dict
    
    def get_test_loss(self):
        return self.test_loss_dict



def Run_One_Model(params,parameter_bound):
    data_input = Utils.DataInput(data_dir=params['input_dir'])
    data = data_input.load_data()
    data_generator = Utils.DataGenerator(obs_len=params['obs_len'],
                                         pred_len=params['pred_len'],
                                         data_split_ratio=params['split_ratio'])
    data_loader = data_generator.get_data_loader(data=data,params=params)
    trainer = ModelTrainer(params=params, data=data, data_container=data_input, data_loader=data_loader,data_generator =data_generator)
    #이제 여러번 돌릴거임
    bayes_optimizer  = BayesianOptimization(f = trainer.train, pbounds = parameter_bound, random_state=0)           
    utility = UtilityFunction(kind="ei", xi=0.01)
    bayes_optimizer.suggest(utility)
    bayes_optimizer.set_gp_params(alpha=1e-4)#alpha가 커진다=uncertainty의 영향을 더 고려한다
    bayes_optimizer.maximize(init_points=10,n_iter=20)             
    print("run_bayes_optimizer_complete") 
    columns=["model","pred_len","epoch","lr","n_head","lstm_layer","hidden_unit","Val_loss","Test_loss"]
    val_loss = -bayes_optimizer.max["target"]
    lr = bayes_optimizer.max["params"]["lr"]
    n_head = int(round(bayes_optimizer.max["params"]["n_head"]))
    lstm_layer = int(round(bayes_optimizer.max["params"]["lstm_layer"]))
    hidden_unit = int(round(bayes_optimizer.max["params"]["hidden_unit"]))
    epoch = int(trainer.get_epoch()[val_loss])
    Best_Net = trainer.get_model()[val_loss]
    test_loss = trainer.get_test_loss()[val_loss]

    Result = [params["model"],params["pred_len"],epoch,lr,n_head,lstm_layer,hidden_unit,val_loss,test_loss]
    df =  pd.DataFrame(data=[Result], columns=columns)
    #df.to_csv("{}/best_models/{}_hyperparameter_{}_model_{}_pred.csv".format(params["output_dir"],len(params['node']),params["model"],params["pred_len"]))
    #model_save_path = "{}/best_models/hyper/{}_model_{}_model_{}_pred.pth".format(params["output_dir"],len(params['node']),params["model"],params["pred_len"])
    #model_save_path2 = "{}/hyper/hyper/{}_load_model_{}_model_{}_pred.pth".format(params["output_dir"],len(params['node']),params["model"],params["pred_len"])
    '''
    torch.save({
            'epoch': epoch,
            'model_state_dict': Best_Net.state_dict(),
            'test loss': test_loss,
            }, model_save_path)'''
    #torch.save(Best_Net, model_save_path2)
    return Result
    
def Bayesian_main(params):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0) 

    parameter_bound = {"lr": (0.0001, 0.01), "hidden_unit": (2,5),"n_head":(1,2),"lstm_layer":(1,2)}
    All_Result = pd.DataFrame(columns=["model", "pred_len", "epoch", "lr", "n_head","lstm_layer","hidden_unit", "Val_loss", "Test_loss"])
    for pred_len in [3]:
        params["pred_len"] = pred_len
        for model in ["MPGAT","deep_learning",  "C_ij"]: #"MPGAT","deep_learning",  "C_ij"
            params["model"] = model
            start_time = datetime.now()
            Result = Run_One_Model(params, parameter_bound)
            All_Result.loc[len(All_Result)] = Result  # Append the result 
            print('Model training completed. Time:', datetime.now() - start_time)
            All_Result.to_csv("./hyper/Hyperparameter.csv")  # Save results after all iterations
        All_Result.loc[len(All_Result)] = ["compartment",pred_len,1,1,1,1,1,1,1]
        All_Result.to_csv("./hyper/Hyperparameter.csv")
    print(All_Result)
    return All_Result

def compute_metrics(predictions, actuals):
    mse = ((predictions - actuals) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    return mse, rmse, mae


def Run_seed(Result, params):
    results = []  
    for index, row in Result.iterrows():
        # Update the parameters with the current row values
        params.update({
            "model": row["model"],
            "n_head": row["n_head"],
            "layer" : row["lstm_layer"],
            "lr": row["lr"],
            "hidden_unit": row["hidden_unit"],
            "pred_len":row["pred_len"]
        })
        
        # Load data
        data_input = Utils.DataInput(data_dir=params['input_dir'])
        data = data_input.load_data()
        data_generator = Utils.DataGenerator(obs_len=params['obs_len'], pred_len=params['pred_len'], data_split_ratio=params['split_ratio'])
        data_loader = data_generator.get_data_loader(data=data, params=params)
        trainer = ModelTrainer(params=params, data=data, data_container=data_input, data_loader=data_loader, data_generator=data_generator)
        
        model_results = []
        save = {}
        
        # Set seed_num based on model type
        if params["model"] == "compartment":
            seed_num = 1
        else:
            seed_num = 30
        
        # Initialize storage for predictions, actuals, etc.
        save["train"] = np.zeros((seed_num, data_generator.len["train"], params["pred_len"], 16, 4,4))
        save["validate"] = np.zeros((seed_num, data_generator.len["validate"], params["pred_len"], 16,4, 4))
        save["test"] = np.zeros((seed_num, data_generator.len["test"], params["pred_len"], 16,4, 4))
        save["contact"] = np.zeros((seed_num, data_generator.len["train"] + data_generator.len["validate"] + data_generator.len["test"], 16, 16))
        
        for seed in range(seed_num):
            # Set seeds for reproducibility
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            
            # Train the model and get the network with the lowest validation loss
            val_loss = -trainer.train(params["lr"], params["hidden_unit"],params["n_head"],params["layer"])
            Net = trainer.get_model()[val_loss]
            contact = []
            
            for mode in ["train", "validate","test"]:
                predictions, actuals, beta_pred = [], [], []
                
                for x_node, x_SIR, y_true, matrix in data_loader[mode]:
                    # Forward pass
                    b, y_pred, h = Net(x_node, x_SIR, matrix)
                    # Process predictions and actuals based on model type
                    if params["model"] == "deep_learning":
                        predictions.append(data_generator.prepro.inverse_transform(y_pred.detach()).cpu().numpy())
                        actuals.append(data_generator.prepro.inverse_transform(y_true).cpu().numpy())

                    else:
                        predictions.append(y_pred.cpu().detach().numpy())
                        actuals.append(y_true.cpu().detach().numpy())
                    
                    beta_pred.append(b.cpu().detach().numpy())
                    contact.append(h.cpu().detach().numpy()[:,:,:])
                
                # Convert lists to numpy arrays
                predictions = np.concatenate(predictions)
                actuals = np.concatenate(actuals)
                beta_pred = np.concatenate(beta_pred)
                
                # Store predictions, actuals, and beta_pred
                save[mode][seed, :, :,:, :, 0] = predictions
                save[mode][seed, :, :,:, :, 1] = actuals
                save[mode][seed, :, :,:, 0, 2] = beta_pred
                
                # Calculate metrics for all modes
                mse, rmse, mae = compute_metrics(predictions[:, -1,:, :], actuals[:, -1,:, :])
                model_results.append({
                    'pred_len': params['pred_len'], 
                    'model': params['model'], 
                    'seed': seed, 
                    'mode': mode,  # Store the mode information
                    'mse': mse, 
                    'rmse': rmse, 
                    'mae': mae
                })
                
            # Concatenate contact information
            
            contact = np.concatenate(contact, axis=0)
            contact_mean = np.mean(contact, axis=0)  # (city, city)
            if params["model"] == "MPUGAT":
                np.savetxt(f"./seed/contact_mean_{params['pred_len']}.csv", contact_mean, delimiter=",")
            save["contact"][seed, :, :, :] = contact
        
        # Save prediction and actual data to file
        pred_act_file = os.path.join(f"./seed/preds_acts_{params['model']}_{params['pred_len']}.npy")
        np.save(pred_act_file, save)
        
        # Create a DataFrame from the results and calculate summary statistics
        df_metrics = pd.DataFrame(model_results)
        summary = df_metrics.groupby(['mode']).agg({
            'mse': ['mean', 'std'], 
            'rmse': ['mean', 'std'], 
            'mae': ['mean', 'std']
        })
        
        # Rename columns
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary['model'] = params['model']
        summary['pred_len'] = params['pred_len']
        
        # Append the summary to the results list
        results.append(summary)
    
    # Concatenate all results into a single DataFrame
    results_df = pd.concat(results)
    results_df.reset_index(inplace=True)
    results_df.set_index(['mode', 'model', 'pred_len'], inplace=True)
    
    # Save the summary results to a CSV file
    results_df.to_csv('./seed/model_performance_summary.csv')



if __name__ == '__main__':

    params =  {}
    params["N"] = 6
    params["GPU"] = "cpu"#'cuda:0'""
    params["input_dir"] = '../data'
    params["split_ratio"] = [8, 1, 1]
    params["batch_size"] = 30
    params["num_epochs"] = 1000
    params["loss"] = 'MAE'
    params["optimizer"] = 'Adam'
    params["weight_decay"] = 1e-9
    params["obs_len"] = 15
    
    Result = Bayesian_main(params)
    Result = pd.read_csv("./hyper/Hyperparameter.csv")
    Run_seed(Result,params)


 
 
 
    # %%
