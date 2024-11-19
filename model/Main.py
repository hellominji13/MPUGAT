

import os
import argparse
import numpy as np
from datetime import datetime
from torch import nn, optim
from torch.nn import functional as F
import torch
import Utils, Multi_model
#import torch_optimizer as optim  
#%%
class ModelTrainer(object):
    def __init__(self, params: dict, data: dict, data_container):
        self.params = params
        self.data_container = data_container
        self.commuter = torch.from_numpy(data['commute'].astype(np.float32)).to(params['GPU'])
        self.model = self.get_model().to(params['GPU'])
        self.criterion = self.get_loss()
        self.optimizer = self.get_optimizer()
        self.task_level = 1

    def get_model(self):
        model = Multi_model.Model(num_feature = self.params['N'], in_len=self.params['obs_len'],out_len=self.params['pred_len'],layers=2
                                    , dropout=0.3, hidden_unit=params["hidden_unit"], model = self.params["model"])
        return model

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
        return optimizer

    def train(self, data_loader: dict, modes: list, early_stop_patience=10):
        checkpoint = {'epoch': 0, 'state_dict': self.model.state_dict()}
        val_loss = np.inf
        patience_count = early_stop_patience

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} model training begins:')

        for epoch in range(1, 1 + self.params['num_epochs']):
            starttime = datetime.now()
            running_loss = {mode: 0.0 for mode in modes}
            for mode in modes:
                if mode == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                step = 0
                for x_node, x_known, x_SIR, y_true, matrix in data_loader[mode]:
                    with torch.set_grad_enabled(mode=(mode == 'train')):
                        beta,y_pred,h = self.model(x_node, x_known, x_SIR, matrix)
                        if mode == 'train':
                            
                            self.optimizer.zero_grad()
                            if epoch % 2 == 0 and self.task_level <=  self.params["pred_len"]:
                                self.task_level += 1
                                loss = self.criterion(y_pred[:, :self.task_level, :],
                                                      y_true[:, :self.task_level, :])
                            else:
                                loss = self.criterion(y_pred, y_true)

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                            self.optimizer.step()
                            
                        else:
                            loss = self.criterion(y_pred, y_true)

                    running_loss[mode] += loss * y_true.shape[0]  # loss reduction='mean': batchwise average
                    step += y_true.shape[0]
                    #torch.cuda.empty_cache()
                
                # epoch end: evaluate on validation set for early stopping
                if mode == 'validate':
                    epoch_val_loss = running_loss[mode] / step
                    if epoch_val_loss < val_loss-0.1:
                        print(f'Epoch {epoch}, validation loss drops from {val_loss:.5} to {epoch_val_loss:.5}. '
                              f'Update model checkpoint..', f'used {(datetime.now() - starttime).seconds}s')
                        val_loss = epoch_val_loss
                        #torch.save(checkpoint, self.params['output_dir'] + f'/{self.params["model"]}_od.pkl')
                        patience_count = early_stop_patience
                    else:
                        print(f'Epoch {epoch}, validation loss does not improve from {val_loss:.5}.',
                              f'used {(datetime.now() - starttime).seconds}s')
                        patience_count -= 1
                        if patience_count == 0:
                            return
        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} model training ends.')
        return

    def test(self, data_loader: dict, modes: list):
        #trained_checkpoint = torch.load(self.params['output_dir'] + f'/{self.params["model"]}_od.pkl')
        #self.model.load_state_dict(trained_checkpoint['state_dict'])
        self.model.eval()
        beta, adj= [],[]
        for mode in modes:
            print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            print(f'     {self.params["model"]} model testing on {mode} data begins:')
            forecast, ground_truth = [], []
            for x_node, x_known, x_SIR, y_true, matrix in data_loader[mode]:
                b,y_pred,h = self.model(x_node, x_known, x_SIR, matrix)
                if params["model"] == "deep_learning":
                    forecast.append(data_generator.prepro.inverse_transform(y_pred.detach()).cpu().detach())
                    ground_truth.append(data_generator.prepro.inverse_transform(y_true).cpu().detach())
                else:
                    forecast.append(y_pred.cpu().detach())
                    ground_truth.append(y_true.cpu().detach())
                beta.append(b.cpu().detach())
                adj.append(h.cpu().detach())

            forecast = torch.cat(forecast, dim=0)
            ground_truth = torch.cat(ground_truth, dim=0)
            np.save(self.params['output_dir'] + '/' + self.params['model'] + '_' + mode + '_prediction.npy', forecast)
            np.save(self.params['output_dir'] + '/' + self.params['model'] + '_' + mode + '_groundtruth.npy',
                    ground_truth)

            # evaluate on metrics
            MSE, RMSE, MAE, MAPE, RAE = self.evaluate(forecast, ground_truth)
            f = open(self.params['output_dir'] + '/' + self.params['model'] + '_prediction_scores.txt', 'a')
            f.write("%s, MSE, RMSE, MAE, MAPE, RAE, %.10f, %.10f, %.10f, %.10f, %.10f\n" % (mode, MSE, RMSE, MAE, MAPE, RAE))
            if mode == 'test':
                for i in range(ground_truth.shape[1]):
                    print("%d step" % (i+1))
                    MSE, RMSE, MAE, MAPE, RAE = self.evaluate(forecast[:, i, :], ground_truth[:, i, :])
                    f.write("%d step,  %s, MSE, RMSE, MAE, MAPE, RAE, %.10f, %.10f, %.10f, %.10f, %.10f\n" % (
                        i + 1, mode, MSE, RMSE, MAE, MAPE, RAE))
            f.close()

        print('\n', datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        print(f'     {self.params["model"]} model testing ends.')
        dictt = {}
        dictt["b"] = beta
        dictt["h"] = adj
        np.save(self.params['output_dir'] + '/' +'parameter.npy', dictt)
        return

    @staticmethod
    def evaluate(y_pred: np.array, y_true: np.array):
        def MSE(y_pred: np.array, y_true: np.array):
            return F.mse_loss(y_pred, y_true)

        def RMSE(y_pred: np.array, y_true: np.array):
            return torch.sqrt(F.mse_loss(y_pred, y_true))

        def MAE(y_pred: np.array, y_true: np.array):
            return F.l1_loss(y_pred, y_true)

        def MAPE(y_pred: np.array, y_true: np.array):  # avoid zero division
            return mape(y_pred, y_true + 1.0)

        def RAE(y_pred: np.array, y_true: np.array):
            return rae(y_pred, y_true)

        print('MSE:', round(MSE(y_pred, y_true).item(), 4))
        print('RMSE:', round(RMSE(y_pred, y_true).item(), 4))
        print('MAE:', round(MAE(y_pred, y_true).item(), 4))
        print('MAPE:', round((MAPE(y_pred, y_true).item() * 100), 4), '%')
        print('RAE:', round(RAE(y_pred, y_true).item(), 4))
        return MSE(y_pred, y_true), RMSE(y_pred, y_true), MAE(y_pred, y_true), MAPE(y_pred, y_true), RAE(y_pred, y_true)

def mape(preds, labels):
    loss = torch.abs(preds-labels)/labels
    return torch.mean(loss)

def rae(preds, labels):
    loss = torch.abs(preds-labels)/(torch.abs(labels-labels.mean()).sum())
    return torch.sum(loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Prediction')

    # command line arguments
    parser.add_argument('-GPU', '--GPU', type=str, help='Specify GPU usage', default='cuda:0') #cuda:0
    parser.add_argument('-in', '--input_dir', type=str, default='../data')
    parser.add_argument('-out', '--output_dir', type=str, default='./output')
    parser.add_argument('-model', '--model', type=str, help='Specify model', choices=['deep_learning',"MPUGAT","MPGAT"], default='MPGAT')
    parser.add_argument('-graph', '--graph_type', type=str, help='Specify graph learning type',
                        choices=['Adaptive', 'Dynamic'], default='Adaptive')
    parser.add_argument('-obs', '--obs_len', type=int, help='Length of observation sequence', default=14)
    parser.add_argument('-pred', '--pred_len', type=int, help='Length of prediction sequence', default=1)
    parser.add_argument('-split', '--split_ratio', type=float, nargs='+',
                        help='Relative data split ratio in train : validate : test'
                             ' Example: -split 6 1 1', default=[9, 1, 0])
    parser.add_argument('-batch', '--batch_size', type=int, default=10)
    parser.add_argument('-epoch', '--num_epochs', type=int, default=2000)
    parser.add_argument('-loss', '--loss', type=str, help='Specify loss function',
                        choices=['MSE', 'MAE'], default='MAE')
    parser.add_argument('-optim', '--optimizer', type=str, help='Specify optimizer', default='Adam')
    parser.add_argument('-lr', '--learn_rate', type=float, default=0.016436376562616366)
    parser.add_argument('-test', '--test_only', type=int, default=0)  # 1 for test only
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-9)
    parser.add_argument('-rep', '--reproduce', type=bool, default=True)
    parser.add_argument('-rd', '--random_seed', type=int, default=0)
    parser.add_argument('-hidden_unit', '--hidden_unit', type=int, default=14)

    params = parser.parse_args().__dict__  # save in dict

    if params['reproduce'] is True:
        torch.manual_seed(params['random_seed'])
        torch.cuda.manual_seed(params['random_seed'])
        np.random.seed(params['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # paths
    os.makedirs(params['output_dir'], exist_ok=True)#UnboundLocalError
    data_input = Utils.DataInput(data_dir=params['input_dir'])
    data = data_input.load_data()
    print(data['node'].shape)
    params['N'] = data['node'].shape[-1]

    # get data loader
    data_generator = Utils.DataGenerator(obs_len=params['obs_len'],
                                         pred_len=params['pred_len'],
                                         data_split_ratio=params['split_ratio'])
    data_loader = data_generator.get_data_loader(data=data, params=params)

    # get model
    trainer = ModelTrainer(params=params, data=data, data_container=data_input)
    trainer.train(data_loader=data_loader,
                      modes=['train', 'validate'])
    trainer.test(data_loader=data_loader,
                 modes=['train', 'validate', 'test'])


# %%
