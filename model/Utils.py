#%%
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os 

class DataInput(object):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_data(self):
        PATH = self.data_dir + '/14_Data(20-03-01_21-11-23).npy' 
        data = np.load(PATH, allow_pickle='TRUE').item()
        data_node = np.concatenate((data['compart'][...,:-1],data["movement"][...,[1,2]]),axis=2)
        #data_node = data['compart'][...,[2]]

        dataset = dict()
        dataset['node'] = data_node
        dataset['SEIR'] = data['compart'][...,:-1]
        dataset['y'] = data['compart'][...,:-1]#.squeeze()#[...,[2]]
        dataset['commute'] = data["corr"]
        return dataset


class StandardScaler():
    def __init__(self, mean, std, min, max):
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max

    def transform(self, data):
        #data = (data - self.mean) / self.std
        data = (data - self.min)/(self.max-self.min)
        return data

    def inverse_transform(self, data):
        if data.is_cuda:
            data = data.cpu()
        data = data*(self.max-self.min)+ self.min
        #data = (data * self.std) + self.mean 
        return data


class ODDataset(Dataset):
    def __init__(self, inputs: dict, output: torch.Tensor, mode: str, mode_len: dict):
        self.mode = mode
        self.mode_len = mode_len
        self.inputs, self.output = self.prepare_xy(inputs, output)

    def __len__(self):
        return self.mode_len[self.mode]

    def __getitem__(self, item):
        return self.inputs['x_node'][item] ,self.inputs['x_SIR'][item],self.output[item], self.inputs['matrix']

    def prepare_xy(self, inputs: dict, output: torch.Tensor):
        if self.mode == 'train':
            start_idx = 0
        elif self.mode == 'validate':
            start_idx = self.mode_len['train']
        else:  # test
            start_idx = self.mode_len['train'] + self.mode_len['validate']
        x = dict()
        x['x_SIR'] = inputs['x_SIR'][start_idx: (start_idx + self.mode_len[self.mode]),...]
        x['x_node'] = inputs['x_node'][start_idx: (start_idx + self.mode_len[self.mode]),...]
        x['matrix'] = inputs['matrix']
        y = output[start_idx: start_idx + self.mode_len[self.mode]]
        return x, y

class DataGenerator(object):
    def __init__(self, obs_len: int, pred_len, data_split_ratio: tuple):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.data_split_ratio = data_split_ratio
        

    def split2len(self, data_len: int):
        mode_len = dict()
        mode_len['train'] = int(self.data_split_ratio[0] / sum(self.data_split_ratio) * data_len)
        mode_len['validate'] = int(self.data_split_ratio[1] / sum(self.data_split_ratio) * data_len)
        mode_len['test'] = data_len - mode_len['train'] - mode_len['validate']
        self.len = mode_len
        return mode_len

    def get_data_loader(self, data: dict, params: dict):
        x_node, x_SIR, y = self.get_feats(data)
        commute = data["commute"]
        commute = np.asarray(commute, dtype=np.int64)
        x_node = np.asarray(x_node)
        x_SIR = np.asarray(x_SIR)
        y = np.asarray(y)
        mode_len = self.split2len(data_len=y.shape[0])
        
        for i in range(x_node.shape[-1]):
            scaler = StandardScaler(mean=x_node[:mode_len['train'],..., i].mean(axis=0),std=x_node[:mode_len['train'],..., i].std(axis=0), 
                                        min=x_node[:mode_len['train'],..., i].min(axis=0),max=x_node[:mode_len['train'],..., i].max(axis=0))
            x_node[...,i] = scaler.transform(x_node[...,i])
        if params["model"] == "deep_learning":
            self.prepro = StandardScaler(mean=y[:mode_len['train'],...].mean(axis=0),
                                        std=y[:mode_len['train'],...].std(axis=0), 
                                        min = y[:mode_len['train'],...].min(axis=0), 
                                        max = y[:mode_len['train'],...].max(axis=0))
            y[...] = self.prepro.transform(y[...])
        feat_dict = dict()
        feat_dict["matrix"] = torch.from_numpy(commute).float().to(params['GPU'])
        feat_dict['x_node'] = torch.from_numpy(x_node).float().to(params['GPU'])
        feat_dict['x_SIR'] = torch.from_numpy(x_SIR).float().to(params['GPU'])
        y = torch.from_numpy(y).float().to(params['GPU'])
        print('Data split:', mode_len)

        data_loader = dict()  # data_loader for [train, validate, test]
        for mode in ['train', 'validate', 'test']:
            dataset = ODDataset(inputs=feat_dict, output=y, mode=mode, mode_len=mode_len)
            print('Data loader', '|', mode, '|', 'input node features:', dataset.inputs['x_node'].shape, '|'
                  'output:', dataset.output.shape)
            if mode == 'train':
                data_loader[mode] = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=False)
            else:
                data_loader[mode] = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=False)
        return data_loader

    def get_feats(self, data: dict):
        x_node, x_SIR, y = [], [], []
        for i in range(self.obs_len, data['node'].shape[0] - self.pred_len + 1):
            x_node.append(data['node'][i - self.obs_len: i,...])
            x_SIR.append(data['SEIR'][i - self.obs_len: i,...])
            y.append(data['y'][i: i + self.pred_len,...])
        return x_node,x_SIR, y


# %%
