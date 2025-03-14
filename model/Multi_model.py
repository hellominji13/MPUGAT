'''
Part of code is adapted from https://github.com/nnzhan/Graph-WaveNet
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#GAT https://nn.labml.ai/graphs/gat/index.html
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GAT(nn.Module):
    def __init__(self, in_features, out_features, n_heads, dropout,is_concat=False):
        super().__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads
        if is_concat:
            self.n_hidden = out_features
        else:
            assert out_features % self.n_heads == 0
            self.n_hidden = out_features // self.n_heads
        self.in_features = in_features
        self.linear = nn.Linear(in_features, in_features * self.n_heads , bias=True)
        self.linear2 = nn.Linear(in_features, in_features * self.n_heads, bias=False)
        self.act =nn.LeakyReLU(0.2)
        self.softmax_2 = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)
        self.ouput_layer_mlp = nn.Linear(self.n_heads*16*self.in_features,self.n_hidden,bias=True)
        # head, city 부분을 하나로 aggreation 하는 코드 
        self.check = 0
        self.a = nn.Parameter(torch.tensor(1,dtype=torch.float32).to(device))
        self.b = nn.Parameter(torch.tensor(1,dtype=torch.float32).to(device))

    def forward(self, x,matrix,N,Update):
        n_nodes = x.shape[1]  # number of nodes
        batch_size = x.shape[0]
        matrix = matrix/(1e4*abs(self.b))
        # Linear transformation
        wg = self.linear(x).view(batch_size, n_nodes, self.n_heads, self.in_features)
        g =  self.linear2(x).view(batch_size, n_nodes, self.n_heads, self.in_features)
        g_reshaped = F.normalize(wg, p=2, dim=-1) 
        g_transposed = F.normalize(g, p=2, dim=-1) 
        e = torch.einsum("bnhk,bmhk->bnmh", g_transposed, g_reshaped)
        e = e.view(batch_size, n_nodes, n_nodes,  self.n_heads)
        eye = torch.eye(n_nodes).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1, self.n_heads).to(device)  # Shape: (1, n_nodes, n_nodes, 1)
        e = e + self.a*eye
        attn_res = torch.tanh(torch.einsum('bjhf,bjih->bihf',  g, e))
        e=self.softmax_2(e)
        check = e.mean(dim=-1, keepdim=False)
        attn_res = torch.mean(attn_res.contiguous().view(batch_size, self.n_heads, 16, self.in_features), dim=(1,2,3)).unsqueeze(-1)
        attn_res=attn_res.unsqueeze(1).repeat(1,16,1)
        if Update == "MPUGAT":
            e = e.mean(dim=-1, keepdim=False)
            from_i = torch.einsum('...it,...tk->...ik', e, matrix)
            from_j = torch.einsum('...jt,...tk->...jk', e, matrix)
            e = torch.einsum('...ik,...jk->...ij', from_i, from_j)
            e = (e+e.transpose(1,2))/2

        if Update == "C_ij":
            e = matrix
        if Update == "MPGAT":
            e = matrix
        if Update == "deep_learning":
            e = None        
        return attn_res ,e,check

class Deeplearning(nn.Module):
    def __init__(self, num_feature, in_len, out_len, layers, dropout, hidden_unit):
        super(Deeplearning, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Tanh() 
        self.out_len = out_len
        # LSTM setup
        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size=num_feature if i == 0 else hidden_unit, hidden_size=hidden_unit, batch_first=True)
            for i in range(layers)]) 
        self.hidden_unit = hidden_unit
    def forward(self, x, gat=True):
        batch_size, input_len, num_cities, num_features = x.shape
        
        x = x.permute(0, 2, 1, 3)  #  batch x city x input_len x feature 
        x = x.contiguous().view(batch_size * num_cities, input_len, num_features)  # Merge batch and city

        for lstm in self.lstm_layers:
            
            x, _ = lstm(x) # batch x city , seq_len , feature -> batch x city , seq, hidden_unit
            x=self.act(x)
 
        
        x = x.view(batch_size, num_cities, input_len, self.hidden_unit)
        num_steps_to_use = 1  # 마지막 시퀀스만 사용
        x = x[:, :, -num_steps_to_use:, :]  # (batch_size, num_cities, num_steps_to_use, hidden_unit)
        output = x.view(batch_size, num_cities, -1)  # (batch_size, num_cities, num_steps_to_use * hidden_unit)        #lstm 의  마지막 타임스텝 출력만 사용 #batch , city , hidden_unit
        return output

class lstm_GAT(nn.Module):
    def __init__(self, num_feature, in_len, out_len, layers, dropout, hidden_unit, n_head):
        super(lstm_GAT, self).__init__()
        self.n_heads = n_head
        self.dropout = dropout
        self.layers = layers
        self.lstm = Deeplearning(num_feature, in_len, hidden_unit, layers, dropout, hidden_unit)
        self.GAT = GAT(in_features =1*hidden_unit, out_features =out_len*self.n_heads, n_heads=self.n_heads, dropout=dropout,is_concat=False)
        self.act = nn.Tanh()
        self.mlp = nn.Linear(1*hidden_unit * 16, 16 * out_len*4)
        self.out_len = out_len
        

    def forward(self, x, matrix,N,Update):
        
        if Update == "deep_learning":
            x = self.lstm(x,True) #batch , city , hidden_unit 출력
            x = x.view(-1, x.size(1) * x.size(2))
            x = self.mlp(x)
            x = x.view(-1, 16, 4,self.out_len)
            h,check = None,None
        else:
            x = self.lstm(x,True)
            #batch_size, input_len, num_cities, num_features = x.shape
            x,h,check = self.GAT(x,matrix,N,Update)
            x = F.sigmoid(x)
        return x,h,check  

class Model(nn.Module):
    def __init__(self, num_feature, in_len, out_len, layers, dropout, hidden_unit, model,n_head):
        super(Model, self).__init__()
        self.lstm_GAT = lstm_GAT(num_feature, in_len, out_len, layers, dropout, hidden_unit,n_head)
        self.model = model  
        self.out_len = out_len
    def forward(self, x_node, x_SIR, matrix):
        #x_sir:bxout_lenxcityxfeature
        outputs_state = []
        outputs_beta = []
        S, E, I, R = x_SIR[:,-1,:, 0], x_SIR[:,-1,:, 1], x_SIR[:,-1,:, 2], x_SIR[:,-1,:, 3]
        N = S + E + I + R  
        if self.model == "MPUGAT":
            pred_beta,h ,check = self.lstm_GAT(x_node, matrix ,N,self.model) #x_node : batch x in_len x city  x featrue / matrix : city x city 
            h = h/(N.unsqueeze(1).repeat(1,16,1))
            for i in range(self.out_len):
                beta = pred_beta[:,:,0]
                Infected = beta * torch.bmm(I.unsqueeze(1), h).squeeze(1)*(S/N)
                S_t = S - Infected
                E_t = E + Infected - torch.tensor(1/6.5)* E
                I_t = I + torch.tensor(1/6.5)* E - torch.tensor(1/4)* I
                R_t = R + torch.tensor(1/4)* I
                current_state = torch.stack([S_t, E_t, I_t, R_t], dim=-1)  
                outputs_state.append(current_state)   
                outputs_beta.append(beta)
                S, E, I, R = S_t, E_t, I_t, R_t
            
        if self.model == "C_ij":
            matrix = (matrix+matrix.transpose(1,2))/2
            pred_beta,h ,check = self.lstm_GAT(x_node, matrix,N,self.model) #x_node : batch x in_len x city  x featrue / matrix : city x city 
            h = h/(N.unsqueeze(1).repeat(1,16,1))
            for i in range(self.out_len):
                beta = pred_beta[:,:,0]
                Infected = beta * torch.bmm(I.unsqueeze(1), h).squeeze(1)*(S/N)
                S_t = S - Infected
                E_t = E + Infected - torch.tensor(1/6.5)* E
                I_t = I + torch.tensor(1/6.5)* E - torch.tensor(1/4)* I
                R_t = R + torch.tensor(1/4)* I
                current_state = torch.stack([S_t, E_t, I_t, R_t], dim=-1)  
                outputs_state.append(current_state)   
                outputs_beta.append(beta)
                S, E, I, R = S_t, E_t, I_t, R_t

        if self.model == "MPGAT":
            pred_beta,h ,check = self.lstm_GAT(x_node, matrix,N,self.model) #x_node : batch x in_len x city  x featrue / matrix : city x city 
            h = np.loadtxt(f"./seed/contact_mean_{self.out_len}.csv", delimiter=",")  # CSV 파일 읽기
            h = torch.from_numpy(h).float().to(device).unsqueeze(0).repeat(pred_beta.shape[0],1,1)
            for i in range(self.out_len):
                beta = pred_beta[:,:,0]
                Infected = beta * torch.bmm(I.unsqueeze(1), h).squeeze(1)*(S/N)
                S_t = S - Infected
                E_t = E + Infected - torch.tensor(1/6.5)* E
                I_t = I + torch.tensor(1/6.5)* E - torch.tensor(1/4)* I
                R_t = R + torch.tensor(1/4)* I
                current_state = torch.stack([S_t, E_t, I_t, R_t], dim=-1)  
                outputs_state.append(current_state)                
                outputs_beta.append(beta)
                S, E, I, R = S_t, E_t, I_t, R_t

        elif self.model == "deep_learning":
            pred_E,_,_ = self.lstm_GAT(x_node, matrix,N,self.model)
            for i in range(self.out_len):
                S_t = pred_E[..., 0,i]
                E_t = pred_E[..., 1,i]
                I_t = pred_E[..., 2,i]
                R_t = pred_E[..., 3,i]
                outputs_beta.append(torch.zeros_like(E_t))  
                current_state = torch.stack([S_t, E_t, I_t, R_t], dim=-1)  
                outputs_state.append(current_state)                
                h = matrix.squeeze(-1)
                
        elif self.model == "compartment":
            #combined_normalized = matrix / matrix.sum(dim=1, keepdim=True) + matrix.t() / matrix.sum(dim=0, keepdim=True)
            for i in range(self.out_len):
                S_t = x_SIR[:,-1,:, 0]
                E_t = x_SIR[:,-1,:, 1]
                I_t = x_SIR[:,-1,:, 2]
                R_t = x_SIR[:,-1,:, 3]
                outputs_beta.append(torch.zeros_like(I_t)) 
                current_state = torch.stack([S_t, E_t, I_t, R_t], dim=-1)  
                outputs_state.append(current_state)                
                h = matrix.squeeze(-1)
                check=h
        outputs_state = torch.stack(outputs_state, dim=1)
        outputs_beta = torch.stack(outputs_beta, dim=1)
        return outputs_beta, outputs_state, h


