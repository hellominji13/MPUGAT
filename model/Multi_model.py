'''
Part of code is adapted from https://github.com/nnzhan/Graph-WaveNet
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

#GAT https://nn.labml.ai/graphs/gat/index.html
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GAT(nn.Module):
    def __init__(self, in_features, out_features, n_heads, dropout,is_concat=False):
        super().__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads
        if is_concat:
            self.n_hidden = in_features
        else:
            assert in_features % n_heads == 0
            self.n_hidden = in_features // n_heads
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=True)
        self.linear2 = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=True)
        self.act =nn.LeakyReLU(0.2)
        self.softmax_1 = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)
        self.ouput_layer_mlp = nn.Linear(in_features*16,out_features,bias=True)
        self.check = 0
        self.w = nn.Parameter(torch.randn(16, 16,  self.n_heads).to(device))
        self.a = nn.Parameter(torch.tensor(5, dtype=torch.float32).to(device))


    def forward(self, x,matrix,N,Update):
        n_nodes = x.shape[1]  # number of nodes
        batch_size = x.shape[0]
        
        # Linear transformation

        wg = self.linear(x).view(batch_size, n_nodes, self.n_heads, self.n_hidden)
        #g =  self.linear2(x).view(batch_size, n_nodes, self.n_heads, self.n_hidden)
        g = x.view(batch_size, n_nodes, self.n_heads, self.n_hidden) 
        #g_repeat = g.repeat(1,n_nodes, 1, 1)
        #g_repeat_interleave = g.repeat_interleave(n_nodes, dim=1)
        g_reshaped = g.view(batch_size, self.n_heads * n_nodes, self.n_hidden)
        g_reshaped = F.normalize(g_reshaped, p=2, dim=-1) 
        g_transposed= wg.view(batch_size, self.n_heads * n_nodes, self.n_hidden)
        g_transposed = F.normalize(g_transposed, p=2, dim=-1) 
        e = torch.bmm(g_reshaped, g_transposed.transpose(1,2)) 
        e = e.view(batch_size, n_nodes, n_nodes,  self.n_heads)
        
        eye = torch.eye(n_nodes).unsqueeze(0).unsqueeze(-1).to(device)  # Shape: (1, n_nodes, n_nodes, 1)
        e = e*(1+ eye*self.a)  #e*eye*self.a
        a = e 
        e=self.softmax_1(e)
        matrix = matrix.unsqueeze(-1)
        check = matrix
        if Update == "MPUGAT":
            
            from_i = torch.einsum('...itp,...tkp->...ikp', e, matrix)
            from_j = torch.einsum('...jkp,...kkp->...jkp', e, matrix)
            e = torch.einsum('...ikp,...jkp->...ijp', from_i, from_j)
            e = (e+e.transpose(1,2))/2
            attn_res = torch.tanh(torch.einsum('bjhf,bjih->bihf', g, a))
            attn_res = self.ouput_layer_mlp(attn_res.reshape(batch_size,-1))
            attn_res = attn_res.reshape(batch_size, -1).unsqueeze(1).repeat(1,16,1)
        if Update == "MPGAT":
            e = matrix
            e = (e+e.transpose(1,2))/2
            attn_res = torch.tanh(torch.einsum('bjhf,bjih->bihf', g, a))
            attn_res = self.ouput_layer_mlp(attn_res.reshape(batch_size,-1))
            attn_res = attn_res.reshape(batch_size, -1).unsqueeze(1).repeat(1,16,1)
        if Update == "deep_learning":
            e = None
            attn_res = torch.einsum('bjhf,bjih->bihf',  g, a)
            attn_res = self.act(attn_res)
            attn_res = self.ouput_layer_mlp(attn_res.reshape(batch_size,-1)).reshape(batch_size,16,-1)

        
        #cde = matrix
        #e =  torch.bmm(torch.bmm(e.squeeze(-1),matrix.squeeze(-1)),(e.transpose(1,2).squeeze(-1))*(N.unsqueeze(1).repeat(1,16,1)))
        #e = (e+e.transpose(1,2))#.unsqueeze(-1)
        
        
        #attn_res = torch.einsum('bijh,bjhf->bihf',  e, attn_res)
        #attn_res = F.tanh(attn_res)
        
        return attn_res ,e,check
        #(batch_size, self.n_heads * self.n_hidden)
class Deeplearning(nn.Module):
    def __init__(self, num_feature, in_len, out_len, layers, dropout, hidden_unit):
        super(Deeplearning, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Tanh() #Tanh negative_slope=0.2
        self.out_len = out_len
        # LSTM setup
        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size=num_feature if i == 0 else hidden_unit, hidden_size=hidden_unit, batch_first=True)
            for i in range(layers)]) 
        self.hidden_unit = hidden_unit
    def forward(self, x, gat=True):
        batch_size, input_len, num_cities, num_features = x.shape
        
        x = x.permute(0, 2, 1, 3)  #  batch x city x input_len x feature 
        #x = x.unsqueeze(-1).repeat(1,1,1,1,input_len)
        x = x.reshape(batch_size * num_cities, input_len, num_features)  # Merge batch and city

        for lstm in self.lstm_layers:
            
            x, _ = lstm(x) # batch x city , seq_len , feature -> batch x city , seq, hidden_unit
            x=self.act(x)
 
        x = x.reshape(batch_size , num_cities, input_len, self.hidden_unit)
        output = x[:,:, -1, :].reshape(batch_size,num_cities,-1)
        #lstm 의  마지막 타임스텝 출력만 사용 #batch , city , hidden_unit
        return output

class lstm_GAT(nn.Module):
    def __init__(self, num_feature, in_len, out_len, layers, dropout, hidden_unit):
        super(lstm_GAT, self).__init__()
        self.n_heads = 1
        self.dropout = dropout
        self.layers = layers
        self.lstm = Deeplearning(num_feature, in_len, hidden_unit, layers, dropout, hidden_unit)
        self.GAT = GAT(in_features =hidden_unit, out_features =out_len*self.n_heads, n_heads=self.n_heads, dropout=dropout,is_concat=False)
        self.act1 = nn.Tanh()
        self.act=nn.LeakyReLU(0.2)
        self.mlp = nn.Linear(hidden_unit * 16, 16 * out_len)
        self.out_len = out_len
        

    def forward(self, x, matrix,N,Update):
        
        if Update == "deep_learning":
            x = self.lstm(x,True) #batch , city , hidden_unit 출력
            x = x.view(-1, x.size(1) * x.size(2))
            x = self.mlp(x)
            x = x.view(-1, 16, self.out_len)
            h,check = None,None
        else:
            x = self.lstm(x,True)
            #batch_size, input_len, num_cities, num_features = x.shape
            #x= x.reshape(batch_size,num_cities,-1)
            x,h,check = self.GAT(x,matrix,N,Update)
            h = h.squeeze(-1)
        x = F.sigmoid(F.tanh(x)*4)
        return x,h,check  

class Model(nn.Module):
    def __init__(self, num_feature, in_len, out_len, layers, dropout, hidden_unit, model):
        super(Model, self).__init__()
        self.deep_learning = Deeplearning(num_feature, in_len, out_len, layers, dropout, hidden_unit)
        self.lstm_GAT = lstm_GAT(num_feature, in_len, out_len, layers, dropout, hidden_unit)
        #self.GAT_lstm = GAT_lstm(num_feature, in_len, out_len, layers, dropout, hidden_unit)
        self.model = model  
        self.out_len = out_len
    def forward(self, x_node, x_known, x_SIR, matrix):
        #x_sir:bxout_lenxcityxfeature
        outputs_E = []
        outputs_beta = []
        S, E, I, R = x_SIR[:,-1,:, 0], x_SIR[:,-1,:, 1], x_SIR[:,-1,:, 2], x_SIR[:,-1,:, 3]
        N = S + E + I + R  
        if self.model == "MPUGAT":
            pred_beta,h ,check = self.lstm_GAT(x_node, matrix,N,self.model) #x_node : batch x in_len x city  x featrue / matrix : city x city 
            h = h/(N.unsqueeze(2).repeat(1,1,16))
            scale_i = h.max() / 10
            h = h/scale_i
            pred_beta = pred_beta.mean(1).unsqueeze(1).repeat(1,16,1)
            #combined_normalized = h / h.sum(dim=2, keepdim=True) + h.transpose(1,2) / h.sum(dim=1, keepdim=True)
            for i in range(self.out_len):
                beta = pred_beta[:,:,i]
                Infected = beta * torch.bmm(h,I.unsqueeze(-1)).squeeze(-1)*(S/N)
                S_t = S - Infected
                E_t = E + Infected - torch.tensor(1/5.2)* E
                I_t = I + torch.tensor(1/5.2)* E - x_known[:, i, 3].unsqueeze(1)* I
                R_t = R + x_known[:, i, 3].unsqueeze(1)* I
                outputs_E.append(E_t)
                outputs_beta.append(beta)
                S, E, I, R = S_t, E_t, I_t, R_t
            


        if self.model == "MPGAT":
            pred_beta,h ,check = self.lstm_GAT(x_node, matrix,N,self.model) #x_node : batch x in_len x city  x featrue / matrix : city x city 
            pred_beta = pred_beta.mean(1).unsqueeze(1).repeat(1,16,1)
            h = matrix/(N.unsqueeze(-1).repeat(1,1,16))
            scale_i = h.max() / 10
            h = h/scale_i
            #combined_normalized = h / h.sum(dim=2, keepdim=True) + h.transpose(1,2) / h.sum(dim=1, keepdim=True)
            for i in range(self.out_len):
                beta = pred_beta[:,:,i]
                Infected = beta * torch.bmm(h ,I.unsqueeze(-1)).squeeze(-1)*(S/N)
                S_t = S - Infected
                E_t = E + Infected - torch.tensor(1/5.2)* E
                I_t = I + torch.tensor(1/5.2)* E - x_known[:, i, 3].unsqueeze(1)* I
                R_t = R + x_known[:, i, 3].unsqueeze(1)* I
                outputs_E.append(E_t)
                outputs_beta.append(beta)
                S, E, I, R = S_t, E_t, I_t, R_t

        elif self.model == "deep_learning":
            pred_I,_,_ = self.lstm_GAT(x_node, matrix,N,self.model)
            for i in range(self.out_len):
                outputs_beta.append(torch.zeros_like(pred_I[..., i]))  
                E_t = pred_I[..., i]
                outputs_E.append(N-S_t-I_t-R_t)
                h = matrix.squeeze(-1)
                
        elif self.model == "compartment":
            #combined_normalized = matrix / matrix.sum(dim=1, keepdim=True) + matrix.t() / matrix.sum(dim=0, keepdim=True)
            for i in range(self.out_len):
                I_t = x_SIR[:,-1,:, 2]
                outputs_beta.append(torch.zeros_like(I_t)) 
                outputs_E.append(I_t)
                h = matrix.squeeze(-1)

        outputs_E = torch.stack(outputs_E, dim=1)
        outputs_beta = torch.stack(outputs_beta, dim=1)
        return outputs_beta, outputs_E,h



class GAT_lstm(nn.Module):
    def __init__(self, num_feature, in_len, out_len, layers, dropout, hidden_unit):
        super(GAT_lstm, self).__init__()
        self.dropout = dropout
        self.layers = layers
        self.lstm = Deeplearning(num_feature, in_len, out_len, layers, dropout, hidden_unit)
        self.GAT = GAT(in_features = in_len*num_feature, out_features =in_len*num_feature, n_heads=1, dropout=dropout,is_concat=False)
        self.act = nn.Sigmoid()
        self.linear = nn.Linear(num_feature, 1, bias=False)
    def forward(self, x, matrix):
        batch_size, input_len, num_cities, num_features = x.shape
        x = x.permute(0, 2, 1, 3) #  batch x city x input_len x feature
        x = self.linear(x.reshape(-1,num_features)).reshape(batch_size,num_cities,input_len)
        x,h = self.GAT(x.reshape(batch_size,num_cities,-1),matrix) 
        x = self.lstm(x.reshape(batch_size,num_cities,input_len,num_features),False)    
        x = self.act(x*6)
        h = h.squeeze(-1)
        return x,h