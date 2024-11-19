
import numpy as np
import matplotlib.pyplot as plt
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

cities = ['Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju', 'Daejeon',
          'Ulsan', 'Gyeonggi', 'Gangwon', 'Chungbuk', 'Chungnam', 'Jeonbuk',
          'Jeonnam', 'Gyeongbuk', 'Gyeongnam', 'Jeju']
colors = {'MPUGAT': 'red', 'MPGAT': 'blue', 'deep_learning': 'green', 'compartment': "gray"}  # 모델별 색상 정의

def calculate_errors(pred, act):
    mse = np.mean(np.square(act - pred), axis=0)
    mape =np.mean(np.abs((act - pred) / (act+1e-10)) * 100, axis=0) * 100
    rmse = np.sqrt(mse)
    return mse, mape, rmse


def plot_errors(model_errors, pred_len):
    x = np.arange(len(cities))
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - bar_width/2, np.mean(model_errors["MPUGAT_train"]['rmse'], axis=0), bar_width, label='Dynamic contact pattern', color=colors['MPUGAT'])
    rects2 = ax.bar(x + bar_width/2, np.mean(model_errors["MPGAT_train"]['rmse'], axis=0), bar_width, label='Sttatic contact pattern', color=colors['MPGAT'])

    ax.set_xlabel('Cities', fontsize=15)
    ax.set_ylabel('MSE', fontsize=15)
    ax.set_title('Fitting Accuracy', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=15)

    plt.tight_layout()
    plt.savefig("Fitting_Accuracy_Train.png")
    plt.show()

    x = np.arange(len(cities))
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - bar_width/2, np.mean(model_errors["MPUGAT_validate"]['rmse'], axis=0), bar_width, label='Dynamic contact pattern', color=colors['MPUGAT'])
    rects2 = ax.bar(x + bar_width/2, np.mean(model_errors["MPGAT_validate"]['rmse'], axis=0), bar_width, label='Sttatic contact pattern', color=colors['MPGAT'])

    ax.set_xlabel('Cities', fontsize=15)
    ax.set_ylabel('MSE', fontsize=15)
    ax.set_title('Fitting Accuracy', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=45, ha='right', fontsize=12)
    ax.legend(fontsize=15)

    plt.tight_layout()
    plt.savefig("Fitting_Accuracy_Validate.png")
    plt.show()


    '''
    #Prediction 일때 
    days = np.arange(1, pred_len + 1)
    fig, ax = plt.subplots(4, 4, figsize=(20, 15))
    fig.suptitle(f'RMSE Error Metrics over {pred_len} Days Prediction', fontsize=16)

    bar_width = 0.2
    index = np.arange(len(days))
    for w,model in enumerate(model_errors) : 
        for i  in range(16):
            row = i // 4
            col = i % 4
            ax[row, col].bar(index+ w * bar_width, model_errors[model]['rmse'][:,i], width=bar_width, label=f'MAPE {model}', color=colors[model])
            ax[row, col].set_title(f'{cities[i]}', fontsize=12)
            ax[row, col].set_xticks(index + bar_width / 2)
            ax[row, col].set_xticklabels([f'{j}day' for j in days])  # Custom x-axis labels
            ax[row, col].legend()
            ax[row, col].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"Error_Metrics_{pred_len}_days.png")
    plt.show()
    '''

def plot():
    base_path = './seed/'
    colors = {'MPUGAT': 'red','MPGAT': 'blue', 'deep_learning': 'green','compartment':"gray"}  # 모델별 색상 정의
    
    for pred_len in [1]: # , 21

        model_errors = {}

        for model in ["MPUGAT","MPGAT"]: # "compartment","deep_learning","MPGAT", 
            plt.figure(figsize=(10, 6))  # 그래프 크기 조정
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))#, dpi=300
            fig.suptitle("beta(t)", size=18)
            filename = f'preds_acts_{model}_{pred_len}.npy'
            full_path = base_path + filename

            start = 0 

            for kind in ["train", "validate", "test"]: #
                data = np.load(full_path, allow_pickle=True).item()

                pred = np.array(data[kind][:, :, 0, 0, 2]) #seed, data_num, pred_len, i, 2(beta)
                mean_preds = np.mean(pred, axis=0)
                std_preds = np.std(pred, axis=0)
                end_date = datetime.date(2021, 11, 23)
                data_length = len(mean_preds)
                if kind == "train":
                    data_length = len(mean_preds) + len(data["test"][0, :, 0, 0, 2]) + len(data["validate"][0, :, 0, 0, 2])
                elif kind == "validate":
                    data_length = len(mean_preds) + len(data["test"][0, :, 0, 0, 2])

                start_date = end_date - datetime.timedelta(days=data_length - 1)
    
                date_range = [start_date + datetime.timedelta(days=x) for x in range(len(mean_preds))]

                ax.plot(date_range, mean_preds, label=f'Prediction {model}', color=colors[model])
                ax.fill_between(date_range, mean_preds - 1.64 * std_preds, mean_preds + 1.64 * std_preds, color=colors[model], alpha=0.2)
                start += pred.shape[1]

            plt.legend(loc='upper left', bbox_to_anchor=(2, 1))  # 범례를 그래프 외부 상단 왼쪽에 배치
            plt.xticks(rotation=45)
            plt.savefig(f"Beta_Result_{pred_len}-{model}.png")
            plt.show()

        plt.figure(figsize=(10, 6))  # 그래프 크기 조정
        fig, ax = plt.subplots(4, 4, figsize=(20, 15))#, dpi=300
        fig.suptitle("I(t)", size=30)        
        
        for model in ["MPUGAT","MPGAT"]:#"deep_learning","MPGAT","deep_learning","compartment", "MPGAT" 
            filename = f'preds_acts_{model}_{pred_len}.npy'
            full_path = base_path + filename
            data = np.load(full_path, allow_pickle=True).item()
            for kind in ["train", "validate"]: 
                errors = {'mse': np.zeros((16,pred_len)), 'mape':  np.zeros((16,pred_len)), 'rmse':  np.zeros((16,pred_len))}
                for i in range(16):
                    pred = np.array(data[kind][:, :, -1, i, 0]) #seed, data_num, pred_len, i, 0 (I)
                    act = np.array(data[kind][0, :, -1, i, 1]) #seed, data_num, pred_len, i, 1 (act)
                    mean_preds = np.mean(pred, axis=0)
                    std_preds = np.std(pred, axis=0)
                    
                    data_length = len(mean_preds) 
                    end_date = datetime.date(2021, 11, 23)
                    if kind == "train":
                        data_length = len(mean_preds) + len(data["test"][0, :, -1, i, 0]) + len(data["validate"][0, :, -1, i, 0])
                    elif kind == "validate":
                        data_length = len(mean_preds) + len(data["test"][0, :, -1, i, 0])
                    
                    start_date = end_date - datetime.timedelta(days=data_length - 1)
        
                    date_range = [start_date + datetime.timedelta(days=x) for x in range(len(mean_preds))]
                    ax[i // 4, i % 4].set_title(cities[i], fontsize=20)
                    ax[i // 4, i % 4].plot(date_range, act, label='Real', color="black")
                    ax[i // 4, i % 4].plot(date_range, mean_preds, label=f'Fitting', color=colors[model])
                    ax[i // 4, i % 4].fill_between(date_range, mean_preds - 1.64 * std_preds, mean_preds + 1.64 * std_preds, color=colors[model], alpha=0.2)
                    ax[i // 4, i % 4].tick_params(axis='x', rotation=45)
                    mse, mape, rmse = calculate_errors(np.mean(np.array(data[kind][:, :, :, :, 0]),axis=0), np.array(data[kind][0, :, :, :, 1]))

                    errors['mse'] = mse
                    errors['mape']= mape
                    errors['rmse']= rmse
                    model_errors["{}_{}".format(model,kind)] = errors
        lines_labels = [ax[0, 0].get_legend_handles_labels()]  # 첫 번째 서브플롯에서 범례 정보 가져오기
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels, loc='upper center', fontsize=15)

        plt.xlabel('Time Steps', fontsize=20)
        plt.ylabel('I(t)', fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 상단에 여유 공간 확보
        plt.savefig("predictions_plot.png")
        plt.show()
        plot_errors(model_errors, pred_len)


plot()