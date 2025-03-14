#%%

import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib.patches import Rectangle


cities = ['Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju', 'Daejeon',
          'Ulsan', 'Gyeonggi', 'Gangwon', 'Chungbuk', 'Chungnam', 'Jeonbuk',
          'Jeonnam', 'Gyeongbuk', 'Gyeongnam', 'Jeju']

colors = {'MPUGAT': 'red', 'MPGAT': 'orange', 'C_ij': 'blue', 'deep_learning': 'green', 'compartment': "gray"}
#%%
# MPUGAT의 Train + Val + Test fitting 
def plot_predictions(data, model):

    fig, ax = plt.subplots(4, 4, figsize=(14, 8))
    fig.suptitle("MPUGAT model fit for I(t)", size=30)        

    for i in range(16):
        for kind in ["train","validate","test"]:
            pred = np.array(data[kind][:, :, -1, i,2, 0])
            act = np.array(data[kind][0, :, -1, i,2, 1])
            mean_preds = np.mean(pred, axis=0)
            std_preds = np.std(pred, axis=0)/np.sqrt(30)

            data_length = len(mean_preds) 
            end_date = datetime.date(2021, 11, 23)
            if kind == "train":
                data_length = len(mean_preds) + len(data["test"][0, :, -1, i, 0,0]) + len(data["validate"][0, :, -1, i, 0,0])
            elif kind == "validate":
                data_length =  len(data["validate"][0, :, -1, i,0, 0]) + len(data["test"][0, :, -1, i,0, 0])
            elif kind == "test":
                data_length =  len(data["test"][0, :, -1, i,0, 0])

            start_date = end_date - datetime.timedelta(days=data_length - 1)
            date_range = [start_date + datetime.timedelta(days=x) for x in range(len(data[kind][0, :, -1, i, 0,0]))]

            ax[i // 4, i % 4].set_title(cities[i], fontsize=15)
            ax[i // 4, i % 4].plot(date_range, act, label='Real', color="black")
            ax[i // 4, i % 4].plot(date_range, mean_preds, label=f'Fitting MPUGAT', color=colors[model])
            ax[i // 4, i % 4].fill_between(date_range, mean_preds - 1.67 * std_preds, mean_preds + 1.67 * std_preds, color=colors[model], alpha=0.2)
            ax[i // 4, i % 4].tick_params(axis='x', rotation=45)
            ax[i // 4, i % 4].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

            # 구간 경계선 추가
            if kind == "train":
                train_end = start_date + datetime.timedelta(days=len(data["train"][0, :, -1, i, 0,0])-1)
                ax[i // 4, i % 4].axvline(x=train_end, color='black', linestyle='--', alpha=0.5)
            elif kind == "validate":
                val_end = start_date + datetime.timedelta(days=len(data["validate"][0, :, -1, i,0, 0])-1)
                ax[i // 4, i % 4].axvline(x=val_end, color='black', linestyle='--', alpha=0.5)

    # y-axis label (I(t)) : 전체 그림의 y축 중앙에 표시
    fig.text(-0.04, 0.5, 'I(t)', va='center', rotation='vertical', fontsize=20)

    # x-axis label (Time Steps) : 전체 그림의 x축 중앙에 표시
    fig.text(0.5, -0.04, 'Time steps', ha='center', fontsize=20)

    # legend 수정 (Real, Fitting MPUGAT만 표시)
    lines_labels = [ax[0, 0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend_labels = ['Real', 'Fitting MPUGAT']
    fig.legend(lines[:2], legend_labels, loc='upper right', fontsize=20)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("./fig/predictions_plot.pdf", dpi=300)
    plt.show()
#%%
def plot_predictions(data, model):
    fig, ax = plt.subplots(4, 4, figsize=(16, 10), sharex='col')  # sharex 파라미터 추가
    fig.suptitle("MPUGAT model fit for I(t)", size=30)        

    # 맨 아래 행을 제외한 모든 행의 x축 레이블 숨기기
    for i in range(12):  # 마지막 행을 제외한 모든 행
        ax[i // 4, i % 4].xaxis.set_tick_params(labelbottom=False)

    for i in range(16):
        for kind in ["train","validate","test"]:
            pred = np.array(data[kind][:, :, -1, i, 2, 0])
            act = np.array(data[kind][0, :, -1, i, 2, 1])
            mean_preds = np.mean(pred, axis=0)
            std_preds = np.std(pred, axis=0)/np.sqrt(30)

            data_length = len(mean_preds) 
            end_date = datetime.date(2021, 11, 23)
            if kind == "train":
                data_length = len(mean_preds) + len(data["test"][0, :, -1, i, 0, 0]) + len(data["validate"][0, :, -1, i, 0, 0])
            elif kind == "validate":
                data_length = len(data["validate"][0, :, -1, i, 0, 0]) + len(data["test"][0, :, -1, i, 0, 0])
            elif kind == "test":
                data_length = len(data["test"][0, :, -1, i, 0, 0])

            start_date = end_date - datetime.timedelta(days=data_length - 1)
            date_range = [start_date + datetime.timedelta(days=x) for x in range(len(data[kind][0, :, -1, i, 0, 0]))]

            ax[i // 4, i % 4].set_title(cities[i], fontsize=15)
            ax[i // 4, i % 4].plot(date_range, act, label='Real', color="black")
            ax[i // 4, i % 4].plot(date_range, mean_preds, label=f'Fitting MPUGAT', color=colors[model])
            ax[i // 4, i % 4].fill_between(date_range, mean_preds - 1.67 * std_preds, mean_preds + 1.67 * std_preds, color=colors[model], alpha=0.2)
            
            # 맨 아래 행에만 x축 포맷 적용
            if i >= 12:  # 마지막 행만
                ax[i // 4, i % 4].tick_params(axis='x', rotation=45)
                ax[i // 4, i % 4].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

            # 구간 경계선 추가
            if kind == "train":
                train_end = start_date + datetime.timedelta(days=len(data["train"][0, :, -1, i, 0, 0])-1)
                ax[i // 4, i % 4].axvline(x=train_end, color='black', linestyle='--', alpha=0.5)
            elif kind == "validate":
                val_end = start_date + datetime.timedelta(days=len(data["validate"][0, :, -1, i, 0, 0])-1)
                ax[i // 4, i % 4].axvline(x=val_end, color='black', linestyle='--', alpha=0.5)

    # 각 열에 공통 x축 레이블 추가
    fig.text(0.5, -0.04, 'Time steps', ha='center', fontsize=20)

    # y축 레이블
    fig.text(0.01, 0.5, 'Number of Infected people', va='center', rotation='vertical', fontsize=20)

    # 범례 추가
    lines_labels = [ax[0, 0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend_labels = ['Real', 'Fitting MPUGAT']
    fig.legend(lines[:2], legend_labels, loc='upper right', fontsize=20)

    plt.tight_layout(rect=[0.02, 0.03, 1, 0.95])  # 여백 조정
    plt.subplots_adjust(hspace=0.3, wspace=0.2)   # 서브플롯 간 간격 축소
    plt.savefig("./fig/predictions_plot.pdf", dpi=300, bbox_inches='tight')
    plt.show()



#%%
def plot_predictions_test_sum(data):
   models = ['deep_learning', 'compartment', 'C_ij', "MPGAT"]
   colors = {'MPUGAT': 'red', 'MPGAT': 'red', 'C_ij': 'blue', 'deep_learning': 'green', 'compartment': "gray"}
   model_labels = {'MPUGAT': 'MPUGAT', 'MPGAT': 'MPGAT', 'C_ij': 'Simple hybrid model', 'deep_learning': 'Deep learning', 'compartment': "Compartment"}

   fig, ax = plt.subplots(figsize=(14, 5))
   fig.suptitle("Aggregate fitting results for test and validation I(t)", size=30)        

   for model in models:
       # 테스트 데이터 처리
       kind = "test"
       total_mean_preds_test = np.zeros(len(data[model][kind][0, :, -1, 0,0, 0]))
       total_std_preds_test = np.zeros(len(data[model][kind][0, :, -1, 0, 0,0]))
       total_act_test = np.zeros(len(data[model][kind][0, :, -1, 0, 0, 1]))
       
       # 검증 데이터 처리
       total_mean_preds_val = np.zeros(len(data[model]["validate"][0, :, -1, 0,0, 0]))
       total_std_preds_val = np.zeros(len(data[model]["validate"][0, :, -1, 0,0, 0]))
       total_act_val = np.zeros(len(data[model]["validate"][0, :, -1, 0,0, 1]))
       
       for i in range(16):
           # 테스트 데이터 합산
           pred_test = np.array(data[model][kind][:, :, -1, i, 2,0])
           act_test = np.array(data[model][kind][0, :, -1, i, 2,1])
           mean_preds_test = np.mean(pred_test, axis=0)
           std_preds_test = np.std(pred_test, axis=0)/np.sqrt(30)
           
           total_mean_preds_test += mean_preds_test
           total_std_preds_test += std_preds_test
           total_act_test += act_test

           # 검증 데이터 합산
           pred_val = np.array(data[model]["validate"][:, :, -1, i,2, 0])
           act_val = np.array(data[model]["validate"][0, :, -1, i, 2,1])
           mean_preds_val = np.mean(pred_val, axis=0)
           std_preds_val = np.std(pred_val, axis=0)/np.sqrt(30)
           
           total_mean_preds_val += mean_preds_val
           total_std_preds_val += std_preds_val
           total_act_val += act_val

       # 테스트 기간 날짜 범위 계산
       data_length_test = len(total_mean_preds_test)
       end_date = datetime.date(2021, 11, 23)
       start_date = end_date - datetime.timedelta(days=data_length_test - 1)
       date_range_test = [start_date + datetime.timedelta(days=x) for x in range(data_length_test)]

       # 검증 기간 날짜 범위 계산
       validate_start_date = start_date - datetime.timedelta(days=(len(total_mean_preds_val)-1))
       date_range_val = [validate_start_date + datetime.timedelta(days=x) for x in range(len(total_mean_preds_val))]

       # 데이터 플로팅
       if model == models[0]:  # 실제 데이터는 한 번만 플로팅
           ax.scatter(date_range_test, total_act_test, label='Real', color="black", s=20)
           ax.scatter(date_range_val, total_act_val, color="black", s=20)
           # validation과 test 사이에 세로선 추가
           ax.axvline(x=start_date, color='black', linestyle='-', alpha=0.5, linewidth=2)
       
       # 테스트 데이터 플로팅
       ax.plot(date_range_test, total_mean_preds_test, label=f'{model_labels[model]}', color=colors[model])
       ax.fill_between(date_range_test, 
                      total_mean_preds_test - 1.67 * total_std_preds_test, 
                      total_mean_preds_test + 1.67 * total_std_preds_test, 
                      color=colors[model], alpha=0.2)

       # 검증 데이터 플로팅 (점선)
       ax.plot(date_range_val, total_mean_preds_val, linestyle='--', color=colors[model])
       ax.fill_between(date_range_val, 
                      total_mean_preds_val - 1.67 * total_std_preds_val, 
                      total_mean_preds_val + 1.67 * total_std_preds_val, 
                      color=colors[model], alpha=0.2)

   ax.tick_params(axis='x', rotation=45)
   ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
   
   # 범례 생성
   ax.legend(fontsize=20)
   
   plt.xlabel('Time steps', fontsize=15)
   plt.ylabel('Sum of I(t)', fontsize=15)
   plt.tight_layout()
   plt.savefig("./fig/predictions_plot_all_models_sum.pdf", dpi=300)
   plt.show()


#%%
def total_metrics(all_models_data):
    results = []
    for model_name, data in all_models_data.items():
        for set_name in ["train", "validate", "test"]:
            #(30, 480, 3, 16, 4, 4)
            pred_all = np.array(data[set_name][:, :, -1, :,:, 0])  # (samples, time, cities)
            act_all = np.array(data[set_name][0, :, -1, :,:, 1])   # (time, cities)
            
            # 각 sample별로 MSE, RMSE, MAE 계산
            metrics_per_sample = []
            for sample in range(pred_all.shape[0]):
                sample_pred = pred_all[sample]  # (time, cities)
                mse = np.mean((sample_pred - act_all) ** 2)
                rmse = np.sqrt(mse)
                mask = act_all > 1
                mape = np.mean(np.abs(sample_pred[mask] - act_all[mask]) / act_all[mask]) * 100
                mae = np.mean(np.abs(sample_pred - act_all))
                # MASE
                naive_errors = np.abs(act_all[3:] - act_all[:-3])
                mae_naive = np.mean(naive_errors)
                eps = 1e-10
                mase = mae / (mae_naive + eps)
                metrics_per_sample.append([mse, rmse, mae])
            
            metrics_per_sample = np.array(metrics_per_sample)  # (samples, 3)
            
            # 각 메트릭의 평균과 표준편차 계산
            mean_metrics = np.mean(metrics_per_sample, axis=0)
            std_metrics = np.std(metrics_per_sample, axis=0)
            
            result = f"{model_name} {set_name}:\n"
            result += f"MSE: {mean_metrics[0]:.4f} ± {std_metrics[0]:.4f}\n"
            result += f"RMSE: {mean_metrics[1]:.4f} ± {std_metrics[1]:.4f}\n"
            result += f"MAE: {mean_metrics[2]:.4f} ± {std_metrics[2]:.4f}\n"
            results.append(result)
    
    with open('./fig/total_metrics.txt', 'w') as f:
        f.write('\n'.join(results))
    
    print('\n'.join(results))

#%%

def parse_and_plot_metrics(file_path):
   # Parse data from file
   data = {}
   current_model = None
   current_mode = None
   
   with open(file_path, 'r') as f:
       for line in f:
           line = line.strip()
           if not line: continue
           
           if ':' in line and any(x in line.lower() for x in ['train', 'validate', 'test']):
               parts = line.split(':')
               model_mode = parts[0].lower()
               for mode in ['train', 'validate', 'test']:
                   if mode in model_mode:
                       current_model = model_mode.replace(mode, '').strip('_').strip()
                       current_mode = mode.capitalize()
                       
               if current_model not in data:
                   data[current_model] = {}
               if current_mode not in data[current_model]:
                   data[current_model][current_mode] = {}
                   
           elif current_model and current_mode and ':' in line:
               metric, values = line.split(':')
               metric = metric.strip().lower()
               mean, std = map(float, values.split('±'))
               std = 1.67*std / np.sqrt(30)
               data[current_model][current_mode][metric] = [mean, std]

   # Plot setup
   fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
   width = 0.2
   metrics = ['mse', 'rmse', 'mae']
   models = ['mpugat', 'mpgat', 'c_ij', 'compartment', 'deep_learning']
   colors = ['#69b3e7', '#4169e1', '#505050','#a9a9a9', '#d3d3d3']  # 회색 계열 수정
   labels = ['MPUGAT', 'MPGAT', 'Simple Hybrid Model', 'Compartment Model', 'Deep Learning']
   modes = ['Train', 'Validate', 'Test']
   # Plotting
   for ax, mode in zip([ax1, ax2, ax3], modes):
       for i, (metric, pos) in enumerate(zip(metrics, [0, 1, 2])):
           for j, (model, color) in enumerate(zip(models, colors)):
               mean = data[model][mode][metric][0]
               std = data[model][mode][metric][1]
               ax.bar(pos + j*width, mean, width, yerr=std, color=color, 
                     label=labels[j] if i == 0 else None, capsize=3)
               
       ax.set_xticks(np.arange(len(metrics)) + width*1.5)
       ax.set_xticklabels(metrics)
       ax.set_yscale('log')
       ax.set_title(mode)
       if mode == 'Train':
           ax.legend(bbox_to_anchor=(1, 1))
           
   plt.savefig("./fig/error_plot.pdf", dpi=300)        
   plt.tight_layout()
   plt.show()

#%%
def plot_contact_matrices(data_dict):
   # Calculate mean over samples and time
   beta_train = data_dict["C_ij"]["train"][:, :, 0, 0,0, 2]  
   beta_val = data_dict["C_ij"]["validate"][:, :, 0, 0, 0,2]
   beta_test = data_dict["C_ij"]["test"][:, :, 0, 0,0, 2]
   c_ij_beta_combined = np.concatenate([beta_train, beta_val, beta_test], axis=1)  # (30, 601)
   c_ij_beta_combined = np.repeat(np.repeat(c_ij_beta_combined[:, :, np.newaxis], 16, axis=2)[..., np.newaxis], 16, axis=3)

   beta_train = data_dict["MPUGAT"]["train"][:, :, 0, 0,0, 2]  
   beta_val = data_dict["MPUGAT"]["validate"][:, :, 0, 0,0, 2]
   beta_test = data_dict["MPUGAT"]["test"][:, :, 0, 0,0, 2]
   mpugat_beta_combined = np.concatenate([beta_train, beta_val, beta_test], axis=1)  # (30, 601)
   mpugat_beta_combined = np.repeat(np.repeat(mpugat_beta_combined[:, :, np.newaxis], 16, axis=2)[..., :, np.newaxis], 16, axis=3)

   mpugat_contact = np.mean(np.mean(data_dict["MPUGAT"]["contact"]*mpugat_beta_combined, axis=0), axis=0)
   c_ij_contact = np.mean(np.mean(data_dict["C_ij"]["contact"]*c_ij_beta_combined, axis=0), axis=0)
   
   # Create figure with two subplots with equal aspect ratio
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
   
   # Plot for Simple Hybrid Model (C_ij)
   sns.heatmap(c_ij_contact, ax=ax1, cmap='Blues', vmin=0, vmax=np.max(c_ij_contact),
               xticklabels=cities, yticklabels=cities, square=True)  # square=True 추가
   ax1.set_title('Simple Hybrid Model\n' + 
              r'$\bar{T}_{SHM} = \operatorname{Mean}_t \left( \frac{M_{ij}}{N_i} \times \bar{\beta}_{SHM}(t) \right)$',fontsize=18)
   ax1.tick_params(axis='x', rotation=75)
   ax1.tick_params(axis='y', rotation=0)
   
   # Plot for MPUGAT
   sns.heatmap(mpugat_contact, ax=ax2, cmap='Blues', vmin=0, vmax=np.max(mpugat_contact),
               xticklabels=cities, yticklabels=cities, square=True)  # square=True 추가
   ax2.set_title('MPUGAT\n' + r'$T_{MPUGAT}=\operatorname{Mean}_t \left(\frac{C_{MPUGAT,ij}(t)}{N_i}\times\beta_{MPUGAT}(t)\right)$',fontsize=18)
   ax2.tick_params(axis='x', rotation=75)
   ax2.tick_params(axis='y', rotation=0)
   plt.savefig("./fig/T_ij.pdf", dpi=300)        
   plt.tight_layout()
   plt.show()

#%%
def plot_beta_values(data_dict):
    # Create figure and axis
    plt.figure(figsize=(10, 5))
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    # Colors for different models
    colors = {'MPUGAT': 'red', 'MPGAT': 'orange', 'C_ij': 'blue', 'deep_learning': 'green', 'compartment': "gray"}
    
    # Define social distancing periods
    sd_periods = [
        {"start": "2020-03-22", "end": "2020-04-20", "strength": "Strong"},
        {"start": "2020-04-20", "end": "2020-05-06", "strength": "Medium"},
        {"start": "2020-05-06", "end": "2020-06-15", "strength": "Weak"},
        {"start": "2020-10-12", "end": "2020-11-23", "strength": "Weak"},
        {"start": "2020-11-23", "end": "2020-12-22", "strength": "Medium"},
        {"start": "2020-12-22", "end": "2021-02-14", "strength": "Strong"}
    ]
    
    # Plot for MPUGAT
    beta_train = data_dict["MPUGAT"]["train"][:, :, 0, 0,0, 2]  
    beta_val = data_dict["MPUGAT"]["validate"][:, :, 0, 0,0, 2]
    beta_test = data_dict["MPUGAT"]["test"][:, :, 0, 0,0, 2]
    pred_mpugat = np.concatenate([beta_train, beta_val, beta_test], axis=1)
    mean_preds_mpugat = np.mean(pred_mpugat, axis=0)
    std_preds_mpugat = np.std(pred_mpugat, axis=0)/np.sqrt(30)
    
    # Plot for C_ij
    beta_train_cij = data_dict["C_ij"]["train"][:, :, 0, 0,0, 2]  
    beta_val_cij = data_dict["C_ij"]["validate"][:, :, 0, 0,0, 2]
    beta_test_cij = data_dict["C_ij"]["test"][:, :, 0, 0,0, 2]
    pred_cij = np.concatenate([beta_train_cij, beta_val_cij, beta_test_cij], axis=1)
    mean_preds_cij = np.mean(pred_cij, axis=0)
    std_preds_cij = np.std(pred_cij, axis=0)/np.sqrt(30)
    
    # Generate date range
    end_date = datetime.datetime.strptime("2021-11-23", "%Y-%m-%d").date()
    data_length = len(mean_preds_mpugat)
    start_date = end_date - datetime.timedelta(days=data_length - 1)
    date_range = [start_date + datetime.timedelta(days=x) for x in range(len(mean_preds_mpugat))]
    
    # Convert dates to matplotlib format for plotting
    dates_num = mdates.date2num(date_range)
    
    plot_len = len(dates_num[:]) 
    # Plot mean and confidence interval for MPUGAT
    ax.plot(dates_num[:plot_len], mean_preds_mpugat[:plot_len], color=colors['MPUGAT'], label='MPUGAT', linewidth=2)
    ax.fill_between(dates_num[:plot_len], 
                    mean_preds_mpugat[:plot_len] - 1.67 * std_preds_mpugat[:plot_len],
                    mean_preds_mpugat[:plot_len] + 1.67 * std_preds_mpugat[:plot_len],
                    color=colors['MPUGAT'], alpha=0.2)
    ax.set_ylim(0.4, 0.6)

    
    # Plot mean and confidence interval for C_ij
    '''
    ax.plot(dates_num[:plot_len], mean_preds_cij[:plot_len], color=colors['C_ij'], label='C_ij', linewidth=2)
    ax.fill_between(dates_num[:plot_len], 
                    mean_preds_cij[:plot_len] - 1.64 * std_preds_cij[:plot_len],
                    mean_preds_cij[:plot_len] + 1.64 * std_preds_cij[:plot_len],
                    color=colors['C_ij'], alpha=0.2)
    '''
    # Add social distancing period highlights
    for period in sd_periods:
        start_date = datetime.datetime.strptime(period["start"], "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(period["end"], "%Y-%m-%d").date()
        
        color = "#FF9999" if period["strength"] == "Weak" else \
                "#FF3333" if period["strength"] == "Medium" else \
                "#CC0000"
        
        # Convert to matplotlib dates
        start_num = mdates.date2num(start_date)
        end_num = mdates.date2num(end_date)
        
        rect = Rectangle((start_num, ax.get_ylim()[0]),
                        end_num - start_num,
                        ax.get_ylim()[1] - ax.get_ylim()[0],
                        facecolor=color, alpha=0.3)
        ax.add_patch(rect)
    
    # Configure date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    # Customize plot
    ax.set_title(r'$\beta_{MPUGAT}$' + ' values over time \n'+ 'with social distance', fontsize=20)
    ax.set_xlabel('Time',fontsize=12)
    ax.set_ylabel('Beta values',fontsize=12)
    
    # Add legend
    ax.legend(loc='upper right')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("./fig/Beta_Result_with_SD.pdf", bbox_inches='tight', dpi=300)
    plt.show()

# Usage
plot_beta_values(data_dict)

#%%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import datetime
from matplotlib.patches import Rectangle

PATH = './14_Data(20-03-01_21-11-23).npy'
data = np.load(PATH, allow_pickle='TRUE').item()
SEIR_data = data["compart"][:, :, :]
N = np.sum(SEIR_data[0, :, :], axis=1) 
movement_facebook = np.mean((data["movement"][:, :, 2]*N), axis=1)[17:]

# MPUGAT data
beta_train_mpugat = data_dict["MPUGAT"]["train"][:, :, 0, 0, 0, 2]  
beta_val_mpugat = data_dict["MPUGAT"]["validate"][:, :, 0, 0, 0, 2]
beta_test_mpugat = data_dict["MPUGAT"]["test"][:, :, 0, 0, 0, 2]
pred_mpugat = np.concatenate([beta_train_mpugat, beta_val_mpugat, beta_test_mpugat], axis=1)
mean_preds_mpugat = np.mean(pred_mpugat, axis=0)
std_preds_mpugat = np.std(pred_mpugat, axis=0)/np.sqrt(30)

# MPGAT data
beta_train_mpgat = data_dict["MPGAT"]["train"][:, :, 0, 0, 0, 2]  
beta_val_mpgat = data_dict["MPGAT"]["validate"][:, :, 0, 0, 0, 2]
beta_test_mpgat = data_dict["MPGAT"]["test"][:, :, 0, 0, 0, 2]
pred_mpgat = np.concatenate([beta_train_mpgat, beta_val_mpgat, beta_test_mpgat], axis=1)
mean_preds_mpgat = np.mean(pred_mpgat, axis=0)
std_preds_mpgat = np.std(pred_mpgat, axis=0)/np.sqrt(30)

# SHM (C_ij) data
beta_train_shm = data_dict["C_ij"]["train"][:, :, 0, 0, 0, 2]  
beta_val_shm = data_dict["C_ij"]["validate"][:, :, 0, 0, 0, 2]
beta_test_shm = data_dict["C_ij"]["test"][:, :, 0, 0, 0, 2]
pred_shm = np.concatenate([beta_train_shm, beta_val_shm, beta_test_shm], axis=1)
mean_preds_shm = np.mean(pred_shm, axis=0)
std_preds_shm = np.std(pred_shm, axis=0)/np.sqrt(30)

movement_skt = pd.read_csv("모바일인구이동_원자료.csv")

# Extract activity data from data_dict
act_data_train = np.sum(np.array(data_dict["MPUGAT"]["train"][0, :, -1, :, 2, 1]), axis=1)
act_data_val = np.sum(np.array(data_dict["MPUGAT"]["validate"][0, :, -1, :, 2, 1]), axis=1)
act_data_test = np.sum(np.array(data_dict["MPUGAT"]["test"][0, :, -1, :, 2, 1]), axis=1)
act_data = np.concatenate([act_data_train, act_data_val, act_data_test], axis=0)

# Colors for different models
colors = {'MPUGAT': 'red', 'MPGAT': 'orange', 'C_ij': 'blue', 'deep_learning': 'green', 'compartment': "gray"}

# Define social distancing periods
sd_periods = [
    {"start": "2020-03-22", "end": "2020-04-20", "strength": "Strong"},
    {"start": "2020-04-20", "end": "2020-05-06", "strength": "Medium"},
    {"start": "2020-05-06", "end": "2020-06-15", "strength": "Weak"},
    {"start": "2020-10-12", "end": "2020-11-23", "strength": "Weak"},
    {"start": "2020-11-23", "end": "2020-12-22", "strength": "Medium"},
    {"start": "2020-12-22", "end": "2021-02-14", "strength": "Strong"}
]

# Create the figure with two subplots
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Generate date range
end_date = datetime.datetime.strptime("2021-11-23", "%Y-%m-%d").date()
data_length = len(mean_preds_mpugat)
start_date = end_date - datetime.timedelta(days=data_length - 1)
date_range = [start_date + datetime.timedelta(days=x) for x in range(len(mean_preds_mpugat))]

# Convert dates to matplotlib format for plotting
dates_num = mdates.date2num(date_range)

#------------------------
# TOP PLOT: Modified with separate y-axes
#------------------------
# 1. 오른쪽 메인 축 (SHM - 실제 데이터용이지만 눈금과 레이블은 숨김)
fig.subplots_adjust(right=0.75)  # 오른쪽 여백 확보
ax_top.yaxis.set_visible(False)  # y축 눈금과 레이블 모두 숨김
ax_top.set_ylim(0.65, 0.75)  # SHM 범위로 설정

# SHM 그래프
ax_top.plot(dates_num[:plot_len], mean_preds_shm[:plot_len], color="gray", linewidth=2, label=r'SHM $\beta$')
ax_top.fill_between(dates_num[:plot_len], 
                    mean_preds_shm[:plot_len] - 1.67 * std_preds_shm[:plot_len],
                    mean_preds_shm[:plot_len] + 1.67 * std_preds_shm[:plot_len],
                    color="gray", alpha=0.2)

# 2. 왼쪽 보조축 (MPUGAT, MPGAT)
ax_top_mp = ax_top.twinx()
ax_top_mp.yaxis.tick_left()
ax_top_mp.yaxis.set_label_position('left')
ax_top_mp.spines["right"].set_visible(False)
ax_top_mp.spines["left"].set_visible(True)
ax_top_mp.set_ylabel('MPUGAT, MPGAT beta values', color="red", fontsize=12)
ax_top_mp.tick_params(axis='y', labelcolor="red")
ax_top_mp.set_ylim(0.4, 0.6)

# MPUGAT 그래프
ax_top_mp.plot(dates_num[:plot_len], mean_preds_mpugat[:plot_len], color="red", linewidth=2, label=r'MPUGAT $\beta$')
ax_top_mp.fill_between(dates_num[:plot_len],
                       mean_preds_mpugat[:plot_len] - 1.67 * std_preds_mpugat[:plot_len],
                       mean_preds_mpugat[:plot_len] + 1.67 * std_preds_mpugat[:plot_len],
                       color="red", alpha=0.2)

# MPGAT 그래프
ax_top_mp.plot(dates_num[:plot_len], mean_preds_mpgat[:plot_len], color="orange", linewidth=2, label=r'MPGAT $\beta$')
ax_top_mp.fill_between(dates_num[:plot_len],
                       mean_preds_mpgat[:plot_len] - 1.67 * std_preds_mpgat[:plot_len],
                       mean_preds_mpgat[:plot_len] + 1.67 * std_preds_mpgat[:plot_len],
                       color="orange", alpha=0.2)

# 3. 왼쪽에 SHM 가짜 축 추가
ax_top_shm_fake = ax_top.twinx()
ax_top_shm_fake.spines["left"].set_visible(False)
ax_top_shm_fake.spines["right"].set_visible(True)
ax_top_shm_fake.spines["right"].set_position(('outward', 0))  # 그림에 바로 붙이기
ax_top_shm_fake.yaxis.tick_right()  # 오른쪽에 눈금 표시
ax_top_shm_fake.yaxis.set_label_position('right')  # 오른쪽에 레이블 위치
ax_top_shm_fake.set_ylabel('SHM beta value', color="blue", fontsize=12)
ax_top_shm_fake.tick_params(axis='y', labelcolor="blue")
ax_top_shm_fake.set_ylim(0.65, 0.75)  # 원하는 범위로 설정

# 4. I(t) 오른쪽 보조축 설정
ax_top_act = ax_top.twinx()
ax_top_act.spines["right"].set_position(("outward", 80))
ax_top_act.set_ylabel('Number of infected people', color='black', fontsize=12)
ax_top_act.tick_params(axis='y', labelcolor='black')
ax_top_act.plot(dates_num[:len(act_data)], act_data, 'k--', linewidth=2, label='I(t)')

# 눈금 설정 (간격 맞추기)
ax_top_mp.set_yticks([0.42, 0.46, 0.50, 0.54, 0.58])  # MPUGAT/MPGAT 간격

# 범례 추가
lines1, labels1 = ax_top.get_legend_handles_labels()
lines2, labels2 = ax_top_mp.get_legend_handles_labels()
lines3, labels3 = ax_top_act.get_legend_handles_labels()
# 가짜 축에서는 범례 정보를 가져오지 않음
ax_top.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='lower right', fontsize=12)
ax_top.set_title(r'Three Inference $\beta$ from MPUGAT, MPGAT, SHM with Infected Population I(t)', fontsize=20, pad=20)

#------------------------
# BOTTOM PLOT: Beta with Movement Trends (Unchanged)
#------------------------

# First axis: Beta
ax_bottom.plot(dates_num[:plot_len], mean_preds_mpugat[:plot_len], color="red", 
              label=r"MPUGAT $\beta$", linewidth=2)
ax_bottom.fill_between(dates_num[:plot_len], 
                    mean_preds_mpugat[:plot_len] - 1.67 * std_preds_mpugat[:plot_len],
                    mean_preds_mpugat[:plot_len] + 1.67 * std_preds_mpugat[:plot_len],
                    color="red", alpha=0.2)
ax_bottom.set_ylabel("Beta", color="red", fontsize=12)
ax_bottom.tick_params(axis='y', labelcolor="red")
ax_bottom.set_ylim(0.4, 0.6)

# Convert movement data dates
beta_dates = pd.to_datetime(date_range)
start_date_pd = pd.to_datetime(start_date)
end_date_pd = pd.to_datetime(end_date)

# Facebook movement data (daily data)
date_range_pd = pd.date_range(start=start_date_pd, end=end_date_pd, freq='D')
# Ensure movement_facebook length matches date_range_pd
movement_facebook_adjusted = movement_facebook
if len(movement_facebook) > len(date_range_pd):
    movement_facebook_adjusted = movement_facebook[:len(date_range_pd)]
else:
    # Pad with NaN values if needed
    pad_length = len(date_range_pd) - len(movement_facebook)
    if pad_length > 0:
        movement_facebook_adjusted = np.append(movement_facebook, [np.nan] * pad_length)

movement_facebook_df = pd.DataFrame({
    "Date": date_range_pd,
    "movement_facebook": movement_facebook_adjusted
})

# Second axis: Facebook movement
ax_fb = ax_bottom.twinx()
ax_fb.plot(movement_facebook_df["Date"], movement_facebook_df["movement_facebook"], 
          color="gray", label="Facebook Movement", linewidth=2, linestyle="dashed")
ax_fb.set_ylabel("Movement (Facebook)", color="black", fontsize=12)
ax_fb.tick_params(axis='y', labelcolor="black")

# SKT movement data (weekly data)
# Prepare SKT data - ensure it's properly aligned with the date range
movement_skt["Date"] = pd.to_datetime(movement_skt["Date"])
movement_skt["전체"] = movement_skt["전체"].rolling(window=2).mean().shift(-2)
mask = (movement_skt["Date"] >= start_date_pd) & (movement_skt["Date"] <= end_date_pd)
filtered_skt = movement_skt.loc[mask]

# Third axis: SKT movement
ax_skt = ax_bottom.twinx()
ax_skt.spines["right"].set_position(("outward", 80))  # Adjust position of third Y axis
ax_skt.plot(filtered_skt["Date"], filtered_skt["전체"], 
           color="black", label="SKT Movement", linewidth=2, linestyle="dotted")
ax_skt.set_ylabel("Movement (SKT)", color="black", fontsize=12)
ax_skt.tick_params(axis='y', labelcolor="black")
ax_skt.get_yaxis().get_offset_text().set_position((1.1,1.1))

# Add social distancing period highlights to bottom plot
for period in sd_periods:
    start_date_sd = datetime.datetime.strptime(period["start"], "%Y-%m-%d").date()
    end_date_sd = datetime.datetime.strptime(period["end"], "%Y-%m-%d").date()
    
    color = "#FF9999" if period["strength"] == "Weak" else \
            "#FF3333" if period["strength"] == "Medium" else \
            "#CC0000"
    
    # Convert to matplotlib dates
    start_num = mdates.date2num(start_date_sd)
    end_num = mdates.date2num(end_date_sd)
    
    rect = Rectangle((start_num, ax_bottom.get_ylim()[0]),
                    end_num - start_num,
                    ax_bottom.get_ylim()[1] - ax_bottom.get_ylim()[0],
                    facecolor=color, alpha=0.3)
    ax_bottom.add_patch(rect)

# Customize bottom plot
ax_bottom.set_title(r'$\beta_{MPUGAT}$ with Movement Data (Facebook, SKT) and Social Distancing Periods', fontsize=20, pad=20)
ax_bottom.set_xlabel("Date", fontsize=12)

# Add legend for bottom plot
lines1, labels1 = ax_bottom.get_legend_handles_labels()
lines2, labels2 = ax_fb.get_legend_handles_labels()
lines3, labels3 = ax_skt.get_legend_handles_labels()
ax_bottom.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc="lower right", fontsize=12)

# X-axis date formatting
ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax_bottom.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)

plt.tight_layout()
plt.grid(True)
plt.savefig("./fig/Beta_Movement_SD.pdf", bbox_inches='tight', dpi=300)
plt.show()
# %%
