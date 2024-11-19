#%%
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import date,timedelta
import matplotlib.dates as mdates
import os 
import seaborn as sns
#%%

def moving_avg(data, window,date_list):
    # Convert data to a NumPy array if it's a list
    if isinstance(data, list):
        data = np.array(data)
    # Compute the moving averages using NumPy's convolution function for efficiency
    weights = np.ones(window) / window
    new_data = np.apply_along_axis(lambda x: np.convolve(x, weights, mode='valid'), axis=0, arr=data)
    date_list = date_list[:-window+1]
    return new_data,date_list

#%% Confrimed case를 불러오자 
window= 15
dic = {}
date_range_dict = {}
N = pd.read_csv("시군구_인구수.csv", encoding="cp949",header=None)
N = N.values

#%%
df = pd.read_csv("시도시별 코로나 데이터.csv")
origin_date_list =pd.to_datetime(df.iloc[0,4:].T.index) 
df = df.replace("-","0")
sejong_rows = df[df["Category"] == "Sejong"]
chungnam_rows = df[df["Category"] == "Chungnam"]

for col in sejong_rows.columns:
    if col != "Category":  # 'Category' 열은 수치 데이터가 아니므로 제외
        chungnam_rows[col] += sejong_rows[col].values
df.loc[df["Category"] == "Chungnam"] = chungnam_rows
df = df[df["Category"] != "Sejong"]
city_name = df["Category"][1:-1].values
C = df.iloc[1:-1,4:].values.astype(float).T
C,date_list = moving_avg(C, window,origin_date_list)
city_name     
#%% 기타 parameter 설정
omicron = pd.read_excel("voc_percent (4).xlsx")
omicron["start_date"] = pd.to_datetime(omicron["start_date"])
omicron["omicron"] = np.sum(omicron.iloc[:,5:].values, axis=1) 
omicron_start = pd.to_datetime('2022-1-16')

def calculate_beta0(date):
    omicron_present = omicron[(omicron["start_date"] <= date) & (omicron["start_date"] > date - pd.Timedelta(days=7))]["omicron"]/100
    if not omicron_present.empty:
        return 2 * omicron_present.iloc[0] + 1 * (1 - omicron_present.iloc[0])
    return 1  # default value if no omicron data is available for the period
def calculate_alpha(date):
    omicron_present = omicron[(omicron["start_date"] <= date) & (omicron["start_date"] > date - pd.Timedelta(days=7))]["omicron"]/100
    if not omicron_present.empty:
        return (1 / 3.5) * omicron_present.iloc[0] + (1 / 6.5) * (1 - omicron_present.iloc[0])
    return (1 / 6.5)  # default if no omicron data is available

beta0 = [calculate_beta0(date) for date in origin_date_list]
alpha = [calculate_alpha(date) for date in origin_date_list]
v = [0.5 if date < pd.to_datetime('2022-04-17') else 0 for date in origin_date_list]
q = [1/3 for date in origin_date_list] 

#moving average 
movingaverage = True
if movingaverage==True:
    alpha,_ = moving_avg(alpha, window,origin_date_list)
    q,_ = moving_avg(q, window,origin_date_list)
    v,_ = moving_avg(v, window,origin_date_list)
    beta0,_ = moving_avg(beta0, window,origin_date_list)
else:
    date_list = origin_date_list

#%% Compartment 부분 제작
I = np.empty(C.shape)
R = np.empty(C.shape)
E = np.empty(C.shape)
R[0,:] = 0
method = "backward"
N = N.squeeze()
if method == "backward":
    for t in range(1, C.shape[0]):  # 행 기준으로 반복
        R[t, :] = R[t-1, :] + C[t, :]
        I[t, :] = (R[t, :] - R[t-1, :]) / q[t]
        E[t, :] = (I[t, :] * (q[t] + 1) - I[t-1, :]) / alpha[t]
    if movingaverage:
        R, date_list = moving_avg(R, window, date_list)
        E, _ = moving_avg(E, window, date_list)
        I, _ = moving_avg(I, window, date_list)
    S = np.full(I.shape, np.nan)
    S[0, :] = N
    for t in range(len(date_list)-1):
        S[t, :] = N - R[t, :] - I[t, :] - E[t, :]
else:
    for t in range(C.shape[0]-1):  # 행 기준으로 -1 해서 반복
        R[t+1, :] = R[t, :] + C[t+1, :]
        I[t, :] = (R[t+1, :] - R[t, :]) / q[t]
        E[t, :] = (I[t+1, :] + (q[t, :] - 1) * I[t, :]) / alpha[t]
    if movingaverage:
        R, date_list = moving_avg(R, window, date_list)
        E, _ = moving_avg(E, window, date_list)
        I, _ = moving_avg(I, window, date_list)
    S = np.full(C.shape, np.nan)
    S[0, :] = N
    for t in range(C.shape[0]-1):
        S[t, :] = N - R[t, :] - I[t, :] - E[t, :]


#%%
C,_ = moving_avg(C, window,origin_date_list)

dic["compart"] = np.stack((S,E,I,R,C),axis=2)  
dic["known"] = np.stack((I.sum(axis=1),beta0[-len(I):],v[-len(I):],alpha[-len(I):],q[-len(I):]),axis=1)  
explain = f"""method : {method}, moving_average : {movingaverage}
compart : S,E,I,R (time x 4)
knwon : beta0,v,alpha,q (time x 4)"""
date_range_dict["compart"] = date_list
date_range_dict["known"] = date_list
#%%
#facebook 데이터 평균
daily_averages = pd.read_csv('Korea Facebook data add pop.csv')
date_list2 = pd.to_datetime(daily_averages["Index"]).unique()

code = daily_averages["code"].unique()
#daily_averages['weighted_change']  = daily_averages['weighted_change']/N
#daily_averages['weighted_ratio']  = daily_averages['weighted_ratio']/N
daily_averages['weighted_product_of_changes'] = daily_averages['weighted_change'] * daily_averages['weighted_ratio']*10
daily_averages = daily_averages[["weighted_product_of_changes","weighted_ratio","weighted_change","code","Index"]]
daily_averages = daily_averages.groupby(['Index',"code"]).sum()
daily_averages = daily_averages.sort_index()
array = np.array(daily_averages).reshape(len(date_list2),-1,3)
movement ,date_list2 = moving_avg(array, window,date_list2)
dic["movement"] = movement
name = ["weighted_product_of_changes","weighted_ratio","weighted_change","code","Index"]
date_range_dict["facebook movement"] = date_list2
explain += f"""\n facebook movment : {name}"""
date_range_dict["movement"] = date_list2

#%%
#date list 정리

start_date = pd.to_datetime('2020-03-01')
end_date = pd.to_datetime('2021-11-23')
new_dic = {}
for key, dates in date_range_dict.items():
    mask = (dates >= start_date) & (dates <= end_date)
    selected_dates = dates[mask]
    start_diff = (start_date-dates[0]).days 
    end_diff = (dates[-1]-end_date).days 

    if key in dic:
        print(len(dic[key]))
        new_dic[key] = dic[key][start_diff:-end_diff]
    # 날짜 범위 업데이트
        date_range_dict[key] = selected_dates
        print(new_dic[key])
        print(len(date_range_dict[key]))
new_dic["explain"] = explain + f"\n start time {selected_dates[0]}, end time {selected_dates[-1]}"
#%%
#Train까지corr값
'''
weighted_product_of_changes = new_dic["movement"][:-120, :, 0] 
correlation_matrix = np.corrcoef(weighted_product_of_changes.T)
new_dic["corr"] = correlation_matrix
'''
traffic = pd.read_excel("traffic.xlsx")
new_dic["corr"] = traffic.iloc[1:,2:-1].values
traffic_df = pd.DataFrame(new_dic["corr"], columns=city_name)
traffic_df.index = city_name
traffic_df = traffic_df.apply(pd.to_numeric)
traffic_df
#%%
plt.figure(figsize=(7, 6))
sns.heatmap(traffic_df, cmap='Blues', annot=False, linewidths=0)  # Use 'Blues' for a white to blue color map
plt.title('Traffic(Movement) Data', fontsize=20)
plt.xlabel('Cities (To)', fontsize=16)
plt.ylabel('Cities (From)', fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

#%%
np.save(f"Facebook_{window}_not_ratio_Facebook_Multi_Data({str(selected_dates[0])[2:10]}_{str(selected_dates[-1])[2:10]}).npy", new_dic)


#%%
print(len(date_range_dict["known"]), len(new_dic["known"][:,1:]))
plt.plot(selected_dates,new_dic["known"][:,1:])
plt.legend(["beta0","v","alpha","q"])
plt.xticks(rotation=45)
#%%
#plot of Compartment
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(selected_dates, new_dic["compart"][:,:,1])
plt.title('E(t)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
plt.plot(selected_dates, new_dic["compart"][:,:,2])
plt.title('I(t)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.xticks(rotation=45)


plt.subplot(2, 2, 3)
plt.plot(selected_dates, new_dic["compart"][:,:,0])
plt.title('S(t)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.xticks(rotation=45)

#%%
plt.figure(figsize=(12,5),dpi=300)
for idx, i in enumerate(movement[:, :, -1].T):
    print(i.shape)
    plt.plot(date_list2, i, label=city_name[idx])  # Use city_name for labels

plt.xlabel("Time steps", fontsize=16)  # Increase x-axis font size
plt.ylabel("(Stay x Change) ratio", fontsize=16)  # Increase y-axis font size
plt.title("Facebook Movement: Product of Changes", size=20)
plt.xticks(rotation=45, fontsize=14)  # Increase x-ticks font size
plt.yticks(fontsize=14)  # Increase y-ticks font size
plt.grid(True)

# Move the legend outside the plot, increase font size, and set ncol to 2 for two columns
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, ncol=2)

# Adjust layout to prevent clipping
plt.tight_layout()

# Show the plot
plt.show()


plt.figure(figsize=(12,5),dpi=300)
for idx, i in enumerate(new_dic["compart"][:, :, -1].T):
    print(i.shape)
    plt.plot(selected_dates, i, label=city_name[idx])  # Use city_name for labels

plt.title('Number of Confirmed ase', fontsize=20)  # Increase title font size
plt.xlabel('Time steps', fontsize=16)  # Increase x-axis font size
plt.ylabel('Q(t)', fontsize=16)  # Increase y-axis font size
plt.xticks(rotation=45, fontsize=14)  # Increase x-ticks font size
plt.yticks(fontsize=14)  # Increase y-ticks font size
plt.grid(True)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, ncol=2)
plt.tight_layout()

# %%

plt.figure(figsize=(10, 8))  # You can adjust the size to fit your needs
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='Blues', 
            xticklabels=city_name, yticklabels=city_name)
plt.title('Correlation Matrix of Cities Over Time',size=20)
plt.xticks(rotation=45,fontsize=12)  # Rotates the x labels to avoid overlap
plt.yticks(rotation=0,fontsize=12)   # Ensures y labels are horizontal
plt.show()

# %%
