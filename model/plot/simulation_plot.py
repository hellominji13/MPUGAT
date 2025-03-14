#%%
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

def simulate_seir(beta_array, h, S0, E0, I0, R0, N, steps):
    """
    Parameters:
    beta_array: shape (30, steps)
    h: shape (30, steps, 16, 16)
    S0, E0, I0, R0: shape (30, 16)
    N: shape (30, 16)
    """
    S, E, I, R = S0, E0, I0, R0  # Each has shape (30, 16)
    N = S+E+I+R
    I_history = []
    I_history.append(I0)
    for t in range(steps-1):
        beta = beta_array[:, t]*0.5
        h_t = h[:, t,:,:]
        # Reshape for broadcasting
        beta = beta.reshape(30, 1)  # Shape: (30, 1)
        # Calculate infection (I is on the left side of multiplication)
        contact = np.matmul(I.reshape(30, 1, 16), h_t).squeeze(1)  # Shape: (30, 16)
        Infected = beta * contact * (S/N)
        # Update SEIR
        S_t = S - Infected
        R_t = R + (1/4) * I
        I_t = I + (1/6.5) * E - (1/4) * I
        E_t = N - S_t - I_t - R_t
        
        I_history.append(I_t)
        S, E, I, R = S_t, E_t, I_t, R_t
    
    return np.array(I_history)  # Shape: (steps, 30, 16)

def plot_city_simulations(data_dict, cutoff_date_str="2020-07-01", pre_months=2, post_months=1):
    """
    각 도시별로 SEIR 시뮬레이션을 4x4 그리드로 플롯합니다.
    
    Parameters:
    data_dict: 모델 데이터 딕셔너리
    cutoff_date_str: 기준 날짜 (문자열, 'YYYY-MM-DD' 형식)
    pre_months: 기준 날짜 이전에 보여줄 개월 수
    post_months: 기준 날짜 이후에 시뮬레이션할 개월 수
    """
    # 도시 이름 설정
    cities = ['Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju', 'Daejeon',
              'Ulsan', 'Gyeonggi', 'Gangwon', 'Chungbuk', 'Chungnam', 'Jeonbuk',
              'Jeonnam', 'Gyeongbuk', 'Gyeongnam', 'Jeju']
    
    # SEIR 데이터 로드
    PATH = './14_Data(20-03-01_21-11-23).npy'
    data = np.load(PATH, allow_pickle='TRUE').item()
    SEIR_data = data["compart"][:, :, :]
    
    # 날짜 계산
    end_date = datetime.datetime.strptime("2021-11-08", "%Y-%m-%d").date()
    total_days = SEIR_data.shape[0]
    start_date = end_date - datetime.timedelta(days=total_days - 1)
    
    # 전체 날짜 범위 생성
    full_date_range = [start_date + datetime.timedelta(days=x) for x in range(total_days)]
    
    # 기준 날짜 설정
    cutoff_date = datetime.datetime.strptime(cutoff_date_str, "%Y-%m-%d").date()
    
    # 기준 날짜의 인덱스 찾기
    cutoff_idx = next((i for i, date in enumerate(full_date_range) if date >= cutoff_date), 0)
    
    # 이전 2개월과 이후 1개월 날짜 범위 설정
    pre_days = pre_months * 20  # 약 2개월
    post_days = post_months * 10  # 약 1개월 (시뮬레이션)
    
    # 플롯 범위 설정
    plot_start_idx = max(0, cutoff_idx - pre_days)
    plot_end_idx = min(total_days, cutoff_idx + post_days)
    
    # 시뮬레이션 기간 설정
    simulation_days = plot_end_idx - cutoff_idx
    
    print(f"기준 날짜: {cutoff_date}")
    print(f"플롯 시작일: {full_date_range[plot_start_idx]}")
    print(f"플롯 종료일: {full_date_range[plot_end_idx-1]}")
    print(f"시뮬레이션 기간: {simulation_days} 일")
    
    # 시뮬레이션 날짜 범위 생성
    date_range_sim = [cutoff_date + datetime.timedelta(days=x) for x in range(simulation_days)]
    
    # Prepare beta values for simulation period
    # MPUGAT 모델
    beta_train = data_dict["MPUGAT"]["train"][:, :, 0, 0, 0, 2]  
    beta_val = data_dict["MPUGAT"]["validate"][:, :, 0, 0, 0, 2]
    beta_test = data_dict["MPUGAT"]["test"][:, :, 0, 0, 0, 2]
    mpugat_beta_combined = np.concatenate([beta_train, beta_val, beta_test], axis=1)
    mpugat_beta = mpugat_beta_combined[:, cutoff_idx:cutoff_idx+simulation_days]
    
    # C_ij 모델
    beta_train = data_dict["C_ij"]["train"][:, :, 0, 0, 0, 2]  
    beta_val = data_dict["C_ij"]["validate"][:, :, 0, 0, 0, 2]
    beta_test = data_dict["C_ij"]["test"][:, :, 0, 0, 0, 2]
    c_ij_beta_combined = np.concatenate([beta_train, beta_val, beta_test], axis=1)
    c_ij_beta = c_ij_beta_combined[:, cutoff_idx:cutoff_idx+simulation_days]
    
    # 접촉 행렬 준비
    mpugat_contact = data_dict["MPUGAT"]["contact"][:, cutoff_idx:cutoff_idx+simulation_days, :, :]
    c_ij_contact = data_dict["C_ij"]["contact"][:, cutoff_idx:cutoff_idx+simulation_days, :, :]
    
    # 실제 데이터 가져오기
    real_I = SEIR_data[:, :, 2]  # I 값
    real_I_for_plot = np.tile(real_I[None, :, :], (30, 1, 1))  # Shape: (30, time, 16)
    
    # 초기 조건 설정
    N = np.sum(SEIR_data[0, :, :], axis=1)  # 각 도시의 t=0에서의 S+E+I+R 합
    N = np.tile(N, (30, 1))  # Shape: (30, 16)
    
    # 기준 날짜의 실제 값을 초기 조건으로 사용
    I0 = SEIR_data[cutoff_idx, :, 2]  # 기준 날짜의 실제 I 값
    I0 = np.tile(I0, (30, 1))  # Shape: (30, 16)
    E0 = SEIR_data[cutoff_idx, :, 1]
    E0 = np.tile(E0, (30, 1))
    R0 = SEIR_data[cutoff_idx, :, 3]
    R0 = np.tile(R0, (30, 1))
    S0 = SEIR_data[cutoff_idx, :, 0]
    S0 = np.tile(S0, (30, 1))
    
    # 시뮬레이션 실행
    mpugat_contact[..., 0, 0] = mpugat_contact[..., 0, 0]
    c_ij_beta[..., -1, -1] = c_ij_beta[..., -1, -1] 
    I_mpugat = simulate_seir(mpugat_beta, mpugat_contact, S0, E0, I0, R0, N, simulation_days)
    I_cij = simulate_seir(c_ij_beta, c_ij_contact, S0, E0, I0, R0, N, simulation_days)
    
    # 4x4 그리드 플롯 생성
    fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True)
    
    for i, city in enumerate(cities):
        row, col = i // 4, i % 4
        ax = axes[row, col]
        
        # 실제 데이터 플롯 (플롯 범위)
        city_real_data = real_I[plot_start_idx:plot_end_idx-simulation_days, i]
        ax.plot(full_date_range[plot_start_idx:plot_end_idx-simulation_days], city_real_data, 
                color='black', linestyle='-', label='Real Data')
        
        # 시뮬레이션 데이터 플롯
        city_mpugat_mean = np.mean(I_mpugat[:, :, i], axis=1)
        city_mpugat_std = np.std(I_mpugat[:, :, i], axis=1)/np.sqrt(30)
        city_cij_mean = np.mean(I_cij[:, :, i], axis=1)
        city_cij_std = np.std(I_cij[:, :, i], axis=1)/np.sqrt(30)
        
        ax.plot(date_range_sim, city_mpugat_mean, label='MPUGAT', color='red')
        ax.fill_between(date_range_sim, 
                        city_mpugat_mean - 1.67 * city_mpugat_std,
                        city_mpugat_mean + 1.67 * city_mpugat_std,
                        color='red', alpha=0.2)
        
        ax.plot(date_range_sim, city_cij_mean, label='C_ij', color='blue')
        ax.fill_between(date_range_sim, 
                        city_cij_mean - 1.67 * city_cij_std,
                        city_cij_mean + 1.67 * city_cij_std,
                        color='blue', alpha=0.2)
        
        # 기준선 표시
        ax.axvline(x=cutoff_date, color='gray', linestyle='--', alpha=0.5)
        
        # 도시 이름 표시
        ax.set_title(f'{city}', fontsize=14)
        
        # x축 포맷팅
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.tick_params(axis='x', rotation=45)
        
        # y축 레이블
        if col == 0:
            ax.set_ylabel('Infected', fontsize=12)
        
        # 첫 번째 플롯에만 범례 추가
        if i == 0:
            ax.legend(fontsize=10)
    
    # 전체 제목
    plt.suptitle(f'SEIR Simulation by City (Cutoff: {cutoff_date_str})', fontsize=18)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 제목 공간 확보
    plt.savefig(f"./city_simulations_{cutoff_date_str}.png", bbox_inches='tight', dpi=300)
    plt.show()
    
    return

# 함수 사용 예시:
# 2020년 7월 1일을 기준으로 시뮬레이션
plot_city_simulations(data_dict, cutoff_date_str="2020-07-01", pre_months=2, post_months=1)
#%%
def plot_city_simulations(data_dict, cutoff_date_str="2020-07-01", pre_months=2, post_months=1):
    import numpy as np
    import datetime
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    cities = ['Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju', 'Daejeon',
              'Ulsan', 'Gyeonggi', 'Gangwon', 'Chungbuk', 'Chungnam', 'Jeonbuk',
              'Jeonnam', 'Gyeongbuk', 'Gyeongnam', 'Jeju']
    
    PATH = './14_Data(20-03-01_21-11-23).npy'
    data = np.load(PATH, allow_pickle='TRUE').item()
    SEIR_data = data["compart"][:, :, :]
    
    end_date = datetime.datetime.strptime("2021-11-08", "%Y-%m-%d").date()
    total_days = SEIR_data.shape[0]
    start_date = end_date - datetime.timedelta(days=total_days - 1)
    full_date_range = [start_date + datetime.timedelta(days=x) for x in range(total_days)]
    
    cutoff_date = datetime.datetime.strptime(cutoff_date_str, "%Y-%m-%d").date()
    cutoff_idx = next((i for i, date in enumerate(full_date_range) if date >= cutoff_date), 0)
    
    # 기준 날짜의 실제 값을 초기 조건으로 사용
    I0 = SEIR_data[cutoff_idx, :, 2]  # 기준 날짜의 실제 I 값
    I0 = np.tile(I0, (30, 1))  # Shape: (30, 16)
    E0 = SEIR_data[cutoff_idx, :, 1]
    E0 = np.tile(E0, (30, 1))
    R0 = SEIR_data[cutoff_idx, :, 3]
    R0 = np.tile(R0, (30, 1))
    S0 = SEIR_data[cutoff_idx, :, 0]
    S0 = np.tile(S0, (30, 1))
    N = np.sum(SEIR_data[0, :, :], axis=1)
    
    pre_days = pre_months * 20
    post_days = post_months * 10
    plot_start_idx = max(0, cutoff_idx - pre_days)
    plot_end_idx = min(total_days, cutoff_idx + post_days)
    simulation_days = plot_end_idx - cutoff_idx
    date_range_sim = [cutoff_date + datetime.timedelta(days=x) for x in range(simulation_days)]
    
    def extract_beta(data_dict, model_name, factor):
        beta_train = data_dict[model_name]["train"][:, :, 0, 0, 0, 2] * factor
        beta_val = data_dict[model_name]["validate"][:, :, 0, 0, 0, 2] * factor
        beta_test = data_dict[model_name]["test"][:, :, 0, 0, 0, 2] * factor
        beta_combined = np.concatenate([beta_train, beta_val, beta_test], axis=1)
        return beta_combined[:, cutoff_idx:cutoff_idx+simulation_days]
    
    factors = [0.5, 2]
    colors = ['red', 'blue']
    labels = {'MPUGAT': 'MPUGAT', 'C_ij': 'Simple Hybrid Model'}
    linestyles = ['--', '-']
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True)
    
    for i, city in enumerate(cities):
        row, col = i // 4, i % 4
        ax = axes[row, col]
        real_dates = full_date_range[cutoff_idx - 20:cutoff_idx]
        real_values = SEIR_data[cutoff_idx - 20:cutoff_idx, i, 2]
        ax.plot(real_dates, real_values, label='Real Data', color='black', linestyle='dotted')
        
        for model, color in zip(["MPUGAT", "C_ij"], colors):
            for factor, linestyle in zip(factors, linestyles):
                beta = extract_beta(data_dict, model, factor)
                contact = data_dict[model]["contact"][:, cutoff_idx:cutoff_idx+simulation_days, :, :]
                
                I_sim = simulate_seir(beta, contact, S0, E0, I0, R0, N, simulation_days)
                city_mean = np.mean(I_sim[:, :, i], axis=1)
                city_std = np.std(I_sim[:, :, i], axis=1)/np.sqrt(30)
                
                ax.plot(date_range_sim, city_mean, label=f'{labels[model]} (β × {factor})', color=color, linestyle=linestyle)
                ax.fill_between(date_range_sim, city_mean - 1.67 * city_std, city_mean + 1.67 * city_std, color=color, alpha=0.2)
        
        ax.axvline(x=cutoff_date, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{city}', fontsize=14)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.tick_params(axis='x', rotation=45)
        if col == 0:
            ax.set_ylabel('Infected', fontsize=12)
        if i == 0:
            ax.legend(fontsize=10)
    
    plt.suptitle(f'SEIR Simulation by City (Cutoff: {cutoff_date_str})', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"./city_simulations_{cutoff_date_str}.png", bbox_inches='tight', dpi=300)
    plt.show()
    
    return


plot_city_simulations(data_dict, cutoff_date_str="2020-07-01", pre_months=2, post_months=1)

#%%
# Run the simulation
base_path = './seed/'
for pred_len in [3]:  # , 21
    data_dict = {} 
    for model in ["deep_learning","compartment","C_ij", "MPGAT","MPUGAT"]:
        filename = f'preds_acts_{model}_{pred_len}.npy'
        full_path = base_path + filename
        data = np.load(full_path, allow_pickle=True).item()
        data_dict[model] = data

# %%
