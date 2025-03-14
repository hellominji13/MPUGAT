#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.dates as mdates
import datetime
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches

# 기존 함수는 그대로 유지
def calculate_infection_ratio(data_dict, data_path, start_date=None, end_date=None):
    """
    특정 기간 동안의 도시 내/외부 감염 비율 계산
    
    Parameters:
    data_dict: 모델 데이터 딕셔너리
    data_path: SEIR 데이터 파일 경로
    start_date: 시작 날짜 (문자열, 'YYYY-MM-DD' 형식)
    end_date: 종료 날짜 (문자열, 'YYYY-MM-DD' 형식)
    
    Returns:
    ratio: 도시 내 감염 비율 (%)
    date_range: 날짜 범위
    start_idx, end_idx: 시작/종료 인덱스
    """
    # SEIR 데이터 로드
    data = np.load(data_path, allow_pickle='TRUE').item()
    SEIR_data = data["compart"]
    
    # 도시 이름 설정
    cities = ['Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju', 'Daejeon',
              'Ulsan', 'Gyeonggi', 'Gangwon', 'Chungbuk', 'Chungnam', 'Jeonbuk',
              'Jeonnam', 'Gyeongbuk', 'Gyeongnam', 'Jeju']
    
    # 날짜 계산
    end_date_data = datetime.datetime.strptime("2021-11-23", "%Y-%m-%d").date()
    total_days = SEIR_data.shape[0]
    start_date_data = end_date_data - datetime.timedelta(days=total_days - 1)
    
    # 전체 날짜 범위 생성
    date_range = [start_date_data + datetime.timedelta(days=x) for x in range(total_days)]
    
    # 사용자가 지정한 기간 인덱스 찾기
    if start_date is not None:
        start_date_obj = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        start_idx = next((i for i, date in enumerate(date_range) if date >= start_date_obj), 0)
    else:
        start_idx = 0
        
    if end_date is not None:
        end_date_obj = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
        end_idx = next((i for i, date in enumerate(date_range) if date > end_date_obj), total_days)
    else:
        end_idx = total_days
    
    print(f"분석 기간: {date_range[start_idx].strftime('%Y-%m-%d')} ~ {date_range[end_idx-1].strftime('%Y-%m-%d')}")
    
    # SEIR 데이터 추출
    I = SEIR_data[:, :, 2]
    E = SEIR_data[:, :, 1]
    R = SEIR_data[:, :, 3]
    S = SEIR_data[:, :, 0]
    N = S + E + I + R
    
    # beta 및 접촉 행렬 준비
    beta_train = data_dict["MPUGAT"]["train"][:, :, 0, 0, 0, 2]  
    beta_val = data_dict["MPUGAT"]["validate"][:, :, 0, 0, 0, 2]
    beta_test = data_dict["MPUGAT"]["test"][:, :, 0, 0, 0, 2]
    beta_combined = np.concatenate([beta_train, beta_val, beta_test], axis=1)
    
    mpugat_contact = data_dict["MPUGAT"]["contact"]
    
    # 도시 내/외부 감염 계산 배열 초기화
    infected_within_city = np.zeros((total_days, 30, 16))
    infected_from_outside = np.zeros((total_days, 30, 16))
    
    # 각 시점에 대해 계산
    for t in range(min(total_days, len(beta_combined[0]))):
        beta_t = beta_combined[:, t]
        h_t = mpugat_contact[:, t, :, :]
        
        beta_t = beta_t.reshape(30, 1)  # Shape: (30, 1)
        
        # t 시점의 I, S, N 값
        I_t = I[t].reshape(1, 16)
        I_t = np.tile(I_t, (30, 1))  # 30개 seed로 복제
        S_t = S[t].reshape(1, 16)
        S_t = np.tile(S_t, (30, 1))
        N_t = N[t].reshape(1, 16)
        N_t = np.tile(N_t, (30, 1))
        
        # 총 접촉 계산
        contact = np.matmul(I_t.reshape(30, 1, 16), h_t).squeeze(1)  # Shape: (30, 16)
        total_infected = beta_t * contact * (S_t / N_t)
        
        # 도시 내부 감염 계산 (h_t의 대각 요소)
        diag_h_t = np.zeros_like(h_t)
        for i in range(30):
            np.fill_diagonal(diag_h_t[i], np.diagonal(h_t[i]))
        within_contact = np.matmul(I_t.reshape(30, 1, 16), diag_h_t).squeeze(1)
        infected_within = beta_t * within_contact * (S_t / N_t)
        
        # 도시 외부에서의 감염 계산
        infected_outside = total_infected - infected_within
        
        # 저장
        infected_within_city[t] = infected_within
        infected_from_outside[t] = infected_outside
    
    # 분석 기간 동안의 평균 계산
    period_within_city = infected_within_city[start_idx:end_idx]
    period_outside_city = infected_from_outside[start_idx:end_idx]
    
     #print(period_within_city.shape) #(164, 30, 16)
    # 기간에 대한 총합 계산
    mean_within_city = np.sum(period_within_city, axis=0)  # (기간, 16,)
    mean_outside_city = np.sum(period_outside_city, axis=0)  # (기간 ,16,)
       # seed에 대한 평균 계산
    total_within_city  = np.mean(mean_within_city, axis=0)  # (16)
    total_outside_city  = np.mean(mean_outside_city, axis=0)  # (16)
    
    # 비율 계산
    total_infections = total_within_city + total_outside_city
    ratio = np.zeros(16)
    
    # 0으로 나누기 방지
    mask = total_infections > 0
    ratio[mask] = total_within_city[mask]  /total_infections[mask]  #N[0,:] #[mask] / total_infections[mask] * 100  # 퍼센트로 변환
    
    # 결과 출력
    result_df = pd.DataFrame({
        'City': cities,
        'Within_City_Infections': total_within_city,
        'Outside_City_Infections': total_outside_city,
        'Total_Infections': total_infections,
        'Within_City_Ratio(%)': ratio
    })
    
    return ratio, date_range, start_idx, end_idx

# 3개 기간에 대한 지도 시각화 함수
def create_three_period_maps(data_dict, data_path):
    """
    3개 기간에 대한 도시 내 감염 비율 지도 시각화
    
    Parameters:
    data_dict: 모델 데이터 딕셔너리
    data_path: SEIR 데이터 파일 경로
    """
    # 도시 이름 설정
    cities = ['Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju', 'Daejeon',
              'Ulsan', 'Gyeonggi', 'Gangwon', 'Chungbuk', 'Chungnam', 'Jeonbuk',
              'Jeonnam', 'Gyeongbuk', 'Gyeongnam', 'Jeju']
    
    # 3개 기간 정의
    '''
    periods = [
        {'name': 'First period', 'start': '2020-01-20', 'end': '2020-08-11'},
        {'name': 'Second period', 'start': '2020-08-12', 'end': '2020-11-12'},
        {'name': 'Third period', 'start': '2020-11-13', 'end': '2021-07-06'}
    ]
    '''
    periods = [
        {'name': 'First Wave', 'start': '2020-03-01', 'end': '2020-04-20'},
        {'name': 'Second Wave', 'start': '2020-07-28', 'end': '2020-08-12'},
        {'name': 'Third Wave', 'start': '2020-11-03', 'end': '2021-02-01'},

    ]
    # 모든 기간에 대한 비율 계산
    ratios = []
    for period in periods:
        ratio, date_range, start_idx, end_idx = calculate_infection_ratio(
            data_dict, data_path, 
            start_date=period['start'], 
            end_date=period['end']
        )
        ratios.append(ratio)
    
    # 한국 행정구역 지도 로드
    korea = gpd.read_file('gadm41_KOR_1.json')
    
    # 작은 섬 제거 함수
    def filter_small_islands(geometry):
        if geometry.geom_type == 'MultiPolygon':
            polygons = [poly for poly in geometry.geoms if poly.area > 0.01]
            if polygons:
                from shapely.geometry import MultiPolygon
                return MultiPolygon(polygons)
        return geometry
    
    # 작은 섬 제거
    korea.geometry = korea.geometry.apply(filter_small_islands)
    
    # 세종과 충남 지역 병합
    chungnam_mask = korea['NAME_1'] == 'Chungcheongnam-do'
    sejong_mask = korea['NAME_1'] == 'Sejong'
    
    if sejong_mask.any():  # 세종시가 있는 경우에만 병합
        merged_geom = korea[chungnam_mask | sejong_mask].unary_union
        
        merged_row = pd.DataFrame({
            'NAME_1': ['Chungcheongnam-do'],
            'geometry': [merged_geom]
        })
        
        korea = korea[~(chungnam_mask | sejong_mask)]
        korea = pd.concat([korea, merged_row], ignore_index=True)
    
    # 지역명 매핑
    name_mapping = {
        'Busan': 'Busan',
        'Chungcheongbuk-do': 'Chungbuk',
        'Chungcheongnam-do': 'Chungnam',
        'Daegu': 'Daegu',
        'Daejeon': 'Daejeon',
        'Gangwon-do': 'Gangwon',
        'Gwangju': 'Gwangju',
        'Gyeonggi-do': 'Gyeonggi',
        'Gyeongsangbuk-do': 'Gyeongbuk',
        'Gyeongsangnam-do': 'Gyeongnam',
        'Incheon': 'Incheon',
        'Jeju': 'Jeju',
        'Jeollabuk-do': 'Jeonbuk',
        'Jeollanam-do': 'Jeonnam',
        'Seoul': 'Seoul',
        'Ulsan': 'Ulsan'
    }
    
    korea['NAME_1'] = korea['NAME_1'].map(name_mapping)
    
    # 전체 figure 생성
    plt.figure(figsize=(15, 7))
    
    # 색상 설정
    colors_ratio = ['#FFF5F0', '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A', '#EF3B2C', '#CB181D', '#A50F15', '#67000D']
    cmap_ratio = LinearSegmentedColormap.from_list("infection_ratio", colors_ratio, N=256)
    
    # 강조 표시할 지역
    highlight_regions = {
        0: [],#['Gyeonggi',"Chungnam",'Daegu'],             # 첫 번째 기간에는 대구 강조
        1: [], #['Gyeonggi',"Chungnam","Busan"],          # 두 번째 기간에는 경기도 강조
        2: [] #["Gyeongnam", 'Busan'] # 세 번째 기간에는 경북, 부산 강조
    }
    
    # 3개 기간에 대한 시각화
    for i, (period, ratio) in enumerate(zip(periods, ratios)):
        # 비율 데이터프레임 생성
        ratio_df = pd.DataFrame({
            'NAME_1': cities,
            'infection_ratio': ratio
        })
        
        # 지도 데이터와 병합
        korea_ratio = korea.merge(ratio_df, how='left', on='NAME_1')
        
        # 서브플롯 위치 설정
        ax = plt.subplot(1, 3, i+1)
        
        # 지도 그리기
        korea_ratio.plot(
            column='infection_ratio', 
            cmap=cmap_ratio, 
            linewidth=0.8, 
            edgecolor='black', 
            ax=ax,
            vmin=0,
            vmax=np.max(ratio),
            legend=True if i == 2 else False  # 마지막 지도에만 범례 표시
        )
        
        # 강조 표시할 지역
        if i in highlight_regions:
            for region in highlight_regions[i]:
                region_geom = korea_ratio[korea_ratio['NAME_1'] == region].geometry
                if not region_geom.empty:
                    # 빨간색 테두리로 강조
                    bbox = region_geom.bounds.iloc[0]
                    rect = patches.Rectangle((bbox.minx, bbox.miny), 
                                           bbox.maxx - bbox.minx, 
                                           bbox.maxy - bbox.miny, 
                                           linewidth=2, 
                                           edgecolor='red', 
                                           facecolor='none')
                    ax.add_patch(rect)
        
        # 제목 설정
        ax.set_title(f'{period["name"]}\n{period["start"]} ~ {period["end"]}', fontsize=12)
        ax.axis('off')
        
        # 지역명 추가 - 텍스트가 너무 많아서 복잡해질 수 있으므로 주요 지역만 표시
        important_regions = ['Seoul', 'Busan', 'Daegu', 'Chungnam', 'Gyeongbuk',"Gyeongnam"]
        for idx, row in korea_ratio.iterrows():
            if row['NAME_1'] in important_regions:
                centroid = row.geometry.centroid
                ratio_val = row['infection_ratio']
                
                # 비율 값이 유효한 경우에만 텍스트 표시
                if not np.isnan(ratio_val):
                    # 배경색 결정 (어두운 지역은 밝은 텍스트, 밝은 지역은 어두운 텍스트)
                    if ratio_val > 50:  # 임계값 조정 가능
                        text_color = 'white'
                    else:
                        text_color = 'black'
                    
                    ax.annotate(
                        f"{row['NAME_1']}",
                        xy=(centroid.x, centroid.y),
                        ha='center',
                        va='center',
                        fontsize=10,
                        color=text_color,
                        weight='bold',
                        bbox=dict(facecolor='none', alpha=0.7, edgecolor='none', pad=1.5)
                    )
    
    # 메인 제목
    plt.suptitle('Number of Infected People between same resdient / Total Number of Infected People', fontsize=16, y=0.95)
    

    
    plt.tight_layout(rect=[0, 0, 0.9, 0.9])
    plt.savefig('./fig/three_period_infection_ratio_map.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# 사용 예시:
# 데이터 경로
data_path = './14_Data(20-03-01_21-11-23).npy'

# 3개 기간에 대한 감염 비율 지도 시각화
create_three_period_maps(data_dict, data_path)
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
from matplotlib.gridspec import GridSpec

def create_simplified_visualization(data_dict, data_path):
    """
    3개 기간에 대한 도시 내 감염 비율 지도와 감염자 수 추이 그래프를 결합한 시각화
    
    Parameters:
    data_dict: 모델 데이터 딕셔너리
    data_path: SEIR 데이터 파일 경로
    """
    # 도시 이름 설정
    cities = ['Seoul', 'Busan', 'Daegu', 'Incheon', 'Gwangju', 'Daejeon',
              'Ulsan', 'Gyeonggi', 'Gangwon', 'Chungbuk', 'Chungnam', 'Jeonbuk',
              'Jeonnam', 'Gyeongbuk', 'Gyeongnam', 'Jeju']
    
    # 3개 기간 정의
    periods = [
        {'name': 'First Wave', 'start': '2020-03-01', 'end': '2020-04-20'},
        {'name': 'Second Wave', 'start': '2020-07-28', 'end': '2020-10-12'},
        {'name': 'Third Wave', 'start': '2020-11-03', 'end': '2021-02-01'},
    ]
    
    # SEIR 데이터 로드
    data = np.load(data_path, allow_pickle=True).item()
    SEIR_data = data["compart"]
    
    # I값 추출 (감염자 수)
    I = SEIR_data[:, :, 2]  # SEIR 데이터에서 I 컴포넌트 (감염자) 추출
    
    # 날짜 계산
    end_date_data = datetime.datetime.strptime("2021-11-23", "%Y-%m-%d").date()
    total_days = SEIR_data.shape[0]
    start_date_data = end_date_data - datetime.timedelta(days=total_days - 1)
    
    # 전체 날짜 범위 생성
    full_date_range = [start_date_data + datetime.timedelta(days=x) for x in range(total_days)]
    
    # 초기 1년 데이터만 선택 (시작일부터 1년)
    one_year_end_idx = min(365, total_days)  # 1년 또는 데이터 전체 길이 중 작은 값
    
    # 1년 데이터만 선택
    I_one_year = I[:one_year_end_idx]
    date_range = full_date_range[:one_year_end_idx]
    date_range_pd = pd.DatetimeIndex(date_range)
    
    # 모든 기간에 대한 비율 계산
    ratios = []
    period_indices = []
    for period in periods:
        ratio, date_subset, start_idx, end_idx = calculate_infection_ratio(
            data_dict, data_path, 
            start_date=period['start'], 
            end_date=period['end']
        )
        ratios.append(ratio)
        period_indices.append((start_idx, end_idx))
    
    # 한국 행정구역 지도 로드
    korea = gpd.read_file('gadm41_KOR_1.json')
    
    # 작은 섬 제거 함수
    def filter_small_islands(geometry):
        if geometry.geom_type == 'MultiPolygon':
            polygons = [poly for poly in geometry.geoms if poly.area > 0.01]
            if polygons:
                from shapely.geometry import MultiPolygon
                return MultiPolygon(polygons)
        return geometry
    
    # 작은 섬 제거
    korea.geometry = korea.geometry.apply(filter_small_islands)
    
    # 세종과 충남 지역 병합
    chungnam_mask = korea['NAME_1'] == 'Chungcheongnam-do'
    sejong_mask = korea['NAME_1'] == 'Sejong'
    
    if sejong_mask.any():  # 세종시가 있는 경우에만 병합
        merged_geom = korea[chungnam_mask | sejong_mask].unary_union
        
        merged_row = pd.DataFrame({
            'NAME_1': ['Chungcheongnam-do'],
            'geometry': [merged_geom]
        })
        
        korea = korea[~(chungnam_mask | sejong_mask)]
        korea = pd.concat([korea, merged_row], ignore_index=True)
    
    # 지역명 매핑
    name_mapping = {
        'Busan': 'Busan',
        'Chungcheongbuk-do': 'Chungbuk',
        'Chungcheongnam-do': 'Chungnam',
        'Daegu': 'Daegu',
        'Daejeon': 'Daejeon',
        'Gangwon-do': 'Gangwon',
        'Gwangju': 'Gwangju',
        'Gyeonggi-do': 'Gyeonggi',
        'Gyeongsangbuk-do': 'Gyeongbuk',
        'Gyeongsangnam-do': 'Gyeongnam',
        'Incheon': 'Incheon',
        'Jeju': 'Jeju',
        'Jeollabuk-do': 'Jeonbuk',
        'Jeollanam-do': 'Jeonnam',
        'Seoul': 'Seoul',
        'Ulsan': 'Ulsan'
    }
    
    korea['NAME_1'] = korea['NAME_1'].map(name_mapping)
    
    # GridSpec을 사용하여 레이아웃 생성 (더 조밀하게)
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3, height_ratios=[2, 1], hspace=0.1, wspace=0.05)
    
    # 색상 설정
    colors_ratio = ['#FFF5F0', '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A', '#EF3B2C', '#CB181D', '#A50F15', '#67000D']
    cmap_ratio = LinearSegmentedColormap.from_list("infection_ratio", colors_ratio, N=256)
    
    # 강조 표시할 지역
    highlight_regions = {
        0: [],#['Gyeonggi',"Chungnam",'Daegu'],             # 첫 번째 기간에는 대구 강조
        1: [], #['Gyeonggi',"Chungnam","Busan"],          # 두 번째 기간에는 경기도 강조
        2: [] #["Gyeongnam", 'Busan'] # 세 번째 기간에는 경북, 부산 강조
    }
    
    # 감염자 수 총합 계산 (전국) - 1년 데이터만
    total_daily_infected = np.sum(I_one_year, axis=1)
    
    # 3개 기간에 대한 지도 시각화
    for i, (period, ratio) in enumerate(zip(periods, ratios)):
        # 비율 데이터프레임 생성
        ratio_df = pd.DataFrame({
            'NAME_1': cities,
            'infection_ratio': ratio
        })
        
        # 지도 데이터와 병합
        korea_ratio = korea.merge(ratio_df, how='left', on='NAME_1')
        
        # 서브플롯 위치 설정
        ax = fig.add_subplot(gs[0, i])
        
        # 지도 그리기
        korea_ratio.plot(
            column='infection_ratio', 
            cmap=cmap_ratio, 
            linewidth=0.8, 
            edgecolor='black', 
            ax=ax,
            vmin=0,
            vmax=max([np.max(r) for r in ratios]),  # 모든 지도에 동일한 색상 범위 적용
            legend=True if i == 2 else False,       # 마지막 지도에만 범례 표시
        )
        
        # 강조 표시할 지역
        if i in highlight_regions:
            for region in highlight_regions[i]:
                region_geom = korea_ratio[korea_ratio['NAME_1'] == region].geometry
                if not region_geom.empty:
                    # 빨간색 테두리로 강조
                    bbox = region_geom.bounds.iloc[0]
                    rect = patches.Rectangle((bbox.minx, bbox.miny), 
                                          bbox.maxx - bbox.minx, 
                                          bbox.maxy - bbox.miny, 
                                          linewidth=2, 
                                          edgecolor='red', 
                                          facecolor='none')
                    ax.add_patch(rect)
        
        # 제목 설정
        ax.set_title(f'{period["name"]}\n{period["start"]} ~ {period["end"]}', fontsize=12)
        ax.axis('off')
        
        # 지역명 추가 - 텍스트가 너무 많아서 복잡해질 수 있으므로 주요 지역만 표시
        important_regions = ['Seoul', 'Busan', 'Daegu', 'Chungnam', 'Gyeongbuk', 'Gyeongnam']
        for idx, row in korea_ratio.iterrows():
            if row['NAME_1'] in important_regions:
                centroid = row.geometry.centroid
                ratio_val = row['infection_ratio']
                
                # 비율 값이 유효한 경우에만 텍스트 표시
                if not np.isnan(ratio_val):
                    # 배경색 결정 (어두운 지역은 밝은 텍스트, 밝은 지역은 어두운 텍스트)
                    text_color = 'white' if ratio_val > 30 else 'black'
                    
                    ax.annotate(
                        f"{row['NAME_1']}",
                        xy=(centroid.x, centroid.y),
                        ha='center',
                        va='center',
                        fontsize=9,
                        color=text_color,
                        weight='bold',
                        bbox=dict(facecolor='none', alpha=0.7, edgecolor='none', pad=1.5)
                    )
    
    # 감염자 수 추이 그래프 생성 (하단에 모든 열을 차지하도록)
    ax_graph = fig.add_subplot(gs[1, :])
    
    # 1년 데이터에 대한 감염자 수 그래프
    ax_graph.plot(date_range_pd, total_daily_infected, 'k-', linewidth=1.5)
    
    # 각 Wave 기간 색상으로 표시 - 처음 1년 데이터만 해당하는 wave
    wave_colors = ['#d1e7ff', '#add8e6', '#a9a9a9']
    wave_periods = [
        {'name': '1st wave', 'start': '2020-01-20', 'end': '2020-04-20'},
        {'name': '2nd wave', 'start': '2020-07-28', 'end': '2020-10-12'},
        {'name': '3rd wave', 'start': '2020-11-03', 'end': '2021-02-01'},
    ]
    
    for j, period in enumerate(wave_periods):
        start_date = datetime.datetime.strptime(period['start'], "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(period['end'], "%Y-%m-%d").date()
        
        # 날짜 범위에 해당하는 인덱스 찾기 - 1년 데이터 범위 내에서만
        # 시작 인덱스 찾기
        start_idx = next((i for i, d in enumerate(date_range) if d >= start_date), None)
        if start_idx is None:
            continue  # 이 기간은 1년 데이터에 포함되지 않음
        
        # 종료 인덱스 찾기
        end_idx = next((i for i, d in enumerate(date_range) if d > end_date), len(date_range))
        
        ax_graph.fill_between(
            date_range_pd[start_idx:end_idx], 0, total_daily_infected[start_idx:end_idx], 
            color=wave_colors[j], alpha=0.5, 
            label=f"{period['name']} ({period['start']} - {period['end']})"
        )
        
        # 각 웨이브 이름 표시
        # 날짜 범위에 해당하는 중간 지점 찾기
        mid_idx = start_idx + (end_idx - start_idx) // 2
        mid_date = date_range_pd[mid_idx]
        y_pos = max(total_daily_infected) * 0.9
        
        ax_graph.text(
            mid_date, y_pos,
            period['name'],
            ha='center', va='center',
            fontsize=10, weight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    # 지도에 표시된 기간 강조 (빨간색 네모)
    for i, period in enumerate(periods):
        start_idx, end_idx = period_indices[i]
        # 1년 데이터 내에 있는 기간만 표시
        if start_idx < one_year_end_idx:
            end_idx = min(end_idx, one_year_end_idx)
            ax_graph.axvspan(
                date_range_pd[start_idx], date_range_pd[min(end_idx-1, one_year_end_idx-1)],
                color='red', alpha=0.2, edgecolor='red', linewidth=2
            )
    
    # 그래프 스타일 설정
    ax_graph.set_xlabel('Date', fontsize=12)
    ax_graph.set_ylabel('Number of Infected People (I(t))', fontsize=12)  # 레이블 변경
    ax_graph.grid(True, linestyle='--', alpha=0.7)
    ax_graph.spines['top'].set_visible(False)
    ax_graph.spines['right'].set_visible(False)
    
    # x축 날짜 포맷 설정
    from matplotlib.dates import MonthLocator, DateFormatter
    ax_graph.xaxis.set_major_locator(MonthLocator(interval=1))  # 1개월 간격으로 표시
    ax_graph.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    plt.setp(ax_graph.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 메인 제목
    fig.suptitle('Infections from Same Resident Contact / Total Infections in Residents', fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('./fig/simplified_covid_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.show()

create_simplified_visualization(data_dict, data_path)

# %%
base_path = './seed/'
for pred_len in [3]:  # , 21
    data_dict = {} 
    for model in ["deep_learning","compartment","C_ij", "MPGAT","MPUGAT"]:
        filename = f'preds_acts_{model}_{pred_len}.npy'
        full_path = base_path + filename
        data = np.load(full_path, allow_pickle=True).item()
        data_dict[model] = data
# %%
data_path = './14_Data(20-03-01_21-11-23).npy'

# 특정 기간에 대한 감염 비율 계산 및 지도 시각화
result_df, mean_within, mean_outside = calculate_infection_ratio(
    data_dict,               # 모델 데이터 딕셔너리
    data_path,               # SEIR 데이터 파일 경로
    start_date='2021-07-01', # 분석 시작 날짜
    end_date='2021-08-01'    # 분석 종료 날짜
)

# %%
