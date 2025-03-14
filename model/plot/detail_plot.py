#%%
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib.patches import Rectangle


#%%
def plot_h_beta():
    """
    Plot the (-1,-1) element of h matrix multiplied by beta for MPUGAT model
    """
    # Extract h values and beta for MPUGAT
    mpugat_h = data_dict["MPUGAT"]["contact"]
    
    # Get beta values
    beta_train = data_dict["MPUGAT"]["train"][:, :, 0, 0,0, 2]  
    beta_val = data_dict["MPUGAT"]["validate"][:, :, 0, 0,0, 2]
    beta_test = data_dict["MPUGAT"]["test"][:, :, 0, 0,0, 2]
    beta = np.concatenate([beta_train, beta_val, beta_test], axis=1)
    
    # Get the (-1,-1) elements
    mpugat_h_last = mpugat_h[:,:,0,0]  # Shape: (30, simulation_days)
    #mpugat_h_last = np.sum(mpugat_h[:,:,0,1:],axis=-1)  # Shape: (30, simulation_days)
    # Multiply h by beta
    #mpugat_h_last = np.sum(mpugat_h_last, axis=-1)
    h_times_beta = mpugat_h_last #* beta
    
    # Calculate mean and std
    result_mean = np.mean(mpugat_h_last, axis=0)
    result_std = np.std(h_times_beta, axis=0)/np.sqrt(30)
    
    # Create date range for x-axis
    end_date = datetime.datetime.strptime("2021-11-23", "%Y-%m-%d").date()
    cutoff_date = end_date - datetime.timedelta(days=len(result_mean) - 1)
    date_range = [cutoff_date + datetime.timedelta(days=x) for x in range(len(result_mean))]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot result
    plt.plot(date_range, result_mean, label='MPUGAT h(-1,-1) × β', color='red')
    plt.fill_between(date_range, 
                     result_mean - 1.64 * result_std,
                     result_mean + 1.64 * result_std,
                     color='red', alpha=0.2)
    
    plt.title('H Matrix (0,0,) Element × β Over Time')
    plt.xlabel('Date')
    plt.ylabel('H × β Value')
    plt.legend()
    
    # 2달 간격으로 x축 설정
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("./seed/h_beta_comparison.png", bbox_inches='tight', dpi=300)
    plt.show()

    # Print some statistics
    print("\nStatistics for H(-1,-1) × β:")
    print(f"Mean: {np.mean(result_mean):.4f}")
    print(f"Std: {np.mean(result_std):.4f}")

# Run the function
plot_h_beta()
#%%
base_path = './seed/'
for pred_len in [3]:  # , 21
    data_dict = {} 
    for model in ["deep_learning","compartment","C_ij", "MPGAT","MPUGAT"]:
        filename = f'preds_acts_{model}_{pred_len}.npy'
        full_path = base_path + filename
        data = np.load(full_path, allow_pickle=True).item()
        data_dict[model] = data


# %%
