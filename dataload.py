import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def plot_confusion_matrix(y_true, y_pred, class_names=None, 
                          title='Confusion Matrix', 
                          cmap='Blues', 
                          normalize=False, 
                          show_percentage=False,
                          figsize=(8, 6),
                          save_path=None):

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, 2)
    

    _, ax = plt.subplots(figsize=figsize)
    

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    

    values_format = '.2f' if normalize else 'd'
    if show_percentage and not normalize:
        values_format = None  

    disp.plot(
        ax=ax,
        cmap=cmap,
        values_format=values_format,
        colorbar=False,
    )
    

    if show_percentage and not normalize:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                total = cm.sum(axis=1)[i]
                percentage = f"{cm[i, j]/total:.1%}" if total > 0 else "0%"
                ax.text(j, i, f"{cm[i, j]}\n({percentage})",
                        ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
    

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    

    plt.tight_layout()
    

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def load_classification_data(folder_path, N_per_class,snr=None, noise_type='white',seed=1,):
    """
    Args:
        folder_path: Folder path containing CSV files
        N_per_class: The number of randomly selected samples for each class
        snr: signal-to-noise ratio (If there is no noise, please fill in None)
        noise_type: Noise type(white,impulse,pink)
    
    return:
        X: data shape:(n_samples, n_features)
        y: label shape:(n_samples,)
    """
    

    random.seed(seed)
    np.random.seed(seed)
    

    all_data = []
    all_labels = []
    class_names = []
    

    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        raise ValueError(f"No files with CSV suffix found")

    file_data = []
    all_rows_count = []
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path, header=None)
        file_data.append(df)
        all_rows_count.append(len(df))
    if snr is not None:
        combined_df = pd.concat(file_data, ignore_index=True)
        np_array = combined_df.to_numpy()
        np_array=addnoise(np_array,snr,noise_type)+np_array
        combined_df = pd.DataFrame(np_array)
        split_dfs = []
        start_idx = 0
        for rows in all_rows_count:
            end_idx = start_idx + rows
            split_df = combined_df.iloc[start_idx:end_idx]
            split_dfs.append(split_df)
            start_idx = end_idx
        file_data=split_dfs

    for data,csv_file in zip(file_data,csv_files):
        class_name = csv_file[:-4]
        class_names.append(class_name)

        
        try:
            df = data

            total_rows = len(df)
            
            if total_rows == 0:
                print(f"Warning: The file {csv_file} is empty")
                continue
                
            if total_rows < N_per_class:
                print(f"Warning: Class {class_name} only has {total_rows} rows, read all")
                sampled_indices = list(range(total_rows))
            else:

                sampled_indices = random.sample(range(total_rows), N_per_class)

            sampled_data = df.iloc[sampled_indices].values
            

            labels = [class_name] * len(sampled_data)

            all_data.append(sampled_data)
            all_labels.extend(labels)

            
        except Exception as e:
            print(f"Error reading file {csv_file}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data was successfully read")
    
    X = np.vstack(all_data)
    y = np.array(all_labels)
    
    unique_classes = np.unique(y)
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    y_encoded = np.array([class_to_idx[label] for label in y])
    y_encoded = np.array([class_to_idx[label] for label in y])
    
    return X, y_encoded


def addnoise(X,snr,noise_type):
    '''
    Args:
        snr: signal-to-noise ratio (If there is no noise, please fill in None)
        noise_type: Noise type(white,impulse,pink)
    '''
    total_signal_power = np.mean(X**2)
    
    if total_signal_power == 0:
        raise ValueError("The total signal power is zero, and SNR cannot be calculated")
    
    snr_linear = 10**(snr / 10)
    total_noise_power = total_signal_power / snr_linear
    if noise_type=='white':
        noise = np.random.normal(0, np.sqrt(total_noise_power), X.shape)
    elif noise_type=='impulse':
        L = X.shape[-1]
        n_imp = int(0.08 * L)

        noise = np.zeros(X.shape, dtype=np.float32)
        idx = np.random.rand(*noise.shape).argpartition(-n_imp, axis=-1)[..., -n_imp:]
        np.put_along_axis(noise, idx, np.random.randn(*idx.shape) * 5, axis=-1)

        noise = noise / np.sqrt(np.mean(noise ** 2)) * np.sqrt(total_noise_power)
    elif noise_type=='pink':
        try:
            size = list(X.shape)
        except TypeError:
            size = [X.shape]
        samples = size[-1]
        f = np.fft.rfftfreq(samples)
        fmin =  1./samples
        s_scale = f
        ix = np.sum(s_scale < fmin)
        if ix and ix < len(s_scale):
            s_scale[:ix] = s_scale[ix]
        s_scale = s_scale**(-1/2.)
        w = s_scale[1:].copy()
        w[-1] *= (1 + (samples % 2)) / 2.
        sigma = 2 * np.sqrt(np.sum(w**2)) / samples
        size[-1] = len(f)
        dims_to_add = len(size) - 1
        s_scale = s_scale[(None,) * dims_to_add + (Ellipsis,)]
        rng = np.random.default_rng()
        sr = rng.normal(scale=s_scale, size=size)
        si = rng.normal(scale=s_scale, size=size)
        if not (samples % 2):
            si[..., -1] = 0
            sr[..., -1] *= np.sqrt(2)
        si[..., 0] = 0
        sr[..., 0] *= np.sqrt(2)
        s = sr + 1J * si
        pink_noise = np.fft.irfft(s, n=samples, axis=-1) / sigma
        noise=pink_noise/ np.std(pink_noise) * np.sqrt(total_noise_power)
    else:
        raise Exception("please check the noisy name")
    
    return noise


def split_data_with_stratify(X, y, rates, random_state=42):
    """
    Args:
        X: data shape(N, L)
        y: label shape(N,)
        rates: ratio proportion (train_ratio, valid_ratio, test_ratio) The total proportion should be 1
    
    return:
        X_train, X_valid, X_test, y_train, y_valid, y_test
    """
    

    assert abs(sum(rates) - 1.0) < 1e-10, "The total proportion should be 1"
    
    train_ratio, valid_ratio, test_ratio = rates

    
    if test_ratio!=0: 

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_ratio,
            stratify=y,
            random_state=random_state
        )
    else:
        X_temp, X_test, y_temp, y_test = X,[], y,[] #The test set is empty
    if valid_ratio!=0: 
        temp_train_ratio = train_ratio + valid_ratio
        valid_in_temp_ratio = valid_ratio / temp_train_ratio
        
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_temp, y_temp,
            test_size=valid_in_temp_ratio,
            stratify=y_temp,
            random_state=random_state
        )
    else:
        X_train, X_valid, y_train, y_valid=X_temp, [], y_temp, [] #The valid set is empty
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test
def prepro(folder_path, N,rates,snr=None,noise_type='white'):
    """
    Args:
        folder_path: The dataset folder
        N: The number of each class
        rates: ratio proportion (train_ratio, valid_ratio, test_ratio) The total proportion should be 1
        snr: signal-to-noise ratio (If there is no noise, please fill in None)
        noise_type: Noise type(white,impulse,pink)

    return:
        X_train, X_valid, X_test, y_train, y_valid, y_test
    """
    X,Y=load_classification_data(folder_path,N,snr,noise_type)
    train,valid,test,trainlabel,valid_label,testlabel=split_data_with_stratify(X,Y,rates)
    return train,valid,test,trainlabel,valid_label,testlabel