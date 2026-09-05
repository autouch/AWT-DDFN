import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import dataload
import numpy as np
from ptflops import get_model_complexity_info
from models.mymodel import AWT_DDFN
import sys
import os
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

class Logger(object):
    def __init__(self, filename="terminal_output.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush() 

def main(snr,save_path,noise_type,device):
    fig_path=save_path+'fig/'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path, exist_ok=True)
    save_model_path=save_path+'saved_model/'
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path, exist_ok=True)
    train_x, valid_x, test_x,train_y,valid_y, test_y=dataload.prepro("AWTL-DDFN1/N15_M07_F10",1000,[0.6,0.2,0.2],snr,noise_type)
    classes=11
    model = AWT_DDFN(classes=classes).to(device)


    class VibrationDataset(Dataset):
        def __init__(self, features, labels):
            self.features = features.unsqueeze(-2)
            self.labels = labels
            
        def __len__(self):
            return len(self.features)
        
        def __getitem__(self, idx):
            return self.features[idx],self.labels[idx]
        
    macs, params = get_model_complexity_info(model, (1, 2048), as_strings=False, print_per_layer_stat=False)

    [train_x, train_y, test_x, test_y,valid_x,valid_y]=[torch.from_numpy(train_x).to(torch.float),torch.from_numpy(train_y).long(),torch.from_numpy(test_x).to(torch.float),torch.from_numpy(test_y).long(),torch.from_numpy(valid_x).to(torch.float),torch.from_numpy(valid_y).long()]

    config = {
        'batch_size': 128,
        'lr': 3e-4,
        'epochs': 300,
    }

    train_dataset = VibrationDataset(train_x,train_y)
    valid_dataset = VibrationDataset(valid_x,valid_y)
    test_dataset = VibrationDataset(test_x,test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    print(f"FLOPs: {macs}")
    print(f"parameters: {params}")
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = CosineAnnealingLR(
    optimizer, 
    T_max=200,        
    eta_min=1e-5     
)
    criterion = nn.CrossEntropyLoss()
    lossfunction=criterion
    def train():
        model.train()
        for signals,labels in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            q_values= model(signals)
            loss = lossfunction(q_values, labels)
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(),      1)
            optimizer.step()
        scheduler.step()

    def evaluate(data_loader,dataset):
        model.eval()
        correct = 0
        total_loss = 0
        true=[]
        predict=[]
        with torch.no_grad():
            for signals, labels in data_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                outputs= model(signals)
                loss = lossfunction(outputs, labels)
                total_loss += loss.item()*outputs.shape[0]
                true+=labels.cpu().tolist()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                predict+=predicted.cpu().tolist()
        return np.array(true),np.array(predict),total_loss / len(dataset)

    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(config['epochs']):
        train()
        valid_labels,valid_predicted,valid_loss = evaluate(valid_loader,valid_dataset)
        valid_accuracy=accuracy_score(valid_labels,valid_predicted)
        if (len(valid_accuracies) == 0 or valid_accuracy>=max(valid_accuracies)):
            best_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
            }
            torch.save(best_checkpoint, save_model_path+'model_best_weights.pth')
        test_labels,test_predicted,test_loss = evaluate(test_loader,test_dataset)
        test_accuracy=accuracy_score(test_labels,test_predicted)
        valid_accuracies.append(valid_accuracy)
        valid_losses.append(valid_loss)
        labels,predicted,train_loss = evaluate(train_loader,train_dataset)
        train_accuracy=accuracy_score(labels,predicted,)
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        print(f"Epoch [{epoch+1}/{config['epochs']}] Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_accuracy:.2%}| Valid Acc: {valid_accuracy:.2%} | Test Acc: {test_accuracy:.2%}")
    last_checkpoint = {
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
    }
    torch.save(last_checkpoint, save_model_path+'model_last_weights.pth')
    best_checkpoint = torch.load(save_model_path+'model_best_weights.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    test_labels,test_predicted,_=evaluate(test_loader,test_dataset)
    accuracy=accuracy_score(test_labels,test_predicted)
    precision,recall,f1,_=precision_recall_fscore_support(test_labels,test_predicted,average='macro')
    print("best_model test result:")
    print(f"Acc: {accuracy:.2%},Pre: {precision:.2%},Rec: {recall:.2%},F1: {f1:.2%}")
    dataload.plot_confusion_matrix(test_labels,test_predicted,save_path=fig_path+'con_best.png')
    print(f"best model epoch:{best_checkpoint['epoch']}")
    last_checkpoint = torch.load(save_model_path+'model_last_weights.pth')
    model.load_state_dict(last_checkpoint['model_state_dict'])
    test_labels,test_predicted,_=evaluate(test_loader,test_dataset)
    dataload.plot_confusion_matrix(test_labels,test_predicted,save_path=fig_path+'con_last.png')
    accuracy=accuracy_score(test_labels,test_predicted)
    precision,recall,f1,_=precision_recall_fscore_support(test_labels,test_predicted,average='macro')
    print("last_model test result:")
    print(f"Acc: {accuracy:.2%},Pre: {precision:.2%},Rec: {recall:.2%},F1: {f1:.2%}")

    plt.figure(figsize=(6, 5))
    plt.plot(list(range(0, config['epochs'])), train_accuracies, color='b', label='Training Acc')
    plt.plot(list(range(0, config['epochs'])), valid_accuracies, color='r', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path+'acc_plot.png',dpi=1000, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(list(range(0, config['epochs'])), train_losses, color='b', label='Training Loss')
    plt.plot(list(range(0, config['epochs'])), valid_losses, color='r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path+'loss_plot.png',dpi=1000, bbox_inches='tight')
    plt.close()
if __name__ == "__main__":
    loops=1
    snr_list=list(np.arange(-10,3,3))
    noise_type='white'
    original_stdout = sys.stdout
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for snr in snr_list:
        for loop in range(loops):
            sys.stdout = original_stdout
            save_path='AWTL-DDFN1/result/'+noise_type+'/'+str(snr)+'/'+str(loop)+'/'
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            sys.stdout = Logger(save_path+"logs.txt")
            print(snr,loop)
            main(snr,save_path,noise_type,device)