import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import os
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from model import DeepAutoencoder, SimpleAutoencoder, IsolationForestModel, LOFModel
from utils import generate_dataset, load_sound_file, extract_signal_features

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


default_config_path = 'config.yaml'
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--config', type=str, default=default_config_path, help='Path to config file')
parser.add_argument('--dataset', type=str,default='slider',help='Name of the dataset')
parser.add_argument('--train_dir', type=str, default=r'.\\Data\\{dataset}\\train',help='Path to training directory')
parser.add_argument('--test_normal_dir', type=str, default=r'.\\Data\\{dataset}\\test\\no',help='Path to normal test directory')
parser.add_argument('--test_anomaly_dir', type=str, default=r'.\\Data\\{dataset}\\test\\ab',help='Path to anomaly test directory')
parser.add_argument('--n_mels', type=int,default=64, help='Number of mel bands')
parser.add_argument('--frames', type=int, default=5,help='Number of frames')
parser.add_argument('--n_fft', type=int, default=1024,help='FFT window size')
parser.add_argument('--hop_length', type=int, default=512,help='Hop length')
parser.add_argument('--num_epochs', type=int, default=40,help='Num epochs')
parser.add_argument('--use_conv', type=int, default=1,help='Whether to use conv layers')
parser.add_argument('--use_attention', type=int, default=1,help='Whether to use multi-head attention')
parser.add_argument('--n_heads', type=int, default=4,help='Number of heads in multi-head attention')
parser.add_argument('--model_type', type=str, default=None, help='Model type: deep_autoencoder, simple_autoencoder, isolation_forest, lof')
parser.add_argument('--hidden_dims', type=int, default=None, help='Hidden dimensions for simple autoencoder')
parser.add_argument('--n_estimators', type=int, default=None, help='Number of estimators for Isolation Forest')
parser.add_argument('--n_neighbors', type=int, default=None, help='Number of neighbors for LOF')

args = parser.parse_args()

# 加载配置文件
config = load_config(args.config)
if args.dataset:
    config['dataset'] = args.dataset
config['train_dir'] = config['train_dir'].format(dataset=config['dataset'])
config['test_normal_dir'] = config['test_normal_dir'].format(dataset=config['dataset'])
config['test_anomaly_dir'] = config['test_anomaly_dir'].format(dataset=config['dataset'])

if args.n_mels is not None:
    config['n_mels'] = args.n_mels
if args.frames is not None:
    config['frames'] = args.frames
if args.n_fft is not None:
    config['n_fft'] = args.n_fft
if args.hop_length is not None:
    config['hop_length'] = args.hop_length
if args.num_epochs is not None:
    config['num_epochs'] = args.num_epochs
if args.use_conv is not None:
    config['use_conv'] = args.use_conv
if args.use_attention is not None:
    config['use_attention'] = args.use_attention
if args.n_heads is not None:
    config['n_heads'] = args.n_heads
if args.model_type is not None:
    config['model_type'] = args.model_type
if args.hidden_dims is not None:
    config['hidden_dims'] = args.hidden_dims
if args.n_estimators is not None:
    config['n_estimators'] = args.n_estimators
if args.n_neighbors is not None:
    config['n_neighbors'] = args.n_neighbors

# 打印最终的配置
print("Final Configuration:")
for key, value in config.items():
    print(f"{key}: {value}")

train_files = [os.path.join(config['train_dir'], file) for file in os.listdir(config['train_dir']) if file.endswith('.wav')]
test_normal_files = [os.path.join(config['test_normal_dir'], file) for file in os.listdir(config['test_normal_dir']) if
                     file.endswith('.wav')]
test_anomaly_files = [os.path.join(config['test_anomaly_dir'], file) for file in os.listdir(config['test_anomaly_dir']) if
                      file.endswith('.wav')]

X_train = generate_dataset(train_files, n_mels=config['n_mels'], frames=config['frames'], n_fft=config['n_fft'], hop_length=config['hop_length'])
X_test_normal = generate_dataset(test_normal_files, n_mels=config['n_mels'], frames=config['frames'], n_fft=config['n_fft'], hop_length=config['hop_length'])
X_test_anomaly = generate_dataset(test_anomaly_files, n_mels=config['n_mels'], frames=config['frames'], n_fft=config['n_fft'], hop_length=config['hop_length'])

X_train_tensor = torch.FloatTensor(X_train)
train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)


input_shape = config['n_mels'] * (config['frames']-1)

# 根据配置选择模型
if config['model_type'] == 'deep_autoencoder':
    model = DeepAutoencoder(input_shape, use_conv=config['use_conv'], use_attention=config['use_attention'], n_heads=config['n_heads'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练深度自编码器
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        epochs = config['num_epochs']
        for batch in train_loader:
            inputs, targets = batch
            cen_inputs = inputs[:, config['n_mels'] * (config['frames']//2):config['n_mels'] * (config['frames']//2+1)]
            cen_targets = targets[:, config['n_mels'] * (config['frames']//2):config['n_mels'] * (config['frames']//2+1)]
            inputs = torch.cat((inputs[:, :config['n_mels'] * (config['frames']//2)], inputs[:, config['n_mels'] * (config['frames']//2+1):]), dim=1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, cen_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')

elif config['model_type'] == 'simple_autoencoder':
    model = SimpleAutoencoder(input_shape, hidden_dims=config['hidden_dims'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    # 使用fit方法训练简单自编码器
    model.fit(X_train, epochs=config['num_epochs'])

elif config['model_type'] == 'isolation_forest':
    model = IsolationForestModel(
        input_dims=input_shape,
        n_estimators=config['n_estimators'],
        contamination=config['contamination']
    )
    # 训练隔离森林模型
    model.fit(X_train)
    print("Isolation Forest model trained successfully")

elif config['model_type'] == 'lof':
    model = LOFModel(
        input_dims=input_shape,
        n_neighbors=config['n_neighbors'],
        contamination=config['contamination']
    )
    # 训练LOF模型
    model.fit(X_train)
    print("LOF model trained successfully")

else:
    raise ValueError(f"Unknown model type: {config['model_type']}")

print(f"Model type: {config['model_type']} initialized and trained successfully")

# 计算异常分数的函数，适用于所有模型类型
def get_anomaly_scores(model, test_files, test_labels, n_mels, frames, n_fft, hop_length):
    anomaly_scores = []
    
    # 对于深度自编码器模型的特殊处理
    if model.model_type == 'deep_autoencoder':
        model.eval()
        with torch.no_grad():
            for eval_filename in tqdm(test_files, total=len(test_files)):
                signal, sr = load_sound_file(eval_filename)
                eval_features = extract_signal_features(
                    signal,
                    sr,
                    n_mels=n_mels,
                    frames=frames,
                    n_fft=n_fft)

                features_tensor = torch.FloatTensor(eval_features)
                cen_test = features_tensor[:, n_mels * (frames//2):n_mels * (frames//2+1)]
                features_inputs = torch.cat((features_tensor[:, :n_mels * (frames//2)], features_tensor[:, n_mels * (frames//2+1):]), dim=1)
                prediction = model(features_inputs)

                mse = torch.mean(torch.mean((cen_test - prediction) ** 2, dim=1))
                anomaly_scores.append(mse.item())
    
    # 对于简单自编码器模型
    elif model.model_type == 'simple_autoencoder':
        model.eval()
        with torch.no_grad():
            for eval_filename in tqdm(test_files, total=len(test_files)):
                signal, sr = load_sound_file(eval_filename)
                eval_features = extract_signal_features(
                    signal,
                    sr,
                    n_mels=n_mels,
                    frames=frames,
                    n_fft=n_fft)

                features_tensor = torch.FloatTensor(eval_features)
                prediction = model(features_tensor)
                mse = torch.mean(torch.mean((features_tensor - prediction) ** 2, dim=1))
                anomaly_scores.append(mse.item())
    
    # 对于Isolation Forest和LOF模型
    elif model.model_type in ['isolation_forest', 'lof']:
        for eval_filename in tqdm(test_files, total=len(test_files)):
            signal, sr = load_sound_file(eval_filename)
            eval_features = extract_signal_features(
                signal,
                sr,
                n_mels=n_mels,
                frames=frames,
                n_fft=n_fft)
            
            # 获取每个文件的平均异常分数
            file_scores = model.get_anomaly_score(eval_features)
            anomaly_scores.append(np.mean(file_scores))
    
    return anomaly_scores


test_files = test_normal_files + test_anomaly_files
test_labels = [0] * len(test_normal_files) + [1] * len(test_anomaly_files)

reconstruction_errors = get_anomaly_scores(model, test_files, test_labels, config['n_mels'], config['frames'], config['n_fft'], config['hop_length'])
"""
thresh = np.percentile(reconstruction_errors, 50)
pred_labels = np.array(reconstruction_errors) > thresh
"""


# 计算 AUC 分数
precision, recall, _ = precision_recall_curve(test_labels, reconstruction_errors)
auc_score = auc(recall, precision)
print("AUC Score:", auc_score)
dataset_str=str(config['dataset'])
conv=str(config['use_conv'])
attention=str(config['use_attention'])
n_heads=str(config['n_heads'])

name=dataset_str+'_'+conv+'_'+attention+'_'+n_heads
save_dir = './results/' + name + '/'
os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，则创建
# 准备数据
data = np.column_stack((range(len(reconstruction_errors)), reconstruction_errors))
bin_width = 0.25
bins = np.arange(min(reconstruction_errors), max(reconstruction_errors) + bin_width, bin_width)

# 绘制直方图
fig = plt.figure(figsize=(10, 6), dpi=100)
plt.hist(data[np.array(test_labels) == 0][:, 1], bins=bins, color='b', label='Normal signals', edgecolor='#FFFFFF')
plt.hist(data[np.array(test_labels) == 1][:, 1], bins=bins, color='r', label='Anomaly signals', edgecolor='#FFFFFF')

# 添加阈值线
#plt.axvline(x=thresh, color='red', linestyle='--', label=f'Threshold: {thresh:.2f}')

# 添加标签和标题
plt.xlabel("MSE")
plt.ylabel("# Samples")
plt.title('Reconstruction error distribution on the testing set', fontsize=16)
plt.legend()
plt.savefig(os.path.join(save_dir, 'MSE.png'), dpi=100)

fpr, tpr, _ = roc_curve(test_labels, reconstruction_errors)

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.text(0.6, 0.4, 'AUC = %0.2f' % auc_score, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.legend(loc="lower right")
plt.savefig(os.path.join(save_dir, 'ROC.png'), dpi=100)
