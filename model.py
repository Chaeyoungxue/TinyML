import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dims, use_conv=1, use_attention=1, n_heads=4):
        super(Autoencoder, self).__init__()
        self.use_conv = use_conv
        self.use_attention = use_attention
        self.n_heads = n_heads
        if self.use_conv == 1:
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.Linear1 = nn.Linear(input_dims, 128)
        self.Linear2 = nn.Linear(128, 64)
        self.Linear3 = nn.Linear(64, 64)
        self.Linear4 = nn.Linear(64, 32)
        if self.use_attention == 1:
            self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=n_heads)

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x):
        if self.use_conv ==1:
            x_sequence = x.unsqueeze(1)
            # 卷积层
            conv1_output = self.conv1(x_sequence)
            conv2_output = self.conv2(conv1_output)
            conv3_output = self.conv3(conv2_output)
            conv1_output = conv1_output.squeeze(1)
            conv2_output = conv2_output.squeeze(1)
            conv3_output = conv3_output.squeeze(1)

            x = self.Linear1(x)
            x = x + conv1_output
            x = self.Linear2(x)
            x = x + conv2_output
            x = self.Linear3(x)
            x = self.Linear4(x)
            x = x + conv3_output
        else:
            x = self.Linear1(x)
            x = self.Linear2(x)
            x = self.Linear3(x)
            x = self.Linear4(x)

        if self.use_attention == 1:
            x = x.unsqueeze(0)
            # 多头注意力机制
            attn_output, _ = self.attention(x, x, x)
            attn_output = attn_output.squeeze(0)
            # 解码器
            x = self.decoder(attn_output)
        else:
            x = self.decoder(x)
        return x