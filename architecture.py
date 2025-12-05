import torch
import torch.nn as nn

class LayerNorm2d(nn.Module):
    """
    LayerNorm, який працює з тензорами зображень (N, C, H, W).
    У PyTorch звичайний LayerNorm чекає канали в кінці, тому ми робимо permute.
    """
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        # Еквівалент (x * weight + bias) але для 4D тензора
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class SimpleGate(nn.Module):
    """
    Ключова фішка NAFNet. Замість ReLU/GELU.
    Ділить канали навпіл і перемножує їх.
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    """
    Основний будівельний блок.
    Містить Simplified Channel Attention (SCA) та LayerScale.
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel) # Depthwise

        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)

        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1)
        )

        # SimpleGate
        self.sg = SimpleGate()

        # Feed Forward Network (FFN) part
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # Layer Scale (параметри, що навчаються, для стабільності)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        # Частина 1: Spatial Mixing
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x) # Множення на увагу
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta # Residual Connection 1

        # Частина 2: Channel Mixing (FFN)
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma # Residual Connection 2

class NAFNet(nn.Module):
    """
    Збираємо все в U-Net подібну структуру.
    """
    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1], dec_blk_nums=[1, 1, 1]):
        super().__init__()

        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width

        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2)) # Downsampling (stride 2)
            chan = chan * 2

        # Middle
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        # Decoder
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2) # Upsampling
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

    def forward(self, inp):
        # 1. Зберігаємо вхід для додавання в кінці (Global Residual)
        # inp - це ваша зашумлена картинка

        # 2. Початок обробки (Feature Extraction)
        x = self.intro(inp)

        # 3. Енкодер (стиснення + обробка)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # 4. Середина (Bottleneck)
        x = self.middle_blks(x)

        # 5. Декодер (відновлення розміру)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip  # Skip connection всередині U-Net
            x = decoder(x)

        # 6. Фінальна проекція у 3 канали
        x = self.ending(x)

        # 7. ГОЛОВНИЙ МОМЕНТ:
        # Ми додаємо вхід (inp) до результату мережі (x).
        # Це означає, що мережа (x) вивчила "негативний шум".
        # Result = NoisyImage + (Correction)
        return x + inp