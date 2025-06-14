import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from scipy.signal import butter, filtfilt
from scipy.signal import firwin
import numpy as np

# -------------------------------------------------------------
# 1. Feature Extractor (CBAM + Sub‑band + depthwise + Variance)
# -------------------------------------------------------------
class ChannelAttention(nn.Module):
    """
    채널 어텐션 모듈 (CBAM의 Channel Gate)
    - 입력: (B, C, T)
    - 출력: (B, C, T) (채널별 스케일링 적용)
    """
    def __init__(self, gate_channels, reduction_ratio=16):
        super().__init__()
        # 1D 풀링: (B, C, T) → (B, C, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        # MLP: C → C/r → C
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, gate_channels // reduction_ratio, bias=True),  # W0
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels, bias=True),  # W1
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        x: Tensor of shape (B, C, T)
        returns: Tensor of shape (B, C, T)
        """
        b, c, t = x.size()
        
        avg = self.avg_pool(x).view(b, c)  # (B, C)
        max = self.max_pool(x).view(b, c)  # (B, C)
        
        avg_out = self.mlp(avg)           # (B, C)
        max_out = self.mlp(max)           # (B, C)
        
        scale = self.sig(avg_out + max_out)      # (B, C)
        
        scale = scale.view(b, c, 1)       # (B, C, 1)
        return x * scale                  # (B, C, T) 브로드캐스트 곱
        
class SpatialAttention(nn.Module):
    """
    공간 어텐션 모듈 (CBAM의 Spatial Gate)
    - 입력: (B, C, T)
    - 출력: (B, C, T) (시간축 1×1 conv 후 스케일링)
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: Tensor, shape (B, C, T)
        returns: Tensor, same shape (B, C, T)
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        concat = torch.cat([avg_out, max_out], dim=1)
        
        scale = self.sigmoid(self.conv(concat))
        return x * scale

class CBAM(nn.Module):
    """
    CBAM 모듈: ChannelAttention + SpatialAttention 순차 적용
    """
    def __init__(self, gate_channels, reduction_ratio=4):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelAttention(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialAttention()

    def forward(self, x):
        x = self.ChannelGate(x)
        x = self.SpatialGate(x)
        return x

class FIRBandPass(nn.Module):
    """
    FFT 기반 서브밴드 필터링 (FIR Hamming 윈도우)
    - subbands: [(low, high), ...]
    - fs: 샘플링 주파수
    - order: 필터 차수 (numtaps = order+1)
    - 입력: (B, C, T)
    - 출력: (B, m, C, T)
    """
    def __init__(self, subbands, fs, T, order=50):
        super().__init__()
        self.subbands = subbands
        self.fs = fs
        self.order = order
        self.numtaps = order + 1

        # 각 밴드별 FIR 필터 탭 계산
        taps_list = []
        for fl, fh in subbands:
            taps = firwin(self.numtaps, [fl, fh], pass_zero=False, fs=fs, window='hamming')
            taps_tensor = torch.tensor(taps, dtype=torch.float32).view(1, 1, self.numtaps)
            taps_list.append(taps_tensor)

        # 버퍼로 등록 (m,1,numtaps)
        self.register_buffer('taps', torch.stack(taps_list, dim=0))

    def forward(self, x):
        """
        x: Tensor of shape (B, C, T)
        returns: Tensor of shape (B, m, C, T)
        where m = number of subbands
        """
        B, C, T = x.shape
        pad = self.numtaps - 1

        # Causal filtering: pad only at the front
        x_padded = F.pad(x, (pad, 0))  # shape (B, C, T + pad)
        outputs = []

        # Apply each subband filter via group convolution
        for taps in self.taps:  # taps: (1, 1, numtaps)
            # Expand to (C, 1, numtaps) so each channel uses the same filter
            kernel = taps.repeat(C, 1, 1)
            # Perform group conv: groups=C ensures channel-wise filtering
            y = F.conv1d(x_padded, kernel, groups=C)
            # y shape: (B, C, T)
            outputs.append(y)

        # Stack outputs: (B, m, C, T)
        return torch.stack(outputs, dim=1)


class DepthwiseConv(nn.Module):
    """
    서브밴드별 depthwise 2D convolution
    - 입력: (B, m, C, T)
    - 출력: (B, m*depth, 1, T)
    """
    def __init__(self, bands: int, depth_multiplier: int, num_electrodes: int, max_norm=1.0):
        super().__init__()
        self.bands = bands               # = in_channels
        self.depth = depth_multiplier   # = depth_multiplier
        # depthwise conv2d: out_channels = in_channels * depth_multiplier
        # groups = in_channels 으로 해야 각 채널마다 별도 필터 적용
        self.max_norm = max_norm
        self.dw = nn.Conv2d(
            in_channels=bands,
            out_channels=bands * depth_multiplier,
            kernel_size=(num_electrodes, 1),
            groups=bands,
            bias=False
        )
        self.bn_dw = nn.BatchNorm2d(bands * depth_multiplier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, m, C, T)
        y = self.dw(x)
        return y

class VarianceLayer(nn.Module):
    """
    비중첩 시간창 분산 계산
    - 입력: (B, C, 1, T)
    - 출력: (B, C, 1, K) where K = T//w
    """
    def __init__(self, w: int):
        super(VarianceLayer, self).__init__()
        self.w = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 차원 정리
        #   x: (B, C, 1, T) → squeeze하여 (B, C, T)
        B, C, _, T = x.shape
        x = x.view(B, C, T)

        # 2) 창 개수 계산 및 불필요한 마지막 부분 자르기
        K = T // self.w
        x = x[:, :, :K * self.w]          # (B, C, K*w)

        # 3) (B, C, K, w) 형태로 뷰 변경
        x = x.view(B, C, K, self.w)

        # 4) 시간평균 µ(k) 계산: (B, C, K, 1)
        mu = x.mean(dim=-1, keepdim=True)

        # 5) 분산 계산: (B, C, K)
        var = ((x - mu) ** 2).mean(dim=-1)
        # 6) 출력 형태 맞추기: (B, C, 1, K)
        y = var.unsqueeze(2)
        return y


class FeatureExtractor(nn.Module):
    """
    전체 특성 추출기:
    1) CBAM
    2) FIRBandpass → (B,m,C,T)
    3) DepthwiseConv → (B,m*depth,1,T)
    4) VarianceLayer → (B,m*depth,1,T//w)
    """
    def __init__(self, C=22, depth=2, subbands=((8,12),(12,16),(16,20),(20,24),(24,28),(28,30)),
                  var_win=250, fs=250, fc_out=64):
        super().__init__()
        self.fs = fs
        self.subbands = subbands
        self.fft_bp = FIRBandPass(self.subbands, fs=self.fs, T=1000)
        
        self.depth_convs = DepthwiseConv(bands=len(subbands), depth_multiplier=depth, num_electrodes=C)
        self.cbams = CBAM(gate_channels=C)
        self.var = VarianceLayer(var_win)
        self.fc = nn.Linear(depth * len(subbands) * (1000 // var_win), fc_out)

    def forward(self, x):   # x: (B, C, T)
        B, C, T = x.shape
        x = self.cbams(x)
        
        feats = self.fft_bp(x)         # (B, m, C, T)
        
        feats = self.depth_convs(feats)             # (B, d, m, 1, T)
        out = self.var(feats)                     # (B, d, m, 1, T//w)

        return out                    

# -------------------------------------------------------------
# 2. Domain Discriminator (Critic)
# -------------------------------------------------------------
# class Critic(nn.Module):
    # """
    # WGAN-GP Critic 네트워크:
    # - 입력: (B, in_dim, 1, T')
    # - 출력: (B,1) critic score
    # """
#     def __init__(self, in_dim, hid=[512,256]):
#         super().__init__()
#         layers = []
#         dims = [in_dim] + hid
#         for in_d, out_d in zip(dims[:-1], dims[1:]):
#             layers += [nn.Linear(in_d, out_d),
#                        nn.LeakyReLU(0.2, True)]
#         layers += [nn.Linear(hid[-1], 1)]
#         self.net = nn.Sequential(*layers)

#     def forward(self, h):
#         return self.net(h).view(-1, 1)

class Critic(nn.Module):
    """
    WGAN-GP Critic 네트워크:
    - 입력: (B, in_dim, 1, T')
    - 출력: (B,1) critic score
    """
    def __init__(self, in_dim: int = 12, hiddens: list = [64, 128]):
        super(Critic, self).__init__()
        layers = []
        prev_channels = in_dim
        # Two downsampling conv2d blocks: kernel (1,2), stride (1,2)
        for feature in hiddens:
            layers.append(
                nn.Conv2d(
                    in_channels=prev_channels,
                    out_channels=feature,
                    kernel_size=(1, 2),
                    stride=(1, 2),
                    padding=(0, 0)
                )
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_channels = feature
        # Final conv to collapse to a single scalar per sample
        layers.append(
            nn.Conv2d(
                in_channels=prev_channels,
                out_channels=1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0)
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x shape: (batch_size, 12, 1, 4)
        Returns: (batch_size,) tensor of critic scores.
        """
        out = self.model(x)
        return out.view(-1, 1)


# Gradient penalty
def gradient_penalty(critic, h_s, h_t, lambda_gp):
    """
    WGAN-GP gradient penalty 계산
    """
    alpha = torch.rand(h_s.size(0), 1, device=h_s.device)
    h_hat = alpha * h_s + (1 - alpha) * h_t
    h_hat.requires_grad_(True)
    d_hat = critic(h_hat)
    grad_out = torch.ones_like(d_hat)
    grad_h = grad(d_hat, h_hat, grad_out, create_graph=True, retain_graph=True)[0]
    return lambda_gp * ((grad_h.norm(2, 1) - 1) ** 2).mean()

# -------------------------------------------------------------
# 3. Classifier (2×FC)
# -------------------------------------------------------------
class Classifier(nn.Module):
    """
    분류기: 2단 FC + softmax
    """
    def __init__(self,in_dim, n_cls, hidden=512):
        super().__init__()
        self.fc1=nn.Linear(in_dim,hidden)
        self.fc2=nn.Linear(hidden,n_cls)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(p=0.6)
    def forward(self,h):
        h = h.flatten(1)  # Flatten to (B, in_dim)
        h = self.fc1(h)
        # h = self.bn1(h)
        # h = F.relu(h)
        # h = self.dropout(h)
        return F.softmax(self.fc2(h), dim=1)

# -------------------------------------------------------------
# 4. DANet wrapper
# -------------------------------------------------------------
class DANet(nn.Module):
    """
    Domain Adaptation Network (F + D + C)
    """
    def __init__(self, C=22, n_cls=4, depth=2, citic_hid=512, classifier_hidden=64, feat_dim=None):
        super().__init__()
        self.F = FeatureExtractor(C=C, depth=depth)
        if feat_dim is None:
            # 만일 누락되면 안전장치
            sample = torch.zeros(1, C, 1000)
            feat_dim = self.F(sample).flatten(1).shape[1]
        # self.D = Critic(feat_dim, hid=citic_hid)
        self.D = Critic(in_dim=depth*6)
        self.C = Classifier(feat_dim, n_cls, hidden=classifier_hidden)
    def forward(self, x):
        h = self.F(x)
        return h, self.D(h), self.C(h)

# -------------------------------------------------------------
# 5. single training iteration
# -------------------------------------------------------------
def train_iter(model, src_x, src_y, tgt_x, opts,
               lambda_gp=10, mu=1.0, critic_steps=5):
    """
    1) Critic 업데이트 (WGAN-GP)
    2) Feature+Classifier 업데이트
    """
    net = model.module if isinstance(model, torch.nn.DataParallel) else model

    Fnet, Clf, Cri = net.F, net.C, net.D
    opt_fc, opt_d = opts
    device = next(Fnet.parameters()).device
    src_x, src_y, tgt_x = src_x.to(device), src_y.to(device), tgt_x.to(device)
    
    model.D.train()
    wd_critic_list, wd_feat_list = [], []
    # ── (i) critic update ─────────────────────────────
    for _ in range(critic_steps):
        # 특징은 gradient 차단
        with torch.no_grad():
            h_s, _, _ = model(src_x)
            h_t, _, _ = model(tgt_x)
            h_s = h_s.detach()
            h_t = h_t.detach()

        wd_critic = model.D(h_s).mean() - model.D(h_t).mean()
        gp = gradient_penalty(Cri, h_s, h_t, lambda_gp)
        loss_d = -(wd_critic) + gp          # gradient ASCENT
        
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()
        
        wd_critic_list.append(wd_critic.item())   # Critic 직후의 wd 기록
    wd_critic_mean = float(np.mean(wd_critic_list))
    
    # ── (ii) feature + classifier update ─────────────
    model.train()
    opt_fc.zero_grad()
    
    h_s = model.F(src_x)
    logit = model.C(h_s)  # Classifier output
    loss_c = F.cross_entropy(logit, src_y)
    
    h_t = model.F(tgt_x)
    wd_feat = model.D(h_s).mean() - model.D(h_t).mean()

    loss_f = (wd_feat * mu) + loss_c
    loss_f.backward()
    opt_fc.step()
    
    # DepthwiseConv 가중치 Clip (max_norm)
    with torch.no_grad():
        w = model.F.depth_convs.dw.weight  # shape: (out_ch, 1, kh, kw)
        # Compute L2 norm over each filter
        w_flat = w.view(w.size(0), -1)
        norms = w_flat.norm(p=2, dim=1, keepdim=True)
        # Clip norms to max_norm
        desired = torch.clamp(norms, max=1.0)
        # Scale weights
        w_flat = w_flat * (desired / (1e-8 + norms))
        model.F.depth_convs.dw.weight.copy_(w_flat.view_as(w))

    wd_feat_list.append(wd_feat.item())   # Feature 단계 직후의 wd 기록

    return {
        'loss_D': loss_d.item(),
        'wd_critic': wd_critic_mean,
        'wd_feat': wd_feat.item(),
        'cls_loss': loss_c.item(),
        'loss_f': loss_f.item()
    }

# 테스트용 main 함수
def main():
    B, C, T = 4, 22, 1000
    x = torch.randn(B, C, T)

    # 1) CBAM 모듈 (4D 더미 입력)
    feat = CBAM(gate_channels=22)(x)
    print("ChannelAttention:", ChannelAttention(gate_channels=22)(x).shape)
    print("SpatialAttention:", SpatialAttention()(x).shape)
    print("CBAM:", feat.shape)

    # 2) Feature Extractor
    feat_extractor = FeatureExtractor(C=C)
    feat = feat_extractor(x)
    print("Feature Extractor output:", feat.shape)  # (B, 64)
    # 3) Critic / Classifier
    critic = Critic(in_dim=feat.shape[1])
    print("Critic output:", critic(feat).shape)      # (B,)
    clf = Classifier(in_dim=feat.flatten(1).shape[1], n_cls=4)
    print("Classifier output:", clf(feat).shape)     # (B,4)

    # 4) DANet end-to-end
    net = DANet(C=C, n_cls=4)
    feat2 = net.F(x)
    print("DANet Feature:", feat2.shape)
    print("DANet Critic:", net.D(feat2).shape)
    print("DANet Classifier:", net.C(feat2).shape)
    h, d, c =net(x)
    print("DANet outputs:", h.shape, d.shape, c.shape)

if __name__ == "__main__":
    main()
