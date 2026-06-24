import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy import interpolate

# ===================== 全局参数【匹配你的合成代码】=====================
# 你的原始光谱点数
RAW_N_POINTS = 1000
# DiffRaman标准图像尺寸 M=32 → 1024像素
M = 32
IMG_POINTS = M * M  # 1024

# DDPM超参 完全沿用清华DiffRaman
T_DIFF = 500
BETA_START = 1e-4
BETA_END = 0.02
LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 800
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 光谱双向转换（适配1000→1024，可逆） =====================
def spec_1000_to_2d(spec_1000: np.ndarray) -> np.ndarray:
    """你的1000点光谱 → 32×32灰度图(1024点)"""
    old_axis = np.linspace(0, 1, RAW_N_POINTS)
    new_axis = np.linspace(0, 1, IMG_POINTS)
    interp_func = interpolate.interp1d(old_axis, spec_1000, fill_value="extrapolate")
    L = interp_func(new_axis)
    min_L, max_L = L.min(), L.max()
    L_norm = (L - min_L) / (max_L + 1e-8 - min_L) * 255
    return L_norm.astype(np.uint8).reshape(M, M)

def img_to_spec_1000(img_2d: np.ndarray) -> np.ndarray:
    """32×32图 → 还原回1000点原始光谱长度"""
    flat_1024 = img_2d.flatten().astype(np.float32)
    old_axis = np.linspace(0, 1, IMG_POINTS)
    new_axis = np.linspace(0, 1, RAW_N_POINTS)
    interp = interpolate.interp1d(old_axis, flat_1024, fill_value="extrapolate")
    return interp(new_axis)

# 图像归一 0~255 → [-1,1]
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

# ===================== 数据集：读取你生成的npy文件 =====================
class CARSRamanNpyDataset(Dataset):
    def __init__(self, cars_np, raman_np):
        self.cars_data = cars_np  # (N, 1000, 1)
        self.raman_data = raman_np # (N, 1000)

    def __len__(self):
        return len(self.cars_data)

    def __getitem__(self, idx):
        cars_1d = self.cars_data[idx, :, 0]
        raman_1d = self.raman_data[idx]
        cond_img = spec_1000_to_2d(cars_1d)
        target_img = spec_1000_to_2d(raman_1d)
        cond = trans(cond_img)
        target = trans(target_img)
        return cond, target

# ===================== UNet / DDPM 模型（无修改，直接复用） =====================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.seq(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mp = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.mp(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(torch.cat([x1, x2], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x): return self.c(x)

class CondUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(2, 64)
        self.d1 = Down(64, 128)
        self.d2 = Down(128, 256)
        self.d3 = Down(256, 512)
        self.u1 = Up(512, 256)
        self.u2 = Up(256, 128)
        self.u3 = Up(128, 64)
        self.out = OutConv(64, 1)
        self.time_emb = nn.Sequential(nn.Linear(1,512), nn.ReLU(), nn.Linear(512,512))
    def forward(self, x, t):
        bs = x.shape[0]
        t_emb = self.time_emb(t.float().unsqueeze(-1)).view(bs,512,1,1)
        x1 = self.inc(x)
        x2 = self.d1(x1) + t_emb[:,:128]
        x3 = self.d2(x2) + t_emb[:,:256]
        x4 = self.d3(x3) + t_emb
        x = self.u1(x4, x3)
        x = self.u2(x, x2)
        x = self.u3(x, x1)
        return self.out(x)

class DDPM:
    def __init__(self):
        self.T = T_DIFF
        self.beta = torch.linspace(BETA_START, BETA_END, self.T).to(DEVICE)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(DEVICE)
    def forward_noise(self, x0, t):
        bs = x0.shape[0]
        a_bar = self.alpha_bar[t].reshape(bs,1,1,1)
        eps = torch.randn_like(x0)
        xt = torch.sqrt(a_bar)*x0 + torch.sqrt(1-a_bar)*eps
        return xt, eps
    def reverse_one_step(self, xt, t, cond, model):
        bs = xt.shape[0]
        input_cat = torch.cat([xt, cond], dim=1)
        eps_pred = model(input_cat, t)
        at = self.alpha[t].reshape(bs,1,1,1)
        abt = self.alpha_bar[t].reshape(bs,1,1,1)
        bt = self.beta[t].reshape(bs,1,1,1)
        mu = 1/torch.sqrt(at) * (xt - bt / torch.sqrt(1-abt) * eps_pred)
        if t[0] > 0:
            z = torch.randn_like(xt)
            return mu + torch.sqrt(bt)*z
        return mu
    def sample_raman(self, cond_img, model):
        bs = cond_img.shape[0]
        xt = torch.randn((bs,1,M,M), device=DEVICE)
        for tidx in range(self.T-1, -1, -1):
            t = torch.full((bs,), tidx, device=DEVICE)
            xt = self.reverse_one_step(xt, t, cond_img, model)
        return xt

# ===================== 训练流程（读取你生成的npy） =====================
def train_diffusion():
    print(f"Train device: {DEVICE}")
    # 读取你用代码生成的数据集
    cars_all = np.load("synthetic_data/cars_10000.npy")
    raman_all = np.load("synthetic_data/raman_10000.npy")
    train_ds = CARSRamanNpyDataset(cars_all, raman_all)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = CondUNet().to(DEVICE)
    ddpm = DDPM()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for ep in range(EPOCHS):
        total_loss = 0.0
        for cond, x0 in train_loader:
            cond, x0 = cond.to(DEVICE), x0.to(DEVICE)
            bs = x0.shape[0]
            t = torch.randint(0, T_DIFF, (bs,), device=DEVICE)
            xt, eps_real = ddpm.forward_noise(x0, t)
            input_cat = torch.cat([xt, cond], dim=1)
            eps_pred = model(input_cat, t)
            loss = loss_fn(eps_pred, eps_real)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        if (ep+1) % 20 == 0:
            print(f"Epoch {ep+1:4d} | MSE Loss: {avg_loss:.6f}")
        if (ep+1) % 100 == 0:
            torch.save(model.state_dict(), f"diff_ep{ep+1}.pth")
    torch.save(model.state_dict(), "diff_final.pth")
    print("训练完成，权重 diff_final.pth")

# ===================== 批量生成扩充数据集（输出1000点，适配DA-DMD） =====================
def generate_new_pairs(gen_count=5000, weight="diff_final.pth"):
    ddpm = DDPM()
    model = CondUNet().to(DEVICE)
    model.load_state_dict(torch.load(weight, map_location=DEVICE))
    model.eval()

    # 加载原始cars用来做条件输入
    raw_cars = np.load("synthetic_data/cars_10000.npy")
    out_cars = []
    out_raman = []

    with torch.no_grad():
        for i in range(gen_count):
            # 随机抽取一条真实生成的CARS作为条件
            rand_idx = np.random.randint(0, len(raw_cars))
            car_1000 = raw_cars[rand_idx, :, 0]
            cond_img = spec_1000_to_2d(car_1000)
            cond_tensor = trans(cond_img).unsqueeze(0).to(DEVICE)

            # 扩散生成拉曼图像
            pred_img = ddpm.sample_raman(cond_tensor, model)
            pred_img = pred_img.squeeze().cpu()
            pred_img = (pred_img * 0.5 + 0.5) * 255
            pred_img = pred_img.numpy().astype(np.uint8)

            # 还原回1000点光谱（和你原始数据维度完全一致）
            raman_1000 = img_to_spec_1000(pred_img)
            out_cars.append(car_1000)
            out_raman.append(raman_1000)

    # 保存格式和你原始完全一致
    out_cars = np.array(out_cars)[:, None]  # (N,1000,1)
    out_raman = np.array(out_raman)         # (N,1000)
    np.save("synthetic_data/expand_cars.npy", out_cars)
    np.save("synthetic_data/expand_raman.npy", out_raman)
    print(f"生成{gen_count}组扩充数据，保存在synthetic_data文件夹")

# ===================== 合并原始+扩充，直接给DA-DMD使用 =====================
def merge_for_dadmd():
    # 原始1万数据
    raw_cars = np.load("synthetic_data/cars_10000.npy")
    raw_raman = np.load("synthetic_data/raman_10000.npy")
    # 扩散新增数据
    add_cars = np.load("synthetic_data/expand_cars.npy")
    add_raman = np.load("synthetic_data/expand_raman.npy")
    # 拼接
    all_cars = np.concatenate([raw_cars, add_cars], axis=0)
    all_raman = np.concatenate([raw_raman, add_raman], axis=0)
    np.save("synthetic_data/mix_cars.npy", all_cars)
    np.save("synthetic_data/mix_raman.npy", all_raman)
    print(f"合并后总样本量: {len(all_cars)}")

if __name__ == "__main__":
    # 第一步：先运行你自己的生成代码，产出 cars_10000.npy / raman_10000.npy
    # 第二步：执行训练扩散
    train_diffusion()

    # 训练完取消注释生成扩充数据
    # generate_new_pairs(gen_count=5000)

    # 生成完成后合并数据集给DA-DMD
    # merge_for_dadmd()