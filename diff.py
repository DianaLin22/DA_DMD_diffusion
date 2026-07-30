import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy import interpolate

# ====================== Global Hyperparameters (Shared Base Settings) ======================
RAW_N_POINTS = 656
M = 32
IMG_POINTS = M * M

# Common diffusion hyperparameters for DDPM
T_DIFF = 500
BETA_START = 1e-4
BETA_END = 0.02
LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====================== [Original 2D Spectrum <-> Image Conversion Fully Preserved] ======================
def spec_1000_to_2d(spec_1000: np.ndarray) -> np.ndarray:
    old_axis = np.linspace(0, 1, RAW_N_POINTS)
    new_axis = np.linspace(0, 1, IMG_POINTS)
    interp_func = interpolate.interp1d(old_axis, spec_1000, fill_value="extrapolate")
    L = interp_func(new_axis)
    min_L, max_L = L.min(), L.max()
    L_norm = (L - min_L) / (max_L + 1e-8 - min_L) * 255
    return L_norm.astype(np.uint8).reshape(M, M)

def img_to_spec_1000(img_2d: np.ndarray) -> np.ndarray:
    flat_1024 = img_2d.flatten().astype(np.float32)
    old_axis = np.linspace(0, 1, IMG_POINTS)
    new_axis = np.linspace(0, 1, RAW_N_POINTS)
    interp = interpolate.interp1d(old_axis, flat_1024, fill_value="extrapolate")
    return interp(new_axis)

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

# 2D Custom Dataset Class for CARS & Raman Spectra Pairs
class CARSRamanNpyDataset(Dataset):
    def __init__(self, cars_np, raman_np):
        self.cars_data = cars_np
        self.raman_data = raman_np

    def __len__(self):
        return len(self.cars_data)

    def __getitem__(self, idx):
        cars_1d = self.cars_data[idx, :]
        raman_1d = self.raman_data[idx]
        cond_img = spec_1000_to_2d(cars_1d)
        target_img = spec_1000_to_2d(raman_1d)
        cond = trans(cond_img)
        target = trans(target_img)
        return cond, target

# 2D UNet Building Blocks (Unmodified Original Implementation)
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
        # Time step embedding layer
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

# 2D DDPM Diffusion Core Class
class DDPM:
    def __init__(self):
        self.T = T_DIFF
        self.beta = torch.linspace(BETA_START, BETA_END, self.T).to(DEVICE)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(DEVICE)

    # Forward diffusion process: add Gaussian noise to clean sample x0
    def forward_noise(self, x0, t):
        bs = x0.shape[0]
        a_bar = self.alpha_bar[t].reshape(bs,1,1,1)
        eps = torch.randn_like(x0)
        xt = torch.sqrt(a_bar)*x0 + torch.sqrt(1-a_bar)*eps
        return xt, eps

    # Single reverse denoising step
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

    # Generate Raman image conditioned on input CARS image
    def sample_raman(self, cond_img, model):
        bs = cond_img.shape[0]
        xt = torch.randn((bs,1,M,M), device=DEVICE)
        for tidx in range(self.T-1, -1, -1):
            t = torch.full((bs,), tidx, device=DEVICE)
            xt = self.reverse_one_step(xt, t, cond_img, model)
        return xt

# Original training pipeline for 2D diffusion model
def train_diffusion_2d():
    print(f"[2D Diffusion] Train device: {DEVICE}")
    cars_all = np.load("synthetic_data/2_cars_2000.npy")
    raman_all = np.load("synthetic_data/2_raman_2000.npy")
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
    torch.save(model.state_dict(), "pretrained_models/diff_2d.pt")
    print("2D Diffusion finished, weight saved as diff_2d.pt")

# Original generation function for 2D data augmentation
def generate_2d_aug(gen_count=2000, weight="pretrained_models/diff_2d.pt"):
    ddpm = DDPM()
    model = CondUNet().to(DEVICE)
    model.load_state_dict(torch.load(weight, map_location=DEVICE))
    model.eval()

    raw_cars = np.load("synthetic_data/2_cars_2000.npy")
    out_cars = []
    out_raman = []

    with torch.no_grad():
        for i in range(gen_count):
            if i % 100 == 0:
                print(f"[2D Generate] Progress: {i}/{gen_count}")
            rand_idx = np.random.randint(0, len(raw_cars))
            car_1000 = raw_cars[rand_idx, :]
            cond_img = spec_1000_to_2d(car_1000)
            cond_tensor = trans(cond_img).unsqueeze(0).to(DEVICE)

            pred_img = ddpm.sample_raman(cond_tensor, model)
            pred_img = pred_img.squeeze().cpu()
            pred_img = (pred_img * 0.5 + 0.5) * 255
            pred_img = pred_img.numpy().astype(np.uint8)

            raman_1000 = img_to_spec_1000(pred_img)
            out_cars.append(car_1000)
            out_raman.append(raman_1000)

    out_cars = np.array(out_cars)
    out_raman = np.array(out_raman)
    np.save("synthetic_data/expand_cars_2d.npy", out_cars)
    np.save("synthetic_data/expand_raman_2d.npy", out_raman)
    print(f"2D augment {gen_count} pairs saved")

# ====================== [Newly Added Complete 1D Diffusion Module] ======================
# 1. 1D Spectrum Dataset: load raw 1000-dimensional spectra directly without 2D conversion
class OneDSpecDataset(Dataset):
    def __init__(self, cars_np, raman_np):
        self.cars = cars_np
        self.raman = raman_np

    def __len__(self):
        return len(self.cars)

    def __getitem__(self, idx):
        # Normalize spectrum range to [-1, 1] for stable diffusion training
        c = self.cars[idx]
        r = self.raman[idx]
        c_norm = (c - c.min()) / (c.max() - c.min() + 1e-8) * 2 - 1
        r_norm = (r - r.min()) / (r.max() - r.min() + 1e-8) * 2 - 1
        # Output tensor shape: [1, 1000]
        return torch.from_numpy(c_norm).float().unsqueeze(0), \
               torch.from_numpy(r_norm).float().unsqueeze(0)

# 2. Basic 1D Convolution Block for 1D UNet
class DoubleConv1d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.seq(x)

class Down1d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv = DoubleConv1d(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up1d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = DoubleConv1d(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(torch.cat([x1, x2], dim=1))

# 3. Conditional 1D UNet
# Input channel = 2: concatenated noisy spectrum + conditional CARS spectrum
class CondUNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv1d(2, 64)
        self.d1 = Down1d(64, 128)
        self.d2 = Down1d(128, 256)
        self.d3 = Down1d(256, 512)

        self.u1 = Up1d(512, 256)
        self.u2 = Up1d(256, 128)
        self.u3 = Up1d(128, 64)
        self.out = nn.Conv1d(64, 1, kernel_size=1)

        # Time step embedding network
        self.time_emb = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, x, t):
        bs = x.shape[0]
        t_emb = self.time_emb(t.float().unsqueeze(-1)).view(bs, 512, 1)

        x1 = self.inc(x)
        x2 = self.d1(x1) + t_emb[:, :128]
        x3 = self.d2(x2) + t_emb[:, :256]
        x4 = self.d3(x3) + t_emb

        x = self.u1(x4, x3)
        x = self.u2(x, x2)
        x = self.u3(x, x1)
        return self.out(x)

# 4. 1D DDPM Implementation
# Reuse standard diffusion mathematical formulas, only adapted for 1D tensor dimensions
class DDPM1D:
    def __init__(self):
        self.T = T_DIFF
        self.beta = torch.linspace(BETA_START, BETA_END, self.T).to(DEVICE)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(DEVICE)

    # Forward noising process for 1D spectra
    def forward_noise(self, x0, t):
        bs = x0.shape[0]
        a_bar = self.alpha_bar[t].reshape(bs, 1, 1)
        eps = torch.randn_like(x0)
        xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps
        return xt, eps

    # Single reverse denoising step for 1D data
    def reverse_one_step(self, xt, t, cond, model):
        bs = xt.shape[0]
        input_cat = torch.cat([xt, cond], dim=1)
        eps_pred = model(input_cat, t)

        at = self.alpha[t].reshape(bs,1,1)
        abt = self.alpha_bar[t].reshape(bs,1,1)
        bt = self.beta[t].reshape(bs,1,1)

        mu = 1 / torch.sqrt(at) * (xt - bt / torch.sqrt(1 - abt) * eps_pred)
        if t[0] > 0:
            z = torch.randn_like(xt)
            return mu + torch.sqrt(bt) * z
        return mu

    # Generate 1D Raman spectrum conditioned on input CARS spectrum
    def sample_1d(self, cond_spec, model):
        bs = cond_spec.shape[0]
        xt = torch.randn((bs, 1, RAW_N_POINTS), device=DEVICE)
        for tidx in range(self.T-1, -1, -1):
            t = torch.full((bs,), tidx, device=DEVICE)
            xt = self.reverse_one_step(xt, t, cond_spec, model)
        return xt

# Main training entry for 1D diffusion model
def train_diffusion_1d():
    print(f"[1D Diffusion] Train device: {DEVICE}")
    cars_all = np.load("synthetic_data/cars_5000.npy")
    raman_all = np.load("synthetic_data/raman_5000.npy")
    ds = OneDSpecDataset(cars_all, raman_all)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = CondUNet1D().to(DEVICE)
    ddpm = DDPM1D()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for ep in range(EPOCHS):
        total_loss = 0.0
        for cond, x0 in loader:
            cond, x0 = cond.to(DEVICE), x0.to(DEVICE)
            bs = x0.shape[0]
            t = torch.randint(0, T_DIFF, (bs,), device=DEVICE)
            xt, eps_real = ddpm.forward_noise(x0, t)
            inp = torch.cat([xt, cond], dim=1)
            eps_pred = model(inp, t)
            loss = loss_fn(eps_pred, eps_real)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        if (ep+1) % 20 == 0:
            print(f"Epoch {ep+1:4d} | MSE Loss: {avg_loss:.6f}")
    torch.save(model.state_dict(), "pretrained_models/diff_1d.pt")
    print("1D Diffusion training done, saved diff_1d.pt")

# 1D Diffusion generation for dataset augmentation
def generate_1d_aug(gen_count, weight="pretrained_models/diff_1d.pt"):
    ddpm = DDPM1D()
    model = CondUNet1D().to(DEVICE)
    model.load_state_dict(torch.load(weight, map_location=DEVICE))
    model.eval()

    raw_cars = np.load("synthetic_data/cars_5000.npy")
    out_cars = []
    out_raman = []

    with torch.no_grad():
        for i in range(gen_count):
            if i % 100 == 0:
                print(f"[1D Generate] Progress {i}/{gen_count}")
            idx = np.random.randint(0, len(raw_cars))
            car_np = raw_cars[idx]
            # Normalize spectrum to [-1, 1] range
            c_norm = (car_np - car_np.min()) / (car_np.max() - car_np.min() + 1e-8) * 2 - 1
            cond_t = torch.from_numpy(c_norm).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            pred_t = ddpm.sample_1d(cond_t, model)
            pred_t = pred_t.squeeze().cpu().numpy()
            # Rescale generated spectrum back to original value range
            pred_rescale = (pred_t + 1) / 2
            pred_rescale = pred_rescale * (car_np.max() - car_np.min()) + car_np.min()

            out_cars.append(car_np)
            out_raman.append(pred_rescale)

    out_cars = np.array(out_cars)
    out_raman = np.array(out_raman)
    np.save("synthetic_data/expand_cars_1d_656.npy", out_cars)
    np.save("synthetic_data/expand_raman_1d_656.npy", out_raman)
    print(f"1D diffusion generate {gen_count} samples saved")

# ====================== Program Main Entry: Enable target function as needed ======================
if __name__ == "__main__":
    # ========== Choose one training pipeline to execute ==========
    # train_diffusion_2d() # Original image-based 2D diffusion
    # train_diffusion_1d() # Newly implemented 1D spectrum diffusion

    # ========== Choose one generation pipeline to execute ==========
    # generate_2d_aug(gen_count=2000)
    generate_1d_aug(gen_count=3000)