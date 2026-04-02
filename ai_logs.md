# AI Tool Usage Log - COMP6001 Assignment 1

Tools used: Claude, ChatGPT, Gemini

---

## Task 2 - Image Deblurring

---

### Log 1 - Understanding the task and what methods to use

**Tool:** ChatGPT

**Prompt:**
> the assignment says to implement image deblurring using classical methods like wiener or richardson lucy and also deep learning methods. i have the gopro dataset which has blurred and sharp image pairs. what should my overall approach be and what methods should i implement

**Output:**
> For a project like this, a good structure is to implement the classical methods first as baselines, then a deep learning model to compare against. For classical methods, both Wiener filtering and Richardson-Lucy deconvolution are standard choices. Wiener filtering is a frequency-domain approach that balances noise suppression and deblurring - it's fast and simple to use via skimage.restoration.wiener. Richardson-Lucy is an iterative Bayesian method that can produce sharper results but is slower and more prone to ringing artifacts at high iterations.
>
> For the deep learning side, you have two main options: train a U-Net style architecture from scratch on the GoPro pairs, or use a pretrained model like NAFNet which was specifically trained on GoPro data and achieves state-of-the-art results. Doing both gives you a good comparison - your own model vs a strong pretrained baseline.
>
> The GoPro dataset is perfect for this since it has paired blurred/sharp images, so your loss function can directly minimize the pixel difference between the network output and the sharp ground truth. Use PSNR and SSIM as your evaluation metrics - PSNR measures pixel-level error and SSIM measures structural similarity, together they give a complete picture of restoration quality.
>
> Suggested pipeline: load paired images -> apply classical filters -> train U-Net -> run NAFNet inference -> compare all four methods on PSNR/SSIM on the test set.

**How it was used:**
Shaped the overall plan for Task 2. Decided to implement Wiener + Richardson-Lucy as classical baselines, build a Residual U-Net from scratch, and use pretrained NAFNet as the deep learning comparison.

---

### Log 2 - How to make PyTorch training reproducible

**Tool:** ChatGPT

**Prompt:**
> how do i make my pytorch experiment fully reproducible, i want to seed everything

**Output:**
> To get full reproducibility in PyTorch you need to seed several things:
>
>     import random
>     import numpy as np
>     import torch
>
>     def set_seed(seed=42):
>         random.seed(seed)
>         np.random.seed(seed)
>         torch.manual_seed(seed)
>         if torch.cuda.is_available():
>             torch.cuda.manual_seed_all(seed)
>             torch.backends.cudnn.deterministic = True
>             torch.backends.cudnn.benchmark = False
>
> The cudnn.deterministic = True forces CUDA to use deterministic algorithms. The cudnn.benchmark = False stops CUDA from auto-selecting the fastest algorithm which varies between runs. Note that setting these will make training slightly slower but results will be identical across runs. Call this function before anything else in your script.

**How it was used:**
Directly used as set_seed(67) at the top of the notebook.

---

### Log 3 - Building the paired image dataset class

**Tool:** Claude

**Prompt:**
> i need a pytorch Dataset class for the gopro dataset. it has a blur folder and a sharp folder with matching filenames. i need it to load pairs, apply random crop and flip augmentations during training, and center crop during eval. the augmentations need to be applied identically to both the blur and sharp image

**Output:**
> The key challenge here is applying the same random transform to both images. The cleanest way is to stack them along the channel dimension, apply the transform once, then split them back:
>
>     from pathlib import Path
>     import cv2
>     import torch
>     from torch.utils.data import Dataset
>     from torchvision.transforms import v2
>
>     class GoproDataset(Dataset):
>         EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
>
>         def __init__(self, blur_dir, sharp_dir, is_train=False, crop_size=256, max_samples=None):
>             self.is_train = is_train
>             sharp_lookup = {p.name: p for p in Path(sharp_dir).rglob("*") if p.suffix.lower() in self.EXTS}
>             self.pairs = [
>                 (bp, sharp_lookup[bp.name])
>                 for bp in sorted(Path(blur_dir).rglob("*"))
>                 if bp.suffix.lower() in self.EXTS and bp.name in sharp_lookup
>             ]
>             if max_samples:
>                 self.pairs = self.pairs[:max_samples]
>             self.train_transform = v2.Compose([
>                 v2.RandomCrop(crop_size),
>                 v2.RandomHorizontalFlip(p=0.5),
>                 v2.RandomVerticalFlip(p=0.2),
>             ])
>             self.eval_transform = v2.CenterCrop(crop_size)
>
>         def __len__(self):
>             return len(self.pairs)
>
>         def __getitem__(self, idx):
>             bp, sp = self.pairs[idx]
>             blur = cv2.cvtColor(cv2.imread(str(bp)), cv2.COLOR_BGR2RGB)
>             sharp = cv2.cvtColor(cv2.imread(str(sp)), cv2.COLOR_BGR2RGB)
>             blur_t = torch.from_numpy(blur.transpose(2, 0, 1)).float() / 255.0
>             sharp_t = torch.from_numpy(sharp.transpose(2, 0, 1)).float() / 255.0
>             if self.is_train:
>                 stacked = torch.cat([blur_t, sharp_t], dim=0)
>                 stacked = self.train_transform(stacked)
>                 blur_t, sharp_t = stacked[:3], stacked[3:]
>             else:
>                 blur_t = self.eval_transform(blur_t)
>                 sharp_t = self.eval_transform(sharp_t)
>             return {"blur": blur_t, "sharp": sharp_t, "name": bp.name}
>
> Note: OpenCV reads as BGR so convert to RGB immediately. Normalise to [0,1] by dividing by 255. The stacking trick ensures RandomCrop picks the same crop coordinates for both images since it is one tensor.

**How it was used:**
Became the DynamicGoproDataset class. The stacking approach for consistent augmentation was the key insight used directly.

---

### Log 4 - Classical deblurring implementation

**Tool:** Gemini

**Prompt:**
> how do i implement wiener filter and richardson lucy deconvolution in python for image deblurring. i have a blurred image as a numpy array, and i need to apply both methods channel by channel. what PSF should i use for gopro motion blur

**Output:**
> Both methods are available in skimage.restoration. For the PSF, since GoPro motion blur is predominantly caused by camera motion, a simple horizontal motion kernel is a reasonable approximation:
>
>     from skimage.restoration import wiener, richardson_lucy
>     import numpy as np
>
>     def apply_classical_filters(img_np):
>         psf_size = 5
>         psf = np.zeros((psf_size, psf_size))
>         psf[psf_size // 2, :] = 1.0
>         psf /= psf.sum()
>
>         wiener_out = np.zeros_like(img_np)
>         lucy_out = np.zeros_like(img_np)
>
>         for ch in range(img_np.shape[2]):
>             wiener_out[:, :, ch] = wiener(img_np[:, :, ch], psf, balance=0.01, clip=True)
>             lucy_out[:, :, ch] = richardson_lucy(img_np[:, :, ch], psf, num_iter=30, clip=True)
>
>         return wiener_out, lucy_out
>
> The balance parameter in Wiener controls regularisation - lower values mean more aggressive deblurring but amplifies noise, higher means smoother result. 0.01 is a reasonable default. For Richardson-Lucy, num_iter=30 is a typical starting point. Always set clip=True to keep pixel values in valid range.
>
> Note: Both methods require you to know the PSF. In practice the true PSF for each GoPro image is different since it depends on actual camera motion during that shot, so using a fixed approximation means results will be limited. This is expected for classical methods and is part of why deep learning outperforms them here.

**How it was used:**
Directly implemented as apply_classical_filters(). Used psf_size=5, balance=0.01, num_iter=30 exactly as suggested.

---

### Log 5 - Building the Residual U-Net architecture

**Tool:** Claude

**Prompt:**
> i want to build a residual unet from scratch in pytorch for image deblurring. the model should take a 3 channel blurred image and output a 3 channel deblurred image. i want residual connections in the conv blocks, maxpool for downsampling, bilinear upsampling in the decoder, and skip connections from encoder to decoder

**Output:**
> Here is a clean implementation with 3 encoder stages and a matching decoder:
>
>     import torch
>     import torch.nn as nn
>     import torch.nn.functional as F
>
>     class ResConvBlock(nn.Module):
>         def __init__(self, in_ch, out_ch):
>             super().__init__()
>             self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
>             self.bn1 = nn.BatchNorm2d(out_ch)
>             self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
>             self.bn2 = nn.BatchNorm2d(out_ch)
>             self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
>
>         def forward(self, x):
>             res = self.shortcut(x)
>             x = F.relu(self.bn1(self.conv1(x)))
>             x = self.bn2(self.conv2(x))
>             return F.relu(x + res)
>
>     class AdvancedUNet(nn.Module):
>         def __init__(self):
>             super().__init__()
>             self.enc1 = ResConvBlock(3, 32)
>             self.enc2 = ResConvBlock(32, 64)
>             self.enc3 = ResConvBlock(64, 128)
>             self.pool = nn.MaxPool2d(2)
>             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
>             self.dec2 = ResConvBlock(128 + 64, 64)
>             self.dec1 = ResConvBlock(64 + 32, 32)
>             self.final = nn.Conv2d(32, 3, 1)
>
>         def forward(self, x):
>             e1 = self.enc1(x)
>             e2 = self.enc2(self.pool(e1))
>             e3 = self.enc3(self.pool(e2))
>             d2 = self.dec2(torch.cat([self.up(e3), e2], dim=1))
>             d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
>             return torch.sigmoid(self.final(d1))
>
> The shortcut in ResConvBlock uses a 1x1 conv when channels change which is the standard residual block pattern from ResNet. The decoder concatenates the upsampled feature map with the skip connection from the matching encoder stage - this is what allows the network to recover spatial details. torch.sigmoid on the final output constrains values to [0,1] which matches the normalised input.
>
> One thing to watch: if your input dimensions are odd numbers, MaxPool will floor them and the upsampled size will not match the skip connection size, causing a RuntimeError on torch.cat. Use input crop sizes that are multiples of 4 to avoid this.

**How it was used:**
Used exactly as shown. Set crop size to 192 in the dataset to avoid shape mismatches as warned.

---

### Log 6 - Training loop with validation and best checkpoint

**Tool:** ChatGPT

**Prompt:**
> write me a pytorch training loop for my unet deblurring model. i need it to have a train phase and validation phase each epoch, compute psnr and ssim on the val set, print logs every few epochs, and save the best model based on lowest val loss

**Output:**
> 
>     def train(model, train_loader, val_loader, val_subset, epochs=15, device='cuda'):
>         optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
>         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
>         criterion = nn.MSELoss()
>         best_loss = float('inf')
>
>         for epoch in range(1, epochs + 1):
>             model.train()
>             train_loss = 0
>             for batch in train_loader:
>                 b_img, s_img = batch['blur'].to(device), batch['sharp'].to(device)
>                 optimizer.zero_grad()
>                 preds = model(b_img)
>                 loss = criterion(preds, s_img)
>                 loss.backward()
>                 optimizer.step()
>                 train_loss += loss.item()
>             scheduler.step()
>
>             model.eval()
>             val_loss, val_psnr, val_ssim = 0, 0, 0
>             with torch.no_grad():
>                 for batch in val_loader:
>                     b_img, s_img = batch['blur'].to(device), batch['sharp'].to(device)
>                     preds = model(b_img)
>                     val_loss += criterion(preds, s_img).item()
>                     for b in range(b_img.size(0)):
>                         p_np = preds[b].clamp(0,1).cpu().permute(1,2,0).numpy()
>                         s_np = s_img[b].cpu().permute(1,2,0).numpy()
>                         val_psnr += psnr_fn(s_np, p_np, data_range=1.0)
>                         val_ssim += ssim_fn(s_np, p_np, data_range=1.0, channel_axis=2)
>
>             if epoch % 3 == 0 or epoch == epochs:
>                 print(f"Epoch [{epoch}/{epochs}] | Train MSE: {train_loss/len(train_loader):.4f} | "
>                       f"Val MSE: {val_loss/len(val_loader):.4f} | "
>                       f"Val PSNR: {val_psnr/len(val_subset):.2f} dB | "
>                       f"Val SSIM: {val_ssim/len(val_subset):.4f}")
>
>             avg_v_loss = val_loss / len(val_loader)
>             if avg_v_loss < best_loss:
>                 best_loss = avg_v_loss
>                 torch.save(model.state_dict(), 'unet_best.pth')
>
>         model.load_state_dict(torch.load('unet_best.pth', map_location=device))
>         return model.eval()
>
> CosineAnnealingLR smoothly decays the learning rate which works well for short training runs. Logging every 3 epochs keeps output readable. The reload at the end ensures you always return the best model, not the last epoch.

**How it was used:**
Adapted into train_unet_routine(). The structure including CosineAnnealingLR, per-image PSNR/SSIM accumulation, and best checkpoint logic was used directly.

---

### Log 7 - Understanding and implementing NAFNet

**Tool:** Claude

**Prompt:**
> i need to implement NAFNet for image deblurring and load pretrained weights from a checkpoint file. can you explain how the architecture works and give me the pytorch implementation. the checkpoint has keys params_ema and params

**Output:**
> NAFNet replaces traditional activations like ReLU with a SimpleGate mechanism - it splits the feature map in half along the channel dimension and multiplies the two halves element-wise. This is surprisingly effective while being computationally cheaper. The architecture follows a U-Net style encoder-decoder with NAFBlocks at each stage.
>
>     class SimpleGate(nn.Module):
>         def forward(self, x):
>             x1, x2 = x.chunk(2, dim=1)
>             return x1 * x2
>
>     class LayerNorm2d(nn.LayerNorm):
>         def forward(self, x):
>             return super().forward(x.permute(0,2,3,1)).permute(0,3,1,2)
>
>     class NAFBlock(nn.Module):
>         def __init__(self, c):
>             super().__init__()
>             self.norm1 = LayerNorm2d(c)
>             self.norm2 = LayerNorm2d(c)
>             self.conv1 = nn.Conv2d(c, c*2, 1)
>             self.conv2 = nn.Conv2d(c*2, c*2, 3, 1, 1, groups=c*2)
>             self.conv3 = nn.Conv2d(c, c, 1)
>             self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
>             self.sg = SimpleGate()
>             self.conv4 = nn.Conv2d(c, c*2, 1)
>             self.conv5 = nn.Conv2d(c, c, 1)
>             self.beta = nn.Parameter(torch.ones(1, c, 1, 1))
>             self.gamma = nn.Parameter(torch.ones(1, c, 1, 1))
>
>         def forward(self, inp):
>             x = self.sg(self.conv2(self.conv1(self.norm1(inp))))
>             y = inp + self.conv3(x * self.sca(x)) * self.beta
>             return y + self.conv5(self.sg(self.conv4(self.norm2(y)))) * self.gamma
>
> For loading the pretrained checkpoint, infer the architecture from the state dict instead of hardcoding:
>
>     def load_nafnet(path, device):
>         ckpt = torch.load(path, map_location=device)
>         state = ckpt.get("params_ema") or ckpt.get("params") or ckpt
>         width = state["intro.weight"].shape[0]
>         num_stages = len({k.split(".")[1] for k in state if k.startswith("downs.")})
>
>         def nblocks(prefix):
>             idxs = {int(k[len(prefix):].split(".")[0])
>                     for k in state if k.startswith(prefix) and k[len(prefix):].split(".")[0].isdigit()}
>             return max(idxs) + 1 if idxs else 0
>
>         enc_blks = [nblocks(f"encoders.{i}.") for i in range(num_stages)]
>         dec_blks = [nblocks(f"decoders.{i}.") for i in range(num_stages)]
>         model = NAFNet(width=width, enc_blks=enc_blks,
>                        middle_blk_num=nblocks("middle_blks."), dec_blks=dec_blks)
>         model.load_state_dict(state, strict=False)
>         return model.eval().to(device)
>
> The forward pass needs to pad input to the nearest multiple of 2^num_stages using reflect padding, then crop back:
>
>     def forward(self, inp):
>         B, C, H, W = inp.shape
>         ph = (self.padder_size - H % self.padder_size) % self.padder_size
>         pw = (self.padder_size - W % self.padder_size) % self.padder_size
>         x = F.pad(inp, (0, pw, 0, ph), mode='reflect')
>         # encoder decoder ...
>         return output[:, :, :H, :W]

**How it was used:**
The full NAFNet implementation including SimpleGate, LayerNorm2d, NAFBlock, and the dynamic weight loading with nblocks() helper came from this. The padding logic in forward() was used directly.

---

### Log 8 - Evaluation and 6x6 comparison grid

**Tool:** ChatGPT

**Prompt:**
> i want to evaluate all 4 deblurring methods on the test set and compute average psnr and ssim for each. also i want to generate a 6x6 grid plot showing 6 reference images across rows and 6 columns: blurred, wiener, lucy, resunet output, nafnet output, sharp gt

**Output:**
> For the evaluation loop, run all four methods per image and accumulate metrics:
>
>     results = {"Wiener": [], "Lucy": [], "UNet": [], "NAFNet": []}
>
>     for batch in test_loader:
>         blur_t = batch["blur"][0]
>         sharp_np = batch["sharp"][0].permute(1,2,0).numpy()
>         w_t, l_t = apply_classical_filters(blur_t)
>         blur_gpu = blur_t.unsqueeze(0).to(device)
>         with torch.inference_mode():
>             u_t = unet_model(blur_gpu).clamp(0,1).squeeze(0).cpu()
>             n_t = nafnet(blur_gpu).clamp(0,1).squeeze(0).cpu()
>         preds = {"Wiener": w_t.permute(1,2,0).numpy(), "Lucy": l_t.permute(1,2,0).numpy(),
>                  "UNet": u_t.permute(1,2,0).numpy(), "NAFNet": n_t.permute(1,2,0).numpy()}
>         for method, pred_np in preds.items():
>             psnr = psnr_fn(sharp_np, pred_np, data_range=1.0)
>             ssim = ssim_fn(sharp_np, pred_np, data_range=1.0, channel_axis=2)
>             results[method].append({"psnr": psnr, "ssim": ssim})
>
> For the 6x6 grid:
>
>     fig, axes = plt.subplots(6, 6, figsize=(24, 24))
>     col_titles = ["Blurred", "Wiener", "Lucy-Richardson", "ResUNet Output", "NAFNet Output", "Sharp (GT)"]
>
>     for row_idx, data in enumerate(ref_plot_data):
>         imgs = [data["blur"], data["wiener"], data["lucy"], data["unet"], data["nafnet"], data["sharp"]]
>         for col_idx, img in enumerate(imgs):
>             ax = axes[row_idx, col_idx]
>             ax.imshow(np.clip(img, 0, 1))
>             ax.set_xticks([]); ax.set_yticks([])
>             if row_idx == 0:
>                 ax.set_title(col_titles[col_idx], fontsize=16, fontweight="bold", pad=10)
>             if col_idx == 0:
>                 ax.set_ylabel(row_label, rotation=0, labelpad=50, va="center")
>     plt.tight_layout()
>     plt.savefig("task2_reference_6x6_grid.png", dpi=150, bbox_inches="tight")
>
> Use torch.inference_mode() instead of no_grad() for inference - it is slightly faster and disables more autograd tracking. Use np.clip(img, 0, 1) before imshow to avoid matplotlib warnings about out-of-range values.

**How it was used:**
The evaluation loop structure and 6x6 grid layout with torch.inference_mode() came from this directly.

---

## Task 3 - Object Detection and Analysis

---

### Log 9 - Planning the detection analysis approach

**Tool:** Gemini

**Prompt:**
> task 3 says to run a pretrained object detector on blurred and deblurred images and analyse how deblurring affects detection performance. no retraining needed. what should i measure and how do i use the sharp images as ground truth if they dont have annotations

**Output:**
> Since you do not have human annotations, you can use the detector output on the sharp images as a ground truth proxy. Run the same YOLOv8 model on all three versions (blurred, deblurred, sharp), treat the sharp predictions as your reference, and measure how closely the blurred and deblurred predictions match them.
>
> What to measure: detection count per image, confidence score distributions for each condition, Precision/Recall/F1 score by matching predictions to sharp GT using IoU >= 0.5, and latency per image.
>
> For matching: sort predictions by confidence descending, then for each prediction find the highest-IoU unmatched GT box of the same class above 0.5 threshold. This is greedy matching used in AP computation.
>
> Good visualisations: a side-by-side grid of 6 reference images x 3 columns (blurred detections, deblurred detections, sharp GT detections), an IoU distribution boxplot showing how accurately boxes are placed, a confidence shift scatter where each point is one matched object showing blur confidence vs deblur confidence with y=x diagonal, and a per-class F1 bar chart for the top 10 classes.

**How it was used:**
Defined the full analysis plan for Task 3 - the proxy GT approach, metrics, and the four visualisations.

---

### Log 10 - YOLOv8 inference and helper functions

**Tool:** Claude

**Prompt:**
> i want to run yolov8m inference on numpy rgb images and get back a list of detections with label, confidence and xyxy bounding box. also write me an iou function, a greedy box matching function, and a function to compute precision recall and f1 from tp count

**Output:**
> 
>     from ultralytics import YOLO
>     import time
>
>     detector = YOLO("yolov8m.pt")
>
>     def detect(img_rgb, conf=0.25, imgsz=640):
>         t0 = time.perf_counter()
>         res = detector(img_rgb, conf=conf, imgsz=imgsz, verbose=False)[0]
>         lat = time.perf_counter() - t0
>         dets = [
>             {"label": detector.names[int(b.cls)], "conf": float(b.conf), "xyxy": b.xyxy[0].tolist()}
>             for b in res.boxes
>         ]
>         return dets, lat
>
>     def iou(a, b):
>         ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
>         ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
>         inter = max(0, ix2-ix1) * max(0, iy2-iy1)
>         if inter == 0: return 0.0
>         return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
>
>     def match_count(dets_pred, dets_gt, iou_thresh=0.5):
>         matched, used = 0, set()
>         for dp in dets_pred:
>             for j, dg in enumerate(dets_gt):
>                 if j not in used and dp["label"] == dg["label"] and iou(dp["xyxy"], dg["xyxy"]) >= iou_thresh:
>                     matched += 1; used.add(j); break
>         return matched
>
>     def calculate_metrics(tp, n_pred, n_gt):
>         precision = tp / n_pred if n_pred > 0 else 0.0
>         recall = tp / n_gt if n_gt > 0 else 0.0
>         f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
>         return precision, recall, f1
>
>     def draw_boxes(img_rgb, dets, color):
>         out = img_rgb.copy()
>         for d in dets:
>             x1, y1, x2, y2 = [int(v) for v in d["xyxy"]]
>             cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
>             cv2.putText(out, f"{d['label']} {d['conf']:.2f}", (x1, max(y1-6,12)),
>                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
>         return out

**How it was used:**
All four functions used directly with minor variable name changes.

---

### Log 11 - Detailed match info for advanced plots

**Tool:** Claude

**Prompt:**
> i also need a more detailed version of the matching function that returns for each matched gt box the iou value, confidence score, and label. i need this to build the iou boxplot and confidence scatter plot

**Output:**
>
>     def get_match_details(dets_pred, dets_gt, iou_thresh=0.5):
>         matches = {}
>         used_gt = set()
>         for dp in sorted(dets_pred, key=lambda x: x["conf"], reverse=True):
>             best_iou, best_gt_idx = 0, -1
>             for j, dg in enumerate(dets_gt):
>                 if j not in used_gt and dp["label"] == dg["label"]:
>                     v = iou(dp["xyxy"], dg["xyxy"])
>                     if v >= iou_thresh and v > best_iou:
>                         best_iou, best_gt_idx = v, j
>             if best_gt_idx != -1:
>                 matches[best_gt_idx] = {"iou": best_iou, "conf": dp["conf"], "label": dp["label"]}
>                 used_gt.add(best_gt_idx)
>         return matches
>
> Sort predictions by confidence descending first so higher confidence predictions get priority in matching. The returned dict maps GT index to match details so you can look up by GT box index.

**How it was used:**
Used as get_match_details() to collect IoU and confidence values per matched object across all 1111 test images, feeding the boxplot and scatter plot.

---

### Log 12 - Memory issue during inference loop

**Tool:** Claude

**Prompt:**
> my inference loop over 1111 images is running out of RAM. i am storing blur_rgb, deblur_rgb and sharp_rgb arrays for every image in the results list so i can make the visualisation grid later. but thats too much memory. how do i fix this

**Output:**
> The problem is keeping full image arrays for 1111 images in memory simultaneously. You only need the pixel arrays for the 6 reference images that go in your grid plot - for all others you only need the detection results which are tiny.
>
> Add an is_ref flag for the 6 reference images. Store pixel arrays only when is_ref=True, set them to None for everything else immediately after inference:
>
>     is_ref = bp.name in REF_IMAGES
>
>     results.append({
>         "name": bp.name,
>         "dets_blur": dets_b,
>         "dets_deblur": dets_d,
>         "dets_sharp": dets_s,
>         "_blur":   blur_rgb   if is_ref else None,
>         "_deblur": deblur_rgb if is_ref else None,
>         "_sharp":  sharp_rgb  if is_ref else None,
>     })
>
>     if not is_ref:
>         del blur_rgb, sharp_rgb, deblur_rgb
>     del deblur_t, blur_t
>
>     if i % 50 == 0:
>         gc.collect()
>         torch.cuda.empty_cache()
>
> Also delete the GPU tensor immediately after NAFNet inference with del blur_gpu and call torch.cuda.empty_cache() right after. For the failure case analysis at the end that needs images for non-reference images, reload them on demand at that point rather than keeping them stored.

**How it was used:**
Exactly applied - the is_ref flag, del immediately after non-ref images, and on-demand reload for failure cases all came from this.

---

### Log 13 - Advanced diagnostic plots

**Tool:** ChatGPT

**Prompt:**
> i want to make 3 diagnostic plots: 1) notched boxplot comparing iou distributions of blurred vs deblurred detections against sharp gt 2) scatter plot showing per object confidence change with x=blur confidence y=deblur confidence and y=x diagonal with percentage improved annotated 3) grouped bar chart of per class f1 scores for top 10 classes. give me matplotlib code

**Output:**
>
>     # IoU Boxplot
>     fig, ax = plt.subplots(figsize=(8, 6))
>     bp_plot = ax.boxplot([all_blur_ious, all_deblur_ious],
>                          patch_artist=True, notch=True,
>                          medianprops=dict(color="black", linewidth=2))
>     for patch, color in zip(bp_plot['boxes'], ["#6c88d4", "#e88b3a"]):
>         patch.set_facecolor(color); patch.set_alpha(0.8)
>     ax.set_xticklabels(["Blurred vs Sharp", "Deblurred vs Sharp"])
>     ax.set_title("Detection Boundary Accuracy (IoU against GT)", fontweight="bold")
>     ax.grid(axis='y', linestyle='--', alpha=0.7)
>
>     # Confidence Shift Scatter
>     fig, ax = plt.subplots(figsize=(7, 7))
>     ax.scatter(conf_pairs_blur, conf_pairs_deblur, alpha=0.5, color="#8b5cf6", edgecolor="black", s=40)
>     ax.plot([0,1], [0,1], 'r--', label="No Change (y=x)", linewidth=2)
>     improved = sum(1 for b, d in zip(conf_pairs_blur, conf_pairs_deblur) if d > b)
>     total = len(conf_pairs_blur)
>     ax.text(0.05, 0.95, f"{improved}/{total} ({improved/total*100:.1f}%) improved",
>             transform=ax.transAxes, fontsize=11, verticalalignment='top',
>             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
>
>     # Per-Class F1
>     x = np.arange(len(labels)); width = 0.35
>     fig, ax = plt.subplots(figsize=(14, 6))
>     rects1 = ax.bar(x - width/2, b_f1s, width, label='Blurred', color="#6c88d4")
>     rects2 = ax.bar(x + width/2, d_f1s, width, label='Deblurred', color="#e88b3a")
>     for rects in [rects1, rects2]:
>         for rect in rects:
>             h = rect.get_height()
>             ax.annotate(f'{h:.2f}', xy=(rect.get_x()+rect.get_width()/2, h),
>                         xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

**How it was used:**
All three plots implemented directly from this code.

---

### Log 14 - Failure case visualisation with dark background

**Tool:** ChatGPT

**Prompt:**
> for the failure case analysis i want to find the 4 images where the most objects were missed after deblurring. show them in a matplotlib figure with a dark background. each row is one failure image, 3 columns: blurred detections, deblurred detections, sharp gt detections

**Output:**
>
>     failures_meta = sorted(results, key=lambda r: r["n_sharp"] - r["n_deblur"], reverse=True)[:4]
>     failures_meta = [r for r in failures_meta if r["n_sharp"] - r["n_deblur"] > 0]
>
>     fig, axes = plt.subplots(len(failures_meta), 3, figsize=(18, 5*len(failures_meta)))
>     fig.patch.set_facecolor("#111827")
>     fig.suptitle("Failure Analysis: Missed Detections After Deblurring",
>                  color="white", fontsize=16, fontweight="bold")
>
>     for row_idx, r in enumerate(failures_meta):
>         for col_idx, (img, title) in enumerate(zip(
>             [ann_b, ann_d, ann_s],
>             [f"Blurred - {r['n_blur']} dets",
>              f"Deblurred - {r['n_deblur']} dets",
>              f"Sharp GT - {r['n_sharp']} dets"]
>         )):
>             ax = axes[row_idx][col_idx]
>             ax.imshow(img)
>             if row_idx == 0:
>                 ax.set_title(title, color="white", fontsize=12, pad=10)
>             if col_idx == 0:
>                 ax.set_ylabel(f"{r['name']}\nGap: {r['n_sharp']-r['n_deblur']} missed",
>                               color="white", fontsize=12, fontweight="bold", labelpad=20)
>             ax.set_xticks([]); ax.set_yticks([])
>
>     plt.savefig("failure_cases.png", dpi=150, facecolor=fig.get_facecolor())
>
> Sort by n_sharp - n_deblur descending to rank by missed objects. fig.patch.set_facecolor("#111827") sets the dark background. Pass facecolor=fig.get_facecolor() when saving to preserve it in the file.

**How it was used:**
Used directly including the dark background color, sort logic, and facecolor propagation to savefig.

---

## Task 4 - Dataset Preparation and Fine-Tuning

---

### Log 15 - Pseudo-label strategy

**Tool:** Gemini

**Prompt:**
> task 4 says to prepare a dataset for training a detector on deblurred images. i dont have human annotations. what strategy should i use to get labels and what are the risks

**Output:**
> Since you do not have ground truth annotations, use pseudo-labeling: run your pretrained YOLOv8m on the deblurred images to generate predicted boxes as training labels. Workflow: deblur all training images with NAFNet, run YOLOv8m on each deblurred image to get predicted boxes, save in YOLO format txt files, then fine-tune YOLOv8 on this pseudo-labeled dataset.
>
> The key risk is confirmation bias - you train on labels generated by the same model family, so it inherits existing errors. Mitigate by using a higher confidence threshold for generating labels (0.4 to 0.5) compared to inference (0.25) to filter out uncertain predictions. Empty label files where the detector found nothing are valid and should be kept - they teach the model to predict nothing when appropriate.
>
> Also stratify your train/val split by blur level. Random splits might accidentally put all easy images in train and all hard images in val. Use Laplacian variance to measure blur and split within each blur bin so both splits have balanced difficulty.

**How it was used:**
Defined the full pseudo-label strategy including confidence threshold choice, keeping empty files, and blur-level stratification.

---

### Log 16 - Laplacian variance stratification

**Tool:** ChatGPT

**Prompt:**
> how do i stratify my training dataset by blur intensity before splitting into train and val. measure blur level for each image and split into low mid high bins then do 80/20 split within each bin

**Output:**
>
>     scored = []
>     for bp, sp in pairs:
>         img = cv2.imread(str(bp))
>         if img is not None:
>             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
>             lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
>             scored.append((lap_var, bp, sp))
>
>     scored.sort(key=lambda x: x[0])
>     n = len(scored)
>     t1, t2 = n // 3, 2 * n // 3
>     bins = [("Low", scored[:t1]), ("Mid", scored[t1:t2]), ("High", scored[t2:])]
>
>     TRAIN_RATIO = 0.80
>     train_pairs, val_pairs = [], []
>     for label, grp in bins:
>         k = int(len(grp) * TRAIN_RATIO)
>         train_pairs.extend([(bp, sp) for _, bp, sp in grp[:k]])
>         val_pairs.extend(  [(bp, sp) for _, bp, sp in grp[k:]])
>         print(f"{label}: {len(grp)} total -> {k} train / {len(grp)-k} val")
>
> Laplacian variance is the standard fast sharpness measure. It computes the second derivative of the image. Sharp images have high variance, blurry have low. Splitting into tertiles ensures each bin has equal size and both train and val contain images from all blur levels.

**How it was used:**
Directly implemented as the stratification pipeline. Matched the output exactly: each bin had 701 images splitting to 560 train / 141 val.

---

### Log 17 - Deblurring training images with resume support

**Tool:** Claude

**Prompt:**
> i need to deblur all 2103 training images with nafnet and save to disk. the kaggle session might time out midway. how do i structure this so i can resume without redoing images that are already done

**Output:**
> Check if the output file already exists before processing each image:
>
>     def process_split(split_pairs, deblur_dir, sharp_dir, desc):
>         paths = []
>         for bp, sp in tqdm(split_pairs, desc=f"Deblurring {desc}"):
>             fname = bp.name
>             dp_out = deblur_dir / fname
>             sp_out = sharp_dir / fname
>
>             if not sp_out.exists():
>                 TF.center_crop(Image.open(sp).convert("RGB"), (CROP_SIZE, CROP_SIZE)).save(sp_out)
>
>             if not dp_out.exists():
>                 blur_pil = TF.center_crop(Image.open(bp).convert("RGB"), (CROP_SIZE, CROP_SIZE))
>                 deblur_pil(nafnet, blur_pil).save(dp_out)
>
>             paths.append((fname, dp_out, sp_out))
>         return paths
>
> The if not dp_out.exists() check makes the loop idempotent - if you restart after a crash it skips already processed images. Use PIL for loading and saving since it integrates cleanly with torchvision transforms. Center crop to 512x512 for consistent dimensions.

**How it was used:**
Used as process_split(). Confirmed by Task 5 log output: 1111 deblur skipped, already exist.

---

### Log 18 - Generating pseudo-labels in YOLO format

**Tool:** Gemini

**Prompt:**
> how do i generate yolo format labels from yolov8 predictions and save them as txt files. the format is class_id cx cy w h with normalised coordinates

**Output:**
>
>     CONF_THRESH = 0.40
>
>     def annotate(model, img_paths, label_dir, desc=""):
>         label_dir.mkdir(parents=True, exist_ok=True)
>         labelled = 0
>         for fname, dp, sp in tqdm(img_paths, desc=desc):
>             if not dp.exists():
>                 continue
>             res = model.predict(str(dp), conf=CONF_THRESH, verbose=False)[0]
>             W, H = Image.open(dp).size
>             lines = []
>             for box, cls_id in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
>                 cx = ((box[0]+box[2])/2) / W
>                 cy = ((box[1]+box[3])/2) / H
>                 w  = (box[2]-box[0]) / W
>                 h  = (box[3]-box[1]) / H
>                 lines.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
>             txt = label_dir / f"{Path(fname).stem}.txt"
>             txt.write_text("\n".join(lines))
>             if lines:
>                 labelled += 1
>         return labelled
>
> YOLO format: each line is class_id cx cy w h where all coordinates are normalised by image width/height. Empty txt files are valid - they tell YOLO this image has no objects. Using conf=0.40 for label generation vs 0.25 for inference means only high-confidence predictions become training labels. Always write the file even when empty.

**How it was used:**
Used directly as annotate(). The 1184/1680 train | 313/423 val labelled output confirms 496 background files were correctly kept.

---

### Log 19 - Setting up YOLOv8 fine-tuning

**Tool:** Claude

**Prompt:**
> i want to fine tune yolov8m on my pseudo-labeled deblurred dataset. what hyperparameters, should i freeze any layers, and how do i write the data.yaml. the dataset has 80 classes matching coco

**Output:**
> For fine-tuning a pretrained YOLO model, freezing early layers is important - the first layers learn low-level features that do not change between domains, only higher layers need updating:
>
>     model = YOLO("yolov8m.pt")
>     model.train(
>         data="path/to/data.yaml",
>         epochs=30,
>         imgsz=512,
>         batch=8,
>         lr0=0.0001,
>         freeze=10,
>         patience=20,
>         mosaic=1.0,
>         hsv_h=0.015,
>         hsv_s=0.7,
>         hsv_v=0.4,
>         fliplr=0.5,
>         scale=0.5,
>         translate=0.1,
>     )
>
> For data.yaml, pull class names from the base model so they match exactly:
>
>     names_80 = YOLO("yolov8m.pt").names
>     yaml_content = (
>         f"path: {dataset_path}\n"
>         f"train: images/train\nval: images/val\n"
>         f"nc: 80\nnames: {list(names_80.values())}\n"
>     )
>     (dataset_path / "data.yaml").write_text(yaml_content)
>
> Folder structure needed: images/train, images/val, labels/train, labels/val, data.yaml. The augmentations add diversity since the dataset is relatively small. patience=20 means early stopping triggers if mAP does not improve for 20 epochs.

**How it was used:**
The full training call including all hyperparameters and data.yaml generation came from this. Training stopped at epoch 21 with best at epoch 1 mAP@50=0.633.

---

## Task 5 - Performance Comparison and Critical Analysis

---

### Log 20 - Planning the Task 5 evaluation

**Tool:** Gemini

**Prompt:**
> task 5 wants a comprehensive comparison across blurred, deblurred, and sharp. i have the base yolov8m and my pseudo-label fine-tuned model. what metrics and visualisations should i do

**Output:**
> For a comprehensive comparison across 4 conditions (Blurred, Deblurred, Sharp GT proxy, Pseudo-label FT), compute mAP@50 for each condition against the Sharp GT proxy, average detections per image, and average confidence per image.
>
> Visualisations: a mAP@50 bar chart and average detections bar chart side by side, per-class AP@50 for the pseudo-label FT model showing top 20 classes, precision-recall curves for all three non-GT conditions on the same axes, a detection gap heatmap where rows are blur level bins (Low/Mid/High) and columns are conditions with cell values showing mean detection gap vs Sharp GT, and a blur-level mAP breakdown computing mAP@50 separately for Low/Mid/High blur subsets for each condition.
>
> For the heatmap use a diverging colormap RdYlGn_r with red for high gap and green for low. Annotate each cell with the numeric value. This combination shows both overall performance and whether deblurring helps more for heavily blurred images.

**How it was used:**
Defined all 5 visualisations for Task 5 including the detection gap heatmap and blur-level mAP breakdown.

---

### Log 21 - mAP@50 computation from scratch

**Tool:** Claude

**Prompt:**
> i want to compute map@50 from scratch using my prediction dicts and sharp gt proxy dicts both keyed by filename. each value has boxes, scores, labels lists. give me the full implementation with 11 point interpolation for ap per class

**Output:**
>
>     from collections import defaultdict
>
>     def box_iou(b1, b2):
>         xi1=max(b1[0],b2[0]); yi1=max(b1[1],b2[1])
>         xi2=min(b1[2],b2[2]); yi2=min(b1[3],b2[3])
>         inter=max(0,xi2-xi1)*max(0,yi2-yi1)
>         union=((b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter)
>         return inter/union if union > 0 else 0.0
>
>     def eval_map(pred, gt, iou_t=0.5):
>         cls_preds = defaultdict(list)
>         cls_npos = defaultdict(int)
>
>         for fname in gt:
>             if fname not in pred: continue
>             for lbl in gt[fname]["labels"]:
>                 cls_npos[lbl] += 1
>             gt_match = [False] * len(gt[fname]["boxes"])
>             for box, score, lbl in sorted(
>                 zip(pred[fname]["boxes"], pred[fname]["scores"], pred[fname]["labels"]),
>                 key=lambda x: -x[1]
>             ):
>                 best_iou, best_j = 0, -1
>                 for j, (g_box, g_lbl) in enumerate(zip(gt[fname]["boxes"], gt[fname]["labels"])):
>                     if g_lbl == lbl and not gt_match[j]:
>                         v = box_iou(box, g_box)
>                         if v > best_iou: best_iou, best_j = v, j
>                 if best_iou >= iou_t and best_j >= 0:
>                     gt_match[best_j] = True
>                 cls_preds[lbl].append((score, 1 if best_iou >= iou_t else 0))
>
>         per_class_ap = {}
>         for cls, preds in cls_preds.items():
>             if cls_npos[cls] == 0: continue
>             preds.sort(key=lambda x: -x[0])
>             tp_cum = np.cumsum([p[1] for p in preds])
>             fp_cum = np.cumsum([1-p[1] for p in preds])
>             recalls = tp_cum / cls_npos[cls]
>             precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
>             ap = sum(
>                 max([p for r, p in zip(recalls, precisions) if r >= thr] + [0])
>                 for thr in np.linspace(0, 1, 11)
>             ) / 11.0
>             per_class_ap[cls] = ap
>
>         mAP = np.mean(list(per_class_ap.values())) if per_class_ap else 0.0
>         return mAP, per_class_ap
>
> 11-point interpolation samples recall at 0.0, 0.1 up to 1.0 and takes the max precision at each threshold. The per_class_ap dict lets you plot per-class results and identify which classes benefit most from deblurring.

**How it was used:**
Used exactly as eval_map() and box_iou(). Results: Blurred 0.1873, Deblurred 0.2504, Pseudo-label FT 0.2804.

---

### Log 22 - Precision-recall curves

**Tool:** ChatGPT

**Prompt:**
> write me a function that computes a global precision recall curve for a set of predictions vs a gt dict then plot curves for 3 conditions on the same axes

**Output:**
>
>     def compute_pr(pred, gt, iou_t=0.5):
>         all_preds = []
>         cls_npos = defaultdict(int)
>         for fname in gt:
>             if fname not in pred: continue
>             for lbl in gt[fname]["labels"]: cls_npos[lbl] += 1
>             gt_match = [False] * len(gt[fname]["boxes"])
>             for box, score, lbl in zip(pred[fname]["boxes"], pred[fname]["scores"], pred[fname]["labels"]):
>                 best_iou, best_j = 0, -1
>                 for j, (g_box, g_lbl) in enumerate(zip(gt[fname]["boxes"], gt[fname]["labels"])):
>                     if g_lbl == lbl and not gt_match[j]:
>                         v = box_iou(box, g_box)
>                         if v > best_iou: best_iou, best_j = v, j
>                 if best_iou >= iou_t and best_j >= 0:
>                     gt_match[best_j] = True
>                 all_preds.append((score, 1 if best_iou >= iou_t else 0))
>         total = sum(cls_npos.values())
>         if total == 0 or not all_preds: return np.array([0,1]), np.array([0,0])
>         all_preds.sort(key=lambda x: -x[0])
>         tp = np.cumsum([p[1] for p in all_preds])
>         fp = np.cumsum([1-p[1] for p in all_preds])
>         return tp/total, tp/(tp+fp+1e-9)
>
>     COLORS = {"Blurred": "#1D6FA4", "Deblurred": "#F28E2B", "Pseudo-label FT": "#9467BD"}
>     fig, ax = plt.subplots(figsize=(8, 6))
>     for cond in ["Blurred", "Deblurred", "Pseudo-label FT"]:
>         r, p = compute_pr(all_dets[cond], gt_proxy)
>         ax.plot(r, p, label=cond, color=COLORS[cond], linewidth=2.2)
>     ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
>     ax.set_title("Precision-Recall Curves vs Sharp GT", fontweight="bold")
>     ax.legend(); ax.set_xlim(0,1); ax.set_ylim(0,1)

**How it was used:**
Used as compute_pr() and the PR curve plot. The COLORS dict was adopted for consistent colors across all Task 5 plots.

---

### Log 23 - Detection gap heatmap

**Tool:** Claude

**Prompt:**
> i want a heatmap where rows are blur level bins low mid high and columns are conditions blurred deblurred pseudo-label ft. cell value is mean detection gap vs sharp gt. use diverging colormap red for high gap green for low and annotate each cell

**Output:**
>
>     heatmap_conds = ["Blurred", "Deblurred", "Pseudo-label FT"]
>     blur_bins = ["Low blur", "Mid blur", "High blur"]
>     heatmap_data = np.zeros((len(blur_bins), len(heatmap_conds)))
>
>     for ci, cond in enumerate(heatmap_conds):
>         for bi, bbin in enumerate(blur_bins):
>             fnames_in_bin = [f for f in test_fnames if blur_bin(f) == bbin]
>             gaps = [
>                 all_dets["Sharp (GT)"].get(f, {"n_det":0})["n_det"] -
>                 all_dets[cond].get(f, {"n_det":0})["n_det"]
>                 for f in fnames_in_bin
>             ]
>             heatmap_data[bi, ci] = np.mean(gaps) if gaps else 0
>
>     fig, ax = plt.subplots(figsize=(8, 4))
>     im = ax.imshow(heatmap_data, cmap="RdYlGn_r", aspect="auto", vmin=-1, vmax=3)
>     ax.set_xticks(range(len(heatmap_conds))); ax.set_xticklabels(heatmap_conds, fontsize=11)
>     ax.set_yticks(range(len(blur_bins))); ax.set_yticklabels(blur_bins, fontsize=11)
>     for bi in range(len(blur_bins)):
>         for ci in range(len(heatmap_conds)):
>             ax.text(ci, bi, f"{heatmap_data[bi,ci]:+.2f}", ha="center", va="center",
>                     fontsize=12, fontweight="bold",
>                     color="white" if abs(heatmap_data[bi,ci]) > 1.5 else "black")
>     plt.colorbar(im, ax=ax, label="Avg detection gap")
>
> Use vmin=-1, vmax=3 to centre the colormap. RdYlGn_r reverses the standard colormap so red is bad (high gap) and green is good (low gap). White text on dark cells and black on light cells for readability.

**How it was used:**
Used directly including the colormap, vmin/vmax settings, and the contrast logic for text color.

---

### Log 24 - Blur-level mAP breakdown

**Tool:** ChatGPT

**Prompt:**
> i want to compute map@50 separately for low mid and high blur image subsets and show as a grouped bar chart with 3 conditions side by side. i have an eval_map function that takes pred and gt dicts

**Output:**
>
>     breakdown_data = {}
>     for cond in heatmap_conds:
>         breakdown_data[cond] = {}
>         for bbin in blur_bins:
>             fnames_in_bin = [f for f in test_fnames if blur_bin(f) == bbin]
>             pred_sub = {f: all_dets[cond][f] for f in fnames_in_bin if f in all_dets[cond]}
>             gt_sub   = {f: gt_proxy[f]        for f in fnames_in_bin if f in gt_proxy}
>             mAP, _   = eval_map(pred_sub, gt_sub)
>             breakdown_data[cond][bbin] = mAP
>
>     x = np.arange(len(blur_bins)); width = 0.25
>     fig, ax = plt.subplots(figsize=(10, 5))
>     for i, cond in enumerate(heatmap_conds):
>         vals = [breakdown_data[cond][b] for b in blur_bins]
>         offset = (i - 1) * width
>         bars = ax.bar(x + offset, vals, width, label=cond,
>                       color=COLORS[cond], edgecolor="white", linewidth=0.8)
>         ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8, fontweight="bold")
>     ax.set_xticks(x); ax.set_xticklabels(blur_bins, fontsize=11)
>     ax.set_ylabel("mAP@50"); ax.set_ylim(0, 0.6)
>     ax.set_title("mAP@50 by Blur Intensity Level", fontweight="bold")
>     ax.legend(fontsize=10)
>
> Filter the pred and gt dicts by filename to get per-bin subsets, then pass to existing eval_map. Use (i-1)*width for bar offset so groups are centered around each x tick position.

**How it was used:**
Used directly. Reusing eval_map() with subset dicts was the key insight - no need to rewrite evaluation logic.
