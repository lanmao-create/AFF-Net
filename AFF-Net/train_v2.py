import argparse
import sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import random
import os
import torch
import torch.optim as optim
from torch.autograd import Variable
from IPython.display import clear_output
from model.FTUNetFormer_v2 import ft_unetformer_v2 as FTUNetFormer_v2
from skimage import io
import glob
import re
import torch.nn.functional as F

# --- Training Configuration ---
# Set the training stage:
# 1: Pre-train the model on RGB images only. Saves the best model to 'stage1_rgb_best.pth'.
# 2: Fine-tune the model with both RGB and DSM data. Loads 'stage1_rgb_best.pth' to start.
STAGE = 1
# STAGE = 2
# --- End of Configuration ---

DATASET = 'Vaihingen'

if DATASET == 'Vaihingen':
    from utils import *
elif DATASET == 'Urban':
    from utils_loveda import *

# --- Model Initialization ---
print(f"--- STAGE {STAGE} ---")
# 默认启用频域融合，便于直接测试跨模态频域策略
net = FTUNetFormer_v2(num_classes=N_CLASSES, fusion_type='frequency').cuda()

if STAGE == 2:
    stage1_files = glob.glob('./resultsv/stage1_rgb_epoch_*_MIoU_*.pth')
    stage1_files = [f for f in stage1_files if '_archive' not in f]  # Exclude checkpoints

    best_file = None
    if stage1_files:
        best_MIoU = 0
        for f in stage1_files:
            match = re.search(r'MIoU_(\d+\.\d+)\.pth', f)
            if match:
                MIoU = float(match.group(1))
                if MIoU > best_MIoU:
                    best_MIoU = MIoU
                    best_file = f

    if best_file and os.path.exists(best_file):
        net.load_state_dict(torch.load(best_file), strict=False)
        print(f"Successfully loaded Stage 1 weights from {best_file}")
    else:
        print(f"Warning: Stage 1 weights not found. Starting Stage 2 from scratch.")
        
params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Trainable parameters: {params / 1_000_000:.2f}M")

# --- DataLoaders ---
print("Training set size: ", len(train_ids))
print("Testing set size: ", len(test_ids))
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

# --- Optimizer and Scheduler Setup ---
if STAGE == 1:
    base_lr = 1e-4
    # Freeze DSM and fusion parts
    for name, param in net.named_parameters():
        if 'dsm_backbone' in name or 'fusion_modules' in name:
            param.requires_grad = False
    
    # Differential learning rate for backbone
    params_to_optimize = [
        {"params": [p for n, p in net.named_parameters() if ("rgb_backbone" in n) and p.requires_grad], "lr": base_lr / 10},
        {"params": [p for n, p in net.named_parameters() if ("rgb_backbone" not in n) and p.requires_grad], "lr": base_lr},
    ]
    print("Optimizer configured for STAGE 1: Training RGB path only.")
else: # STAGE == 2
    base_lr = 5e-5 # Smaller LR for fine-tuning
    # Unfreeze all parts
    for param in net.parameters():
        param.requires_grad = True
        
    # Differential LR: smaller for pretrained parts, larger for new parts
    # 核心修改：为小波相关的新模块创建一个独立的参数组，并给予正常的学习率
    wavelet_params = [p for n, p in net.named_parameters() if ("wavelet" in n or "wave_fuse" in n) and p.requires_grad]
    backbone_decoder_params = [p for n, p in net.named_parameters() if ("wavelet" not in n and "wave_fuse" not in n) and (("rgb_backbone" in n or "dsm_backbone" in n or "decoder" in n) and p.requires_grad)]
    fusion_params = [p for n, p in net.named_parameters() if ("fusion_modules" in n and p.requires_grad)]
    
    params_to_optimize = [
        {"params": backbone_decoder_params, "lr": base_lr / 10},
        {"params": fusion_params, "lr": base_lr},
        {"params": wavelet_params, "lr": base_lr}, # 新模块使用正常学习率
    ]
    print("Optimizer configured for STAGE 2: Fine-tuning all paths with dedicated LR for wavelet modules.")

optimizer = optim.AdamW(params_to_optimize, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)


def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE, stage=1, save_dir=None):
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_dsms = (io.imread(DSM_FOLDER.format(id)) for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    
    all_preds, all_gts = [], []
    net.eval()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (img, dsm, gt, gt_e) in enumerate(tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids), leave=False, desc="Testing")):
            pred = np.zeros(img.shape[:2] + (N_CLASSES,))
            
            dsm = np.asarray(dsm, dtype='float32')
            dsm_min, dsm_max = dsm.min(), dsm.max()
            if dsm_max > dsm_min: dsm = (dsm - dsm_min) / (dsm_max - dsm_min)

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=False, desc="Sliding window")):
                rgb_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                rgb_patches = Variable(torch.from_numpy(np.asarray(rgb_patches)).cuda())
                
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.expand_dims(np.asarray(dsm_patches, dtype='float32'), axis=1)
                dsm_patches = Variable(torch.from_numpy(dsm_patches).cuda())
                
                # 接收模型的元组输出，并只取第一个元素（分割结果）
                outs, _ = net(rgb_patches, dsm_patches, stage=stage)
                outs = outs.data.cpu().numpy()
                
                for out, (x, y, w, h) in zip(outs, coords):
                    pred[x:x + w, y:y + h] += out.transpose((1, 2, 0))
            
            pred = np.argmax(pred, axis=-1)
            all_preds.append(pred)
            all_gts.append(gt_e)
            
            # 可选：保存彩色预测图便于对比
            if save_dir:
                img_id = str(test_ids[idx]) if test_ids else str(idx)
                color_pred = convert_to_color(pred)
                io.imsave(os.path.join(save_dir, f"{img_id}_pred.png"), color_pred)

    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1, stage=1):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    iter_ = 0
    MIoU_best = 0.0
    
    for e in range(1, epochs + 1):
        net.train()
        for batch_idx, (data_rgb, data_dsm, boundary, object, target) in enumerate(train_loader):
            data_rgb, data_dsm, target = Variable(data_rgb.cuda()), Variable(data_dsm.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            
            # 接收模型的两个输出: 分割结果和重建图像
            output_seg, recon_rgb = net(data_rgb, data_dsm, stage=stage)
            
            # 1. 计算主分割损失
            loss_seg = loss_calc(output_seg, target, weights)
            
            # 2. 计算辅助重建损失 (MSE Loss)
            loss_recon = F.mse_loss(recon_rgb, data_rgb)
            
            # 3. 加权组合两个损失
            # lambda_recon 是一个超参数，用来平衡两个损失的重要性，可以从0.1开始尝试
            lambda_recon = 0.1 
            total_loss = loss_seg + lambda_recon * loss_recon
            
            total_loss.backward()
            optimizer.step()

            losses[iter_] = total_loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])
            
            if iter_ % 10 == 0:
                clear_output()
                pred = np.argmax(output_seg.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]

                # Calculate the other loss for display purposes only
                with torch.no_grad():
                    if LOSS_FN == 'Focal':
                        main_loss_str = 'Focal Loss'
                        other_loss_str = 'CE Loss'
                        other_criterion = CrossEntropy2d_ignore().cuda()
                        other_loss_val = other_criterion(output_seg, target, weights).item()
                    else:  # Main is CrossEntropy
                        main_loss_str = 'CE Loss'
                        other_loss_str = 'Focal Loss'
                        other_criterion = FocalLoss(gamma=2).cuda()
                        other_loss_val = other_criterion(output_seg, target, weights).item()
                
                print(f'Train (Stage {stage}, Epoch {e}/{epochs}) [{batch_idx}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Total Loss: {total_loss.item():.6f} (Seg: {loss_seg.item():.6f} + Recon: {loss_recon.item():.6f})\t'
                      f'Accuracy: {accuracy(pred, gt):.2f}%')
            iter_ += 1

        if scheduler is not None:
            scheduler.step()

        if e % save_epoch == 0:
            MIoU = test(net, test_ids, all=False, stride=Stride_Size, stage=stage)
            if MIoU > MIoU_best:
                MIoU_best = MIoU
                if stage == 1:
                    save_path = f'./resultsv/stage1_rgb_epoch_{e}_MIoU_{MIoU_best:.4f}.pth'
                else:
                    save_path = f'./resultsv/stage2_fused_epoch_{e}_MIoU_{MIoU_best:.4f}.pth'
                # 保存纯权重（state_dict），保留原有命名与格式
                torch.save(net.state_dict(), save_path)
                # 额外保存完整检查点（包含优化器/调度器等）
                ckpt = {
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                    'epoch': e,
                    'stage': stage,
                    'MIoU_best': MIoU_best,
                }
                ckpt_path = save_path.replace('.pth', '_archive.pth')
                torch.save(ckpt, ckpt_path)
                print(f"*** Best model saved to {save_path} and checkpoint to {ckpt_path} with MIoU: {MIoU_best:.4f} ***")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train/Test FTUNetFormer_v2")
    parser.add_argument('--eval_ckpt', type=str, default=None, help='Path to a .pth checkpoint for evaluation only')
    parser.add_argument('--stage', type=int, default=STAGE, choices=[1, 2], help='Stage for evaluation (1: RGB only, 2: RGB+DSM)')
    parser.add_argument('--eval_only', action='store_true', help='If set, only run evaluation and exit')
    parser.add_argument('--save_preds', type=str, default=None, help='Directory to save predicted color maps during evaluation')
    args = parser.parse_args()

    if not os.path.exists('./resultsv'):
        os.makedirs('./resultsv')
    
    # Evaluation-only path: load weights and run test
    if args.eval_ckpt:
        state = torch.load(args.eval_ckpt, map_location='cuda')
        net.load_state_dict(state, strict=(args.stage == 2))
        print(f"Loaded weights from {args.eval_ckpt}")
        miou = test(net, test_ids, all=False, stride=Stride_Size, stage=args.stage, save_dir=args.save_preds)
        print(f"[Eval stage {args.stage}] MIoU: {miou:.4f}")
        if args.eval_only:
            sys.exit(0)

    epochs = 50 if STAGE == 1 else 50
    train(net, optimizer, epochs, scheduler, stage=STAGE)
    
