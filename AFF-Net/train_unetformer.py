"""
UNetFormer 语义分割训练脚本
专门用于训练UNetFormer模型的简化版本
"""

import numpy as np
import os
import random
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from skimage import io
import itertools

# 导入模型
from model.UNetFormer import UNetFormer

# ===== 配置参数 =====
class Config:
    # 数据集配置
    DATASET = 'Vaihingen'  # 'Vaihingen' 或 'Urban'
    FOLDER = "./ISPRS_dataset/"  # 数据集根目录
    
    # 训练配置
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    
    # 模型配置
    WINDOW_SIZE = (256, 256)  # 输入图像尺寸
    IN_CHANNELS = 3
    DECODE_CHANNELS = 64
    WINDOW_SIZE_ATTENTION = 8  # 注意力窗口大小
    DROPOUT = 0.1
    BACKBONE = 'swsl_resnet18'  # timm 支持的backbone
    PRETRAINED = True
    
    # 损失函数配置
    USE_BOUNDARY_LOSS = True
    USE_OBJECT_LOSS = True
    LAMBDA_BOUNDARY = 0.1
    LAMBDA_OBJECT = 1.0
    
    # 其他配置
    STRIDE_TEST = 32  # 测试时的滑动窗口步长
    SAVE_EPOCH = 5  # 每隔几个epoch保存一次模型
    CACHE = True  # 是否缓存数据到内存
    GPU_ID = "0"
    SEED = 42

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ===== 数据集配置 =====
def get_dataset_config(config):
    if config.DATASET == 'Vaihingen':
        train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
        test_ids = ['5', '21', '15', '30']
        labels = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]
        n_classes = len(labels)
        
        # ISPRS颜色调色板
        palette = {0: (255, 255, 255), 1: (0, 0, 255), 2: (0, 255, 255),
                  3: (0, 255, 0), 4: (255, 255, 0), 5: (255, 0, 0), 6: (0, 0, 0)}
        
        main_folder = config.FOLDER + 'Vaihingen/'
        data_folder = main_folder + 'ISPRS_semantic_labeling_Vaihingen/top/top_mosaic_09cm_area{}.tif'
        label_folder = main_folder + 'ISPRS_semantic_labeling_Vaihingen/gts_for_participants/top_mosaic_09cm_area{}.tif'
        boundary_folder = None  # 没有边界标注
        object_folder = None    # 没有对象标注
        eroded_folder = main_folder + 'ISPRS_semantic_labeing_Vaihingen_ground_truth_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'
        
    elif config.DATASET == 'Urban':
        # Urban数据集配置（此处简化，您可以根据需要完善）
        train_ids = [str(i) for i in range(1366, 2522)]  # 示例范围
        test_ids = [str(i) for i in range(3514, 4190)]   # 示例范围
        labels = ["background", "building", "road", "water", "barren", "forest", "agriculture"]
        n_classes = len(labels)
        
        palette = {0: (255, 255, 255), 1: (255, 0, 0), 2: (255, 255, 0),
                  3: (0, 0, 255), 4: (159, 129, 183), 5: (0, 255, 0), 6: (255, 195, 128)}
        
        main_folder = config.FOLDER + 'Urban/'
        data_folder = main_folder + 'images_png/{}.png'
        label_folder = main_folder + 'masks_png/{}.png'
        boundary_folder = main_folder + 'boundary_pngs/{}_Boundary.png'
        object_folder = main_folder + 'object_pngs/{}_objects.png'
        eroded_folder = main_folder + 'masks_png/{}.png'
    
    return {
        'train_ids': train_ids,
        'test_ids': test_ids,
        'labels': labels,
        'n_classes': n_classes,
        'palette': palette,
        'data_folder': data_folder,
        'label_folder': label_folder,
        'boundary_folder': boundary_folder,
        'object_folder': object_folder,
        'eroded_folder': eroded_folder
    }

# ===== 工具函数 =====
def convert_from_color(arr_3d, palette):
    """RGB颜色编码转换为灰度标签"""
    invert_palette = {v: k for k, v in palette.items()}
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for c, i in invert_palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d

def convert_to_color(arr_2d, palette):
    """数值标签转换为RGB颜色编码"""
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    return arr_3d

def get_random_pos(img, window_shape):
    """在图像中提取随机位置的窗口"""
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

def object_process(object_mask):
    """处理对象掩码，重新分配ID"""
    ids = np.unique(object_mask)
    new_id = 1
    for id in ids[1:]:
        object_mask = np.where(object_mask == id, new_id, object_mask)
        new_id += 1
    return object_mask

# ===== 数据集类 =====
class UNetFormerDataset(torch.utils.data.Dataset):
    def __init__(self, ids, dataset_config, config, augmentation=True):
        super(UNetFormerDataset, self).__init__()
        
        self.ids = ids
        self.config = config
        self.augmentation = augmentation
        self.cache = config.CACHE
        
        # 文件路径列表
        self.data_files = [dataset_config['data_folder'].format(id) for id in ids]
        self.label_files = [dataset_config['label_folder'].format(id) for id in ids]
        # 检查是否有边界和对象标注
        self.has_boundary = dataset_config['boundary_folder'] is not None
        self.has_object = dataset_config['object_folder'] is not None
        self.boundary_files = [dataset_config['boundary_folder'].format(id) for id in ids] if self.has_boundary else None
        self.object_files = [dataset_config['object_folder'].format(id) for id in ids] if self.has_object else None
        
        self.palette = dataset_config['palette']
        
        # 检查文件是否存在
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                print(f"警告: {f} 文件不存在!")
        
        # 缓存字典
        self.data_cache_ = {}
        self.label_cache_ = {}
        self.boundary_cache_ = {}
        self.object_cache_ = {}

    def __len__(self):
        return self.config.BATCH_SIZE * 1000  # 每个epoch的样本数

    def data_augmentation(self, *arrays):
        """数据增强：翻转和镜像"""
        will_flip = random.random() < 0.5
        will_mirror = random.random() < 0.5
        
        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
        return tuple(results)

    def __getitem__(self, i):
        # 随机选择一个tile
        random_idx = random.randint(0, len(self.data_files) - 1)
        
        # 加载数据（带缓存）
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            data = io.imread(self.data_files[random_idx])
            data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data
        
        # 加载标签
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            if self.config.DATASET == 'Urban':
                label = np.asarray(io.imread(self.label_files[random_idx]), dtype='int64') - 1
            else:
                label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx]), self.palette), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label
        
        # 加载边界（如果有的话）
        if self.has_boundary:
            if random_idx in self.boundary_cache_.keys():
                boundary = self.boundary_cache_[random_idx]
            else:
                boundary = np.asarray(io.imread(self.boundary_files[random_idx])) / 255
                boundary = boundary.astype(np.int64)
                if self.cache:
                    self.boundary_cache_[random_idx] = boundary
        else:
            # 创建虚拟边界掩码（全零）
            boundary = np.zeros(label.shape, dtype=np.int64)
        
        # 加载对象（如果有的话）
        if self.has_object:
            if random_idx in self.object_cache_.keys():
                object_mask = self.object_cache_[random_idx]
            else:
                object_mask = np.asarray(io.imread(self.object_files[random_idx]))
                object_mask = object_mask.astype(np.int64)
                if self.cache:
                    self.object_cache_[random_idx] = object_mask
        else:
            # 创建虚拟对象掩码（全零）
            object_mask = np.zeros(label.shape, dtype=np.int64)
        
        # 获取随机patch
        x1, x2, y1, y2 = get_random_pos(data, self.config.WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]
        boundary_p = boundary[x1:x2, y1:y2]
        object_p = object_mask[x1:x2, y1:y2]
        
        # 数据增强
        if self.augmentation:
            data_p, boundary_p, object_p, label_p = self.data_augmentation(data_p, boundary_p, object_p, label_p)
        
        object_p = object_process(object_p)
        
        return (torch.from_numpy(data_p),
                torch.from_numpy(boundary_p),
                torch.from_numpy(object_p),
                torch.from_numpy(label_p))

# ===== 损失函数 =====
class CrossEntropy2d_ignore(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d_ignore, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        n, _, _, _ = pred.shape
        pred = torch.softmax(pred, dim=1)
        class_map = pred.argmax(dim=1).cpu()
        
        # 边界图
        gt_b = F.max_pool2d(
            1 - gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - gt
        
        pred_b = F.max_pool2d(
            1 - class_map, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - class_map
        
        # 扩展边界图
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)
        
        # 重塑
        gt_b = gt_b.view(n, 2, -1)
        pred_b = pred_b.view(n, 2, -1)
        gt_b_ext = gt_b_ext.view(n, 2, -1)
        pred_b_ext = pred_b_ext.view(n, 2, -1)
        
        # 精确度和召回率
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)
        
        # 边界F1分数
        BF1 = 2 * P * R / (P + R + 1e-7)
        loss = torch.mean(1 - BF1)
        return loss

class ObjectLoss(nn.Module):
    def __init__(self, max_object=50):
        super().__init__()
        self.max_object = max_object

    def forward(self, pred, gt):
        num_object = int(torch.max(gt)) + 1
        num_object = min(num_object, self.max_object)
        total_object_loss = 0

        for object_index in range(1, num_object):
            mask = torch.where(gt == object_index, 1, 0).unsqueeze(1).to('cuda')
            num_point = mask.sum(2).sum(2).unsqueeze(2).unsqueeze(2).to('cuda')
            avg_pool = mask / (num_point + 1)

            object_feature = pred.mul(avg_pool)
            avg_feature = object_feature.sum(2).sum(2).unsqueeze(2).unsqueeze(2).repeat(1, 1, gt.shape[1], gt.shape[2])
            avg_feature = avg_feature.mul(mask)

            object_loss = torch.nn.functional.mse_loss(num_point * object_feature, avg_feature, reduction='mean')
            total_object_loss = total_object_loss + object_loss

        return total_object_loss

# ===== 训练和测试函数 =====
def train_epoch(model, train_loader, optimizer, criterion_ce, criterion_boundary, criterion_object, config, device):
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_boundary_loss = 0.0
    total_object_loss = 0.0
    
    for batch_idx, (data, boundary, object_mask, target) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        boundary, object_mask = boundary.to(device), object_mask.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # 计算损失
        loss_ce = criterion_ce(output, target)
        loss = loss_ce
        
        if config.USE_BOUNDARY_LOSS:
            loss_boundary = criterion_boundary(output, boundary)
            loss += config.LAMBDA_BOUNDARY * loss_boundary
            total_boundary_loss += loss_boundary.item()
        
        if config.USE_OBJECT_LOSS:
            loss_object = criterion_object(output, object_mask)
            loss += config.LAMBDA_OBJECT * loss_object
            total_object_loss += loss_object.item()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_ce_loss += loss_ce.item()
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.6f}, CE: {loss_ce.item():.6f}, '
                  f'Boundary: {total_boundary_loss/(batch_idx+1):.6f}, '
                  f'Object: {total_object_loss/(batch_idx+1):.6f}')
    
    return total_loss / len(train_loader)

def calculate_metrics(predictions, targets, labels):
    """计算评估指标"""
    cm = confusion_matrix(targets, predictions, labels=range(len(labels)))
    
    # 总体精度
    total = np.sum(cm)
    accuracy = np.sum(np.diag(cm)) / total * 100
    
    # MIoU
    miou = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    mean_miou = np.nanmean(miou[:len(labels)])
    
    print(f"总体精度: {accuracy:.2f}%")
    print(f"平均MIoU: {mean_miou:.4f}")
    
    return mean_miou

def main():
    # 配置
    config = Config()
    
    # 设置随机种子
    set_seed(config.SEED)
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_ID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 获取数据集配置
    dataset_config = get_dataset_config(config)
    
    # 创建模型
    model = UNetFormer(
        decode_channels=config.DECODE_CHANNELS,
        dropout=config.DROPOUT,
        backbone_name=config.BACKBONE,
        pretrained=config.PRETRAINED,
        window_size=config.WINDOW_SIZE_ATTENTION,
        num_classes=dataset_config['n_classes']
    ).to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params:,}")
    
    # 创建数据加载器
    train_dataset = UNetFormerDataset(dataset_config['train_ids'], dataset_config, config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # Dataset内部已随机
        num_workers=4,
        pin_memory=True
    )
    
    # 定义优化器和调度器
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
    
    # 定义损失函数
    weights = torch.ones(dataset_config['n_classes']).to(device)
    criterion_ce = CrossEntropy2d_ignore().to(device)
    criterion_boundary = BoundaryLoss() if config.USE_BOUNDARY_LOSS else None
    criterion_object = ObjectLoss() if config.USE_OBJECT_LOSS else None
    
    # 创建保存目录
    save_dir = f'./results_{config.DATASET.lower()}/'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"开始训练UNetFormer模型...")
    print(f"数据集: {config.DATASET}")
    print(f"训练样本: {len(dataset_config['train_ids'])}")
    print(f"测试样本: {len(dataset_config['test_ids'])}")
    print(f"类别数: {dataset_config['n_classes']}")
    
    # 训练循环
    best_score = 0.0
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        print("-" * 50)
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, 
            criterion_ce, criterion_boundary, criterion_object, 
            config, device
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"平均训练损失: {train_loss:.6f}, 学习率: {current_lr:.6f}")
        
        # 保存模型
        if epoch % config.SAVE_EPOCH == 0:
            save_path = os.path.join(save_dir, f'unetformer_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': config.__dict__
            }, save_path)
            print(f"模型已保存: {save_path}")
    
    print("训练完成!")

if __name__ == '__main__':
    main() 