import torch
import torch.nn as nn
from model.FTUNetFormer_v2 import ft_unetformer_v2

def test_architecture():
    """
    测试改进版本的FTUNetFormer架构
    """
    print("=" * 60)
    print("测试 FTUNetFormer V2 架构")
    print("=" * 60)
    
    # 测试参数
    batch_size = 2
    height, width = 256, 256
    num_classes = 6
    
    # 创建测试数据
    rgb_input = torch.randn(batch_size, 3, height, width).cuda()
    dsm_input = torch.randn(batch_size, 1, height, width).cuda()
    
    print(f"输入数据形状:")
    print(f"RGB: {rgb_input.shape}")
    print(f"DSM: {dsm_input.shape}")
    print()
    
    # 测试配置1：仅RGB（baseline）
    print("配置1: 仅RGB分支 (baseline)")
    print("-" * 30)
    model1 = ft_unetformer_v2(
        pretrained=False,
        num_classes=num_classes,
        use_fusion=False
    ).cuda()
    
    with torch.no_grad():
        output1 = model1(rgb_input, None)
    
    print(f"输出形状: {output1.shape}")
    print(f"参数数量: {sum(p.numel() for p in model1.parameters()):,}")
    print()
    
    # 测试配置2：简单融合
    print("配置2: 双分支 + 简单融合")
    print("-" * 30)
    model2 = ft_unetformer_v2(
        pretrained=False,
        num_classes=num_classes,
        use_fusion=True,
        fusion_type='simple'
    ).cuda()
    
    with torch.no_grad():
        output2 = model2(rgb_input, dsm_input)
    
    print(f"输出形状: {output2.shape}")
    print(f"参数数量: {sum(p.numel() for p in model2.parameters()):,}")
    print()
    
    # 测试配置3：频率域融合
    print("配置3: 双分支 + 频率域融合")
    print("-" * 30)
    model3 = ft_unetformer_v2(
        pretrained=False,
        num_classes=num_classes,
        use_fusion=True,
        fusion_type='frequency'
    ).cuda()
    
    with torch.no_grad():
        output3 = model3(rgb_input, dsm_input)
    
    print(f"输出形状: {output3.shape}")
    print(f"参数数量: {sum(p.numel() for p in model3.parameters()):,}")
    print()
    
    # 测试分辨率差异处理
    print("配置4: 测试分辨率差异处理")
    print("-" * 30)
    rgb_input_diff = torch.randn(batch_size, 3, 256, 256).cuda()
    dsm_input_diff = torch.randn(batch_size, 1, 128, 128).cuda()  # 不同分辨率
    
    print(f"不同分辨率输入:")
    print(f"RGB: {rgb_input_diff.shape}")
    print(f"DSM: {dsm_input_diff.shape}")
    
    with torch.no_grad():
        output4 = model3(rgb_input_diff, dsm_input_diff)
    
    print(f"输出形状: {output4.shape}")
    print("✓ 分辨率差异处理成功!")
    print()
    
    # 测试渐进式训练
    print("配置5: 测试渐进式训练模式")
    print("-" * 30)
    
    # 第一阶段：仅RGB
    print("阶段1: 仅RGB训练")
    model3.use_fusion = False
    with torch.no_grad():
        output5a = model3(rgb_input, None)
    print(f"输出形状: {output5a.shape}")
    
    # 第二阶段：启用融合
    print("阶段2: 启用DSM融合")
    model3.use_fusion = True
    with torch.no_grad():
        output5b = model3(rgb_input, dsm_input)
    print(f"输出形状: {output5b.shape}")
    print("✓ 渐进式训练模式测试成功!")
    print()
    
    print("=" * 60)
    print("架构测试总结")
    print("=" * 60)
    print("✓ 所有配置都能正常运行")
    print("✓ 分辨率自适应功能正常")
    print("✓ 渐进式训练支持正常")
    print("✓ 不同融合策略都可用")
    print()
    print("建议使用顺序:")
    print("1. 先使用配置2（简单融合）进行快速验证")
    print("2. 如果效果好，再尝试配置3（频率域融合）")
    print("3. 使用渐进式训练策略获得最佳效果")

if __name__ == "__main__":
    test_architecture() 