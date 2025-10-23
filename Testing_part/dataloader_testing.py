import os  
import sys  
import torch  
from PIL import Image  
  
# 添加项目根目录到 Python 路径  
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  
  
from dataloader.stage1_dataset import PretrainDataset_Contact  
  
def test_dataloader():  
    """验证数据加载器是否正常工作"""  
      
    print("=" * 60)  
    print("开始验证数据加载器")  
    print("=" * 60)  
      
    try:  
        # 1. 创建数据集实例  
        print("\n[步骤 1] 创建数据集实例...")  
        dataset = PretrainDataset_Contact(mode='train')  
        print(f"✓ 数据集创建成功")  
        print(f"  数据集大小: {len(dataset)} 个样本")  
          
        if len(dataset) == 0:  
            print("✗ 警告: 数据集为空,请检查 CSV 文件和数据路径")  
            return False  
          
        # 2. 测试加载单个样本  
        print("\n[步骤 2] 测试加载第一个样本...")  
        img = dataset[0]  
        print(f"✓ 样本加载成功")  
        print(f"  图像形状: {img.shape}")  
        print(f"  图像数据类型: {img.dtype}")  
        print(f"  图像值范围: [{img.min():.3f}, {img.max():.3f}]")  
          
        # 验证形状  
        expected_shape = torch.Size([4, 3, 224, 224])  
        if img.shape != expected_shape:  
            print(f"✗ 错误: 图像形状不正确,期望 {expected_shape},实际 {img.shape}")  
            return False  
          
        # 3. 测试加载多个样本  
        print("\n[步骤 3] 测试加载多个样本...")  
        num_test_samples = min(10, len(dataset))  
        for i in range(num_test_samples):  
            try:  
                img = dataset[i]  
                assert img.shape == expected_shape, f"样本 {i} 形状错误"  
                print(f"  样本 {i}: ✓")  
            except Exception as e:  
                print(f"  样本 {i}: ✗ 错误 - {str(e)}")  
                if i < len(dataset.datalist):  
                    print(f"    路径: {dataset.datalist[i]}")  
                return False  
          
        # 4. 测试 DataLoader  
        print("\n[步骤 4] 测试 DataLoader...")  
        from torch.utils.data import DataLoader  
        dataloader = DataLoader(  
            dataset,   
            batch_size=2,   
            shuffle=False,   
            num_workers=0  
        )  
          
        batch = next(iter(dataloader))  
        print(f"✓ DataLoader 工作正常")  
        print(f"  批次形状: {batch.shape}")  
        print(f"  批次大小: {batch.shape[0]}")  
          
        # 5. 检查数据路径  
        print("\n[步骤 5] 检查数据路径...")  
        sample_paths = dataset.datalist[0]  
        print(f"  第一个样本的路径:")  
        for i, path in enumerate(sample_paths):  
            exists = "✓" if os.path.exists(path) else "✗"  
            print(f"    帧 {i}: {exists} {path}")  
          
        # 6. 统计数据来源  
        print("\n[步骤 6] 统计数据来源...")  
        tacquad_count = sum(1 for paths in dataset.datalist if 'Tacquad' in paths[0])  
        tag_count = sum(1 for paths in dataset.datalist if 'TAG' in paths[0])  
        print(f"  Tacquad 样本数: {tacquad_count}")  
        print(f"  TAG 样本数: {tag_count}")  
          
        # 7. 测试数据增强  
        print("\n[步骤 7] 测试数据增强...")  
        dataset_eval = PretrainDataset_Contact(mode='eval')  
        img_eval = dataset_eval[0]  
        print(f"✓ 评估模式数据加载成功")  
        print(f"  评估模式图像形状: {img_eval.shape}")  
          
        print("\n" + "=" * 60)  
        print("✓ 所有验证测试通过!")  
        print("=" * 60)  
        return True  
          
    except Exception as e:  
        print(f"\n✗ 验证失败: {str(e)}")  
        import traceback  
        traceback.print_exc()  
        return False  
  
def test_csv_files():  
    """测试 CSV 文件是否存在且格式正确"""  
    print("\n" + "=" * 60)  
    print("测试 CSV 文件")  
    print("=" * 60)  
      
    csv_files = [  
        '/data/tactile_datasets/S1_dataset/Tacquad/Tacquad.csv',  
        '/data/tactile_datasets/S1_dataset/TAG/TAG.csv'  
    ]  
      
    for csv_file in csv_files:  
        if os.path.exists(csv_file):  
            print(f"✓ {csv_file} 存在")  
            try:  
                import csv  
                with open(csv_file, 'r') as f:  
                    reader = csv.reader(f)  
                    rows = list(reader)  
                    print(f"  行数: {len(rows)}")  
                    if len(rows) > 0:  
                        print(f"  第一行: {rows[0]}")  
                    if len(rows) > 1:  
                        print(f"  第二行: {rows[1]}")  
            except Exception as e:  
                print(f"✗ 读取 CSV 文件失败: {str(e)}")  
        else:  
            print(f"✗ {csv_file} 不存在")  
  
if __name__ == "__main__":  
    # 测试 CSV 文件  
    test_csv_files()  
      
    # 测试数据加载器  
    success = test_dataloader()  
      
    if success:  
        print("\n数据加载验证完成,可以开始训练!")  
        sys.exit(0)  
    else:  
        print("\n数据加载验证失败,请检查配置!")  
        sys.exit(1)