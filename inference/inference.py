"""
简洁版推理脚本 - 可直接调用
"""

import os
import sys
import jittor as jt
from jittor import nn
import numpy as np
from PIL import Image
import json

current_dir = os.path.abspath('.')
project_root = os.path.dirname(current_dir)

if project_root in sys.path:
    sys.path.remove(project_root)

sys.path.insert(0, project_root)

from data_loader import get_dataloader, CLASS_NAMES
from models.vit_model import Visual_Transformer
from config import Config


class TomatoPredictor:
    def __init__(self, model_path, device='gpu'):
        """初始化预测器"""
        self.config = Config()
        jt.flags.use_cuda = 1 if device == 'gpu' and jt.has_cuda else 0
        
        print(f"📦 加载模型: {model_path}")
        
        # 创建模型
        self.model = Visual_Transformer(
            img_size=self.config.IMG_SIZE,
            patch_size=self.config.PATCH_SIZE,
            in_channels=self.config.IN_CHANNELS,
            embed_dim=self.config.EMBED_DIM,
            depth=self.config.NUM_LAYERS,
            num_heads=self.config.NUM_HEADS,
            dropout_rate=0.1,
            hidden_dim=self.config.MLP_Hidden_Dim
        )
        
        # 加载权重
        state_dict = jt.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"✅ 模型加载完成 (设备: {'GPU' if jt.flags.use_cuda else 'CPU'})\n")
    
    def preprocess_image(self, image_path):
        """预处理单张图片"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.config.IMG_SIZE, self.config.IMG_SIZE))
        
        # 转为数组并归一化
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # 标准化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # 转为 (C, H, W)
        img_array = img_array.transpose(2, 0, 1)
        img_tensor = jt.array(img_array[np.newaxis, :])
        
        return img_tensor
    
    def predict_single(self, image_path, top_k=3, verbose=True):
        """
        预测单张图片
        
        Args:
            image_path: 图片路径
            top_k: 返回前k个预测
            verbose: 是否打印结果
            
        Returns:
            [(类别名, 置信度), ...]
        """
        img_tensor = self.preprocess_image(image_path)
        
        with jt.no_grad():
            outputs = self.model(img_tensor)
            probabilities = nn.softmax(outputs, dim=1)[0].numpy()
        
        # 获取 top-k
        top_k_indices = np.argsort(probabilities)[::-1][:top_k]
        predictions = [
            (CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class_{idx}", 
             float(probabilities[idx]))
            for idx in top_k_indices
        ]
        
        if verbose:
            print(f"{'='*70}")
            print(f"🔍 推理: {os.path.basename(image_path)}")
            print(f"{'='*70}\n")
            print("📊 预测结果:\n")
            for i, (class_name, confidence) in enumerate(predictions, 1):
                bar_length = int(confidence * 40)
                bar = '█' * bar_length + '░' * (40 - bar_length)
                print(f"  {i}. {class_name:<30} |{bar}| {confidence*100:>6.2f}%")
            print()
        
        return predictions
    
    def predict_batch(self, image_folder, output_file=None, verbose=True):
        """
        批量推理文件夹中的图片
        
        Args:
            image_folder: 图片文件夹
            output_file: 保存结果的json文件
            verbose: 是否打印进度
        """
        # 获取图片文件
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in os.listdir(image_folder)
            if os.path.splitext(f)[1].lower() in image_exts
        ]
        
        if not image_files:
            print(f"❌ 在 {image_folder} 中未找到图片")
            return None
        
        if verbose:
            print(f"📂 找到 {len(image_files)} 张图片\n")
        
        results = {}
        
        for i, img_file in enumerate(image_files, 1):
            img_path = os.path.join(image_folder, img_file)
            
            try:
                predictions = self.predict_single(img_path, top_k=1, verbose=False)
                class_name, confidence = predictions[0]
                
                results[img_file] = {
                    'predicted_class': class_name,
                    'confidence': confidence
                }
                
                if verbose:
                    print(f"[{i}/{len(image_files)}] {img_file:<30} -> {class_name:<30} ({confidence*100:.2f}%)")
                
            except Exception as e:
                if verbose:
                    print(f"[{i}/{len(image_files)}] {img_file:<30} -> ❌ 错误: {e}")
                results[img_file] = {'error': str(e)}
        
        # 保存结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"\n✅ 结果已保存: {output_file}")
        
        # 统计
        if verbose:
            print(f"\n{'='*70}")
            print("📊 预测统计")
            print(f"{'='*70}")
            
            class_counts = {}
            total = 0
            for result in results.values():
                if 'predicted_class' in result:
                    cls = result['predicted_class']
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                    total += 1
            
            for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / total * 100 if total > 0 else 0
                print(f"  {cls:<30}: {count:>4} ({percentage:>5.1f}%)")
            print()
        
        return results
    
    def evaluate_testset(self, data_root, batch_size=8, verbose=True):
        """
        在测试集上评估
        
        Args:
            data_root: 数据集根目录
            batch_size: 批次大小
            verbose: 是否打印详细信息
        """
        if verbose:
            print("📦 加载测试集...")
        
        test_loader = get_dataloader(
            root_dir=data_root,
            mode='test',
            batch_size=batch_size,
            img_size=self.config.IMG_SIZE,
            shuffle=False,
            num_workers=0,
            sample_ratio=1.0
        )
        
        if verbose:
            print("🔍 评估中...\n")
        
        # 统计变量
        class_correct = np.zeros(self.config.NUM_CLASSES, dtype=np.int64)
        class_total = np.zeros(self.config.NUM_CLASSES, dtype=np.int64)
        
        processed = 0
        
        with jt.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                # 过滤无效标签
                valid_mask = (labels.numpy() >= 0) & (labels.numpy() < self.config.NUM_CLASSES)
                
                if not valid_mask.any():
                    continue
                
                images = images[jt.array(valid_mask)]
                labels = labels[jt.array(valid_mask)]
                
                # 推理
                outputs = self.model(images)
                preds = jt.argmax(outputs, dim=1)[0].numpy()
                labels_np = labels.numpy()
                
                # 统计
                for pred, label in zip(preds, labels_np):
                    label = int(label)
                    pred = int(pred)
                    if 0 <= label < self.config.NUM_CLASSES:
                        class_total[label] += 1
                        if pred == label:
                            class_correct[label] += 1
                
                processed += len(labels_np)
                
                # 进度显示
                if verbose and (batch_idx + 1) % 20 == 0:
                    current_acc = class_correct.sum() / class_total.sum() if class_total.sum() > 0 else 0
                    print(f"  处理: {processed} 样本 | 当前准确率: {current_acc*100:.2f}%")
        
        # 计算结果
        total_correct = int(class_correct.sum())
        total_samples = int(class_total.sum())
        overall_acc = total_correct / total_samples if total_samples > 0 else 0
        
        # 计算平衡准确率
        class_accs = []
        for i in range(self.config.NUM_CLASSES):
            if class_total[i] > 0:
                class_accs.append(class_correct[i] / class_total[i])
        balanced_acc = np.mean(class_accs) if class_accs else 0
        
        # 打印结果
        if verbose:
            print(f"\n{'='*70}")
            print("📊 评估结果")
            print(f"{'='*70}")
            print(f"总样本数: {total_samples}")
            print(f"正确预测: {total_correct}")
            print(f"总体准确率: {overall_acc*100:.2f}%")
            print(f"平衡准确率: {balanced_acc*100:.2f}%")
            
            print(f"\n{'='*70}")
            print("📋 各类别详细结果")
            print(f"{'='*70}")
            print(f"{'类别':<30} {'准确率':>10} {'正确/总数':>15}")
            print(f"{'-'*70}")
            
            for i in range(self.config.NUM_CLASSES):
                if i < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[i]
                    if class_total[i] > 0:
                        acc = class_correct[i] / class_total[i] * 100
                        print(f"{class_name:<30} {acc:>9.2f}% {int(class_correct[i]):>7}/{int(class_total[i]):<7}")
                    else:
                        print(f"{class_name:<30} {'N/A':>10} {'0/0':>15}")
            print()
        
        return {
            'overall_accuracy': overall_acc,
            'balanced_accuracy': balanced_acc,
            'class_correct': class_correct,
            'class_total': class_total,
            'total_samples': total_samples
        }


# ========== 便捷函数 ==========

def predict_image(model_path, image_path, top_k=5, device='gpu'):
    """快速预测单张图片"""
    predictor = TomatoPredictor(model_path, device=device)
    return predictor.predict_single(image_path, top_k=top_k)


def predict_folder(model_path, folder_path, output_file=None, device='gpu'):
    """快速预测文件夹"""
    predictor = TomatoPredictor(model_path, device=device)
    return predictor.predict_batch(folder_path, output_file=output_file)


def evaluate_model(model_path, data_root, batch_size=8, device='gpu'):
    """快速评估模型"""
    predictor = TomatoPredictor(model_path, device=device)
    return predictor.evaluate_testset(data_root, batch_size=batch_size)