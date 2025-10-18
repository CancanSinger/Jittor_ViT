
import sys
import os
import argparse
import json

current_dir = os.path.abspath('.')
project_root = current_dir

if project_root in sys.path:
    sys.path.remove(project_root)

sys.path.insert(0, project_root)

import jittor as jt
from jittor import nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import Config
from models.vit_model import Visual_Transformer
from data_loader import get_data_loaders, create_dummy_data_loaders

config = Config()

def load_model(model_path):
    # 创建模型实例
    model = Visual_Transformer()

    # 加载模型参数
    model.load_state_dict(jt.load(model_path))
    model.eval()

    return model

def preprocess_image(image_path):
    # 图像预处理
    from jittor.transform import Compose, Resize, ToTensor, Normalize

    transform = Compose([
        Resize((config.IMG_SIZE, config.IMG_SIZE)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载图像
    image = Image.open(image_path).convert('RGB')

    # 应用变换
    image = transform(image)

    # 添加批次维度
    image = jt.unsqueeze(image, 0)

    return image

def predict_single_image(model, image_path, class_names=None):
    # 预处理图像
    image = preprocess_image(image_path)

    # 模型推理
    with jt.no_grad():
        outputs = model(image)
        probabilities = nn.softmax(outputs, dim=1)[0]
        _, predicted = jt.argmax(outputs, 1)
        predicted_class = predicted.item()
        confidence = probabilities[predicted_class].item()

    # 准备结果
    result = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities.numpy().tolist()
    }

    # 如果提供了类别名称，添加到结果中
    if class_names:
        result['predicted_class_name'] = class_names[predicted_class]

    return result

def predict_directory(model, input_dir, output_dir, class_names=None):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))

    # 处理每张图像
    results = {}

    for image_path in tqdm(image_files, desc='Processing images'):
        # 预测
        result = predict_single_image(model, image_path, class_names)

        # 保存结果
        relative_path = os.path.relpath(image_path, input_dir)
        results[relative_path] = result

    # 保存结果到JSON文件
    output_file = os.path.join(output_dir, 'predictions.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f'结果已保存到 {output_file}')

    return results

def evaluate_model(model, data_dir, class_names=None):
    # 加载验证数据
    _, val_loader = get_data_loaders(
        train_dir=data_dir,  # 这里使用data_dir作为训练目录，实际会划分出验证集
        batch_size=32
    )

    # 评估模型
    correct = 0
    total = 0
    class_correct = [0] * config.NUM_CLASSES
    class_total = [0] * config.NUM_CLASSES

    with jt.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Evaluating model'):
            outputs = model(inputs)
            _, predicted = jt.argmax(outputs, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # 统计每个类别的准确率
            c = (predicted == targets).squeeze()
            for i in range(targets.size(0)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # 计算整体准确率
    overall_acc = 100 * correct / total

    # 计算每个类别的准确率
    class_accs = []
    for i in range(config.NUM_CLASSES):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            class_name = class_names[i] if class_names else f'Class {i}'
            class_accs.append({'class': class_name, 'accuracy': class_acc})
        else:
            class_name = class_names[i] if class_names else f'Class {i}'
            class_accs.append({'class': class_name, 'accuracy': 0.0})

    # 打印结果
    print(f'整体准确率: {overall_acc:.2f}%')
    print('各类别准确率:')
    for class_acc in class_accs:
        print(f"  {class_acc['class']}: {class_acc['accuracy']:.2f}%")

    return {
        'overall_accuracy': overall_acc,
        'class_accuracies': class_accs
    }

def main():
    parser = argparse.ArgumentParser(description='ViT模型推理')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--input', type=str, required=True, help='输入图像或目录路径')
    parser.add_argument('--output_dir', type=str, default='inference_results', help='输出目录')
    parser.add_argument('--mode', type=str, choices=['single', 'directory', 'evaluate'], default='single', 
                        help='推理模式: single(单张图像), directory(目录), evaluate(评估)')
    parser.add_argument('--data_dir', type=str, default='data/val', help='数据目录(用于评估模式)')
    parser.add_argument('--class_names', type=str, nargs='+', help='类别名称列表')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print(f'加载模型: {args.model_path}')
    model = load_model(args.model_path)

    # 根据模式执行推理
    if args.mode == 'single':
        # 单张图像推理
        print(f'对单张图像进行推理: {args.input}')
        result = predict_single_image(model, args.input, args.class_names)

        # 打印结果
        predicted_class = result['predicted_class']
        confidence = result['confidence']

        if 'predicted_class_name' in result:
            print(f"预测类别: {result['predicted_class_name']} (ID: {predicted_class})")
        else:
            print(f"预测类别 ID: {predicted_class}")

        print(f"置信度: {confidence:.4f}")

        # 保存结果
        output_file = os.path.join(args.output_dir, 'single_prediction.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)

        print(f'结果已保存到 {output_file}')

    elif args.mode == 'directory':
        # 目录推理
        print(f'对目录中的图像进行推理: {args.input}')
        results = predict_directory(model, args.input, args.output_dir, args.class_names)

    elif args.mode == 'evaluate':
        # 模型评估
        print(f'在数据集上评估模型: {args.data_dir}')
        results = evaluate_model(model, args.data_dir, args.class_names)

        # 保存评估结果
        output_file = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f'评估结果已保存到 {output_file}')

if __name__ == '__main__':
    main()
