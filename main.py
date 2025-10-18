
import sys
import os
import argparse
import subprocess

current_dir = os.path.abspath('.')
project_root = current_dir

if project_root in sys.path:
    sys.path.remove(project_root)

sys.path.insert(0, project_root)

def run_command(command):
    """运行命令并打印输出"""
    print(f"执行命令: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 实时打印输出
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    # 获取返回码
    return_code = process.poll()

    # 如果有错误，打印错误信息
    if return_code != 0:
        error_output = process.stderr.read()
        print(f"错误: {error_output}")

    return return_code

def train(args):
    """训练模型"""
    command = [
        'python', 'train.py',
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--weight_decay', str(args.weight_decay),
        '--output_dir', args.output_dir
    ]

    if args.use_dummy_data:
        command.append('--use_dummy_data')
    else:
        command.extend(['--train_dir', args.train_dir])
        if args.val_dir:
            command.extend(['--val_dir', args.val_dir])

    return run_command(command)

def inference(args):
    """推理"""
    command = [
        'python', 'inference.py',
        '--model_path', args.model_path,
        '--input', args.input,
        '--output_dir', args.output_dir,
        '--mode', args.mode
    ]

    if args.data_dir:
        command.extend(['--data_dir', args.data_dir])

    if args.class_names:
        command.extend(['--class_names'] + args.class_names)

    return run_command(command)

def main():
    parser = argparse.ArgumentParser(description='ViT模型训练和推理')
    subparsers = parser.add_subparsers(dest='command', help='命令')

    # 训练参数
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--train_dir', type=str, default='data/train', help='训练数据目录')
    train_parser.add_argument('--val_dir', type=str, default='data/val', help='验证数据目录')
    train_parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    train_parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    train_parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    train_parser.add_argument('--weight_decay', type=float, default=5e-2, help='权重衰减')
    train_parser.add_argument('--use_dummy_data', action='store_true', help='使用虚拟数据进行训练')

    # 推理参数
    inference_parser = subparsers.add_parser('inference', help='模型推理')
    inference_parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    inference_parser.add_argument('--input', type=str, required=True, help='输入图像或目录路径')
    inference_parser.add_argument('--output_dir', type=str, default='inference_results', help='输出目录')
    inference_parser.add_argument('--mode', type=str, choices=['single', 'directory', 'evaluate'], 
                                default='single', help='推理模式')
    inference_parser.add_argument('--data_dir', type=str, default='data/val', help='数据目录(用于评估模式)')
    inference_parser.add_argument('--class_names', type=str, nargs='+', help='类别名称列表')

    # 完整流程参数
    full_parser = subparsers.add_parser('full', help='完整的训练和推理流程')
    full_parser.add_argument('--train_dir', type=str, default='data/train', help='训练数据目录')
    full_parser.add_argument('--val_dir', type=str, default='data/val', help='验证数据目录')
    full_parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    full_parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    full_parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    full_parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    full_parser.add_argument('--weight_decay', type=float, default=5e-2, help='权重衰减')
    full_parser.add_argument('--use_dummy_data', action='store_true', help='使用虚拟数据进行训练')
    full_parser.add_argument('--test_dir', type=str, default=None, help='测试数据目录(用于推理)')
    full_parser.add_argument('--class_names', type=str, nargs='+', help='类别名称列表')

    args = parser.parse_args()

    if args.command == 'train':
        # 训练模型
        print("开始训练模型...")
        return_code = train(args)
        if return_code == 0:
            print("训练完成!")
        else:
            print("训练失败!")
            sys.exit(1)

    elif args.command == 'inference':
        # 推理
        print("开始推理...")
        return_code = inference(args)
        if return_code != 0:
            print("推理失败!")
            sys.exit(1)

    elif args.command == 'full':
        # 完整流程：训练和推理
        print("开始完整的训练和推理流程...")

        # 1. 训练模型
        print("\n步骤1: 训练模型")
        return_code = train(args)
        if return_code != 0:
            print("训练失败!")
            sys.exit(1)

        # 2. 推理
        print("\n步骤2: 模型推理")
        model_path = os.path.join(args.output_dir, 'best_model.pth')

        if args.test_dir:
            # 使用测试目录进行推理
            inference_args = argparse.Namespace(
                model_path=model_path,
                input=args.test_dir,
                output_dir=os.path.join(args.output_dir, 'inference'),
                mode='directory',
                data_dir=None,
                class_names=args.class_names
            )
        else:
            # 使用验证集进行评估
            inference_args = argparse.Namespace(
                model_path=model_path,
                input=args.val_dir,
                output_dir=os.path.join(args.output_dir, 'inference'),
                mode='evaluate',
                data_dir=args.val_dir,
                class_names=args.class_names
            )

        return_code = inference(inference_args)
        if return_code != 0:
            print("推理失败!")
            sys.exit(1)

        print("\n完整的训练和推理流程完成!")

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
