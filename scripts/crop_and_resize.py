#!/usr/bin/env python3
"""
裁剪4:3比例图片的中间正方形部分并缩放到256x256
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def crop_and_resize(
    input_dir,
    output_dir,
    target_size=256,
    image_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff'),
    recursive=True
):
    """
    处理输入目录中的所有图片，支持递归处理子目录
    
    Args:
        input_dir: 输入图片目录
        output_dir: 输出图片目录
        target_size: 目标尺寸（默认256x256）
        image_extensions: 图片扩展名
        recursive: 是否递归处理子目录（默认True）
    """
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 获取所有图片文件（支持递归）
    image_files = []
    if recursive:
        pattern = '**/*'
    else:
        pattern = '*'
    
    for item in input_path.glob(pattern):
        if item.is_file():
            if item.suffix.lower() in image_extensions:
                image_files.append(item)
    
    if not image_files:
        logger.warning(f"在 {input_dir} 中未找到图片文件")
        return
    
    logger.info(f"找到 {len(image_files)} 张图片")
    
    success_count = 0
    error_count = 0
    
    for image_file in sorted(image_files):
        try:
            # 计算相对路径，用于在输出目录中保持相同的структуру
            relative_path = image_file.relative_to(input_path)
            output_file = output_path / relative_path
            
            # 创建输出文件的父目录
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 打开图片
            img = Image.open(image_file)
            width, height = img.size
            
            logger.info(f"处理: {relative_path} (分辨率: {width}x{height})")
            
            # 计算中间正方形的尺寸和位置
            # 取宽度和高度的较小值作为正方形边长
            square_size = min(width, height)
            
            # 计算裁剪的起始位置（居中）
            left = (width - square_size) // 2
            top = (height - square_size) // 2
            right = left + square_size
            bottom = top + square_size
            
            # 裁剪中间正方形
            cropped = img.crop((left, top, right, bottom))
            
            # 缩放到目标尺寸
            resized = cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
            # 保存结果
            resized.save(output_file, quality=95)
            
            logger.info(f"✓ 保存: {output_file}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"✗ 处理失败 {image_file.name}: {str(e)}")
            error_count += 1
    
    logger.info(f"\n处理完成！成功: {success_count}，失败: {error_count}")


def main():
    parser = argparse.ArgumentParser(
        description='裁剪4:3比例图片的中间正方形部分并缩放到256x256'
    )
    parser.add_argument(
        'input_dir',
        help='输入图片目录'
    )
    parser.add_argument(
        '-o', '--output',
        default='output_cropped',
        help='输出目录（默认: output_cropped）'
    )
    parser.add_argument(
        '-s', '--size',
        type=int,
        default=256,
        help='目标尺寸（默认: 256）'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='不递归处理子目录'
    )
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.isdir(args.input_dir):
        logger.error(f"错误: 输入目录不存在: {args.input_dir}")
        return
    
    crop_and_resize(
        input_dir=args.input_dir,
        output_dir=args.output,
        target_size=args.size,
        recursive=not args.no_recursive
    )


if __name__ == '__main__':
    main()
