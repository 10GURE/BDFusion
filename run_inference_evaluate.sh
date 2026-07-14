#!/bin/bash

# BDFusion 推理与评估快速启动脚本

echo "=========================================="
echo "    BDFusion 推理与评估系统"
echo "=========================================="
echo ""

# 默认路径
RGB_DIR="./experiments/instance/rgb"
IR_DIR="./experiments/instance/ir"
OUTPUT_DIR="./experiments/instance/output"
GT_DIR="./data/msrs_bd-f/gt_clean"
CHECKPOINT="./checkpoint/last.ckpt"

# 检查必需文件
echo "检查必需文件..."
if [ ! -f "$CHECKPOINT" ]; then
    echo "错误：找不到模型权重文件 $CHECKPOINT"
    exit 1
fi

if [ ! -d "$RGB_DIR" ] || [ ! -d "$IR_DIR" ]; then
    echo "错误：找不到输入目录 $RGB_DIR 或 $IR_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "配置信息："
echo "  RGB输入: $RGB_DIR"
echo "  IR输入: $IR_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  GT参考: $GT_DIR"
echo "  模型权重: $CHECKPOINT"
echo ""

# 运行推理与评估
echo "开始推理与评估..."
python inference_and_evaluate.py \
    --rgb_dir "$RGB_DIR" \
    --ir_dir "$IR_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --gt_dir "$GT_DIR" \
    --checkpoint "$CHECKPOINT"

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  执行成功！"
    echo "=========================================="
    echo ""
    echo "输出文件："
    echo "  - 融合图像: $OUTPUT_DIR/enh_*.png"
    echo "  - 评估指标: $OUTPUT_DIR/metrics.csv"
    echo "  - 统计摘要: $OUTPUT_DIR/metrics_summary.txt"
    echo ""
    
    # 显示统计摘要
    if [ -f "$OUTPUT_DIR/metrics_summary.txt" ]; then
        echo "评估结果摘要："
        cat "$OUTPUT_DIR/metrics_summary.txt"
    fi
else
    echo ""
    echo "执行失败，请检查错误信息。"
    exit 1
fi
