import csv
import json
import math
import re

# 定义输入文件路径和输出 CSV 文件路径
input_file_path = 'tmp.out.sglang.tp8.ep8.dp.new2.cuda512'
output_file_path = 'tmp.out.sglang.tp8.ep8.dp.new2.cuda512.csv'

# 存储所有测试用例数据的列表
all_cases = []

# 手动指定配置信息和性能指标的顺序
predefined_keys = [
    'max_prefill', 'max_running_requests', 'torch_compile', 'is_dp', 'backend', 'dataset_name', 'request_rate',
    'max_concurrency', 'sharegpt_output_len', 'random_input_len', 'random_output_len', 'random_range_ratio',
    'duration', 'completed', 'total_input_tokens', 'total_output_tokens', 'total_output_tokens_retokenized',
    'request_throughput', 'input_throughput', 'output_throughput', 'mean_e2e_latency_ms', 'median_e2e_latency_ms',
    'std_e2e_latency_ms', 'p99_e2e_latency_ms', 'mean_ttft_ms', 'median_ttft_ms', 'std_ttft_ms', 'p99_ttft_ms',
    'mean_tpot_ms', 'median_tpot_ms', 'std_tpot_ms', 'p99_tpot_ms', 'mean_itl_ms', 'median_itl_ms', 'std_itl_ms',
    'p95_itl_ms', 'p99_itl_ms', 'concurrency', 'accept_length'
]

def clean_inf_nan_null(value):
    """统一处理无穷大、NaN、null值"""
    if isinstance(value, float):
        if math.isinf(value):
            return 'Infinity'
        elif math.isnan(value):
            return 'NaN'
    elif value is None:
        return 'null'
    return value

def parse_config_line(config_line):
    """解析单条配置行（max_prefill:...）"""
    config_info = {}
    config_parts = config_line.strip().split(',')
    for part in config_parts:
        if ':' in part:
            key, value = part.split(':', 1)
            # 类型转换
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
            config_info[key] = value
    return config_info

def extract_complete_json_from_lines(lines, start_idx):
    """从指定行开始，提取完整的JSON（处理多行JSON）"""
    json_lines = []
    brace_count = 0
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        # 统计大括号层级
        brace_count += line.count('{')
        brace_count -= line.count('}')
        json_lines.append(line)
        # 当大括号层级归0时，说明JSON结束
        if brace_count == 0:
            break
    # 拼接完整JSON并解析
    if brace_count != 0:
        raise ValueError(f"JSON不完整，剩余大括号层级: {brace_count} (起始行: {start_idx+1})")
    complete_json_str = ''.join(json_lines)
    return json.loads(complete_json_str), i + 1  # 返回解析后的JSON和下一个起始行

try:
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        i = 0
        case_num = 0

        # 循环解析每一条case（配置行 + JSON块）
        while i < len(lines):
            line = lines[i].strip()
            # 匹配case的配置行（以max_prefill开头）
            if line.startswith('max_prefill'):
                case_num += 1
                print(f"正在解析第 {case_num} 条case (行号: {i+1})")

                # 1. 解析配置行
                config_info = parse_config_line(line)

                # 2. 找到下一行开始的JSON并解析
                i += 1
                if i >= len(lines):
                    raise ValueError(f"第 {case_num} 条case缺少JSON数据（配置行后无内容）")

                try:
                    metrics_info, i = extract_complete_json_from_lines(lines, i)
                except json.JSONDecodeError as e:
                    print(f"⚠️ 第 {case_num} 条case JSON解析失败，跳过 - {e}")
                    i += 1
                    continue
                except ValueError as e:
                    print(f"⚠️ 第 {case_num} 条case JSON不完整，跳过 - {e}")
                    i += 1
                    continue

                # 3. 扁平化JSON数据
                flat_metrics = {}
                top_level_fields = [
                    'backend', 'dataset_name', 'request_rate', 'max_concurrency',
                    'sharegpt_output_len', 'random_input_len', 'random_output_len',
                    'random_range_ratio', 'duration', 'completed', 'total_input_tokens',
                    'total_output_tokens', 'total_output_tokens_retokenized', 'request_throughput',
                    'input_throughput', 'output_throughput', 'mean_e2e_latency_ms', 'median_e2e_latency_ms',
                    'std_e2e_latency_ms', 'p99_e2e_latency_ms', 'mean_ttft_ms', 'median_ttft_ms',
                    'std_ttft_ms', 'p99_ttft_ms', 'mean_tpot_ms', 'median_tpot_ms', 'std_tpot_ms',
                    'p99_tpot_ms', 'mean_itl_ms', 'median_itl_ms', 'std_itl_ms', 'p95_itl_ms',
                    'p99_itl_ms', 'concurrency', 'accept_length'
                ]
                for field in top_level_fields:
                    if field in metrics_info:
                        flat_metrics[field] = clean_inf_nan_null(metrics_info[field])

                # 4. 合并配置和性能数据
                case_info = {**config_info, **flat_metrics}

                # 确保所有预定义字段都存在
                for key in predefined_keys:
                    if key not in case_info:
                        case_info[key] = ''

                all_cases.append(case_info)
            else:
                i += 1  # 非配置行，跳过

    # 写入CSV文件
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=predefined_keys)
        writer.writeheader()
        for case in all_cases:
            writer.writerow(case)

    print(f"\n✅ 解析完成！共处理 {case_num} 条case，成功解析 {len(all_cases)} 条")
    print(f"✅ 数据已写入 {output_file_path}")

except FileNotFoundError:
    print(f"❌ 错误：未找到输入文件 '{input_file_path}'")
except Exception as e:
    print(f"❌ 未知错误: {e}")
    # 调试时开启，打印完整错误堆栈
    import traceback
    traceback.print_exc()
