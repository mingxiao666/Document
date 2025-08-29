import csv
import json
import math

# 定义输入文件路径和输出 CSV 文件路径
input_file_path ='tmp-6k.out'
output_file_path = 'tmp-6k.out.csv'

# 存储所有测试用例数据的列表
all_cases = []

# 手动指定配置信息和性能指标的顺序
predefined_keys = [
    'max_prefill', 'max_running_requests', 'torch_compile', 'is_dp', 'backend', 'dataset_name', 'request_rate', 'max_concurrency', 'sharegpt_output_len', 'random_input_len', 'random_output_len', 'random_range_ratio', 'duration', 'completed', 'total_input_tokens', 'total_output_tokens', 'total_output_tokens_retokenized', 'request_throughput', 'input_throughput', 'output_throughput', 'mean_e2e_latency_ms', 'median_e2e_latency_ms', 'std_e2e_latency_ms', 'p99_e2e_latency_ms', 'mean_ttft_ms', 'median_ttft_ms', 'std_ttft_ms', 'p99_ttft_ms', 'mean_tpot_ms', 'median_tpot_ms', 'std_tpot_ms', 'p99_tpot_ms', 'mean_itl_ms', 'median_itl_ms', 'std_itl_ms', 'p95_itl_ms', 'p99_itl_ms', 'concurrency', 'accept_length'
]

try:
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('max_prefill'):
                # 解析配置信息
                config_info = {}
                config_parts = line.split(',')
                for part in config_parts:
                    key, value = part.split(':')
                    # 尝试将值转换为合适的数据类型
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

                i += 1
                if i < len(lines):
                    metrics_line = lines[i].strip()
                    try:
                        metrics_info = json.loads(metrics_line)
                        # 处理 Infinity、NaN 和 null 值
                        for key, value in metrics_info.items():
                            if isinstance(value, float) and math.isinf(value):
                                metrics_info[key] = 'Infinity'
                            elif isinstance(value, float) and math.isnan(value):
                                metrics_info[key] = 'NaN'
                            elif value is None:
                                metrics_info[key] = 'null'
                        # 合并配置信息和性能指标信息
                        case_info = {**config_info, **metrics_info}
                        all_cases.append(case_info)
                    except json.JSONDecodeError:
                        print(f"解析 JSON 数据时出错，行内容: {metrics_line}")
            i += 1

    # 打开输出 CSV 文件进行写入
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=predefined_keys)
        # 写入 CSV 文件的表头
        writer.writeheader()
        # 逐行写入每个测试用例的数据
        for case in all_cases:
            # 确保每个键都存在于数据中，若不存在则设为空字符串
            for key in predefined_keys:
                if key not in case:
                    case[key] = ''
            writer.writerow(case)

    print(f"数据已成功写入 {output_file_path}")

except FileNotFoundError:
    print(f"未找到输入文件: {input_file_path}")
except Exception as e:
    print(f"发生未知错误: {e}")
