import csv
import argparse

# 定义要提取的键
KEYS = [
    "backend", "dataset_name", "request_rate", "max_concurrency", "sharegpt_output_len",
    "random_input_len", "random_output_len", "random_range_ratio", "duration",
    "completed", "total_input_tokens", "total_output_tokens",
    "total_output_tokens_retokenized", "request_throughput", "input_throughput",
    "output_throughput", "mean_e2e_latency_ms", "median_e2e_latency_ms",
    "std_e2e_latency_ms", "p99_e2e_latency_ms", "mean_ttft_ms", "median_ttft_ms",
    "std_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms",
    "std_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "median_itl_ms",
    "std_itl_ms", "p95_itl_ms", "p99_itl_ms", "concurrency", "accept_length"
]


def log_to_csv(log_file_path, csv_file_path):
    try:
        with open(log_file_path, 'r') as log_file, open(csv_file_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=KEYS)
            writer.writeheader()

            for line in log_file:
                if "backend" in line:
                    line = line.strip()
                    data = {}
                    # 去掉首尾大括号
                    line = line[1:-1]
                    pairs = line.split(',')
                    for pair in pairs:
                        pair = pair.strip()
                        parts = pair.split(':')
                        if len(parts) == 2:
                            key = parts[0].strip().strip('"')
                            value = parts[1].strip()
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            data[key] = value
                    # 构建行数据
                    row = {key: data.get(key) for key in KEYS}
                    writer.writerow(row)

    except FileNotFoundError:
        print(f"错误: 未找到日志文件 {log_file_path}")
    except Exception as e:
        print(f"错误: 发生未知错误 {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='筛选包含backend的行并将信息存入CSV')
    parser.add_argument('-f', '--file', type=str, required=True, help='日志文件的路径')
    parser.add_argument('-o', '--output', type=str, required=True, help='输出CSV文件的路径')
    args = parser.parse_args()

    log_file_path = args.file
    csv_file_path = args.output
    log_to_csv(log_file_path, csv_file_path)

