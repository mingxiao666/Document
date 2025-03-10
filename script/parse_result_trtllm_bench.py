import csv
import os
import argparse

# 定义要提取的指标的关键字
target_keys = [
    "Number of requests",
    "Number of concurrent requests",
    "Average Input Length (tokens)",
    "Average Output Length (tokens)",
    "TP Size",
    "PP Size",
    "EP Size",
    "Max Runtime Batch Size",
    "Max Runtime Tokens",
    "Scheduling Policy",
    "KV Memory Percentage",
    "Issue Rate (req/sec)",
    "Request Throughput (req/sec)",
    "Total Output Throughput (tokens/sec)",
    "Per User Output Throughput (tokens/sec/user)",
    "Per GPU Output Throughput (tokens/sec/gpu)",
    "Total Latency (ms)",
    "Average request latency (ms)",
    "Per User Output Speed [1/TPOT] (tokens/sec/user)",
    "Average time-to-first-token [TTFT] (ms)",
    "Average time-per-output-token [TPOT] (ms)",
    "[TPOT] MINIMUM",
    "[TPOT] MAXIMUM",
    "[TPOT] AVERAGE",
    "[TPOT] P50",
    "[TPOT] P90",
    "[TPOT] P95",
    "[TPOT] P99",
    "[TTFT] MINIMUM",
    "[TTFT] MAXIMUM",
    "[TTFT] AVERAGE",
    "[TTFT] P50",
    "[TTFT] P90",
    "[TTFT] P95",
    "[TTFT] P99",
    "[Latency] P50",
    "[Latency] P90",
    "[Latency] P95",
    "[Latency] P99",
    "[Latency] MINIMUM",
    "[Latency] MAXIMUM",
    "[Latency] AVERAGE"
]


def extract_metrics_from_file(file_path):
    # 初始化一个字典来存储指标的值，同时记录该指标是否已被记录过
    metrics_dict = {key: {"value": "", "recorded": False} for key in target_keys}
    # 标记是否开始记录指标
    start_recording = False

    with open(file_path, 'r') as file:
        for line in file:
            if "= REQUEST DETAILS" in line:
                start_recording = True
                continue

            if start_recording:
                for key in target_keys:
                    if line.startswith(key) and not metrics_dict[key]["recorded"]:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            metrics_dict[key]["value"] = parts[1].strip()
                            metrics_dict[key]["recorded"] = True

    return {key: metrics["value"] for key, metrics in metrics_dict.items()}


def main():
    parser = argparse.ArgumentParser(description='Extract metrics from log files and save to CSV.')
    parser.add_argument('-d', '--directory', required=True, help='Directory containing log files.')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file.')
    args = parser.parse_args()

    result = []
    for root, dirs, files in os.walk(args.directory):
        for file in files:
            if file.endswith('.log'):
                file_path = os.path.join(root, file)
                metrics = extract_metrics_from_file(file_path)
                metrics["File Name"] = file
                result.append(metrics)

    headers = ["File Name"] + target_keys
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in result:
            writer.writerow(row)

    print(f"指标信息已成功保存到 {args.output} 文件中。")


if __name__ == "__main__":
    main()

