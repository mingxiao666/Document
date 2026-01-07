import re
import argparse
import csv
from collections import defaultdict

class NSysResultParser:
    """通用解析器：增强容错，避免KeyError"""
    def __init__(self, log_path: str):
        self.log_path = log_path
        # 使用defaultdict确保字段不存在时返回0.0，避免KeyError
        self.data = defaultdict(lambda: defaultdict(float))
        # 初始化核心模块默认值
        core_modules = ["Communication", "GEMM", "Attention", "MoE", "Others", "Total"]
        for module in core_modules:
            self.data[module]["percent"] = 0.0
            self.data[module]["time_sec"] = 0.0

    def parse(self):
        """解析日志：增强容错，适配任意格式"""
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"读取日志失败：{e}")
            return self.data

        # 匹配表格行：兼容多种空格/制表符格式
        pattern = r'\| *([\w]+) *\| *([\d\.]+) *\| *([\d\.]+) *\|'
        for line in lines:
            line = line.strip()
            if not line or "Metric" in line or "--------" in line:
                continue
            
            match = re.search(pattern, line)
            if not match:
                continue
            
            try:
                metric = match.group(1).strip()
                percent_str = match.group(2).strip()
                time_str = match.group(3).strip()

                # 容错转换：处理空值/非数字
                percent = float(percent_str) if percent_str and percent_str.replace('.','').isdigit() else 0.0
                time_sec = float(time_str) if time_str and time_str.replace('.','').isdigit() else 0.0

                # 赋值：Total模块无percent，单独处理
                if metric == "Total":
                    self.data[metric]["time_sec"] = time_sec
                else:
                    self.data[metric]["percent"] = percent
                    self.data[metric]["time_sec"] = time_sec
            except Exception as e:
                # 跳过解析失败的行，不影响整体
                continue
        
        return self.data

class NSysGeneralComparator:
    """通用对比器：完全无硬编码，支持CSV导出"""
    def __init__(self, log1_path: str, log2_path: str, log1_name: str = "Result1", log2_name: str = "Result2"):
        # 解析两份日志（容错解析）
        self.log1_parser = NSysResultParser(log1_path)
        self.log1_data = self.log1_parser.parse()
        
        self.log2_parser = NSysResultParser(log2_path)
        self.log2_data = self.log2_parser.parse()

        # 自定义对比名称
        self.log1_name = log1_name
        self.log2_name = log2_name

    def calculate_speedup(self, base_time: float, target_time: float) -> float:
        """计算加速比：容错除零"""
        if abs(target_time) < 1e-6 or abs(base_time) < 1e-6:
            return 0.0
        return round(base_time / target_time, 2)

    def generate_comparison_data(self, custom_config: dict = None):
        """生成对比数据（结构化，用于控制台/CSV输出）"""
        # 默认配置
        default_config = {
            "Model Type": ("FP8", "FP8"),
            "Concurrency": ("64", "64"),
            "ISL/OSL": ("1000 / 1000", "1000 / 1000")
        }
        if custom_config:
            default_config.update(custom_config)

        # 构建结构化数据
        comparison_data = []
        
        # 配置行数据
        for cfg_name, (cfg1, cfg2) in default_config.items():
            comparison_data.append({
                "Category": cfg_name,
                self.log1_name: cfg1,
                self.log2_name: cfg2,
                "Speedup (vs " + self.log1_name + ")": ""
            })

        # 核心模块数据
        modules = ["Communication", "GEMM", "Attention", "MoE", "Others", "Total"]
        for module in modules:
            # 容错获取数据
            log1_p = self.log1_data[module].get("percent", 0.0)
            log1_t = self.log1_data[module].get("time_sec", 0.0)
            log2_p = self.log2_data[module].get("percent", 0.0)
            log2_t = self.log2_data[module].get("time_sec", 0.0)
            
            # 格式化输出
            if module == "Total":
                log1_str = f"~{log1_t:.3f} s" if log1_t > 0 else "-"
                log2_str = f"~{log2_t:.3f} s" if log2_t > 0 else "-"
            else:
                log1_str = f"{log1_p:.1f}% / {log1_t:.3f} s" if log1_t > 0 else "-"
                log2_str = f"{log2_p:.1f}% / {log2_t:.3f} s" if log2_t > 0 else "-"
            
            # 计算加速比
            speedup = self.calculate_speedup(log1_t, log2_t)
            speedup_str = f"{speedup}" if speedup > 0 else "-"

            comparison_data.append({
                "Category": module,
                self.log1_name: log1_str,
                self.log2_name: log2_str,
                "Speedup (vs " + self.log1_name + ")": speedup_str
            })

        return comparison_data

    def generate_console_table(self, comparison_data):
        """生成控制台易读的表格"""
        # 获取表头
        headers = list(comparison_data[0].keys())
        # 计算列宽
        col_widths = {
            "Category": 12,
            self.log1_name: 18,
            self.log2_name: 18,
            "Speedup (vs " + self.log1_name + ")": 15
        }

        # 表头行
        header_line = (
            f"{headers[0]:<{col_widths[headers[0]]}}\t"
            f"{headers[1]:<{col_widths[headers[1]]}}\t"
            f"{headers[2]:<{col_widths[headers[2]]}}\t"
            f"{headers[3]:<{col_widths[headers[3]]}}\n"
        )

        # 数据行
        data_lines = []
        for row in comparison_data:
            data_line = (
                f"{row[headers[0]]:<{col_widths[headers[0]]}}\t"
                f"{row[headers[1]]:<{col_widths[headers[1]]}}\t"
                f"{row[headers[2]]:<{col_widths[headers[2]]}}\t"
                f"{row[headers[3]]:<{col_widths[headers[3]]}}\n"
            )
            data_lines.append(data_line)

        return header_line + ''.join(data_lines)

    def save_to_csv(self, comparison_data, csv_filename):
        """保存对比数据到CSV文件"""
        # 获取表头
        headers = list(comparison_data[0].keys())
        # 写入CSV
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(comparison_data)
        print(f"\nCSV文件已保存到：{csv_filename}")

def main():
    parser = argparse.ArgumentParser(description='通用NSys结果对比工具（支持CSV导出）')
    parser.add_argument('log1', help='基准日志路径（如result1.log）')
    parser.add_argument('log2', help='对比日志路径（如result2.log）')
    parser.add_argument('--log1-name', default='Result1', help='基准日志名称（如H20_TP8EP8）')
    parser.add_argument('--log2-name', default='Result2', help='对比日志名称（如H200_TP8EP8）')
    parser.add_argument('--csv', default='gpu_comparison.csv', help='CSV输出文件名（默认：gpu_comparison.csv）')

    args = parser.parse_args()

    try:
        # 初始化对比器
        comparator = NSysGeneralComparator(
            log1_path=args.log1,
            log2_path=args.log2,
            log1_name=args.log1_name,
            log2_name=args.log2_name
        )

        # 自定义配置（可根据实际修改）
        custom_config = {
            "Model Type": ("FP8", "FP8"),
            "Concurrency": ("64", "64"),
            "ISL/OSL": ("1000 / 1000", "1000 / 1000")
        }

        # 生成结构化对比数据
        comparison_data = comparator.generate_comparison_data(custom_config)
        
        # 输出控制台表格
        print(f"=== {args.log1_name} vs {args.log2_name} 对比表 ===")
        console_table = comparator.generate_console_table(comparison_data)
        print(console_table)

        # 保存到CSV文件
        comparator.save_to_csv(comparison_data, args.csv)

    except Exception as e:
        print(f"生成对比表失败：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
