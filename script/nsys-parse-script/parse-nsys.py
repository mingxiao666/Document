import sys
import re

def parse_nsys_log(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        total_lines = len(lines)
        print("=== 调试信息 ===")
        print(f"[DEBUG] 成功读取日志，共{total_lines}行有效内容")
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到。")
        return
    except Exception as e:
        print(f"错误：读取文件时发生异常 - {e}")
        return

    # === 1. 定义分类关键词（保持顺序：Communication→GEMM→Attention→MoE→Others）===
    categories = [
        ('Communication', ['ncclDevKernel_', 'ncclKernel', 'moe_comm']),
        ('GEMM', [
            'gemm', 'Gemm',          # 通用 GEMM 关键词
            'cutlass::Kernel2',     # 6KD Cutlass GEMM 识别
            'cutlass::device_kernel',# 6KD Cutlass GEMM 识别
            'matMul', 'MatMul'      # 其他矩阵乘法命名
        ]),
        ('Attention', ['flash_attention', 'mha', 'FlashInfer', 'attention', 'mla']),
        ('MoE', ['moe', 'Expert', 'finalizeMoe', 'expandInputRows', 'fused_moe']),
        ('Others', ['.*'])          # 兜底
    ]

    # === 2. 定位 Kernel Summary 区域 ===
    start_marker = " ** CUDA GPU Kernel Summary"
    end_marker = "Processing \\["
    data_lines = []
    in_summary = False
    start_line = -1
    end_line = -1

    for i, line in enumerate(lines):
        if start_marker in line:
            in_summary = True
            start_line = i + 1
            print(f"[DEBUG] 找到Kernel Summary起始行：第{start_line}行 - {line.strip()[:50]}...")
            continue
        
        if in_summary:
            # 匹配结束：遇到 Processing [ 或者下一个 Summary (API Summary)
            if re.search(end_marker, line) or " ** CUDA API Summary" in line:
                end_line = i + 1
                print(f"[DEBUG] 找到Kernel Summary结束行：第{end_line}行 - {line.strip()[:50]}...")
                break
            # 跳过表头和分割线
            if "--------" in line or "Time (%)" in line or "Total Time" in line:
                continue
            stripped = line.strip()
            if stripped:
                data_lines.append(stripped)

    if not data_lines:
        print("警告：未找到 Kernel Summary 数据，请检查日志格式。")
        return
    
    print(f"[DEBUG] 提取到Kernel Summary区域，共{len(data_lines)}行")

    # === 3. 解析与分类（使用最稳健的正则）===
    # 正则解释：
    # ^\s*([\d\.]+)      -> 匹配开头的百分比 (Time %)
    # \s+([\d\,]+)       -> 匹配后面的总时间 (Total Time)
    # .*?                -> 非贪婪匹配中间所有内容 (Instances, Avg, Med, etc.)
    # ([\w_:<].*)$       -> 匹配最后以字母/下划线/冒号开头的内核名称
    pattern = re.compile(r'^\s*([\d\.]+)\s+([\d\,]+)\s+.*?([\w_:<].*)$')
    
    totals = {cat: 0 for cat, _ in categories}
    totals['Others'] = 0
    total_time_ns = 0
    kernel_count = 0
    debug_lines = []

    for line in data_lines:
        match = pattern.match(line)
        if match:
            percent_str = match.group(1)
            time_str = match.group(2).replace(',', '')
            name = match.group(3).strip()
            
            try:
                time_ns = int(time_str)
                # percent = float(percent_str) # 实际计算用总耗时比例，不用日志里的百分比
            except ValueError:
                continue
            
            total_time_ns += time_ns
            kernel_count += 1
            classified = False

            # 按顺序匹配分类
            for cat, keywords in categories:
                if cat == 'Others': continue
                for kw in keywords:
                    if kw.lower() in name.lower():
                        totals[cat] += time_ns
                        # 格式化调试信息（截取名称前40字符）
                        display_name = name[:40] + "..." if len(name) > 40 else name
                        debug_lines.append(f"[DEBUG] {cat}内核：{display_name} | {percent_str}% | {time_str}ns")
                        classified = True
                        break
                if classified: break
            
            if not classified:
                totals['Others'] += time_ns
                display_name = name[:40] + "..." if len(name) > 40 else name
                debug_lines.append(f"[DEBUG] Others内核：{display_name} | {percent_str}% | {time_str}ns")

    # === 4. 输出调试信息（只打印前5条）===
    for msg in debug_lines[:5]:
        print(msg)
    print(f"[DEBUG] 共解析{kernel_count}个有效内核")
    print(f"[DEBUG] 总耗时：{total_time_ns} ns ({total_time_ns/1e9:.3f} s)")
    print("================\n")

    # === 5. 输出表格（纯数字，无%符号，兼容 parse-table.py）===
    # 计算百分比和时间
    comm_p = (totals['Communication'] / total_time_ns) * 100 if total_time_ns else 0.0
    gemm_p = (totals['GEMM'] / total_time_ns) * 100 if total_time_ns else 0.0
    attn_p = (totals['Attention'] / total_time_ns) * 100 if total_time_ns else 0.0
    moe_p = (totals['MoE'] / total_time_ns) * 100 if total_time_ns else 0.0
    others_p = (totals['Others'] / total_time_ns) * 100 if total_time_ns else 0.0
    total_p = 100.0

    comm_t = totals['Communication'] / 1e9
    gemm_t = totals['GEMM'] / 1e9
    attn_t = totals['Attention'] / 1e9
    moe_t = totals['MoE'] / 1e9
    others_t = totals['Others'] / 1e9
    total_t = total_time_ns / 1e9

    # 格式化输出
    print(f"| {'Metric':<12} | {'Percent (%)':<11} | {'Time (s)':<8} |")
    print(f"|{'-'*14}|{'-'*13}|{'-'*10}|")
    print(f"| {'Communication':<12} | {comm_p:>10.1f} | {comm_t:>8.3f} |")
    print(f"| {'GEMM':<12} | {gemm_p:>10.1f} | {gemm_t:>8.3f} |")
    print(f"| {'Attention':<12} | {attn_p:>10.1f} | {attn_t:>8.3f} |")
    print(f"| {'MoE':<12} | {moe_p:>10.1f} | {moe_t:>8.3f} |")
    print(f"| {'Others':<12} | {others_p:>10.1f} | {others_t:>8.3f} |")
    print(f"|{'-'*14}|{'-'*13}|{'-'*10}|")
    print(f"| {'Total':<12} | {total_p:>10.1f} | {total_t:>8.3f} |")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python parse-nsys.py <日志文件路径>")
    else:
        parse_nsys_log(sys.argv[1])
