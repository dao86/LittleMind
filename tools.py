import json
import os
import matplotlib.pyplot as plt

def plot_loss_curve(file, save_path=None):
    """
    绘制训练损失曲线
    参数:
        file : 日志文件路径
        save_path : 保存图像的路径，如果为 None 则不保存
    """
    losses = []
    if os.path.exists(file):
        with open(file, 'r', encoding='utf-8') as log:
            for i, line in enumerate(log):
                data = json.loads(line.strip().replace('\\', '/'))
                losses.append(data["loss"])
    plt.figure(figsize=(20, 12))
    plt.plot(range(1, len(losses) + 1), losses, marker='.', linestyle='-', color='b')

    plt.ylim(0, 10)
    plt.title('损失曲线')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.xticks(range(1, len(losses) + 1))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"保存图片 {save_path}")

    plt.show()

def split_file_into_n_parts(file_path, save_file, write_num, read_num=0):
    """
    将原数据文件分成n份文件，read_num=0读取全部，read_num》0,read_num，分成n份文件，每份 write_num 行
    :param file_path:原数据文件
    :param save_file:保存的文件名前缀
    :param write_num:每份文件的数据行数
    :param read_num:读取原数据文件的数据行数，为0时全部读取
    :return:n份文件
    """
    # 读取原始文件内容
    if (read_num > 0):
        lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in range(read_num):
                line = f.readline()
                if not line:  # 文件结束
                    break
                lines.append(line)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    total_lines = len(lines)

    for i in range(total_lines):
        start = i * write_num
        end = i * write_num + (write_num)
        part_lines = lines[start:end]
        if len(part_lines) == 0:
            break

        start = 1 if start == 0 else start
        output_file = f"{save_file}_{start}-{end}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as out_f:
            out_f.writelines(part_lines)

        print(f"已创建文件: {output_file}")

def csv2json_songci_sft(csv_file_path,jsonl_file_path):
    import csv
    import json
    # 打开CSV文件并逐行写入JSONL文件
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file, \
            open(jsonl_file_path, mode='w', encoding='utf-8') as jsonl_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            congci = row['text1'].split('|')
            cipaiming = songci_title(congci[0])
            json_item = {"conversations": [{"role": "user", "content":cipaiming}, {"role": "assistant", "content": congci[1]}]}
            line = json.dumps(json_item,ensure_ascii=False)
            jsonl_file.write(line + '\n')
    print(f"已成功将 {csv_file_path} 转换为 {jsonl_file_path}")

def csv2json_songci_pre(csv_file_path,jsonl_file_path):
    import csv
    import json
    # 打开CSV文件并逐行写入JSONL文件
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file, \
            open(jsonl_file_path, mode='w', encoding='utf-8') as jsonl_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            congci = row['text1'].split('|')
            # cipaiming = songci_title(congci[0])
            text_sc = '<|im_start|>'+congci[1]+'<|im_end|>'
            json_item = {"text":text_sc}
            line = json.dumps(json_item,ensure_ascii=False)
            jsonl_file.write(line + '\n')
    print(f"已成功将 {csv_file_path} 转换为 {jsonl_file_path}")

def songci_title(ci_pai_ming):
    import json
    import random
    prompt1 = [
        "请做一首宋词，",  "来一个宋代的词，", "宋词风格，", "古代宋朝的词，",""
    ]
    prompt2 = ["词牌名，", "词牌，", "词牌名:", "词牌:",""]
    p1 = random.choice(prompt1)
    p2 = random.choice(prompt2)
    prompt_ret =p1+p2+ci_pai_ming
    return prompt_ret

def read_and_mix_files_head(file1_path, file2_path, output_path=None):
    """
    读取两个文件的内容并混合到前面

    参数:
        file1_path (str): 第一个文件的路径
        file2_path (str): 第二个文件的路径
        output_path (str, optional): 输出文件的路径，如果为None则只打印混合结果
    """
    try:
        # 读取第一个文件
        with open(file1_path, 'r', encoding='utf-8') as f1:
            content1 = f1.read()
            print(f"文件1 ({file1_path}) 内容长度: {len(content1)} 字符")

        # 读取第二个文件
        with open(file2_path, 'r', encoding='utf-8') as f2:
            content2 = f2.read()
            print(f"文件2 ({file2_path}) 内容长度: {len(content2)} 字符")

        # 混合文件内容（交替行）
        lines1 = content1.split('\n')
        lines2 = content2.split('\n')

        mixed_content = []
        max_lines = max(len(lines1), len(lines2))

        for i in range(max_lines):
            if i < len(lines1):
                mixed_content.append(f"{lines1[i]}")
            if i < len(lines2):
                mixed_content.append(f"{lines2[i]}")

        mixed_text = '\n'.join(mixed_content)

        print(f"\n混合后的内容总长度: {len(mixed_text)} 字符")
        print("=" * 50)
        print("混合内容预览（前20行）:")
        print("=" * 50)

        # 显示前20行作为预览
        preview_lines = mixed_text.split('\n')[:20]
        for line in preview_lines:
            print(line)

        if len(mixed_text.split('\n')) > 20:
            print("...")
            remaining_lines = len(mixed_text.split('\n')) - 20
            print(f"还有 {remaining_lines} 行未显示")

        # 如果指定了输出路径，保存混合内容
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(mixed_text)
            print(f"\n混合内容已保存到: {output_path}")

        return mixed_text

    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
        return None
    except Exception as e:
        print(f"错误: {e}")
        return None

def read_and_mix_files_tail(file_a, file_b, output_file):
    """
    读取两个文件的内容并混合到后面
    :param file_a:
    :param file_b:
    :param output_file:
    :return:
    """
    def read_jsonl(file_path):
        """读取 JSONL 文件并返回所有行（去除换行符）"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    lines_a = read_jsonl(file_a)
    lines_b = read_jsonl(file_b)

    len_a = len(lines_a)
    len_b = len(lines_b)
    print(f'len_a:{len_a}---len_b:{len_b}')

    if len_a>=len_b:
        lines_big=lines_a
        lines_small=lines_b
    else:
        lines_big = lines_b
        lines_small = lines_a
    # 计算需要直接保留的头部部分
    head_part = lines_big[:len(lines_big) - len(lines_small)]

    # 需要混合的部分（A 的末尾 len_b 行）
    tail_a = lines_big[len(lines_big) - len(lines_small):]
    tail_b = lines_small[:]

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as out:
        # 先写入头部部分
        for line in head_part:
            out.write(line + '\n')
        print(f'head_part:{len(head_part)}')
        # 再写入交错混合部分
        for i in range(len_b):
            out.write(tail_a[i] + '\n')
            out.write(tail_b[i] + '\n')
        print(f'tail_:{len(tail_a)}')

if __name__ == "__main__":
    # split_file_into_n_parts('./data/pre_test.jsonl', './data/pre_test_split', 1, 100000)
    # plot_loss_curve('./output/log/pre_log_512.log', save_path="././output/log/loss_pre_512.png")
    # csv2json_songci_sft('./data/songci.csv', 'data/songci_sft.jsonl')
    # csv2json_songci_pre('./data/songci.csv', 'data/songci_pre.jsonl')
    # read_and_mix_files_head('./data/songci_sft.jsonl','./data/songci_identity.jsonl','./data/songci_mix.jsonl')
    # read_and_mix_files_tail('./data/songci_sft.jsonl','./data/songci_identity.jsonl','./data/songci_mix_new.jsonl')
    print('tool')