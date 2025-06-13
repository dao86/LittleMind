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

if __name__ == "__main__":
    # split_file_into_n_parts('./data/pre_test.jsonl', './data/pre_test_split', 1, 100000)
    # plot_loss_curve('./output/log/pre_log_512.log', save_path="././output/log/loss_pre_512.png")
    print(f'test')
