import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import time
import random
from datetime import datetime, timedelta

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 定义通信操作类型及其对应的颜色
OPERATIONS = {
    'allreduce': '#1f77b4',    # 蓝色
    'broadcast': '#2ca02c',    # 绿色
    'barrier': '#ff7f0e',      # 橙色
    'allgather': '#d62728',    # 红色
    'reduce': '#9467bd',       # 紫色
    'scatter': '#8c564b',      # 棕色
    'gather': '#e377c2'        # 粉色
}

class CommunicationTimeline:
    def __init__(self, num_ranks, update_interval=100):
        """
        初始化时间线可视化工具
        :param num_ranks: 进程数量
        :param update_interval: 刷新间隔(毫秒)
        """
        self.num_ranks = num_ranks
        self.update_interval = update_interval
        self.start_time = datetime.now()
        self.events = []  # 存储所有通信事件 (开始时间, 结束时间, 进程编号, 操作类型)
        
        # 设置图形
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('分布式通信操作时间线')
        
        # 设置坐标轴
        self.ax.set_ylim(-0.5, num_ranks - 0.5)
        self.ax.set_yticks(range(num_ranks))
        self.ax.set_yticklabels([f'Rank {i}' for i in range(num_ranks)])
        self.ax.set_ylabel('Rank index (Rank)')
        self.ax.set_xlabel('Running (s)')
        self.ax.set_title('Communications Timeline in Distributed Training')
        
        # 添加网格和图例
        self.ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        self._add_legend()
        
        # 动画更新函数
        self.animation = FuncAnimation(
            self.fig, self._update_plot, 
            interval=update_interval, 
            blit=True,
            cache_frame_data=False 
        )
    
    def _add_legend(self):
        """添加图例"""
        handles = [plt.Rectangle((0,0),1,1, facecolor=color) 
                  for color in OPERATIONS.values()]
        self.ax.legend(handles, OPERATIONS.keys(), 
                      loc='upper left', bbox_to_anchor=(1, 1))
    
    def add_event(self, rank, op_type, start_time, end_time):
        """
        添加一个通信事件
        :param rank: 进程编号
        :param op_type: 操作类型
        :param start_time: 开始时间(相对于启动的秒数)
        :param end_time: 结束时间(相对于启动的秒数)
        """
        if op_type not in OPERATIONS:
            raise ValueError(f"未知的操作类型: {op_type}")
        if rank < 0 or rank >= self.num_ranks:
            raise ValueError(f"无效的进程编号: {rank}, 有效范围 0-{self.num_ranks-1}")
            
        self.events.append((start_time, end_time, rank, op_type))
    
    def _update_plot(self, frame):
        """更新图表"""
        # 清除当前轴但保留标题、标签等
        self.ax.clear()
        self._add_legend()
        self.ax.set_ylim(-0.5, self.num_ranks - 0.5)
        self.ax.set_yticks(range(self.num_ranks))
        self.ax.set_yticklabels([f'Rank {i}' for i in range(self.num_ranks)])
        self.ax.set_ylabel('进程编号 (Rank)')
        self.ax.set_xlabel('训练时间 (秒)')
        self.ax.set_title('分布式训练中集体通信操作时间线')
        self.ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # 绘制所有事件
        for start, end, rank, op_type in self.events:
            self.ax.barh(
                y=rank, 
                width=end - start, 
                left=start, 
                height=0.8, 
                color=OPERATIONS[op_type],
                edgecolor='black',
                alpha=0.8
            )
        
        # 动态调整x轴范围
        if self.events:
            max_time = max(end for _, end, _, _ in self.events)
            self.ax.set_xlim(0, max(max_time * 1.1, 10))  # 至少显示10秒
        
        return self.ax.patches  # 返回需要更新的元素
    
    def show(self):
        """显示图表"""
        plt.tight_layout()
        plt.show()


# 模拟分布式训练中的通信操作
def simulate_training(num_ranks=4, duration=30):
    timeline = CommunicationTimeline(num_ranks)
    start_time = time.time()
    current_time = 0
    
    # 模拟训练循环
    while current_time < duration:
        # 每个进程随机执行一些通信操作
        for rank in range(num_ranks):
            # 随机决定是否执行操作
            if random.random() < 0.3:  # 30%的概率执行操作
                op_type = random.choice(list(OPERATIONS.keys()))
                op_start = current_time
                # 操作耗时: 0.1-1秒不等，根据操作类型略有差异
                base_duration = 0.1 if op_type == 'barrier' else 0.2
                op_duration = base_duration + random.random() * (0.9 if op_type != 'allreduce' else 1.8)
                op_end = op_start + op_duration
                
                timeline.add_event(rank, op_type, op_start, op_end)
        
        # 推进时间
        time.sleep(0.1)
        current_time = time.time() - start_time
    
    return timeline


if __name__ == "__main__":
    # 模拟4个进程，30秒的训练过程
    timeline = simulate_training(num_ranks=4, duration=30)
    timeline.show()
