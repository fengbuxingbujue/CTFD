import numpy as np
import torch


class Schedule:
    def __init__(self, schedule, timesteps):
        self.timesteps = timesteps
        self.schedule = schedule

    def cosine_beta_schedule(self, s=0.001):
        timesteps = self.timesteps
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        # 计算余弦调度中的参数 alphas_cumprod
        # 这里的计算是基于余弦函数的变换，它使得调度在时间上平滑地从 0 到 1 变化
        # 具体地，先将时间步数 x 归一化到 [0, 1] 的范围内，然后通过余弦函数进行变换，最后取平方以确保结果在 [0, 1] 范围内。
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def linear_beta_schedule(self):
        timesteps = self.timesteps
        scale = 1000 / timesteps
        beta_start = 1e-6 * scale
        beta_end = 0.02 * scale
        return torch.linspace(beta_start, beta_end, timesteps)
    # 上面的代码中实际上是将一组(起始，终点，时间步)迁移到另一组(起始，终点，时间步)
    # 原本的一组是(1e-6，0.02，1000)，迁移后的一组是(beta_start，beta_end，timesteps)

    def quadratic_beta_schedule(self):
        timesteps = self.timesteps
        scale = 1000 / timesteps
        beta_start = 1e-6 * scale
        beta_end = 0.02 * scale
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2

    def sigmoid_beta_schedule(self):
        timesteps = self.timesteps
        scale = 1000 / timesteps
        beta_start = 1e-6 * scale
        beta_end = 0.02 * scale
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    def get_betas(self):
        if self.schedule == "linear":
            return self.linear_beta_schedule()
        elif self.schedule == 'cosine':
            return self.cosine_beta_schedule()
        else:
            raise NotImplementedError


if __name__ == "__main__":
    schedule = Schedule(schedule="linear", timesteps=100)
    print(schedule.get_betas().shape)
    print(schedule.get_betas())
