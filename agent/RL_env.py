import numpy as np


def formulate_state(bpp, loss, cover_score, message_density, UIQI=0):
    state = [bpp, loss, cover_score, message_density,UIQI]
    return state

# def formulate_state(bpp, loss, a1, a2,UIQI=0,):
#     state = [bpp, loss, a1,a2, UIQI]
#     return state


def compute_reward(psnr, ssim, consumption, ):
    # PSNR 均值是 SSIM 的 10 倍
    w1 = 0.2
    w2 = 2
    w3 = 0.1
    reward = w1 * psnr + w2 * ssim - w3 * consumption
    return reward
