import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
import pywt
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from critic import BasicCritic
from decoder import DenseDecoder,BasicDecoder
from encoder import DenseEncoder, BasicEncoder, ResidualEncoder
from torchvision import datasets, transforms
from IPython.display import clear_output
import torchvision
from torch.optim import Adam
import pytorch_ssim
from tqdm import tqdm
import torch
import os
import gc
import csv
from PIL import ImageFile
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
import pdb
# RL
from agent.RL_env import formulate_state,compute_reward
from agent.Fixed_agent import FixedAgent
from agent.DQN_agent import DQN
from agent.hrl import HRL
from RS_analysis import rs_analysis

ImageFile.LOAD_TRUNCATED_IMAGES = True



# plot('encoder_mse', ep, metrics['val.encoder_mse'], True)
def plot(name, train_epoch, values, save):
    clear_output(wait=True)
    plt.close('all')
    fig = plt.figure()
    fig = plt.ion()
    fig = plt.subplot(1, 1, 1)
    fig = plt.title('epoch: %s -> %s: %s' % (train_epoch, name, values[-1]))
    fig = plt.ylabel(name)
    fig = plt.xlabel('train_loader')  # epoch??
    fig = plt.plot(values)
    fig = plt.grid()
    get_fig = plt.gcf()
    fig = plt.draw()  # draw the plot
    # fig = plt.pause(1)  # show it for 1 second
    if save:
        # now = datetime.datetime.now()
        get_fig.savefig('results/plots/%s_%d.png' %
                        (name, train_epoch))
        
def uiqi(reference_image, distorted_image):
    # 将图像转换为灰度
    transform_to_gray = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])


    transform_to_pil_image = transforms.ToPILImage()


    ref_PIL = transform_to_pil_image(reference_image)
    dist_PIL = transform_to_pil_image(distorted_image)
    
    ref_gray = transform_to_gray(ref_PIL).unsqueeze(0)
    dist_gray = transform_to_gray(dist_PIL).unsqueeze(0)
    

    mu1 = ref_gray.mean()
    mu2 = dist_gray.mean()


    sigma1_sq = ((ref_gray - mu1) ** 2).mean()
    sigma2_sq = ((dist_gray - mu2) ** 2).mean()


    sigma_12 = ((ref_gray - mu1) * (dist_gray - mu2)).mean()

    epsilon = 1e-10


    uiqi_value = (4 * mu1 * mu2 * sigma_12) / ((mu1.pow(2) + mu2.pow(2) + epsilon) * (sigma1_sq + sigma2_sq + epsilon))
    
    return uiqi_value


def main():
    data_dir='div2k'
    file_path = os.path.join('data', 'data.csv')
    epochs = 300
    max_data_depth = 8
    hidden_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cuda'):
        print('run on GPU')
    else:
        print('run on CPU')
    # 指标们
    METRIC_FIELDS = [
        'val.encoder_mse',
        'val.decoder_loss',
        'val.decoder_acc',
        'val.cover_score',
        'val.generated_score',
        'val.ssim',
        'val.psnr',
        'val.bpp',
        'val.consumption',
        'val.message_density',
        'val.UIQI',
        'val.rs',

        'train.message_density',
        'train.UIQI',
        'train.rs',
        'train.encoder_mse',
        'train.decoder_loss',
        'train.decoder_acc',
        'train.cover_score',
        'train.generated_score',
    ]

    mu = [.5, .5, .5]
    sigma = [.5, .5, .5]

    transform = transforms.Compose([transforms.RandomHorizontalFlip(),  # 以一定的概率随机水平翻转图像
                                    transforms.RandomCrop(
                                        360, pad_if_needed=True),  # 会随机裁剪图像。它将图像裁剪为指定的大小（360x360），如果图像的尺寸小于指定大小，则会进行填充。
                                    transforms.ToTensor(),  # 将图像数据转换为张量（tensor）格式
                                    transforms.Normalize(mu,
                                                         sigma)])  # 这个操作对图像进行标准化处理，将图像的像素值减去均值(mu)并除以标准差(sigma)。这样做可以使得图像的每个通道具有零均值和单位方差，有助于模型的训练。
    

    train_set = datasets.ImageFolder(os.path.join(  # 加载数据集，并进行预处理
        data_dir, "train/"), transform=transform)


    valid_set = datasets.ImageFolder(os.path.join(
        data_dir, "val/"), transform=transform)

    algs=['DQN-2','HRL-2']
    modes=['Basic','Residual','Dense']

    fieldnames = [
    'algorithm', 'mode','depth','combination','h0_reward', 'h1_reward', 'reward', 'bpp', 'mse',
    'cover_scores', 'message_density', 'uiqi', 'rs', 'psnr', 'ssim', 'consumption','generated_scores',
    ]

    h1_reward_alg = {alg: [] for alg in algs}
    h0_reward_alg = {alg: [] for alg in algs}
    reward_alg = {alg: [] for alg in algs}
    # reward
    psnr_alg = {alg: [] for alg in algs}
    ssim_alg = {alg: [] for alg in algs}
    consumption_alg = {alg: [] for alg in algs}

    # valid
    bpp_alg = {alg: [] for alg in algs}
    mse_alg = {alg: [] for alg in algs}
    cover_scores_alg = {alg: [] for alg in algs}
    generated_scores_alg = {alg: [] for alg in algs}
    message_density_alg = {alg: [] for alg in algs}
    uiqi_alg = {alg: [] for alg in algs}

    # action
    moede_alg = {alg: [] for alg in algs}
    depth_alg = {alg: [] for alg in algs}
    combination_alg = {alg: [] for alg in algs}
    

    for alg in tqdm(algs):
    
        h1_reward_avg = []
        h0_reward_avg = []
        reward_avg = []
        # state
        # tarin
        bpp_avg = []
        mse_avg = []
        cover_scores_avg = [] 
        generated_scores_avg=[]
        message_density_avg = []
        uiqi_avg = []

        # reward
        psnr_avg = []
        ssim_avg = []
        consumption_avg = []
        

        if alg in ['HRL-1','HRL-2']:
            agent=HRL()
        elif alg in ['DQN-1','DQN-2']:
            agent=DQN(gamma=0.99, lr=0.0001, action_num=24, state_num=5,
                            buffer_size=10000, batch_size=64, INITIAL_EPSILON=0.2, FINAL_EPSILON=0.001,max_episode=1000,
                            replace=1000, chkpt_dir='./chkpt')
        else:
            fixed_mode=modes.index(alg)
            agent=FixedAgent(fixed_mode, max_data_depth)


        encoders = {depth: [] for depth in range(1, 1 + max_data_depth)}
        for depth in encoders.keys():
            for Encoder in [BasicEncoder, ResidualEncoder, DenseEncoder]:
                encoders[depth].append(Encoder(int(depth), hidden_size).to(device))

        decoders = {depth: [] for depth in range(1, 1 + max_data_depth)}
        for depth in decoders.keys():
            for Decoder in [BasicDecoder, BasicDecoder, DenseDecoder]:
                decoders[depth].append(Decoder(int(depth), hidden_size).to(device))
        
        critic = BasicCritic(hidden_size).to(device)  # critic评估器
        cr_optimizer = Adam(critic.parameters(), lr=1e-4)  # critic模型优化

        # s
        # train
        message_density_s_t = []
        uiqi_s_t = []
        # valid
        message_density_s_v = []
        cover_scores_v = []
        generated_scores_v = []
        bpps_v = []
        encode_mse_losses_v = []
        uiqi_s_v = []
        # r
        h1_rewards = []
        h0_rewards = []
        rewards = []
        psnr_s = []
        ssim_s = []
        consumptions = []
        next_state = [0, 0, 0, 0, 0]

        for ep in tqdm(range(epochs)):
            train_size = 100
            valid_size = 50
            train_indices = np.random.choice(range(len(train_set)), train_size, replace=False)
            valid_indices = np.random.choice(range(len(valid_set)), valid_size, replace=False)
            train_subset = torch.utils.data.Subset(train_set, train_indices)
            valid_subset = torch.utils.data.Subset(valid_set, valid_indices)
            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=4, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_subset, batch_size=4, shuffle=False)
            
            state = next_state
            if alg in ['HRL-1','HRL-2']:
                if ep==0:
                    h0_action,h1_action = agent.choose_action(state)
                else:
                    h0_action,h1_action = h0_next_action,h1_next_action
                encode_mode = h0_action
                decode_mode = h0_action
                data_depth = h1_action +1
            else:
                action, _ = agent.choose_action(state)
                encode_mode = action // max_data_depth
                decode_mode = action // max_data_depth
                data_depth = action % max_data_depth + 1
            

            moede_alg[alg].append(encode_mode)
            depth_alg[alg].append(data_depth)   
            if alg in ['HRL-1','HRL-2']:
                combination_alg[alg].append(h0_action*8+h1_action+1)
            else :
                combination_alg[alg].append(action+1)
            encoder = encoders[data_depth][encode_mode]
            decoder = decoders[data_depth][decode_mode]
            en_de_optimizer = Adam(list(decoder.parameters()) +
                                list(encoder.parameters()), lr=1e-4)  # encoder和decoder模型优化
            # 设置指标
            metrics = {field: list() for field in METRIC_FIELDS}
            
            # 训练Critic
            for cover, _ in train_loader:  # 从train_loader中取出图像
                gc.collect()
                cover = cover.to(device)  # 将cover转移到指定设备上

                N, _, H, W = cover.size()  # 这行代码获取输入数据 cover 的形状信息，其中 N 表示批次大小，H 和 W 分别表示输入图像的高度和宽度
                # sampled from the discrete uniform distribution over 0 to 2
                payload = torch.zeros((N, data_depth, H, W),
                                    # 这行代码创建一个形状为 (N, data_depth, H, W) 的全零张量 payload，并随机填充为 0 或 1。
                                    device=device).random_(0, 2)
                generated = encoder.forward(cover, payload)  # 生成隐写图片
                cover_score = torch.mean(critic.forward(cover))  # 原始图片评估
                generated_score = torch.mean(critic.forward(generated))  # 生成图片评估

                cr_optimizer.zero_grad()  # Critic优化
                (cover_score - generated_score).backward(retain_graph=False)  # 损失函数
                cr_optimizer.step()

                for p in critic.parameters():  # 遍历 critic 模型的参数，并将它们限制在一个范围内
                    p.data.clamp_(-0.1, 0.1)
                metrics['train.cover_score'].append(cover_score.item())  # 添加相关参数
                metrics['train.generated_score'].append(generated_score.item())

            rgb_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(device)
            for cover, _ in train_loader:
                gc.collect()
                cover = cover.to(device)
                
                N, _, H, W = cover.size()
                # sampled from the discrete uniform distribution over 0 to 2
                payload = torch.zeros((N, data_depth, H, W),
                                    device=device).random_(0, 2)
                generated = encoder.forward(cover, payload)  # 生成图片
                if decode_mode == 1:
                    generated=generated+cover
                decoded = decoder.forward(generated)  # 隐写信息
                
                # 计算指标
                UIQI = 0
                for c, g in zip(cover, generated):
                    UIQI += uiqi(c, g)
                UIQI /= len(cover)
                message_density = torch.sum(payload) / torch.numel(payload) # 计算消息密度
                encoder_mse = mse_loss(generated, cover)  # 均方误差
                decoder_loss = binary_cross_entropy_with_logits(decoded, payload)  # 交叉熵
                decoder_acc = (decoded >= 0.0).eq(
                    payload >= 0.5).sum().float() / payload.numel()  # 解码准确率
                generated_score = torch.mean(critic.forward(generated))  # 生成图片评估

                en_de_optimizer.zero_grad()
                (100.0 * encoder_mse + decoder_loss +
                generated_score).backward()  # Why 100?   #加大均方误差的权重
                en_de_optimizer.step()
                # 记录指标
                metrics['train.encoder_mse'].append(encoder_mse.item())
                metrics['train.decoder_loss'].append(decoder_loss.item())
                metrics['train.decoder_acc'].append(decoder_acc.item())
                metrics['train.message_density'].append(message_density.item())
                metrics['train.UIQI'].append(UIQI.item())
                metrics['train.rs'].append(rs)

            
            # 验证集
            for cover, _ in valid_loader:
                gc.collect()
                cover = cover.to(device)
                vutils.save_image(cover, "cover.png")
                N, _, H, W = cover.size()
                # sampled from the discrete uniform distribution over 0 to 2
                payload = torch.zeros((N, data_depth, H, W),
                                    device=device).random_(0, 2)
                #隐写开始时间
                start_time = time.time()
                generated = encoder.forward(cover, payload)
                vutils.save_image(generated, "images.png") 
                # residual
                if decode_mode == 1:
                    generated=generated+cover
                decoded = decoder.forward(generated)
                #隐写结束时间
                end_time = time.time()
                time_r = end_time - start_time
                # 计算指标
                UIQI = 0
                for c, g in zip(cover, generated):
                    UIQI += uiqi(c, g)
                    rs += rs_analysis(g)
                UIQI /= len(cover)
                #计算指标
                message_density = torch.sum(payload) / torch.numel(payload) # 计算消息密度
                encoder_mse = mse_loss(generated, cover)
                decoder_loss = binary_cross_entropy_with_logits(decoded, payload)
                decoder_acc = (decoded >= 0.0).eq(
                    payload >= 0.5).sum().float() / payload.numel()
                generated_score = torch.mean(critic.forward(generated))
                cover_score = torch.mean(critic.forward(cover))
                #记录指标
                metrics['val.encoder_mse'].append(encoder_mse.item())
                metrics['val.decoder_loss'].append(decoder_loss.item())
                metrics['val.decoder_acc'].append(decoder_acc.item())
                metrics['val.cover_score'].append(cover_score.item())
                metrics['val.generated_score'].append(generated_score.item())
                metrics['val.message_density'].append(message_density.item())
                metrics['val.UIQI'].append(UIQI.item())
                metrics['val.rs'].append(rs)
                
                metrics['val.consumption'].append(time_r)
                metrics['val.ssim'].append(
                    pytorch_ssim.ssim(cover, generated).item())
                metrics['val.psnr'].append(
                    10 * torch.log10(4 / encoder_mse).item())
                metrics['val.bpp'].append(
                    data_depth * (2 * decoder_acc.item() - 1))
            #指标取均值
            encode_mse_losses_v.append(np.mean(metrics['val.encoder_mse']))
            bpps_v.append(np.mean(metrics['val.bpp']))
            cover_scores_v.append(np.mean(metrics['val.cover_score']))
            generated_scores_v.append(np.mean(metrics['val.generated_score']))
            message_density_s_v.append(np.mean(metrics['val.message_density']))
            uiqi_s_v.append(np.mean(metrics['val.UIQI']))
            # 训练集
            message_density_s_t.append(np.mean(metrics['train.message_density']))  
            uiqi_s_t.append(np.mean(metrics['train.UIQI'])) 
            
            consumptions.append(np.mean(metrics['val.consumption']))
            psnr_s.append(np.mean(metrics['val.psnr']))
            ssim_s.append(np.mean(metrics['val.ssim']))
            #计算奖励 
            reward=compute_reward(psnr_s[-1], ssim_s[-1], consumptions[-1])
            # 计算奖励
            rewards.append(reward)
            if alg in ['HRL-1','HRL-2']:
                h0_reward=compute_reward(psnr_s[-1], ssim_s[-1], consumptions[-1])
                h1_reward=compute_reward(psnr_s[-1], ssim_s[-1], consumptions[-1])
                h0_rewards.append(h0_reward)
                h1_rewards.append(h1_reward) 
            # 更新状态
            next_state=formulate_state(bpps_v[-1], encode_mse_losses_v[-1], generated_scores_v[-1], message_density_s_v[-1], uiqi_s_v[-1])  
            
            #因为下层状态需要上层动作 所以在这里先选出下一次动作
            if alg in ['HRL-1','HRL-2']:
                h0_next_action,h1_next_action = agent.choose_action(next_state)
                # 存储经验
                agent.store_transition(state, h0_action, h1_action, h0_reward, h1_reward, next_state,h0_next_action)
            else :
                agent.store_transition(state, action, reward, next_state)
            # 学习
            agent.learn()

            # 保存模型状态
            name = "EN_DE_%+.3f.dat" % (cover_score.item())
            fname = os.path.join('.', 'results','model', name)
            states = {
                'state_dict_critic': critic.state_dict(),
                'state_dict_encoder': encoder.state_dict(),
                'state_dict_decoder': decoder.state_dict(),
                'en_de_optimizer': en_de_optimizer.state_dict(),
                'cr_optimizer': cr_optimizer.state_dict(),
                'metrics': metrics,
                'train_epoch': ep,
                # 'date': now.strftime("%Y-%m-%d_%H:%M:%S"),
            }
            torch.save(states, fname)

            h0_reward_avg.append(h0_rewards)
            h1_reward_avg.append(h1_rewards)
            reward_avg.append(rewards)
            # state 
            bpp_avg.append(bpps_v)
            mse_avg.append(encode_mse_losses_v)
            cover_scores_avg.append(cover_scores_v)
            generated_scores_avg.append(generated_scores_v)
            message_density_avg.append(message_density_s_v)
            uiqi_avg.append(uiqi_s_v)
            # rs_avg.append(rs_s_v)
            # reward
            psnr_avg.append(psnr_s)
            ssim_avg.append(ssim_s)
            consumption_avg.append(consumptions)

        # 感觉没什么用的取均值
        h0_reward_avg = np.mean(h0_reward_avg, axis=0).tolist()
        h1_reward_avg = np.mean(h1_reward_avg, axis=0).tolist()
        reward_avg = np.mean(reward_avg, axis=0).tolist()
        # state
        bpp_avg = np.mean(bpp_avg, axis=0).tolist()
        mse_avg = np.mean(mse_avg, axis=0).tolist() 
        cover_scores_avg = np.mean(cover_scores_avg, axis=0).tolist()
        generated_scores_avg = np.mean(generated_scores_avg, axis=0).tolist()
        message_density_avg = np.mean(message_density_avg, axis=0).tolist()
        uiqi_avg = np.mean(uiqi_avg, axis=0).tolist()
        # reward
        psnr_avg = np.mean(psnr_avg, axis=0).tolist()
        ssim_avg = np.mean(ssim_avg, axis=0).tolist()
        consumption_avg = np.mean(consumption_avg, axis=0).tolist()
        #记录对应算法数据
        if alg not in ['HRL-1','HRL-2']:
            h0_reward_alg[alg] = reward_avg
            h1_reward_alg[alg] = reward_avg
        else:
            h0_reward_alg[alg] = h0_reward_avg
            h1_reward_alg[alg] = h1_reward_avg
        reward_alg[alg] = reward_avg
        # state
        bpp_alg[alg] = bpp_avg
        mse_alg[alg] = mse_avg
        cover_scores_alg[alg] = cover_scores_avg
        generated_scores_alg[alg]=generated_scores_avg
        message_density_alg[alg] = message_density_avg
        uiqi_alg[alg] = uiqi_avg
        # reward
        psnr_alg[alg] = psnr_avg
        ssim_alg[alg] = ssim_avg
        consumption_alg[alg] = consumption_avg
    

        # 打开CSV文件进行追加写操作
        file_path = os.path.join('data', 'data.csv')

        with open(file_path, mode='a+', newline='') as file:
            file.seek(0)
            reader = csv.DictReader(file)
            existing_fieldnames = reader.fieldnames if reader.fieldnames else []
            print(f"Existing fieldnames: {existing_fieldnames}")
            
            # 找出缺失的表头
            missing_fieldnames = [field for field in fieldnames if field not in existing_fieldnames]
            
            if missing_fieldnames:
                print(f"Missing fieldnames: {missing_fieldnames}")
                # 读取现有数据
                file.seek(0)
                existing_data = list(reader)
                
                # 补全现有数据中的缺失字段
                for row in existing_data:
                    for field in missing_fieldnames:
                        row[field] = '[]'  # 将缺失字段标为空列表
                
                # 写入新的表头和现有数据
                file.seek(0)
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_data)
                file.truncate()
            
            # 移动到文件末尾以追加新数据
            file.seek(0, 2)
            writer = csv.DictWriter(file, fieldnames=fieldnames)
                
            # 写入数据
            writer.writerow({
                'algorithm': alg,
                'mode': moede_alg.get(alg, ''),
                'depth': depth_alg.get(alg, ''),
                'combination': combination_alg.get(alg, ''),
                'h0_reward': h0_reward_alg.get(alg, ''),
                'h1_reward': h1_reward_alg.get(alg, ''),
                'reward': reward_alg.get(alg, ''),
                'bpp': bpp_alg.get(alg, ''),
                'mse': mse_alg.get(alg, ''),
                'cover_scores': cover_scores_alg.get(alg, ''),
                'message_density': message_density_alg.get(alg, ''),
                'uiqi': uiqi_alg.get(alg, ''),
                'psnr': psnr_alg.get(alg, ''),
                'ssim': ssim_alg.get(alg, ''),
                'consumption': consumption_alg.get(alg, ''),
                'generated_scores': generated_scores_alg.get(alg, ''),
            })

    # # save mat
    # import scipy.io as sio
    # sio.savemat('results.mat', {'reward': reward_alg, 'psnr': psnr_alg, 'ssim': ssim_alg, 'consumption': consumption_alg, 'uiqi': uiqi_alg, 'rs': rs_alg, 'mse': mse_alg})


if __name__ == '__main__':
    for func in [
        lambda: os.mkdir(os.path.join('.', 'results')) if not os.path.exists(os.path.join('.', 'results')) else None,
        lambda: os.mkdir(os.path.join('.', 'results','model')) if not os.path.exists(
            os.path.join('.', 'results','model')) else None,
        lambda: os.mkdir(os.path.join('.', 'results','plots')) if not os.path.exists(
            os.path.join('.', 'results','plots')) else None]:  # create directories
        try:
            func()
        except Exception as error:
            print(error)
            continue
    main()
