from agent.DQN_agent import DQN
import torch as T

class HRL(object):  
    def __init__(self):
        #上层 选择编码器
        self.h0_agent = DQN(gamma=0.99, lr=0.0001, action_num=3, state_num=5,
                            buffer_size=10000, batch_size=64, INITIAL_EPSILON=0.2, FINAL_EPSILON=0.001,max_episode=1000,
                            replace=1000, chkpt_dir='./chkpt')
        #下层 选择编码深度
        self.h1_agent = DQN(gamma=0.99, lr=0.0001, action_num=8, state_num=6,
                            buffer_size=10000, batch_size=64, INITIAL_EPSILON=0.2, FINAL_EPSILON=0.001,max_episode=1000,
                            replace=1000, chkpt_dir='./chkpt')
    #动作选择
    def choose_action(self, state):  
        h0_action,_= self.h0_agent.choose_action(state)
        # 下层根据上层经验进行动作选择
        cho_state_0action = state + [h0_action]
        h1_action ,_= self.h1_agent.choose_action(cho_state_0action)
        return h0_action,h1_action
    
    # 存储经验
    def store_transition(self, state, h0_action,h1_action, h0_reward,h1_reward ,next_state,h0_next_action):
        self.h0_agent.store_transition(state, h0_action, h0_reward, next_state)
        store_state_0action=state+[h0_action]
        store_next_state_0action=next_state+[h0_next_action]
        self.h1_agent.store_transition(store_state_0action, h1_action, h1_reward, store_next_state_0action)

    def learn(self):
        self.h0_agent.learn()
        self.h1_agent.learn()