import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise
from environment.vrep_env import Env
#specify parameters here:
episodes=200
is_batch_norm = False #batch normalization switch

def main():
    env= Env(20003)
    steps = 50
    
    #Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
    # agent = DDPG(env, is_batch_norm)
    counter=0
    reward_per_episode = 0    
    num_states = 96*4+2
    num_actions = 2    
    print ("Number of States:", str(num_states))
    print ("Number of Actions:", str(num_actions))
    print ("Number of Steps per episode:", str(steps))
    #saving reward:
    reward_st = np.array([0])
    
    for i in range(episodes):
        print ("==== Starting episode no:",str(i),"====","\n")
        observation = env.reset()
        reward_per_episode = 0
        for t in range(steps):
            action = [-0.4, 0.0]
            observation,reward,done=env.step(action,t)
            reward_per_episode+=reward
            counter+=1
            #check if episode ends:
            if (done):
                print ('EPISODE: ',str(i),' Steps: ',str(t),' Total Reward: ',str(reward_per_episode))
                reward_st = np.append(reward_st,reward_per_episode)
                np.savetxt('episode_reward.txt',reward_st, newline="\n")
                print ('\n\n')
                break

if __name__ == '__main__':
    main()    