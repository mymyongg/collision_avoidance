#Implementation of Deep Deterministic Gradient with Tensor Flow"
# Author: Steven Spielberg Pon Kumar (github.com/stevenpjg)

import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise
from environment.vrep_env import Env
#specify parameters here:
episodes=30000
is_batch_norm = False #batch normalization switch
test = False

def main():
    env= Env(19998)
    steps = 50
    
    #Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
    agent = DDPG(env, is_batch_norm)
    exploration_noise = OUNoise(2)
    counter=0
    reward_per_episode = 0    
    total_reward=0
    num_states = 96*4+2
    num_actions = 2    
    print ("Number of States:", str(num_states))
    print ("Number of Actions:", str(num_actions))
    print ("Number of Steps per episode:", str(steps))
    #saving reward:
    reward_st = np.array([0])

    if test == True:
        agent.actor_net.load_actor('/home/myounghoe/ddpgtf/norepeat/weights/actor/model.ckpt')
        agent.critic_net.load_critic('/home/myounghoe/ddpgtf/norepeat/weights/critic/model.ckpt')
    
    for i in range(episodes):
        print ("==== Starting episode no:",str(i),"====","\n")
        observation = env.reset()
        reward_per_episode = 0
        for t in range(steps):
            x = observation
            action = agent.evaluate_actor(np.reshape(x,[1,num_states]))
            noise = exploration_noise.noise()
            action = action[0] + noise #Select action according to current policy and exploration noise
            if action[0] > 1.:
                action[0] = 1.
            elif action[0] < -1.:
                action[0] = -1.
            if action[1] > 1.:
                action[1] = 1.
            elif action[1] < -1.:
                action[1] = -1.
            print ("Action at step", str(t) ," :",str(action),"\n")
            
            observation,reward,done=env.step(action,t)
            
            #add s_t,s_t+1,action,reward to experience memory
            agent.add_experience(x,observation,action,reward,done)
            #train critic and actor network
            if counter > 64: 
                agent.train()
            reward_per_episode+=reward
            counter+=1
            #check if episode ends:
            if (done):
                print ('EPISODE: ',str(i),' Steps: ',str(t),' Total Reward: ',str(reward_per_episode))
                # print "Printing reward to file"
                exploration_noise.reset() #reinitializing random noise for action exploration
                reward_st = np.append(reward_st,reward_per_episode)
                np.savetxt('episode_reward.txt',reward_st, newline="\n")
                agent.actor_net.save_actor('/home/myounghoe/ddpgtf/norepeat_2action/weights/actor/model.ckpt')
                agent.critic_net.save_critic('/home/myounghoe/ddpgtf/norepeat_2action/weights/critic/model.ckpt')
                print ('\n\n')
                break
    total_reward+=reward_per_episode            
    # print "Average reward per episode {}".format(total_reward / episodes)    


if __name__ == '__main__':
    main()    