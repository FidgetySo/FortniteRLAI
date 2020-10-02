import numpy as np
import torch
import argparse
import os
import math
import sys
import random
import time
import json
import copy

import utils
from logger import Logger

from curl_sac import CurlSacAgent
from torchvision import transforms

import cv2
from mss import mss

import time


import pyautogui
import pydirectinput


pyautogui.FAILSAFE = False
pydirectinput.FAILSAFE = False
def set_pos(x, y):
    pydirectinput.move(x, y)

building = False
class Actions:
    def action(self, choice):
        cursor_x, cursor_y = pydirectinput.position()
        
        if cursor_x >= 1900:
            pydirectinput.keyDown('alt')
            time.sleep(0.015)
            pydirectinput.keyUp('alt')
        elif cursor_x <= 20:
            pydirectinput.keyDown('alt')
            time.sleep(0.015)
            pydirectinput.keyUp('alt')
        if cursor_y >= 1060:
            pydirectinput.keyDown('alt')
            time.sleep(0.015)
            pydirectinput.keyUp('alt')
        elif cursor_y <= 20:
            pydirectinput.keyDown('alt')
            time.sleep(0.015)
            pydirectinput.keyUp('alt')
        if choice == 0:
            pydirectinput.mouseDown(button="left")
            time.sleep(0.1)
            pydirectinput.mouseUp(button="left")
        elif choice == 1:
            set_pos(45, None)
        elif choice == 2:
            set_pos(-45, None)
        elif choice == 3:
            set_pos(None, 45)
        elif choice == 4:
            set_pos(None, -45)
        elif choice == 5:
            pydirectinput.keyDown('shift')
            time.sleep(0.015)
            pydirectinput.keyDown('w')
        elif choice == 6:
            pydirectinput.keyUp('shift')
            time.sleep(0.015)
            pydirectinput.keyUp('w')
        elif choice == 7:
            pyautogui.scroll(1)
            time.sleep(0.01)
        elif choice == 8:
            pyautogui.scroll(-1)
            time.sleep(0.01)
        elif choice == 9:
            pydirectinput.press('e')
        elif choice == 10:
            pydirectinput.keyDown('space')
            time.sleep(0.015)
            pydirectinput.keyUp('space')
        elif choice == 11:
            pydirectinput.press('0')
            if building:
                building = False
            else:
                building = True
            time.sleep(0.015)
            
class Env:
    SIZE = 84 #Image Size
    player = Actions()
    OBSERVATION_SPACE_VALUES = (3, SIZE, SIZE)  # 3 Channel image array with 84x84
    ACTION_SPACE_SIZE = 11

    def reset(self):
        # Reset the variables
        self.episode_step = 0
        observation = np.array(self.get_image())
        observation=cv2.resize(observation,(self.SIZE,self.SIZE),interpolation=cv2.INTER_AREA)
        observation=observation[...,:3]
        observation = observation.transpose().astype(np.uint8)
        return torch.tensor(observation, dtype=torch.float32, device="cpu").div_(255)

    def step(self, action):
        print(action)
        #Add episode step
        self.episode_step += 1
        if self.episode_step < 10:
            previus_health = 1
            self.current_health = 1
        self.player.action(action)

        img = np.array(self.get_image())
        new_observation = cv2.resize(img,(self.SIZE,self.SIZE),interpolation=cv2.INTER_AREA)
        new_observation = new_observation[...,:3]
        new_observation = new_observation.transpose().astype(np.uint8)
        done = False
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        done_check = self.done_check(img)
        reward_damage = self.get_damage(img)

        reward_health =self.get_health(img)
        previus_health = self.current_health
        self.current_health = reward_health / 301.5
        print(self.current_health)
        reward = 0.001
        if reward_damage >= 1:
            reward += 0.35
        if self.current_health < previus_health:
            reward += self.current_health - previus_health
        elif self.current_health > previus_health:
            reward += self.current_health - previus_healt:
        if building:
            if action == 0:
                reward += .075
        else:
            if action == 0 and reward_damage <= 0:
                reward -= 0.1
        if done_check > 5000:
            done = True
            pyautogui.keyUp('shift')
            pyautogui.keyUp('w')
            reward = 0
        return torch.tensor(new_observation, dtype=torch.float32, device="cpu").div_(255), reward, done

    # FOR CNN #
    def get_image(self):
        with mss() as sct:
            img = np.array(sct.grab(sct.monitors[1]))
        return img
    def done_check(self, img):
        black_min=np.array([0,0,0] , np.uint8)
        black_max=np.array([0,0,0] , np.uint8)

        dst=cv2.inRange(img , black_min , black_max)
        no_black =cv2.countNonZero(dst)
        print('The number of black  pixels are: ' + str(no_black))
        return no_black
    def get_damage(self, image_xp):
        image=image_xp[ 407:407 + 215 , 980:980 + 323 , : ]
        blue_min = np.array([95,226,234] , np.uint8)
        blue_max =np.array([95,226,234] , np.uint8)

        blue = cv2.inRange(image , blue_min , blue_max)
        total = cv2.countNonZero(blue)
        white_min =np.array([221,221,217] , np.uint8)
        white_max =np.array([221,221,217] , np.uint8)

        white=cv2.inRange(image , white_min , white_max)
        total += cv2.countNonZero(white)
        
        yellow_min =np.array([232,223,0] , np.uint8)
        yellow_max =np.array([232,223,0] , np.uint8)

        yellow=cv2.inRange(image , yellow_min , yellow_max)
        total += cv2.countNonZero(yellow)
        print('The number of damage  pixels are: ' + str(total))
        return total
    def get_health(self, image):
        image=image[ 1015:1015 + 1 , 306:306 + 304 , : ]
        red_min=np.array([ 8, 170, 0] , np.uint8)
        red_max=np.array([ 150, 255, 75 ] , np.uint8)

        health_dst=cv2.inRange(image , red_min , red_max)
        number_health =cv2.countNonZero(health_dst)
        print('The number of health pixels are: ' + str(number_health))
        return number_health
        
        
def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='fps')
    parser.add_argument('--task_name', default='train')
    parser.add_argument('--pre_transform_image_size', default=84, type=int)

    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=4, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=10000, type=int)
    # train
    parser.add_argument('--agent', default='curl_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--hidden_dim', default=512, type=int)
    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--num_eval_episodes', default=15, type=int)
    # critic
    parser.add_argument('--critic_lr', default=0.003, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float) # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2, type=int) # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', default=0.003, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=100, type=int)
    parser.add_argument('--encoder_lr', default=0.003, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--curl_latent_dim', default=128, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=0.003, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=True, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')

    parser.add_argument('--log_interval', default=100, type=int)
    args = parser.parse_args()
    return args


def evaluate(env, agent, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            while not done:
                # center crop image
                if args.encoder_type == 'pixel':
                    obs = utils.center_crop_image(obs,args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        print("sample_stochastically")
                        action = random.randint(0, 11)
                    else:
                        print("agent selected")
                        action = agent.select_action(obs)
                obs, reward, done = env.step(action)
                episode_reward += reward

            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
        
        L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'curl_sac':
        return CurlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            curl_latent_dim=args.curl_latent_dim

        )
    else:
        assert 'agent is not supported: %s' % args.agent
def main():
    args = parse_args()
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1,1000000)
    utils.set_seed_everywhere(args.seed)
    env = Env()

    # stack several consecutive frames together
    env = utils.FrameStack(env, k=args.frame_stack)
    
    # make directory
    ts = time.gmtime() 
    ts = time.strftime("%m-%d", ts)    
    env_name = args.domain_name + '-' + args.task_name
    exp_name = env_name + '-' + ts + '-im' + str(args.image_size) +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed)  + '-' + args.encoder_type
    args.work_dir = args.work_dir + '/'  + exp_name

    utils.make_dir(args.work_dir)
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    torch.cuda.current_device()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.ACTION_SPACE_SIZE

    obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
    pre_aug_obs_shape = (3*args.frame_stack,args.pre_transform_image_size,args.pre_transform_image_size)
    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )
    #agent.load("D:/curl-master/fps-train-09-29-im84-b64-s829604-pixel/model", 2000)
    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(args.num_train_steps):
        pre_time = time.time()
        # evaluate agent periodically

        if step % args.eval_freq == 0 and step != 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, args.num_eval_episodes, L, step,args)
            if args.save_model:
                agent.save_curl(model_dir, step)
                agent.save(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            print("random")  
            action = random.randint(0, 11)
        else:
            with utils.eval_mode(agent):
                print("random 2")  
                action = random.randint(0, 11)

        # run training update
        if step >= args.init_steps:
            num_updates = 1 
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)
        print("Agent selected 2")
        next_obs, reward, done = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == 500000000 else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1
        time_used=time.time() - pre_time
        print(time_used)
        print("FPS:")
        fps = 1 / time_used
        print(fps)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
