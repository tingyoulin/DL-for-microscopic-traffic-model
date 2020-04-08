from module import Environment, Vehicle
import os, sys, time
import numpy as np
import pandas as pd


class Simulation:
    def __init__(self, input_time_step=5, output_time_step=5):
        self.INPUT_TIME_STEP = input_time_step
        self.OUTPUT_TIME_STEP = output_time_step

    def data_generator(self, model, input_var='v', output_var='v'):
        # 最後的 x, y 以 model 名稱區別
        # final_dir = os.path.join(os.path.split(traj_data_dir)[0], final_dir)
        traj_data_dir='/mnt/c/Users/User/Google Drive/Master Thesis/Trajectory Data'
        # 最終的資料存在 Users/User/Google Drive/Master Thesis/Data'
        final_data_dir = '/mnt/c/Users/User/Google Drive/Master Thesis/Data'
        env_list, x, y = [], None, None
        
        # 對 base資料夾中每個 環境目錄 做事)
        for env_name in os.listdir(traj_data_dir):
            print(env_name)
            env_start_time = time.time()
            env_path = os.path.join(traj_data_dir, env_name)
            final_env_path = os.path.join(final_data_dir, env_name)

            # 跳過traj_data_dir裡 (非環境目錄) 的檔案
            if not os.path.isdir(env_path):
                continue
            # 若存放計算過車輛資料的環境目錄 (final_env_path) 不存在，就建立新的環境目錄
            elif not os.path.isdir(final_env_path):
                os.mkdir(final_env_path)
            
            # 跳過已經計算過的環境 (已經有 X.npy, Y.npy)
            if os.path.isfile(final_env_path + '/{}_X.npy'.format(model)) and os.path.isfile(final_env_path + '/{}_Y.npy'.format(model)):
                env_x = np.load(final_env_path + '/{}_X.npy'.format(model), allow_pickle=True)
                env_y = np.load(final_env_path + '/{}_Y.npy'.format(model), allow_pickle=True)    # 從 .npy 檔載入已經算好的數據
                print('env_x.shape:', env_x.shape)
                print('env_y.shape:', env_y.shape)

                # 如果 x 或 y 是空的
                if x is None or y is None:
                    x = env_x.copy()
                    y = env_y.copy()
                else:
                    x = np.vstack([x, env_x])
                    y = np.vstack([y, env_y])
                continue
            else:
                env_x, env_y = [], []
            
            # 建立新的 Environment 物件並命名
            env = Environment(name=env_name, base_dir=env_path)
            # 對環境資料夾中每個 車輛軌跡檔, road env 檔做事
            for veh_num in os.listdir(env_path):
                # 用 road env 檔建立道路環境設施
                if veh_num == 'road env.txt':
                    df_env = pd.read_csv(os.path.join(traj_data_dir, env_name, veh_num), header=1, encoding='utf-8')

                    # 設定道路設施參數
                    slow_mix = df_env.loc[0, 'slow mix']
                    mix_fast = slow_mix + df_env.loc[0, 'mix fast']
                    center = mix_fast + df_env.loc[0, 'center']
                    env.set_env(curb=0.0, slow_mix=slow_mix, mix_fast=mix_fast, center=center)
                    continue
                # 跳過非 .txt 檔
                elif veh_num.split('.')[-1] != 'txt':
                    continue
                
                traj_file = os.path.join(traj_data_dir, env_name, veh_num)
                vtype = veh_num.split('.')[0].split('_')[1]      # 取得車種
                veh = Vehicle(vtype)                             # 建立新的 Vehicle 物件
                veh.load_data(traj_file)                         # 載入車輛軌跡檔
                env.add_vehicles(veh)                            # 在當前環境加入該車輛
            
            # 將該環境放入 環境list 中
            env_list.append(env)
            print("Loading Trajectory Data Done!!")

            for veh in env.get_data():
                # 跳過不是機車的車輛
                if veh.get_type() != 'motor':
                    continue

                x_features = self.__input(input_var=input_var)
                y_features = self.__output(output_var=output_var)
                # print("veh.data.shape:", veh.data.shape)
                # TODO: 檢查 reindex 有沒有問題
                veh_in = veh.data.reindex(columns=x_features)
                veh_out = veh.data.reindex(columns=y_features)

                "==========Transform to the data format required by time-series model========="
                for index in range(self.INPUT_TIME_STEP, len(veh_in)-self.OUTPUT_TIME_STEP+1):
                    env_x.append(np.array(veh_in.loc[index-self.INPUT_TIME_STEP:index-1, :]))
                    env_y.append(np.array(veh_out.loc[index:index+self.OUTPUT_TIME_STEP-1, :]))

                    # look for bug
                    # if env_x[-1].shape[0] != self.INPUT_TIME_STEP:
                    #     print("env:", env, "veh:", veh)
                    #     print("env_x[-1].shape:", env_x[-1].shape)
                    #     print("[index-self.INPUT_TIME_STEP:index-1]:", [index-self.INPUT_TIME_STEP, index-1])
                    # elif env_y[-1].shape[0] != self.OUTPUT_TIME_STEP:
                    #     print("env:", env, "veh:", veh)
                    #     print("env_y[-1].shape:", env_y[-1].shape)
                    #     print("[index:index+self.OUTPUT_TIME_STEP-1]:", [index, index+self.OUTPUT_TIME_STEP-1])

            env_x = np.array(env_x)
            env_y = np.array(env_y)
            print('env_x.shape:', env_x.shape)
            print('env_y.shape:', env_y.shape)

            # 存.npy檔
            np.save(final_env_path + '/{}_X.npy'.format(model), env_x)
            np.save(final_env_path + '/{}_Y.npy'.format(model), env_y)

            # 將 env_x, env_y 放進 x, y
            if x is None or y is None:
                x = env_x.copy()
                y = env_y.copy()
            else:
                x = np.vstack([x, env_x])
                y = np.vstack([y, env_y])

            print('time spent: %.2f' % (time.time() - env_start_time))    # print 出產生此環境數據所花費的時間

        print('x.shape:', x.shape)
        print('y.shape:', y.shape)
        NUM_INPUT_ATTRIBUTES = x.shape[-1]
        NUM_OUTPUT_ATTRIBUTES = y.shape[-1]
        # reshaping
        x_reshape = np.reshape(x, (x.shape[0], self.INPUT_TIME_STEP, NUM_INPUT_ATTRIBUTES))
        y_reshape = np.reshape(y, (y.shape[0], self.OUTPUT_TIME_STEP, NUM_OUTPUT_ATTRIBUTES))

        return x_reshape, y_reshape, env_list

    def __input(self, input_var):
        features = ['v_lon', 'v_lat', 'a', 'theta',
                    'bike_F', 'bus_F', 'car_F', 'motor_F', 'pickup_F', 'small bus_F', 'rx_F', 'ry_F', 'space_F', 'rv_lon_F', 'rv_lat_F',
                    'bike_LF', 'bus_LF', 'car_LF', 'motor_LF', 'pickup_LF', 'small bus_LF', 'rx_LF', 'ry_LF', 'space_LF', 'rv_lon_LF', 'rv_lat_LF',
                    'bike_RF', 'bus_RF', 'car_RF', 'motor_RF', 'pickup_RF', 'small bus_RF', 'rx_RF', 'ry_RF', 'space_RF', 'rv_lon_RF', 'rv_lat_RF',
                    'bike_LR', 'bus_LR', 'car_LR', 'motor_LR', 'pickup_LR', 'small bus_LR', 'rx_LR', 'ry_LR', 'space_LR', 'rv_lon_LR', 'rv_lat_LR',
                    'bike_RR', 'bus_RR', 'car_RR', 'motor_RR', 'pickup_RR', 'small bus_RR', 'rx_RR', 'ry_RR', 'space_RR', 'rv_lon_RR', 'rv_lat_RR',
                    'curb', 'slow_mix', 'mix_fast', 'center']
        if input_var == 'v':
            features.remove('a')
            features.remove('theta')
            return features
        elif input_var == 'a':
            features.remove('v_lon')
            features.remove('v_lat')
            return features
        else:
            print('The input variables are limited to V or A.')
            sys.exit(0)

    def __output(self, output_var):
        if output_var == 'v':
            return ['v_lon', 'v_lat']
        elif output_var == 'a':
            return ['a', 'theta']
        else:
            print('The output variables are limited to V or A.')
            sys.exit(0)


# env1 = Environment()
# veh1 = Vehicle('car')
# veh1.load_data(os.path.join(base_dir, '01-section/25_car.txt'))
# veh2 = Vehicle('motor')
# veh2.load_data(os.path.join(base_dir, '01-section/26_motor.txt'))
# veh3 = Vehicle('motor')
# veh3.load_data(os.path.join(base_dir, '01-section/27_motor.txt'))
# veh4 = Vehicle('motor')
# veh4.load_data(os.path.join(base_dir, '01-section/28_motor.txt'))
# veh5 = Vehicle('motor')
# veh5.load_data(os.path.join(base_dir, '01-section/29_motor.txt'))

# env1.add_vehicles(veh1, veh2)
# env1.add_vehicles([veh3, veh4, veh5])
# df = env1.get_data()

sim = Simulation(input_time_step=5, output_time_step=1)
X, y, env_list = sim.data_generator(model='lstm', input_var='a', output_var='v')
