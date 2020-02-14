import pandas as pd
import numpy as np
import os, sys
import time


class Simulation:
    def __init__(self, input_time_step=5, output_time_step=5):
        self.INPUT_TIME_STEP = input_time_step
        self.OUTPUT_TIME_STEP = output_time_step

    def data_generator(self, base_dir="C:/Users/User/Google Drive/Master Thesis/Trajectory Data", input_var=None, output_var=None):
        env_list, x, y = [], None, None
        final_data_dir = os.path.split(base_dir)[0] + '/Data'
        # 對 base資料夾中每個 環境資料夾 做事
        for env_name in os.listdir(base_dir):
            print(env_name, end=', ')
            env_start_time = time.time()
            env_path = os.path.join(base_dir, env_name)
            final_env_path = os.path.join(final_data_dir, env_name)

            if not os.path.isdir(env_path):
                continue                    # 跳過 base_dir 裡(非環境目錄)的檔案
            elif not os.path.isdir(final_env_path):
                os.mkdir(final_env_path)    # 若存放計算過車輛資料的環境目錄不存在，就建立新的環境目錄
            # 跳過已經計算過的環境 (已經有 X.npy, Y.npy)
            if os.path.isfile(os.path.join(final_env_path, 'X.npy')) and os.path.isfile(os.path.join(final_env_path, 'Y.npy')):
                env_x = np.load(os.path.join(final_env_path, 'X.npy'), allow_pickle=True)
                env_y = np.load(os.path.join(final_env_path, 'Y.npy'), allow_pickle=True)    # 從 .npy 檔載入已經算好的數據
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
            env = Environment(name=env_name)
            # 對環境資料夾中每個 車輛軌跡檔, road env 檔做事
            for veh_num in os.listdir(env_path):
                # print(veh_num, end=', ')
                # 用 road env 檔建立道路環境設施
                if veh_num == 'road env.txt':
                    df_env = pd.read_csv(os.path.join(base_dir, env_name, veh_num), header=1, encoding='utf-8')
                    slow_mix = df_env.loc[0, 'slow mix']
                    mix_fast = slow_mix + df_env.loc[0, 'mix fast']
                    center = mix_fast + df_env.loc[0, 'center']
                    env.set_env(curb=0.0, slow_mix=slow_mix, mix_fast=mix_fast, center=center)    # 設定道路設施參數
                    continue
                elif veh_num.split('.')[-1] != 'txt':
                    continue              # 跳過非 .txt 檔
                traj_file = os.path.join(base_dir, env_name, veh_num)
                vtype = veh_num.split('.')[0].split('_')[1]      # 取得車種
                veh = Vehicle(vtype)                             # 建立新的 Vehicle 物件
                veh.load_data(traj_file)                         # 載入車輛軌跡檔
                env.add_vehicles(veh)                            # 在當前環境加入該車輛
            # 將該環境放入環境 list 中
            env_list.append(env)

            for veh in env.get_data():
                # 跳過不是機車的車輛
                if veh.get_type() != 'motor':
                    continue
                else:
                    # 將計算好的車輛資料存成.csv
                    veh_name = str(veh.num) + veh.get_type()
                    veh.data.to_csv(os.path.join(final_env_path, veh_name+'.csv'), index=False, encoding='utf-8')
                x_features = self.__input(input_var=input_var)
                y_features = self.__output(output_var=output_var)
                veh_dropna = veh.data.dropna(axis=0, subset=y_features).copy()
                veh_dropna = veh_dropna.reset_index(drop=True)
                veh_dropna_in = veh_dropna.reindex(columns=x_features)
                veh_dropna_out = veh_dropna.reindex(columns=y_features)

                for index in range(self.INPUT_TIME_STEP, len(veh_dropna)-self.OUTPUT_TIME_STEP+1):
                    env_x.append(np.array(veh_dropna_in.loc[index-self.INPUT_TIME_STEP:index-1, :]))
                    env_y.append(np.array(veh_dropna_out.loc[index:index+self.OUTPUT_TIME_STEP-1, :]))
            env_x = np.array(env_x)
            env_y = np.array(env_y)
            np.save(os.path.join(env_path, 'X.npy'), env_x)
            np.save(os.path.join(env_path, 'Y.npy'), env_y)
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

    def __set_features(self, input_var):
        features = ['v_lon', 'v_lat', 'a_lon', 'a_lat',
                    'type_F', 'rx_F', 'ry_F', 'space_F', 'v_lon_F', 'v_lat_F',
                    'type_LF', 'rx_LF', 'ry_LF', 'space_LF', 'v_lon_LF', 'v_lat_LF',
                    'type_RF', 'rx_RF', 'ry_RF', 'space_RF', 'v_lon_RF', 'v_lat_RF',
                    'type_LR', 'rx_LR', 'ry_LR', 'space_LR', 'v_lon_LR', 'v_lat_LR',
                    'type_RR', 'rx_RR', 'ry_RR', 'space_RR', 'v_lon_RR', 'v_lat_RR',
                    'curb', 'slow mix', 'mix fast', 'center']
        if input_var is None or input_var == 'v':
            features.remove('a_lon')
            features.remove('a_lat')
            return features
        elif input_var == 'a':
            features.remove('v_lon')
            features.remove('v_lat')
            return features
        else:
            print('The input variables are limited to V or A.')
            sys.exit(0)

    def __output(self, output_var):
        if output_var is None or output_var == 'v':
            features = ['v_lon', 'v_lat']
            return features
        elif output_var == 'a':
            features = ['a_lon', 'a_lat']
            return features
        else:
            print('The output variables are limited to V or A.')
            sys.exit(0)


class Environment:
    """
        DataFrame的子類別
        存某段時間內所有時點包含的車輛
    """
    def __init__(self, name=None):
        self.__name = name
        self.__data = dict()        # 環境中所有的車輛物件key: 編號
        self.__time_step = dict()   # 整段時間中，所有時點包含的車輛
        self.__start_time = None    # 記錄整段時間中，第一個時點的時間(t(n=0))
        self.__end_time = None      # 記錄整段時間中，最後一個時點的時間
        self.__road_env = {  # 道路環境相關參數
            'curb': 0.0,  # 路緣基準
            'slow_mix': 3.5,  # 慢車道寬
            'mix_fast': 6.5,  # 混合車道寬
            'center': 9.5  # 快車(禁行機車)道寬
        }

    def __str__(self):
        return self.__name   # print Environment 出來時，格式為 name

    def add_vehicles(self, *vehs):
        """
        在DataFrame中增加一輛車，
        :param veh:
        :return:
        """
        # 對每個傳入的 Vehicle物件 做事
        for veh in vehs:
            # 如果傳入整個 list, 則呼叫__veh_is_list()
            if type(veh) == list:
                self.__veh_is_list(veh)
            else:
                # 如果車輛已經在data裡了
                if veh in self.__data.values():
                    print("The vehicle is already in the environment.")
                    continue
                veh.num = len(self.__data)           # 設定車輛編號
                self.__data[len(self.__data)] = veh  # 將車輛添加到__data字典裡
                self.__add_to_time_step(veh)         # 將車輛依所在時點加入到__time_step字典裡
                self.__update_time(veh)              # 更新 start/end time
        return "Number of vehicles  in the environment:", len(self.__data)

    def get_data(self, sim_motor='all', sur_vehicles='all', interval='all'):
        """針對每台機車做計算"""
        # 如果要取得環境中每輛機車的 data, 且所有車都可被當作是環境車
        if sim_motor == 'all' and sur_vehicles == 'all':
            # 每個__data dict裡的 keys (環境中的每輛車)
            for veh in self.__data.values():
                # 跳過不是機車的車輛
                if veh.get_type() != 'motor':
                    continue
                self.__calculate_surrounding_vehicles(veh)   # 計算此機車的環境車輛變數
                self.__calculate_env(veh)                    # 計算此機車的環境變數
        return self.__data.values()   # 回傳 __data.values() (整個環境下的所有車輛)

    def remove_vehicles(self, veh):
        self.__data.pop(veh)

    def set_env(self, **kwarg):
        """手動設定道路環境參數"""
        self.__road_env['curb'] = kwarg['curb']
        self.__road_env['slow_mix'] = kwarg['slow_mix']
        self.__road_env['mix_fast'] = kwarg['mix_fast']
        self.__road_env['center'] = kwarg['center']

        return self.__road_env

    def __add_to_time_step(self, veh):
        """將車輛依所在時點加入到__time_step字典裡"""
        for t in veh.data['t']:
            if t not in self.__time_step.keys():
                self.__time_step[t] = [veh]         # 如果該時點沒有其他車輛，則建立該時點的 t: list()
            else:
                self.__time_step[t].append(veh)     # 如果該時點已有其他車輛，則加入該時點的list

    def __calculate_env(self, veh):
        length, width = veh.get_size()

        for facility in ['curb', 'slow_mix', 'mix_fast', 'center']:
            veh.data[facility] = veh.data['y'] - self.__road_env[facility]
            veh.data.loc[abs(veh.data[facility]) <= (width / 2), facility] = 0.0
            veh.data.loc[(abs(veh.data[facility]) > (width / 2) & veh.data[facility] > 0), facility] - (width / 2)
            veh.data.loc[(abs(veh.data[facility]) > (width / 2) & veh.data[facility] < 0), facility] + (width / 2)

    def __calculate_surrounding_vehicles(self, veh):
        """
        計算每時點的環境車輛相關變數
        :param veh: 本車
        """
        veh.add_columns(cols=['num_F', 'type_F', 'rx_F', 'ry_F', 'space_F', 'rv_lon_F', 'rv_lat_F',
                              'num_LF', 'type_LF', 'rx_LF', 'ry_LF', 'space_LF', 'rv_lon_LF', 'rv_lat_LF',
                              'num_RF', 'type_RF', 'rx_RF', 'ry_RF', 'space_RF', 'rv_lon_RF', 'rv_lat_RF',
                              'num_LR', 'type_LR', 'rx_LR', 'ry_LR', 'space_LR', 'rv_lon_LR', 'rv_lat_LR',
                              'num_RR', 'type_RR', 'rx_RR', 'ry_RR', 'space_RR', 'rv_lon_RR', 'rv_lat_RR'], value=np.nan)    # 創建環境車輛相關欄位

        for ts in veh.data['t']:
            sur_vehicles = self.__drop_self(v_list=self.__time_step[ts], veh=veh)

            for sur_veh in sur_vehicles:
                veh.switch_direction(sur_veh=sur_veh, time_step=ts)   # 計算該時點個方向的周圍車
        pass

    def __drop_self(self, v_list, veh):
        """
        清掉 list中 value 是本車輛的 key-value pair
        :param v_list: 是一個包含在某時階所有車輛的list
        :param veh: 本車
        :return: 不包含本車的list
        """
        new_list = list()
        for v in v_list:
            if v is not veh:
                new_list.append(v)

        return new_list

    def __specified_sim_motor(self, sim_motor):
        pass

    def __specified_surrounding_vehicles(self, env_vehicles):
        pass

    def __update_time(self, veh):
        """更新環境的 start/end time"""
        if self.__start_time is None or self.__start_time > min(veh.data['t']):
            self.__start_time = min(veh.data['t'])
        if self.__end_time is None or self.__end_time > max(veh.data['t']):
            self.__end_time = max(veh.data['t'])

    def __veh_is_list(self, veh):
        for vehicle in veh:
            if vehicle in self.__data.values():
                print("The vehicle is already in the environment.")  # 車輛已經在data裡了
                continue
            vehicle.num = len(self.__data)  # 設定車輛編號
            self.__data[len(self.__data)] = vehicle  # 將車輛添加到__data字典裡
            self.__add_to_time_step(vehicle)         # 將車輛依所在時點加入到__time_step字典裡
            self.__update_time(vehicle)  # 更新start/end time


class Vehicle:
    """
        每一輛車的資訊，包含車種、編號、每時點的(x, y, v, a, theta)
    """
    def __init__(self, vtype='motor'):
        self.data = None        # 每時點的資料
        self.num = None         # 編號
        self.__type = vtype     # 車種
        self.__set_size()       # 設定length, width

    def __str__(self):
        return str(self.num) + '_' + self.__type   # print Vehicle出來時，格式為num_type

    def add_columns(self, cols, value=np.nan):
        for col in cols:
            self.data[col] = value
        pass

    def direction(self, sur_veh, time_step):
        """
        計算本車與環境車輛在某時點的方位關係
        :param sur_veh: 某時點的環境車輛
        :param time_step:
        :return: 某時點與環境車輛的方位關係
        """
        self_veh_x = self.data.loc[self.data['t'] == time_step, 'x'].values[0]
        self_veh_y = self.data.loc[self.data['t'] == time_step, 'y'].values[0]
        self_veh_length, self_veh_width = self.__length, self.__width
        sur_veh_x = sur_veh.data.loc[sur_veh.data['t'] == time_step, 'x'].values[0]
        sur_veh_y = sur_veh.data.loc[sur_veh.data['t'] == time_step, 'y'].values[0]
        sur_veh_length, sur_veh_width = sur_veh.get_size()

        if (sur_veh_x - (sur_veh_length/2)) >= (self_veh_x + (self_veh_length/2)):
            """車尾已超過本車的車頭，則定義為 前車(F, LF, RF)"""
            if (sur_veh_y - (sur_veh_width/2)) > (self_veh_y + (self_veh_width/2)):
                return 'LF'  # 如果前車的右側車身大於本車的左側車身，則定義為 左前車LF
            elif (sur_veh_y + (sur_veh_width/2)) < (self_veh_y - (self_veh_width/2)):
                return 'RF'  # 如果前車的左側車身小於本車的右側車身，則定義為 右前車RF
            else:
                return 'F'  # 如果前兩個狀況都不符合，則定義為 (正)前車F
        else:
            """車尾 沒有 超過本車的車頭，則定義為 後車(LR, RR)"""
            if sur_veh_y >= self_veh_y:
                return 'LR'  # 如果後車的y大於本車的y，則定義為 左後車LR
            elif sur_veh_y < self_veh_y:
                return 'RR'  # 如果後車的y小於本車的y，則定義為 右後車RR

    def distance(self, sur_veh, time_step):
        """
        計算本車與環境車輛在某時點的直線距離
        :param sur_veh: 某時點的環境車輛
        :param time_step:
        :return: 某時點與環境車輛的距離
        """
        self_veh_x = self.data.loc[self.data['t'] == time_step, 'x'].values[0]
        self_veh_y = self.data.loc[self.data['t'] == time_step, 'y'].values[0]
        sur_veh_x = sur_veh.data.loc[sur_veh.data['t'] == time_step, 'x'].values[0]
        sur_veh_y = sur_veh.data.loc[sur_veh.data['t'] == time_step, 'y'].values[0]

        return np.power(np.power(sur_veh_x - self_veh_x, 2) + np.power(sur_veh_y - self_veh_y, 2), 0.5)

    def get_size(self):
        return self.__length, self.__width

    def get_type(self):
        return self.__type

    def load_data(self, file):
        """
        讀取每輛車的Tracker資料
        :param file: Tracker資料(.txt)
        :return: self.data (pandas.DataFrame)
        """
        fin = open(file, encoding='utf-8')
        flines = fin.readlines()  # flines是一個list，每個element是file中的每一行

        for i in range(len(flines)):
            """找DataFrame的header (行開頭是t的)"""
            if flines[i].lstrip()[0] == 't':
                input_df = pd.read_csv(file, header=i, encoding='utf-8')
                self.data = input_df.loc[:, ['t', 'x', 'y']].copy()
                del input_df
                break

        if self.data is None:
            """如果找不到header，就使用自訂的columns name"""
            self.data = pd.read_csv(
                file,
                names=['t', 'x', 'y', 'v_lon', 'v_lat', 'a_lon', 'a_lat', 'theta'],
                encoding='utf-8'
            )
        # self.data['t'] = np.around(self.data['t'], decimals=3)
        self.__calculate()  # 計算速度、加速度、行進角度

        return self.data

    def set_type(self, vtype):
        self.__type = vtype
        self.__set_size()  # 調整length, width

    def switch_direction(self, sur_veh, time_step):
        """
        查看此環境車的方位關係，且如果(該方位沒有其他車輛 或 此環境車距離較短)，則更新相關資訊
        :param sur_veh: 環境車
        :param time_step:
        """
        direction = self.direction(sur_veh, time_step)   # 環境車與本車的方位關係

        if direction is None:     # 用來找bug
            print('self:', self)
            print('sur_veh:', sur_veh)
            print('time_step:', time_step)
        # 某方位的環境車輛與本車的距離
        dist_direction = self.data.loc[self.data['t'] == time_step, 'space_'+direction]
        # 此環境車與本車的相對縱向位置
        sur_veh_rx = (
            sur_veh.data.loc[sur_veh.data['t'] == time_step, 'x'].values[0] - self.data.loc[self.data['t'] == time_step, 'x'].values[0]
        )
        # 此環境車與本車的相對橫向位置
        sur_veh_ry = (
            sur_veh.data.loc[sur_veh.data['t'] == time_step, 'y'].values[0] - self.data.loc[self.data['t'] == time_step, 'y'].values[0]
        )
        # 此環境車與本車的相對縱向速率
        sur_veh_rv_lon = (
                sur_veh.data.loc[sur_veh.data['t'] == time_step, 'v_lon'].values[0] - self.data.loc[sur_veh.data['t'] == time_step, 'v_lon'].values[0]
        )
        # 此環境車與本車的相對橫向速率
        sur_veh_rv_lat = (
                sur_veh.data.loc[sur_veh.data['t'] == time_step, 'v_lat'].values[0] - self.data.loc[sur_veh.data['t'] == time_step, 'v_lat'].values[0]
        )
        # 如果某方位的周圍車與本車的距離是 nan 或 值大於此環境車與本車的距離，則將此周圍車指派此方位周圍車
        if (dist_direction.isnull().values[0]) or (dist_direction.values[0] > self.distance(sur_veh, time_step)):
            self.data.loc[self.data['t'] == time_step, 'num_'+direction] = sur_veh.num
            self.data.loc[self.data['t'] == time_step, 'type_'+direction] = sur_veh.get_type()
            self.data.loc[self.data['t'] == time_step, 'rx_' + direction] = sur_veh_rx
            self.data.loc[self.data['t'] == time_step, 'ry_' + direction] = sur_veh_ry
            self.data.loc[self.data['t'] == time_step, 'space_'+direction] = self.distance(sur_veh, time_step)
            self.data.loc[self.data['t'] == time_step, 'rv_lon_'+direction] = sur_veh_rv_lon
            self.data.loc[self.data['t'] == time_step, 'rv_lat_'+direction] = sur_veh_rv_lat

    def __calculate(self):
        """計算速度、加速度、行進角度"""
        delta_x = self.data['x'].shift(-1) - self.data['x'].shift(1)  # delta_x = x(t+1)-x(t-1)
        delta_y = self.data['y'].shift(-1) - self.data['y'].shift(1)  # delta_y = y(t+1)-y(t-1)

        # self.data['v'] = np.power(
        #     np.power(delta_x / (2 / 30), 2) + np.power(delta_y / (2 / 30), 2), 0.5
        # )  # 速度v = ( {[x(t+1)-x(t-1)]/(2/30)}^2 + {[y(t+1)-y(t-1)]/(2/30)}^2) ^ (0.5)
        #
        self.data['v_lon'] = delta_x / (2 / 30)
        self.data['v_lat'] = delta_y / (2 / 30)
        self.data['a_lon'] = (self.data['v_lon'].shift(-1) - self.data['v_lon'].shift(1)) / (2 / 30)  # 縱向加速度
        self.data['a_lat'] = (self.data['v_lat'].shift(-1) - self.data['v_lat'].shift(1)) / (2 / 30)  # 側向加速度
        self.data['theta'] = np.arctan2(delta_y, delta_x) * 180 / np.pi  # 行進角度

    def __set_size(self):
        if self.__type == 'motor' or self.__type == 'bike':
            self.__length = 2.0
            self.__width = 0.8
        elif self.__type == 'car':
            self.__length = 4.5
            self.__width = 2.0
        elif self.__type == 'pickup':
            self.__length = 5.0
            self.__width = 2.0
        elif self.__type == 'small bus' or self.__type == 'small bus (parking)':
            self.__length = 8.5
            self.__width = 3.0
        elif self.__type == 'bus' or self.__type == 'bus (parking)':
            self.__length = 12.0
            self.__width = 3.0
