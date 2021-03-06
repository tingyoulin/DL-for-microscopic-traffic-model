import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys


class Environment:
    """
        DataFrame的子類別
        存某段時間內所有時點包含的車輛
    """
    def __init__(self, name=None, base_dir=None):
        self.input_var = []
        self.output_var = []
        self.__name = name
        self.__base_dir = base_dir  # 記錄環境所在目錄位置
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
                self.__data[veh.num] = veh  # 將車輛添加到__data字典裡
                self.__add_to_time_step(veh)         # 將車輛依所在時點加入到__time_step字典裡
                self.__update_time(veh)              # 更新 start/end time
        return "Number of vehicles  in the environment:", len(self.__data)

    def get_data(self, sim_motor='all', sur_vehicles='all', interval='all'):
        """針對每台機車做計算"""
        final_data_dir='/mnt/c/Users/User/Google Drive/Master Thesis/Data'
        final_env_path = os.path.join(final_data_dir, self.__name)

        # 如果要取得環境中每輛機車的 data, 且所有車都可被當作是周圍車
        if sim_motor == 'all' and sur_vehicles == 'all':
            # 每個__data dict裡的 keys (環境中的每輛車)
            for veh in self.__data.values():
                veh_name = str(veh.num) + '_' + veh.get_type()

                # 跳過不是機車的車輛
                if veh.get_type() != 'motor':
                    continue
                # 跳過已經計算過且存檔的機車
                elif os.path.isfile(os.path.join(final_env_path, veh_name + '.csv')):
                    veh.data = pd.read_csv(os.path.join(final_env_path, veh_name + '.csv'), encoding='utf-8')
                    continue

                self.__calculate_surrounding_vehicles(veh)   # 計算此機車的環境車輛變數
                self.__calculate_env(veh)                    # 計算此機車的環境變數
                
                # 將計算好的車輛資料存成.csv
                veh.data.to_csv(os.path.join(final_env_path, veh_name+'.csv'), index=False, encoding='utf-8')
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
        del length

        for facility in ['curb', 'slow_mix', 'mix_fast', 'center']:
            veh.data[facility] = veh.data['y'] - self.__road_env[facility]
            veh.data.loc[abs(veh.data[facility]) <= (width / 2), facility] = 0.0                                    # 與標線距離在1/2車寬內
            veh.data.loc[veh.data[facility] > (width / 2), facility] - (width / 2)
            veh.data.loc[veh.data[facility] < -(width / 2), facility] + (width / 2)

    def __calculate_surrounding_vehicles(self, veh):
        """
            計算每時點的環境車輛相關變數
            :param veh: 本車
        """
        sur_cols = ['num', 'bike', 'bus', 'car', 'motor', 'pickup', 'small bus', 
                    'rx', 'ry', 'space', 'rv_lon', 'rv_lat']
        veh.add_columns(cols=sur_cols, is_sur=True, value=0)    # 創建環境車輛相關 columns

        for ts in veh.data['t']:
            sur_vehicles = self.__drop_self(v_list=self.__time_step[ts], veh=veh)

            for sur_veh in sur_vehicles:
                veh.switch_direction(sur_veh=sur_veh, time_step=ts)                 # 計算該時點各方向的周圍車
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
        self.turn = False       # 是否為左/右轉車，預設是直行車
        self.__type = vtype     # 車種
        self.__set_size()       # 設定length, width
        self.__sur_direction = ['F', 'LF', 'RF', 'LR', 'RR']

    def __str__(self):
        return str(self.num) + '_' + self.__type   # print Vehicle出來時，格式為num_type

    def add_columns(self, cols, is_sur=False, value=0):
        if is_sur:
            for sur_direction in self.__sur_direction:
                for col in cols:
                    if col in ['space', 'rx', 'ry']:
                        self.data["%s_%s" % (col, sur_direction)] = 999
                    else:
                        self.data["%s_%s" % (col, sur_direction)] = value

        else:
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
        sur_veh_x = sur_veh.data.loc[sur_veh.data['t'] == time_step, 'x'].values[0]
        sur_veh_y = sur_veh.data.loc[sur_veh.data['t'] == time_step, 'y'].values[0]

        if sur_veh_x >= (self_veh_x + (self.__length/2)):
            """車身中段已超過本車的車頭，則定義為 前車(F, LF, RF)"""
            if (sur_veh_y - (sur_veh.get_size()[1]/2)) > (self_veh_y + (self.__width/2)):
                return 'LF'  # 如果前車的右側車身大於本車的左側車身，則定義為 左前車LF
            elif (sur_veh_y + (sur_veh.get_size()[1]/2)) < (self_veh_y - (self.__width/2)):
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
            """找DataFrame的header (行開頭是 t or step 的)"""
            if flines[i].lstrip()[0] == 't' or flines[i].lstrip()[0:4] == 'step':
                input_df = pd.read_csv(file, header=i, encoding='utf-8')
                input_df.rename(columns={'step': 't'}, inplace=True)
                self.data = input_df.reindex(columns=['t', 'x', 'y'])
                del input_df
                break

        if self.data is None:
            """如果找不到header，就使用自訂的columns name"""
            self.data = pd.read_csv(
                file,
                names=['t', 'x', 'y', 'v_lon', 'v_lat', 'a', 'theta'],
                encoding='utf-8'
            )
        self.num = os.path.split(file)[1].split('_')[0]
        self.__calculate()  # 計算速度、加速度、行進角度

        return self.data

    def plot_trajectory(self):
        
        pass

    def predict(self, model, time_step=5):
        # 初始 5 time steps
        test_a = np.array(self.data.loc[0:4, 5:6])          # a, theta
        test_f = np.array(self.data.loc[0:4, 8:18])         # F
        test_lf = np.array(self.data.loc[0:4, 20:30])       # LF
        test_rf = np.array(self.data.loc[0:4, 32:42])
        test_lr = np.array(self.data.loc[0:4, 44:54])
        test_rr = np.array(self.data.loc[0:4, 56:66])
        test_infras = np.array(self.data.loc[0:4, 68:71])

        current_v_lon = self.data.loc[4, 'v_lon']
        current_v_lat = self.data.loc[4, 'v_lat']

        for time in range(len(self.data)-time_step):
            pred_v_lon, pred_v_lat = model.predict([test_a, test_f, test_lf, test_rf, test_lr, test_rr, test_infras])
            pred_a = (np.power(np.power(pred_v_lon - current_v_lon, 2) + np.power(pred_v_lat - current_v_lat, 2), 0.5)) / (2 / 30)                   # 加速度 (m/s^2)
            pred_theta = np.arctan2(pred_v_lat, pred_v_lon) * 180 / np.pi    # 行進角度

            # 更新下一time step 的 input
            test_a = np.vstack([test_a[-4:], [pred_a, pred_theta]])     # a, theta
            test_f = np.array(self.data.loc[time:time+4, 8:18])         # F
            test_lf = np.array(self.data.loc[time:time+4, 20:30])       # LF
            test_rf = np.array(self.data.loc[time:time+4, 32:42])
            test_lr = np.array(self.data.loc[time:time+4, 44:54])
            test_rr = np.array(self.data.loc[time:time+4, 56:66])
            test_infras = np.array(self.data.loc[time:time+4, 68:71])

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
        ##
        self_veh_t = self.data.loc[self.data['t'] == time_step, :]
        sur_veh_t = sur_veh.data.loc[sur_veh.data['t'] == time_step, :]
        # 此環境車與本車的相對縱向位置
        sur_veh_rx = sur_veh_t['x'].values[0] - self_veh_t['x'].values[0]
        # 此環境車與本車的相對橫向位置
        sur_veh_ry = sur_veh_t['y'].values[0] - self_veh_t['y'].values[0]
        # 某方位的環境車輛與本車的車間距離
        origin_space = self_veh_t['space_' + direction].values[0]
        # 此環境車與本車的車間距離
        space_lon = abs(sur_veh_rx) - (self.__length + sur_veh.get_size()[0]) / 2
        space_lon = 0 if space_lon <= 0 else space_lon
        space_lat = abs(sur_veh_ry) - (self.__width + sur_veh.get_size()[1]) / 2
        space_lat = 0 if space_lat <= 0 else space_lat
        space = np.power(np.power(space_lon, 2) + np.power(space_lat, 2), 0.5)
        origin_space = self_veh_t['space_' + direction].values[0]

        # 如果某方位的原周圍車與本車的距離 (default=999) 大於 此周圍車與本車的距離，則將此周圍車指派此方位周圍車
        if origin_space > space:
            # 此環境車與本車的相對縱向速率
            sur_veh_rv_lon = sur_veh_t['v_lon'].values[0] - self_veh_t['v_lon'].values[0]
            # 此環境車與本車的相對橫向速率
            sur_veh_rv_lat = sur_veh_t['v_lat'].values[0] - self_veh_t['v_lat'].values[0]

            self.data.loc[self.data['t'] == time_step, 'num_' + direction] = sur_veh.num
            self.__sur_type_encode(sur_type=sur_veh.get_type(), direction=direction, time_step=time_step)   # identify the type of surrounding vehicles
            self.data.loc[self.data['t'] == time_step, 'rx_' + direction] = sur_veh_rx
            self.data.loc[self.data['t'] == time_step, 'ry_' + direction] = sur_veh_ry
            self.data.loc[self.data['t'] == time_step, 'space_' + direction] = space
            self.data.loc[self.data['t'] == time_step, 'rv_lon_' + direction] = sur_veh_rv_lon
            self.data.loc[self.data['t'] == time_step, 'rv_lat_' + direction] = sur_veh_rv_lat

    def __calculate(self):
        """計算速度、加速度、行進角度"""
        delta_x = self.data['x'].shift(-1) - self.data['x'].shift(1)  # delta_x = x(t+1)-x(t-1)
        delta_y = self.data['y'].shift(-1) - self.data['y'].shift(1)  # delta_y = y(t+1)-y(t-1)

        self.data['v_lon'] = delta_x / (2 / 30)     # 縱向速度 (m/s)
        self.data['v_lat'] = delta_y / (2 / 30)     # 側向速度 (m/s)
        v = np.power(np.power(self.data['v_lon'], 2) + np.power(self.data['v_lat'], 2), 0.5)
        self.data['a'] = (v.shift(-1) - v.shift(1)) / (2 / 30)  # 加速度 (m/s^2)
        # self.data['a_lon'] = (self.data['v_lon'].shift(-1) - self.data['v_lon'].shift(1)) / (2 / 30)  # 縱向加速度 (m/s^2)
        # self.data['a_lat'] = (self.data['v_lat'].shift(-1) - self.data['v_lat'].shift(1)) / (2 / 30)  # 側向加速度 (m/s^2)
        self.data['theta'] = np.arctan2(delta_y, delta_x) * 180 / np.pi  # 行進角度
        self.data.dropna(axis=0, subset=['a'], inplace=True)    # drop a=nan的row (避免往後計算時出錯)
        self.data.reset_index(drop=True, inplace=True)

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
        elif self.__type == 'smallbus':
            self.__length = 8.5
            self.__width = 3.0
        elif self.__type == 'bus':
            self.__length = 12.0
            self.__width = 3.0

    def __sur_type_encode(self, sur_type, direction, time_step):
        for _type in ['bike', 'bus', 'car', 'motor', 'pickup', 'small bus']:
            self.data.loc[self.data['t'] == time_step, '%s_%s' % (_type, direction)] = 1 if _type == sur_type else 0
