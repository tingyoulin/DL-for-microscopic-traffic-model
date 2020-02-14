from module import Environment, Vehicle, Simulation

# env1 = Environment()
# veh1 = Vehicle('car')
# veh1.load_data('D:/亭佑/Traj. Data (Tracker)/02/10_car.txt')
# veh2 = Vehicle('motor')
# veh2.load_data('D:/亭佑/Traj. Data (Tracker)/02/11_motor.txt')
# veh3 = Vehicle('motor')
# veh3.load_data('D:/亭佑/Traj. Data (Tracker)/02/12_motor.txt')
# veh4 = Vehicle('motor')
# veh4.load_data('D:/亭佑/Traj. Data (Tracker)/02/13_motor.txt')
# veh5 = Vehicle('motor')
# veh5.load_data('D:/亭佑/Traj. Data (Tracker)/02/14_motor.txt')
#
# env1.add_vehicles(veh1, veh2)
# env1.add_vehicles([veh3, veh4, veh5])
# df = env1.get_data()

sim = Simulation()
X, y, env_list = sim.data_generator(base_dir="C:/Users/User/Google Drive/Master Thesis/Trajectoty Data", input_var=None)
