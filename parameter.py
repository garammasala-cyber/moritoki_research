#計算時間に関係するパラメータ
num_epoch = 500 #何回学習させるか
num_data = 1000 #学習用のデータ数
num_testdata = 200
num_partition = 100 #number of partition
learning_rate = 0.01

#結果に関係するパラメータ
a = -5 #一様分布
b = (-1)*a #一様分布
K = 15 #行使価格
T = 1 #expiration
epsilon = 1e-1



h = 1e-2
r = 0.01 #interest rate
q = 0.008 #dividend
alpha = (r-q-1)/2
beta = alpha**2 + r
len_partition = T / num_partition