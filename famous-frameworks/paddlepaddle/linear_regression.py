
import paddle
import paddle.fluid as fluid
import numpy, math, sys


BATCH_SIZE = 20

train_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.uci_housing.train(), buf_size=500),
        batch_size=BATCH_SIZE)

test_reader = paddle.batch(
        paddle.reader.shuffle(paddle.dataset.uci_housing.test(), buf_size=500),
        batch_size=BATCH_SIZE)

x = fluid.layers.data(name='x', shape=[13], dtype='float32') # 定义输入的形状和数据类型
y = fluid.layers.data(name='y', shape=[1], dtype='float32') # 定义输出的形状和数据类型
print(x)
print(y)

y_predict = fluid.layers.fc(input=x, size=1, act=None) # 连接输入和输出的全连接层

main_program = fluid.default_main_program() # 获取默认/全局主函数
startup_program = fluid.default_startup_program() # 获取默认/全局启动程序

cost = fluid.layers.square_error_cost(input=y_predict, label=y) # 利用标签数据和输出的预测数据估计方差
avg_loss = fluid.layers.mean(cost) # 对方差求均值，得到平均损失


sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_loss)

#克隆main_program得到test_program
#有些operator在训练和测试之间的操作是不同的，例如batch_norm，使用参数for_test来区分该程序是用来训练还是用来测试
#该api不会删除任何操作符,请在backward和optimization之前使用
test_program = main_program.clone(for_test=True)



use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace() # 指明executor的执行场所

###executor可以接受传入的program，并根据feed map(输入映射表)和fetch list(结果获取表)向program中添加数据输入算子和结果获取算子。使用close()关闭该executor，调用run(...)执行program。
exe = fluid.Executor(place)







