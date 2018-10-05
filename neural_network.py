import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

class neural_network:
    def __init__(self, precision, max_learning_time, learning_rate, hidden_nodes, sigmoid):
        """
        Parameter initialization.
        precision, max_learning_time : 计算精度ε，最大学习次数M
        learning_rate : 学习率 η （0.1~3）
        # inertance_Alpha : 惯性项系数 α（0.9~1 或者 0即不用）
        """
        self.precision = precision
        self.max_learning_time = max_learning_time
        self.learning_rate = learning_rate

        self.hidden_nodes = hidden_nodes
        self.sigmoid = sigmoid
        print("Parameter initialization is finished... ")

    def __construct_network(self, feature_dimension, class_number):
        """
        Network structure initialization.
        hidden_nodes : 隐层节点数，自定 eg, list=[4,5,4]
        hidden_layer : 隐藏层层数，>=1 len(list)
        input_nodes : 输入层节点，等于样本特征向量的维数
        output_nodes : 输出层节点，等于分类的类别数（多输出型，1-0编码模式）
        weight_list : 每层间的权重

        :param feature_dimension: 样本特征维数
        :param class_number: 分类的类别数
        :return:
        """
        self.input_nodes = feature_dimension
        self.output_nodes = class_number

        # 相邻两层的权系数和偏置, 范围（-0.5,0.5) , “+1” : 加上一行偏置b
        weight_ih = np.random.random_sample(size=(self.input_nodes + 1, self.hidden_nodes[0])) - 0.5  # 输入层到第一隐层
        self.weight_list = [weight_ih]
        for i in range(1, len(self.hidden_nodes)):  # 中间相邻隐层
            weight_hh = np.random.random_sample(size=(self.hidden_nodes[i - 1] + 1, self.hidden_nodes[i])) - 0.5
            self.weight_list.append(weight_hh)
        weight_ho = np.random.random_sample(size=(self.hidden_nodes[-1] + 1, self.output_nodes)) - 0.5  # 最后一隐层到输出层
        self.weight_list.append(weight_ho)

        network = [self.input_nodes]
        network.extend(self.hidden_nodes)
        network.append(self.output_nodes)
        print("Network construction is finished: ", network)


    def __activation(self,a):
        '''
        :param a: 输入
        :return: 激活函数处理后输出
        '''
        ea = np.exp(a)
        e_a = np.exp(-a)
        if self.sigmoid == "log":
            return (e_a + 1) ** (-1)
        if self.sigmoid == "tanh":
            return (ea - e_a) / (ea + e_a)

    def __derivative(self,y):
        '''
        导数
        :param y:
        :return:
        '''
        if self.sigmoid == "log":
            return y * (1 - y)
        if self.sigmoid == "tanh":
            return (1 + y) * (1 - y)

    def __forward_propagation(self, x_k):
        x_input = x_k
        nodes_list = []
        for weight in self.weight_list:
            h_input = x_input.dot(weight)
            h_output = self.__activation(h_input)
            x_input = np.hstack((np.array([[1]]), h_output))
            nodes_list.append((h_input, h_output))  # 每层结点组成的列表list[元组tuple(输入值input，激活值output),...]
        return nodes_list

    def __back_propagation(self, d_o, nodes_list, x_input):
        # 计算输出层神经元敏感度 δo
        y_output = nodes_list[-1][1]
        sensitivity = (d_o - y_output) * self.__derivative(y_output)  # δo=(do-yo)yo'

        # 从后向前计算隐层节点的δh，每层δ(h)的计算依赖于后一层δ(h+1)
        for i in range(len(nodes_list) - 2, -1, -1):
            h_output = nodes_list[i][1]

            # 修正隐层-输出层，隐层-隐层间权值
            self.weight_list[i + 1] += np.vstack((np.array([[1]]), h_output.T)).dot(
                sensitivity) * self.learning_rate

            dh_output = self.__derivative(h_output)
            w_h = self.weight_list[i + 1][1:]
            sensitivity = sensitivity.dot(w_h.T) * dh_output

        # 修正输入-隐层间权值
        self.weight_list[0] += x_input.T.dot(sensitivity) * self.learning_rate

    def fit(self, training_data, destination):
        # 加载训练数据，构造网络结构
        data_quantity, class_number = np.shape(destination)
        data_quantity, feature_dimension = np.shape(training_data)
        self.__construct_network(feature_dimension, class_number)
        x0 = np.ones((data_quantity, 1), dtype=float)
        input_data = np.hstack((x0, training_data))

        print("Training start:")
        E_sum = 0  # 全局误差和
        learning_count = 0  # 学习次数
        k = 0
        # 对各各样本依次计算直至收敛
        while learning_count <= self.max_learning_time:
            learning_count += 1
            # 前向传播：从前向后计算网络各单元,结果存放于nodes_list = [tuple(输入值node_input，激活值node_output),...]
            xi_k = np.array([input_data[k]])
            nodes_list = self.__forward_propagation(xi_k)
            # print(nodes_list)

            # 计算全局误差，是否达到精度要求
            E_sum += np.average((destination[k] - nodes_list[-1][1]) ** 2) / 2
            E = E_sum / learning_count  # 全局误差 E = ½ ∑ (∑(d-yo)²/q) / k
            # print(E)
            if E <= self.precision:
                break

            # 反向传播：从输出层，从后向前计算各隐层的 δi，并修改连接权值wi
            self.__back_propagation(destination[k], nodes_list, xi_k)

            k += 1
            if k >= data_quantity:
                k = 0
        print("Training finished...")
        print("Learning count: ", learning_count, data_quantity)
        print("E : ", E)

    # 对一个样本做预测
    def predict(self, feature):
        feature = np.array([feature])
        x_input = np.hstack((np.array([[1]]), feature))
        calculation = self.__forward_propagation(x_input)
        y_output = list(calculation[-1][1][0, :])
        return y_output.index(max(y_output))



if __name__=="__main__":
    data=load_iris()
    x=data["data"]
    y=data["target"]
    enco=OneHotEncoder(sparse=False)
    y=enco.fit_transform([[i] for i in y])
    nn=neural_network(hidden_nodes=[10,10,20],sigmoid="log",precision=0.0005,max_learning_time=100000,learning_rate=0.1)
    nn.fit(x[:-1],y[:-1])
    print(nn.predict(x[-1]))
