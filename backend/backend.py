import os
from time import sleep

import numpy as np
import torchvision
from PIL import Image
from flask import Flask, request, jsonify, send_file
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

app = Flask(__name__)
df = None
cluster_info = ['No data available.']
dimension_info = ['No data available.']
ml_info = ['No data available.']
data_path = 'None'
train_test_ratio = 0.7
sub_dir_list = []
model = None
train_loader = None
test_loader = None
train_set = None
test_set = None


class MyDataset(torch.utils.data.Dataset):  # 继承torch.utils.data.Dataset，定义自己的数据类
    def __init__(self, datatxt, transform=None):  # 初始化一些需要传入的参数

        fh = open(datatxt, 'r')  # 打开 txt 文件，并读取内容
        imgs = []  # 创建一个名为 img 的空列表
        for line in fh:  # 按行循环 txt 文本中的内容
            line = line.rstrip()
            words = line.split(',')  # 通过逗号对字符串进行切片
            # 把 txt 里的内容读入 imgs 列表保存，words[0]是图片路径，words[1]是label
            imgs.append((words[0], int(words[1])))  #
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]  # 将 imgs 中第 index 个内容分别赋值给 fn 和 label（fn是图片路径，label是标签）
        img = Image.open(fn).convert('RGB')  # 按照路径读入图片

        if self.transform is not None:
            img = self.transform(img)  # 对图片进行 transform （例如，将图片缩小，标准化等）
        return img, label

    def __len__(self):  # 返回的是数据集的长度，也就是多少张图片
        return len(self.imgs)


# 上传csv文件
@app.route('/cluster/upload-csv', methods=['POST'])
def cluster_upload_csv():
    global df
    file = request.files['file']  # 获取上传的文件
    print("上传的文件名:" + file.filename)
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)  # 使用 Pandas 读取 CSV 文件
        # 将标头的数据类型转换为字符串
        df.columns = df.columns.astype(str)
    elif file and file.filename.endswith('.xlsx'):
        df = pd.read_excel(file)  # 使用 Pandas 读取 Excel 文件
        # 将标头的数据类型转换为字符串
        df.columns = df.columns.astype(str)
    else:
        return 'File format not supported.'
    # 在这里进行数据处理和操作
    return 'File uploaded successfully!'


# 返回表格的表头
@app.route('/cluster/data-header', methods=['GET'])
def cluster_data_header():
    global df
    if df is None:
        return 'No data available.'  # 处理没有上传文件时的情况
    data = df.columns.to_list()
    # 将数据转换为字典格式
    return jsonify(data)


# 返回表格数据
@app.route('/cluster/data', methods=['GET'])
def cluster_data():
    global df
    if df is None:
        return 'No data available.'  # 处理没有上传文件时的情况
    data = df.to_dict(orient='records')  # 将数据转换为字典格式
    return jsonify(data)


# 上传聚类的属性列
@app.route('/cluster/upload-tag', methods=['POST'])
def cluster_upload_tag():
    global df, cluster_info
    cluster_info = [['blue', '等待提交...'], ['blue', '提交成功, 服务端正在处理...']]
    data = request.get_json()  # 获取发送的JSON数据
    # data = request.data  # 获取发送的JSON数据
    print(data)

    X = df.loc[:, data['tags']]
    # 在这里对数据进行处理
    # 返回处理结果
    # 对数据进行归一化处理
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    # 确定最优的 k 值
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)

    print(f"各个 k 值对应的轮廓系数为：{silhouette_scores}")
    cluster_info.append(['blue', f"各个 k 值对应的轮廓系数为：{silhouette_scores}"])
    sleep(2)
    best_k = np.argmax(silhouette_scores) + 2
    print(f"最优的 k 值为：{best_k}")
    cluster_info.append(['blue', f"轮廓系数得到的最优的 k 值为：{best_k}"])
    sleep(1)
    if data['customK'] != 0:
        best_k = data['customK']
        cluster_info.append(['blue', f"用户自定义的 k 值为：{best_k}"])

    # kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    kmeans.fit(X_scaled)

    # 可视化聚类结果
    X['cluster'] = kmeans.labels_
    cluster_info.append(['blue', f"正在生成聚类结果图..."])
    sns.pairplot(X, hue='cluster')
    plt.savefig('cluster_plot.png')  # 保存图表到文件
    cluster_info.append(['green', f"聚类完成"])
    sleep(1)

    return jsonify({'result': 'success', 'data': f"{best_k}"})


@app.route('/cluster/get-info', methods=['GET'])
def cluster_get_info():
    global cluster_info
    return jsonify({'info': cluster_info})


@app.route('/cluster/get-img', methods=['GET'])
def cluster_get_img():
    # 获取文件的路径
    file_path = os.path.join(app.root_path, 'cluster_plot.png')
    print("当前文件的路径为：", file_path)
    # 返回文件作为响应
    return send_file(file_path, mimetype='image/png')


@app.route('/cluster/get-icon', methods=['GET'])
def cluster_get_icon():
    img_param = request.args.get('img')
    # 获取文件的路径
    # 根据参数值确定要返回的图片路径
    if img_param == 'load_data':
        file_path = os.path.join(app.root_path, '数据导入.png')
    if img_param == 'cluster':
        file_path = os.path.join(app.root_path, '聚类.png')
    if img_param == 'dimension':
        file_path = os.path.join(app.root_path, '维度.png')
    if img_param == 'model':
        file_path = os.path.join(app.root_path, '模型.png')
    if img_param == 'model-training':
        file_path = os.path.join(app.root_path, '模型训练.png')
    print("当前文件的路径为：", file_path)
    # 返回文件作为响应
    return send_file(file_path, mimetype='image/png')


@app.route('/dimension/upload-tag', methods=['POST'])
def dimension_upload_tag():
    global df, dimension_info
    dimension_info = [['blue', '等待提交...'], ['blue', '提交成功, 服务端正在处理...']]
    data = request.get_json()  # 获取发送的JSON数据
    # data = request.data  # 获取发送的JSON数据
    print(data)
    print(data['tags'])
    Label = df.iloc[data['labelStartRow'] - 1:, data['labelStartColumn'] - 1]
    X = df.iloc[data['startRow'] - 1:, data['startColumn'] - 1:]

    # 是否显示混淆矩阵
    show_conf_mx = False

    def pca(ndim=100):
        # 使用PCA算法将高维数据降至指定的维度(ndim)，并保存在reduced_X中。
        pca_sk = PCA(n_components=ndim)
        reduced_X = pca_sk.fit_transform(X)
        return reduced_X

    def split_train_test(n, test_ratio):
        shuffled_indices = np.random.permutation(n)
        test_set_size = int(n * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return train_indices, test_indices

    # 执行一次朴素贝叶斯分类
    def execute_analysis(reduced_X):
        train_id, test_id = split_train_test(len(reduced_X), 0.3)

        # 将训练集和测试集X矩阵以及相应的标签y向量提取出来。
        X_train, X_test, y_train, y_test = reduced_X[train_id], reduced_X[test_id], Label[train_id + 1], Label[
            test_id + 1]

        # 使用GaussianNB高斯朴素贝叶斯分类器对训练集进行拟合
        md = GaussianNB()
        md.fit(X_train, y_train)

        # 用训练好的模型预测测试集，并计算分类结果的混淆矩阵并输出。
        yfit = md.predict(X_test)

        # 计算分类准确度（accuracy_score）并输出。最后一个print语句输出的是准确率的百分数。
        conf_mx = confusion_matrix(y_test, yfit)

        if show_conf_mx:
            plt.matshow(conf_mx, cmap=plt.cm.gray)
            plt.show()
            print(conf_mx)

        acc = accuracy_score(yfit, y_test)

        return acc * 100

    dims = data['tags']
    dims = [int(dim) for dim in dims]
    pca_results = [[execute_analysis(pca(ndim)) for i in range(10)] for ndim in dims]
    for i, ndim in enumerate(dims):
        mean_acc = sum(pca_results[i]) / len(pca_results[i])
        print(f"PCA降维维数为{ndim}时，10次朴素贝叶斯分类平均精度为{mean_acc:.1f}")
        dimension_info.append(['blue', f"PCA降维维数为{ndim}时，10次朴素贝叶斯分类平均精度为{mean_acc:.1f}"])

    dimension_info.append(['green', f"处理完成"])
    sleep(1)
    return jsonify({'result': 'success', 'data': "{}"})


@app.route('/dimension/get-info', methods=['GET'])
def dimension_get_info():
    global dimension_info
    return jsonify({'info': dimension_info})


@app.route('/ml/upload-path', methods=['POST'])
def ml_upload_path():
    global data_path, sub_dir_list
    data = request.get_json()  # 获取发送的JSON数据
    # data = request.data  # 获取发送的JSON数据
    data_dir_path = data['dirPath']
    # 返回子目录和子目录下的文件数量，存放在一个列表中
    data_path = data_dir_path
    sub_dir_list = []
    for sub_dir in os.listdir(data_dir_path):
        sub_dir_path = os.path.join(data_dir_path, sub_dir)
        if os.path.isdir(sub_dir_path):
            sub_dir_list.append({"label": sub_dir, "amount": len(os.listdir(sub_dir_path))})

    print(data)
    print(sub_dir_list)
    return jsonify({'status': 'success', 'data': sub_dir_list})


@app.route('/ml/load-dataset', methods=['POST'])
def ml_load_dataset():
    global train_test_ratio, data_path, sub_dir_list, train_loader, test_loader, train_set, test_set

    data = request.get_json()
    print(data)

    data['train_test_ratio'] = float(data['train_test_ratio'])

    # train_test_ratio = 0.7  # 划分训练集测试集比例
    train_test_ratio = data['train_test_ratio']  # 划分训练集测试集比例
    print(f"数据集路径：{data_path}, 子目录：{sub_dir_list}")
    # 同时打开两个文件写
    with open('train.txt', mode='w', encoding='utf8') as f:
        with open('test.txt', mode='w', encoding='utf8') as f1:
            for index, item in enumerate(sub_dir_list):
                # 进入每种树皮的文件夹
                item = item['label']
                tree_species = os.listdir(os.path.join(data_path, item))
                # 这种树皮有多少张照片
                count = len(tree_species)
                # 根据上面的 train_test_ratio 变量，取前 70% 作为训练集，后 30% 作为测试集
                train = tree_species[:int(count * train_test_ratio)]
                test = tree_species[int(count * train_test_ratio):]
                for picture in train:
                    f.write(os.path.join(data_path, item, picture))
                    f.write(",")
                    f.write(str(index))
                    f.write('\n')
                for picture in test:
                    f1.write(os.path.join(data_path, item, picture))
                    f1.write(",")
                    f1.write(str(index))
                    f1.write('\n')

    my_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = MyDataset(datatxt="train.txt", transform=my_transform)
    test_set = MyDataset(datatxt="test.txt", transform=my_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=data['batch_size'], shuffle=True)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=data['batch_size'], shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

    batch = next(iter(train_loader))
    # print(batch[0])
    return jsonify({'status': 'success', 'data': '加载数据集成功'})


@app.route('/ml/model-select', methods=['POST'])
def ml_model_select():
    global model, train_loader, test_loader
    # model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
    data = request.get_json()

    if data['model'] == 'ResNet-18':
        model = torchvision.models.resnet18(pretrained=True)
    if data['model'] == 'ResNet-50':
        model = torchvision.models.resnet50(pretrained=True)
    # 其他模型供选择
    # model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True)

    num_features = model.fc.in_features
    print(f"原连接层输入{num_features},输出{model.fc.out_features}")
    # 更改模型的 fc 层
    model.fc = nn.Linear(num_features, 5)

    # 对于模型的每个权重，使其不进行反向传播，即固定参数
    for param in model.parameters():
        param.requires_grad = False

    # 将其中最后的全连接部分的网路参数设置为可反向传播
    for param in model.fc.parameters():
        param.requires_grad = True

    return jsonify({'status': 'success', 'data': ''})


@app.route('/ml/model-train', methods=['POST'])
def ml_model_train():
    global model, ml_info
    ml_info = [['blue', '等待训练开始...'], ['blue', '提交成功, 正在开始训练...']]

    data = request.get_json()
    def get_num_correct(preds, labels):
        # 使用 argmax 获取每个样本的预测结果，使用 eq 比较预测结果和真实标签是否相同，最后使用 sum 函数统计出正确分类的数量
        return preds.argmax(dim=1).eq(labels).sum().item()

    # 定义 Adam 优化器（只更新需要更新的部分参数）
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=data['learningRate'])

    # 记录准确率的列表
    train_accuracy = []
    test_accuracy = []

    # for epoch in range(50):
    for epoch in range(data['epoch']):

        total_loss = 0  # 记录每轮训练的总损失
        train_correct = 0  # 记录每轮训练中正确分类的数量
        test_correct = 0  # 记录每轮测试中正确分类的数量

        # 开始训练
        model.train()  # 设置模型为训练模式，启用 dropout 和批标准化等操作
        for batch in train_loader:
            images, labels = batch[0], batch[1]
            preds = model(images)  # 前向传播，得到预测结果
            loss = F.cross_entropy(preds, labels)  # 计算损失函数

            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新参数

            total_loss += loss.item()  # 累加每个 batch 的损失函数值
            train_correct += get_num_correct(preds, labels)  # 累加每个 batch 中正确分类的数量

        # 开始测试
        model.eval()  # 设置模型为评估模式，关闭 dropout 和批标准化等操作
        with torch.no_grad():  # 关闭梯度计算，减少内存占用
            for batch in test_loader:
                images, labels = batch[0], batch[1]
                preds = model(images)  # 前向传播，得到预测结果
                test_correct += get_num_correct(preds, labels)  # 累加每个 batch 中正确分类的数量

        # 打印本轮训练和测试的结果
        print(
            f'epoch:{epoch} train_correct:{train_correct}/{len(train_set)} loss:{total_loss}  test_correct:{test_correct}/{len(test_set)}')

        ml_info.append(['blue', f'epoch:{epoch} train_correct:{train_correct}/{len(train_set)} loss:{total_loss}  test_correct:{test_correct}/{len(test_set)}'])
        # 记录本轮训练和测试的准确率
        train_accuracy.append(train_correct / len(train_set))
        test_accuracy.append(test_correct / len(test_set))

    ml_info.append(['green', '训练完成'])
    sleep(1)

    return jsonify({'status': 'success', 'data': ''})

@app.route('/ml/get-info', methods=['GET'])
def ml_get_info():
    global ml_info
    return jsonify({'info': ml_info})


if __name__ == '__main__':
    app.run()
