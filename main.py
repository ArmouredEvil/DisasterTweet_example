import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.utils.data
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
import MyRNN

# 选取设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')
train_df.info()
test_df.info()
# 合并数据
all_data = pd.concat([train_df.drop(columns='target'), test_df])
all_data.info()

'''
数据预处理
'''
# 填充缺失值
all_data['keyword'] = all_data['keyword'].fillna('<unk>')
all_data['location'] = all_data['location'].fillna('<unk>')
# 修改keyword列的数据格式
all_data['keyword'] = all_data['keyword'].apply(lambda x: 'keyword:' + x + '  ')
# 修改location列的数据格式
# all_data['location'] = all_data['location'].apply(lambda x: 'location:' + x + '  ')
# 修改text列的数据格式
all_data['text'] = all_data['text'].apply(lambda x: 'text:' + x)
# 提取新的文本
all_data['new_text'] = all_data['keyword'] + all_data['text']

# 提取训练和测试文本
train_text = all_data[:train_df.shape[0]]['new_text'].tolist()
test_text = all_data[train_df.shape[0]:]['new_text'].tolist()
# 提取标签
train_label = train_df['target'].tolist()

# 定义分词方法
tokenizer = get_tokenizer('basic_english')


# 定义词向量生成函数
def yield_tokens(texts):
    for yield_text in texts:
        yield tokenizer(yield_text)


# 构建词汇表
vocab = build_vocab_from_iterator(yield_tokens(train_text), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


# 定义文本处理函数。先进行分词，然后返回分词结果在词汇表的索引
def text_pipeline(text):
    return vocab(tokenizer(text))


# 定义标签处理函数
def label_pipeline(label):
    return int(label)


# 创建数据集
train_dataset = to_map_style_dataset(zip(train_text, train_label))


# 定义训练数据的collate函数, 用于定义处理一个batch内的数据
def collate_train(batch):
    text_list, label_list, length_list = [], [], []
    for (collate_text, collate_label) in batch:
        processed_text = torch.tensor(text_pipeline(collate_text), dtype=torch.int64)
        text_list.append(processed_text)
        label_list.append(label_pipeline(collate_label))
        length_list.append(len(processed_text))
    text_list = torch.nn.utils.rnn.pad_sequence(text_list, padding_value=0, batch_first=True)
    length_list = torch.tensor(length_list, dtype=torch.int64)
    return text_list, torch.tensor(label_list, dtype=torch.int64), length_list


# 创建数据迭代器
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)


'''
训练模型
'''
# 定义模型
input_dim = len(vocab)
hidden_dim = 512
output_dim = 2
embedding_dim = 100
model = MyRNN.MyRNN(input_dim, embedding_dim, hidden_dim, output_dim).to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

# 训练模型
torch.cuda.empty_cache()
for epoch in range(20):
    epoch_loss = 0.0
    for _text, _label, _length in train_loader:
        _text, _label, _length = _text.to(device), _label.to(device), _length.to(device)
        # 确保没有nan和inf
        assert torch.isnan(_text).sum() == 0
        # 前向传播
        output = model(_text, _length)
        # 计算损失函数
        loss = criterion(output, _label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 更新参数
        optimizer.step()
        # 累计损失
        epoch_loss += loss.item()
    # 输出每个epoch的损失
    print(f'Epoch {epoch + 1} loss: {epoch_loss / len(train_dataset)}')


'''
测试模型
'''
# 创建测试数据集
test_dataset = to_map_style_dataset(test_text)


# 定义测试集专用的collate函数
def collate_test(batch):
    text_list, length_list = [], []
    for collate_text in batch:
        processed_text = torch.tensor(text_pipeline(collate_text), dtype=torch.int64)
        text_list.append(processed_text)
        length_list.append(len(processed_text))
    text_list = torch.nn.utils.rnn.pad_sequence(text_list, padding_value=0, batch_first=True)
    length_list = torch.tensor(length_list, dtype=torch.int64)
    return text_list, length_list


# 创建测试数据迭代器
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_test)

# 测试模型
predict_list = []
with torch.no_grad():
    for _text, _length in test_loader:
        _text, _length = _text.to(device), _length.to(device)
        # 前向传播
        output = model(_text, _length)
        # 添加预测结果
        predict_list.extend(torch.argmax(output, dim=1).tolist())

# 转换为Series类型
predict_series = pd.Series(predict_list, name='target')
# 和测试数据的id合并
submission = pd.concat([test_df['id'], predict_series], axis=1)
# 保存csv
submission.to_csv('submission.csv', index=False)

