# -*- encoding:utf-8 -*-
import os

import numpy as np
import torch
import math
from sys import argv
import json
import random
import argparse
import torch.nn as nn
import pickle
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
from optimizers import BertAdam


# 定义分类器
class ChatglmClassifier(nn.Module):
    # 定义构造函数
    def __init__(self, args):  # 定义类的初始化函数，用户传入的参数
        super(ChatglmClassifier, self).__init__()  # 调用父类nn.module的初始化方法，初始化必要的变量和参数
        emb_size=4096
        dropout = args.dropout
        lstm_hidden_size = 200  # lstm_hidden_size的规格
        self.multi_head_attention1 = torch.nn.MultiheadAttention(emb_size, 8 ,dropout=dropout)
        self.multi_head_attention2 = torch.nn.MultiheadAttention(emb_size, 8 ,dropout=dropout)
        self.multi_head_attention3 = torch.nn.MultiheadAttention(emb_size, 8 ,dropout=dropout)
        # self.multi_head_attention4 = torch.nn.MultiheadAttention(emb_size, 8 ,dropout=dropout)
        self.multi_head_attention5 = torch.nn.MultiheadAttention(emb_size, 8 ,dropout=dropout)
        self.sigmoid = nn.Sigmoid()
        self.batchsize=args.batch_size
        # 双向lstm
        bidir = args.high_encoder == 'bi-lstm'
        self.direc = 2 if bidir else 1
        # 将LSTM的各项参数赋给rnn
        self.rnn = nn.LSTM(input_size=4096,  # 输入size=768
                           hidden_size=lstm_hidden_size,  # 赋值hidden_size
                           num_layers=2,  # num_layers=2
                           batch_first=True,  # 布尔赋值
                           bidirectional=bidir
                           )

        self.labels_num = 3
        self.pooling = args.pooling
        self.high_encoder = args.high_encoder

        self.head_num = 1
        if args.pooling == 'attention':
            # 线性回归
            self.attn_weight = nn.Linear(lstm_hidden_size * self.direc, 1)  # 输入与输出维度
        elif args.pooling == 'multi-head':  # else args.pooling取值
            self.emo_vec = self.load_emo_vec(args.emo_vec_path)  # 加载
            self.head_num = self.emo_vec.shape[0]
            self.bilinear = nn.Linear(lstm_hidden_size * 2, self.emo_vec.shape[-1], bias=False)
            self.emo_weight = nn.Linear(self.emo_vec.shape[-1], self.head_num, bias=False)  # 定义权重，无偏置
            self.emo_weight.weight.data = self.emo_vec
            self.attn_weight = nn.Sequential(
                self.bilinear,
                self.emo_weight,
            )
        self.transformer_enconder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=8,dropout=dropout)
        self.transformer_enconder = nn.TransformerEncoder(self.transformer_enconder_layer, num_layers=6)

        self.linear_c = nn.Linear(in_features=emb_size, out_features=emb_size)
        self.linear_r = nn.Linear(in_features=emb_size, out_features=emb_size)
        self.linear_s = nn.Linear(in_features=emb_size, out_features=emb_size)
        # self.linear_e = nn.Linear(in_features=emb_size, out_features=emb_size)
        self.linear_w = nn.Linear(in_features=emb_size, out_features=emb_size)
        self.linear_i = nn.Linear(in_features=emb_size, out_features=emb_size)
        # self.W_e = torch.nn.Linear(emb_size, lstm_hidden_size)
        self.W_s = torch.nn.Linear(emb_size, lstm_hidden_size)
        self.W_c = torch.nn.Linear(emb_size, lstm_hidden_size)
        self.W_r = torch.nn.Linear(emb_size, lstm_hidden_size)
        self.W_t = torch.nn.Linear(emb_size, lstm_hidden_size)
        self.output_layer_1 = nn.Linear(emb_size, lstm_hidden_size)
        self.output_layer_2 = nn.Linear(lstm_hidden_size, self.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)  # softmax维度
        self.criterion = nn.NLLLoss()  # 损失函数（NLLLoss 函数输入 input 之前，需要对 input 进行 log_softmax 处理）

        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(emb_size))
        self.fr=nn.Linear(256, emb_size)
        self.ft=nn.Linear(256, emb_size)


    def att(self, x, d):
        # x=self.fc(x)
        M = d * self.tanh1(x)  #
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = x * alpha
        out = torch.sum(out, 1)
        return out

    # 定义attention函数
    def attention(self, H):
        # mask (batch_size, seq_length)
        # mask = (mask > 0).unsqueeze(-1).repeat(1, 1, self.head_num)
        # mask = mask.float()
        # mask = (1.0 - mask) * -10000.0
        scores = self.attn_weight(H)  # 分数
        hidden_size = H.size(-1)
        scores /= math.sqrt(float(hidden_size))
        # scores += mask
        probs = nn.Softmax(dim=-2)(scores)
        H = H.transpose(-1, -2)
        output = torch.bmm(H, probs)
        output = torch.reshape(output, (-1, hidden_size * self.head_num))
        return output

    def load_emo_vec(self, path):  # 定义load_emo_vec（）函数
        with open(path, 'r', encoding='utf-8') as f:
            emo_vec = json.load(f)  # 打开文件
            return torch.tensor(list(emo_vec.values())[:3])  # 返回张量 取值前三列

    def orthogonal_loss(self, input):  # 定义orthogonal_loss（）函数
        norm_query = input / torch.norm(input, dim=-1, keepdim=True)
        dot_res = torch.matmul(norm_query, norm_query.t())
        dot_res = torch.abs(dot_res)
        reg = torch.sum(dot_res) - torch.trace(dot_res)
        return reg

    def forward(self, inputs, label, user_c, user_r, user_s, text_word):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size] label
            mask: [batch_size x seq_length]
        """
        inputs = rearrange(inputs, 'B S E->S B E')
        # inputs= self.transformer_enconder(inputs)
        user_r = self.sigmoid(self.fr(user_r))
        text_word= self.sigmoid(self.ft(text_word))
        text_word= text_word.transpose(0, 1)
        att_w = self.multi_head_attention5(text_word, inputs, inputs)[0]
        z_att_w = self.sigmoid(self.linear_w(att_w))
        inputs = att_w * self.sigmoid(z_att_w)
        user_c = user_c.transpose(0, 1)
        user_r = user_r.transpose(0, 1)
        user_s = user_s.transpose(0, 1)
        # explain = explain.transpose(0, 1)
        att_c = self.multi_head_attention1(user_c, inputs, inputs)[0]
        att_s = self.multi_head_attention2(user_s, inputs, inputs)[0]
        att_r = self.multi_head_attention3(user_r, inputs, inputs)[0]
        # att_e = self.multi_head_attention4(explain,inputs, inputs)[0]
        z_att_c = self.sigmoid(self.linear_c(att_c))
        z_att_r = self.sigmoid(self.linear_r(att_r))
        z_att_s = self.sigmoid(self.linear_s(att_s))
        # z_att_e = self.sigmoid(self.linear_e(att_e))
        user_c = att_c * z_att_c
        user_r = att_r * z_att_r
        user_s = att_s * z_att_s
        # explain= att_e * z_att_e
        user_c = torch.mean(user_c.transpose(0, 1), dim=1)
        inputs = torch.mean(inputs.transpose(0, 1), dim=1)
        user_s = torch.mean(user_s.transpose(0, 1), dim=1)
        user_r = torch.mean(user_r.transpose(0, 1), dim=1)
        # explain= torch.mean(explain.transpose(0,1), dim=1)
        # mpoa_input = self.sigmoid(torch.cat((self.output_layer_1(inputs),self.W_c(user_c)),1))
        # mpoa_input = self.sigmoid(torch.cat((self.output_layer_1(inputs),self.W_c(user_c),self.W_r(user_r),self.W_s(user_s),self.W_e(explain)),1))
        mpoa_input = self.sigmoid(self.output_layer_1(inputs)+self.W_c(user_c)+self.W_r(user_r)+self.W_s(user_s))
        logits = self.output_layer_2(mpoa_input)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))  # 损失函数
        if self.pooling == "multi-head":  # 多头机制
            loss = 0.9 * loss + 0.1 * self.orthogonal_loss(self.emo_weight.weight)
        return loss, logits


# 定义主函数
def main():
    def set_seed(seed=7):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def read_dataset(path):
        dataset = []
        with open(path, mode="r", encoding="utf-8") as f:  # 打开文件
            for line_id, line in enumerate(f):
                line = line.strip('\n').split('\t')  # 按 Tab键分割文本
                dataset.append(line)
        print("dataset:", len(dataset))
        return dataset

    def batch_loader(data, data_type):
        def get_glm_emb(text, max_len):
            tokens = tokenizer([text], return_tensors="pt")["input_ids"].tolist()[0]
            if len(tokens) > max_len:
                tokens = tokens[:max_len - 2] + tokens[-2:]
            while len(tokens) < max_len:
                tokens.append(3)
            emb = Chatglm.transformer(**{"input_ids": torch.tensor(tokens).unsqueeze(0).to(device)},
                                      output_hidden_states=True).last_hidden_state
            return emb.cpu().detach().squeeze()

        if not os.path.exists(args.dataset + "_/" + data_type+"/batch_0.pkl"):
            tokenizer = AutoTokenizer.from_pretrained("/home/stu1/PythonProjects/ChatGLM/chatglm-6b",
                                                      trust_remote_code=True)
            Chatglm = AutoModel.from_pretrained("/home/stu1/PythonProjects/ChatGLM/chatglm-6b",
                                                trust_remote_code=True).half().to(device)
            Chatglm = Chatglm.eval()
            dataset = []
            random.shuffle(data)
            batch_id = 0
            for line_id, line in enumerate(data):
                if len(dataset) == batch_size or line_id == len(data)-1:
                    batch_file = args.dataset + "_/" + data_type+"/batch_"+str(batch_id)+'.pkl'
                    batch_inputs = torch.tensor([example[0].numpy() for example in dataset])
                    batch_label = torch.LongTensor([example[1] for example in dataset])
                    batch_user_c = torch.tensor([example[2].numpy() for example in dataset])
                    batch_user_r = torch.tensor([example[3].numpy() for example in dataset])
                    batch_user_s = torch.tensor([example[4].numpy() for example in dataset])
                    # batch_explain = torch.tensor([example[5].numpy() for example in dataset])
                    batch_text_word = torch.tensor([example[6].numpy() for example in dataset])
                    with open(batch_file, "wb") as f:
                        pickle.dump([batch_inputs,batch_label,batch_user_c,batch_user_r,batch_user_s,batch_text_word], f)
                    dataset=[]
                    batch_id += 1
                label = int(line[columns["label"]])  # 赋值label
                text = line[columns["text"]]  # 40左右
                sex = line[columns["sex"]]
                address = line[columns["address"]]
                flag = line[columns["flag"]]
                document = line[columns["document"]]  # 100左右
                user_id = int(line[columns["user_id"]])
                user_r1 = user_r1_emb[user_id]
                user_r2 = user_r2_emb[user_id]
                user_r3 = user_r3_emb[user_id]
                user_r = torch.stack([user_r1, user_r2, user_r3])
                # explain = line[columns["explain"]]  # 300左右
                text_words = line[columns["text_word"]]
                if text_words != '':
                    text_words_id = [int(_) for _ in text_words.split(' ')][:32]  # 800多
                else:
                    text_words_id = []
                user_s = sex + "," + address + "," + flag
                inputs = get_glm_emb(text, 32)
                user_s = get_glm_emb(user_s, 32)
                user_c = get_glm_emb(document, 64)
                # explain = get_glm_emb(explain, 256)
                text_word = torch.zeros(32, 256)
                for i, word_id in enumerate(text_words_id):
                    text_word[i] = kg_emb[word_id]
                dataset.append((inputs, label, user_c, user_r, user_s, text_word))
        for i in range(math.ceil(len(data)/batch_size)):
            with open(args.dataset + "_/" + data_type + "/batch_" + str(i) + ".pkl", "rb") as f:
                inputs, label, user_c, user_r, user_s, text_word = pickle.load(f)
                inputs=inputs.type(torch.float32)
                user_c=user_c.type(torch.float32)
                user_s=user_s.type(torch.float32)
                # explain=explain.type(torch.float32)
            yield inputs.to(device), label.to(device), user_c.to(device), user_r.to(device), user_s.to(device), text_word.to(device)

    def evaluate(args, is_test):
        if is_test:
            dataset = read_dataset(args.dataset+"/test.txt")[1:]
            data_type = "test"
        else:
            dataset = read_dataset(args.dataset+"/dev.txt")[1:]
            data_type = "test"

        instances_num = len(dataset)
        if is_test:
            logger("The number of evaluation instances: ", instances_num)

        correct = 0
        confusion = torch.zeros(3, 3, dtype=torch.long)

        model.eval()
        pred_all=[]
        gold_all=[]
        for i, (inputs_batch, label_batch, user_c_batch, user_r_batch, user_s_batch, text_word_bitch)\
                in enumerate(batch_loader(dataset, data_type)):  # 循环

            with torch.no_grad():  # 进行计算图的构建
                loss, logits = model(inputs_batch, label_batch, user_c_batch, user_r_batch,user_s_batch, text_word_bitch)
            logits = nn.Softmax(dim=1)(logits)
            pred = torch.argmax(logits, dim=1)
            gold = label_batch
            pred_all.extend(pred.cpu().numpy().tolist())
            gold_all.extend(gold.cpu().numpy().tolist())
            correct += torch.sum(pred == gold).item()

        logger(classification_report(gold_all, pred_all, digits=3))

    def save_model(model, model_path):
        if hasattr(model, "module"):
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)

    def load_model(model, model_path):
        if hasattr(model, "module"):
            model.module.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        return model

    root = 'logger/'  # 日志文件
    file_name = root + 'logger.txt'
    log_file = open(file_name, 'a', encoding='utf-8')

    def logger(*args):
        str_list = " ".join([str(arg) for arg in args])
        print(str_list)
        log_file.write(str_list + '\n')
        log_file.flush()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 命令行参数解析包
    parser.add_argument("--output_model_path", default="./other/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--dataset", default="data", type=str,
                        help="Path of the trainset.")
    parser.add_argument("--emo_vec_path", type=str, default="other/emo_vector.json")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--high_encoder", choices=["bi-lstm", "lstm", "none"], default="bi-lstm")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last", "attention", "multi-head"],
                        default="mean", help="Pooling type.")
    parser.add_argument("--learning_rate", type=float, default=2e-6,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=15,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to logger prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")
    args = parser.parse_args()
    logger(args)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    model = ChatglmClassifier(args)
    model = model.to(device)

    # Training phase.
    logger("Start training.")
    file1 = open("other/user1.pkl", "rb")
    user_r1_emb = pickle.load(file1).cpu().detach()
    file2 = open("other/user2.pkl", "rb")
    user_r2_emb = pickle.load(file2).cpu().detach()
    file3 = open("other/user3.pkl", "rb")
    user_r3_emb = pickle.load(file3).cpu().detach()
    file1.close()
    file2.close()
    file3.close()
    kg_file = open("other/kg_emb.pkl", "rb")
    kg_emb = pickle.load(kg_file).cpu()
    kg_file.close()

    dataset = read_dataset(args.dataset+"/train.txt")
    columns = dict(zip(dataset[0], range(len(dataset[0]))))
    instances_num = len(dataset)-1
    batch_size = args.batch_size

    logger("Batch size: ", batch_size)
    logger("The number of training instances:", instances_num)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    train_steps = int(instances_num * args.epochs_num / batch_size) + 1
    # optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)
    total_loss = 0.
    best_result = 0.0
    up_epoch = 0

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        data_type = "train"
        for i, (inputs_batch, label_batch, user_c_batch, user_r_batch, user_s_batch, text_word_bitch) in enumerate(
                batch_loader(dataset[1:], data_type)):  # 循环
            model.zero_grad()
            loss, logits = model(inputs_batch, label_batch, user_c_batch, user_r_batch, user_s_batch, text_word_bitch)
            # print("loss",loss)
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                logger("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,total_loss / args.report_steps))
                total_loss = 0.

            loss.backward()
            optimizer.step()
        evaluate(args, False)
        # if result > best_result:
        #     best_result = result
        #     save_model(model, args.output_model_path)
        #     up_epoch = epoch
        # else:
        #     if epoch - up_epoch >= 5:
        #         break
        #     continue

    # Evaluation phase.
    # logger("Test set evaluation.")
    # model = load_model(model, args.output_model_path)
    # evaluate(args, True)
    log_file.close()


if __name__ == "__main__":
    main()
