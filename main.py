import torch
from torch import optim
from torch.utils.data import DataLoader
from model import TranE
from data import TrainSet, TestSet

device = torch.device('cuda:0')
total_epochs = 1

k = 50
alpha = 1e-2
gamma = 2
norm = 2

def main():
    train_dataset = TrainSet()
    test_dataset = TestSet()
    test_dataset.convert_word_to_index(train_dataset.entity_to_index, train_dataset.relation_to_index, test_dataset.raw_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
    transe = TranE(train_dataset.entity_num, train_dataset.relation_num, device, dim=k, norm=norm, gamma=gamma).to(device)
    optimizer = optim.SGD(transe.parameters(), lr=alpha, momentum=0)
    for epoch in range(total_epochs):
        entity_norm = torch.norm(transe.entity_embedding.weight.data, dim=1, keepdim=True)
        transe.entity_embedding.weight.data = transe.entity_embedding.weight.data / entity_norm
        total_loss = 0
        for batch_idx, (pos, neg) in enumerate(train_loader):
            pos, neg = pos.to(device), neg.to(device)
            pos = torch.transpose(pos, 0, 1)
            pos_head, pos_relation, pos_tail = pos[0], pos[1], pos[2]
            neg = torch.transpose(neg, 0, 1)
            neg_head, neg_relation, neg_tail = neg[0], neg[1], neg[2]
            loss = transe(pos_head, pos_relation, pos_tail, neg_head, neg_relation, neg_tail)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("----------------------")
        print(f"Training : epoch {epoch}, loss = {total_loss/train_dataset.__len__()}")
        correct_test = 0
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            data = torch.transpose(data, 0, 1)
            correct_test += transe.tail_predict(data[0], data[1], data[2], k=10)
        print(f"Testing : epoch {epoch}, accuracy {correct_test/test_dataset.__len__()}")
        print("----------------------")
        
if __name__ == '__main__':
    main()
    