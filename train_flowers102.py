import numpy as np
import torch
import torch.nn as nn
import torchvision as tv

if __name__ == '__main__':
    batch_size = 4
    transform = tv.transforms.Compose([
        tv.transforms.Resize((64, 64)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = tv.datasets.Flowers102(root='./datasets', split="train", download=False, transform=transform)
    print("Training samples: ", len(train_dataset))
    
    val_dataset = tv.datasets.Flowers102(root='./datasets', split="val", download=False, transform=transform)
    print("Validation samples: ", len(val_dataset))
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size= batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # for xs, labels in train_dataloader:
    #     print(xs.shape)
    #     print(labels.shape)
    #     break
    
    model = tv.models.vgg11(pretrained=True)
    
    # 将最后一层全连接层替换为102分类的全连接层
    n_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(n_features, 102)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    n_epochs = 100
    for epoch in range(n_epochs):
        print(">> Epoch: ", epoch + 1)
        model.train()
        for xs, labels in train_dataloader:
            xs, labels = xs.to(device), labels.to(device)
            # print(xs.shape)
            # print(labels.shape)
            # break
            optimizer.zero_grad()
            outputs = model(xs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for xs, labels in val_dataloader:
                xs, labels = xs.to(device), labels.to(device)
                outputs = model(xs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print("Accuracy: ", correct / total)