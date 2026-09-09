from dataset import train_loader, test_loader
from model import CNN
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
learning_rate = 0.001
batch_size = 64
epochs=10

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # stochastic gradient desc
#schedular = ExponentialLR(optimizer, gamma=0.9) # learning rate schedular

def train_loop(dataloader, model, loss_f, optimizer):
    size = len(dataloader.dataset)

    model.train()
    for batch, (x,y) in enumerate(dataloader):
        x=x.to(device)
        y=y.to(device)
        pred=model(x)
        loss=loss_f(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch%100==0:
            loss, current = loss.item(), batch * batch_size + len(x)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    #schedular.step()

def test_loop(dataloader, model, loss_f):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss=0
    correct=0

    with torch.no_grad():
        for x,y in dataloader:
            x=x.to(device)
            y=y.to(device)
            pred = model(x)
            test_loss += loss_f(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"testing err : \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct*100, test_loss

accuracies = []
losses = []

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_func, optimizer)
    acc, loss = test_loop(test_loader, model, loss_func)
    accuracies.append(acc)
    losses.append(loss)

epochs_range = range(1, epochs+1)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracies)
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, losses)
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.show()

torch.save(model.state_dict(), 'model_weights.pth')