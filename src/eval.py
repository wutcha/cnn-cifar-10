from dataset import train_data, test_loader
from model import CNN
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

model.load_state_dict(torch.load('model_weights.pth'))

model.eval()

loss_func = nn.CrossEntropyLoss()

correct = 0
test_loss = 0

c = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for x,y in test_loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        all_predictions.append(pred.argmax(1))
        all_labels.append(y)

        test_loss += loss_func(pred, y).item()
        correct += (pred.argmax(1)==y).sum().item()

flatten_predictions = torch.cat(all_predictions).cpu().numpy()
flatten_labels = torch.cat(all_labels).cpu().numpy()
class_names = train_data.classes

test_loss /= len(test_loader)
accuracy = correct / len(test_loader.dataset)

print(f"test_loss: {test_loss} | accuracy: {accuracy}")

cm = confusion_matrix(flatten_predictions, flatten_labels)

display = ConfusionMatrixDisplay(cm, display_labels=class_names)
display.plot(cmap=plt.cm.Blues)
plt.show()