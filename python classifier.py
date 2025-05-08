import torch 
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
y = torch.tensor([0, 0, 1, 1])

model = nn.Linear(2, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch{epoch + 1}, Loss: {loss.item():.4f}')
    
with torch.no_grad():
    outputs = model(X)
    _, predicted = torch.max(outputs, 1)
    print(f'Predicted: {predicted.tolist()}, Actual: {y.tolist()}')