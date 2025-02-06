import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms



# Define a transformation for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset without downloading
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Input layer
        self.relu = nn.ReLU()            # Activation function
        self.fc2 = nn.Linear(128, 10)    # Output layer

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
def train_model(model, train_loader, test_loader, num_epochs=5):
    # Move model to available device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for images, labels in train_loader:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass: compute predictions
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss for monitoring
            running_loss += loss.item()

        # Print epoch loss
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    return model

def predict_digit(model, image_tensor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Example usage:
if __name__ == '__main__':
    model = SimpleNN()
    model = train_model(model, train_loader, test_loader)
    
    # Save the trained model
    torch.save(model.state_dict(), 'mnist_model.pth')

