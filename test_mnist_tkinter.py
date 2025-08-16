import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import torch.nn as nn
import tkinter as tk
from tkinter import ttk


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.first_layer = nn.Linear(784, 256)
        self.first_activation = nn.ReLU()
        self.second_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(128, 10),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.first_layer(x)
        x = self.first_activation(x)
        x = self.second_layer(x)
        x = self.output(x)
        return x


def load_model(model_path):
    checkpoint = torch.load(model_path)
    model = Net()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image)
    return image.unsqueeze(0)


def create_drawing_window():
    root = tk.Tk()
    root.title("Vẽ và dự đoán số")

    canvas = tk.Canvas(root, width=280, height=280, bg='black')
    canvas.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

    drawing = False
    last_x = None
    last_y = None

    def start_drawing(event):
        nonlocal drawing, last_x, last_y
        drawing = True
        last_x = event.x
        last_y = event.y

    def draw(event):
        nonlocal drawing, last_x, last_y
        if drawing:
            x = event.x
            y = event.y
            canvas.create_line(last_x, last_y, x, y, fill='white', width=20)
            last_x = x
            last_y = y
            root.after(100, predict)

    def stop_drawing(event):
        nonlocal drawing
        drawing = False

    def clear_canvas():
        canvas.delete("all")

    def predict():
        x = canvas.winfo_rootx() + canvas.winfo_x()
        y = canvas.winfo_rooty() + canvas.winfo_y()
        x1 = x + canvas.winfo_width()
        y1 = y + canvas.winfo_height()

        image = Image.new('L', (280, 280), color='black')
        draw = ImageDraw.Draw(image)

        for item in canvas.find_all():
            coords = canvas.coords(item)
            if len(coords) == 4:
                draw.line(coords, fill='white', width=20)

        image = image.resize((28, 28))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        img_tensor = transform(image).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            predicted = output.argmax(dim=1).item()
            result_label.config(text=f'Kết quả dự đoán: {predicted}', font=('Arial', 24, 'bold'))

    canvas.bind('<Button-1>', start_drawing)
    canvas.bind('<B1-Motion>', draw)
    canvas.bind('<ButtonRelease-1>', stop_drawing)

    ttk.Button(root, text="Clear", command=clear_canvas).grid(row=1, column=0, columnspan=2, pady=5)
    result_label = ttk.Label(root, text="Kết quả dự đoán:", font=('Arial', 24, 'bold'))
    result_label.grid(row=2, column=0, columnspan=2, pady=10)

    return root


def test_model():
    model_path = 'model_checkpoints/best_model_acc_0.9650.pth'
    global model
    model = load_model(model_path)
    root = create_drawing_window()
    root.mainloop()


if __name__ == "__main__":
    test_model()
