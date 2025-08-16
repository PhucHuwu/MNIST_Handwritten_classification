import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import ttk
from nn import Net


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model = Net()

    model_state = checkpoint['model_state']

    model.first_layer.weight.data = model_state['first_layer']['weight']
    model.first_layer.bias.data = model_state['first_layer']['bias']

    model.second_layer.weight.data = model_state['second_layer']['weight']
    model.second_layer.bias.data = model_state['second_layer']['bias']

    model.output_layer.weight.data = model_state['output_layer']['weight']
    model.output_layer.bias.data = model_state['output_layer']['bias']

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
    root.title("Draw and Predict Digits")

    canvas = tk.Canvas(root, width=280, height=280, bg='black')
    canvas.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

    drawing = False
    last_x = None
    last_y = None
    prediction_job = None

    def start_drawing(event):
        nonlocal drawing, last_x, last_y
        drawing = True
        last_x = event.x
        last_y = event.y

    def draw(event):
        nonlocal drawing, last_x, last_y, prediction_job
        if drawing:
            x = event.x
            y = event.y
            canvas.create_line(last_x, last_y, x, y, fill='white', width=20,
                               capstyle=tk.ROUND, smooth=tk.TRUE)
            last_x = x
            last_y = y

            if prediction_job:
                root.after_cancel(prediction_job)
            prediction_job = root.after(0, predict)

    def stop_drawing(event):
        nonlocal drawing
        drawing = False

    def clear_canvas():
        canvas.delete("all")
        result_label.config(text="Draw a digit to predict", font=('Arial', 18))

    def predict():
        if canvas.find_all():
            image = Image.new('L', (280, 280), color='black')
            draw_pil = ImageDraw.Draw(image)

            for item in canvas.find_all():
                coords = canvas.coords(item)
                if len(coords) == 4:
                    draw_pil.line(coords, fill='white', width=20)

            image = image.resize((28, 28))
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            img_tensor = transform(image).unsqueeze(0)

            model.eval()
            output = model.forward(img_tensor)
            predicted = torch.argmax(output, dim=1).item()
            confidence = torch.max(output, dim=1).values.item()

            result_label.config(text=f'{predicted} (Confidence: {confidence:.3f})',
                                font=('Arial', 18, 'bold'))
        else:
            result_label.config(text="Draw a digit to predict", font=('Arial', 18))

    canvas.bind('<Button-1>', start_drawing)
    canvas.bind('<B1-Motion>', draw)
    canvas.bind('<ButtonRelease-1>', stop_drawing)

    button_frame = tk.Frame(root)
    button_frame.grid(row=1, column=0, columnspan=2, pady=10)

    ttk.Button(button_frame, text="Clear", command=clear_canvas).pack(side=tk.LEFT, padx=5)

    result_label = ttk.Label(root, text="Draw a digit to predict", font=('Arial', 18))
    result_label.grid(row=2, column=0, columnspan=2, pady=10)

    return root


def test_model():
    model_path = 'model_checkpoints/best_model_epoch_27_acc_0.9759.pth'
    global model
    model = load_model(model_path)
    root = create_drawing_window()
    root.mainloop()


if __name__ == "__main__":
    test_model()
