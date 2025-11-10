import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import ttk
import os
from nn import Net


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model = Net()

        model_state = checkpoint['model_state']

        model.first_layer.weight.data = model_state['first_layer']['weight']
        model.first_layer.bias.data = model_state['first_layer']['bias']

        model.second_layer.weight.data = model_state['second_layer']['weight']
        model.second_layer.bias.data = model_state['second_layer']['bias']

        model.output_layer.weight.data = model_state['output_layer']['weight']
        model.output_layer.bias.data = model_state['output_layer']['bias']

        print(f"Model loaded successfully!")
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")

        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")


def preprocess_drawing(canvas_image):
    image = canvas_image.resize((28, 28), Image.Resampling.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def predict_digit(model, image_tensor):

    with torch.no_grad():
        output = model.forward(image_tensor)

        probabilities = output[0]

        confidence, predicted = torch.max(probabilities, dim=0)

    return predicted.item(), confidence.item(), probabilities.numpy()


def create_drawing_window(model):
    root = tk.Tk()
    root.title("MNIST Digit Recognition - Draw and Predict")

    canvas = tk.Canvas(root, width=280, height=280, bg='black', cursor='crosshair')
    canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

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
            prediction_job = root.after(100, predict_from_canvas)

    def stop_drawing(event):
        nonlocal drawing
        drawing = False

    def clear_canvas():
        canvas.delete("all")
        result_label.config(text="Draw a digit (0-9)", font=('Arial', 16))
        prob_text.config(state='normal')
        prob_text.delete(1.0, tk.END)
        prob_text.config(state='disabled')

    def predict_from_canvas():
        if canvas.find_all():
            try:
                image = Image.new('L', (280, 280), color='black')
                draw_pil = ImageDraw.Draw(image)

                for item in canvas.find_all():
                    coords = canvas.coords(item)
                    if len(coords) == 4:
                        draw_pil.line(coords, fill='white', width=20)

                img_tensor = preprocess_drawing(image)

                predicted, confidence, probabilities = predict_digit(model, img_tensor)

                result_text = f'Predicted: {predicted}\nConfidence: {confidence*100:.2f}%'
                result_label.config(text=result_text, font=('Arial', 16, 'bold'))

                prob_text.config(state='normal')
                prob_text.delete(1.0, tk.END)
                prob_text.insert(tk.END, "Class Probabilities:\n" + "-"*25 + "\n")
                for i, prob in enumerate(probabilities):
                    marker = "← " if i == predicted else "  "
                    prob_text.insert(tk.END, f"{marker}Digit {i}: {prob*100:5.2f}%\n")
                prob_text.config(state='disabled')

            except Exception as e:
                result_label.config(text=f"Error: {str(e)}", font=('Arial', 12))
        else:
            result_label.config(text="Draw a digit (0-9)", font=('Arial', 16))

    canvas.bind('<Button-1>', start_drawing)
    canvas.bind('<B1-Motion>', draw)
    canvas.bind('<ButtonRelease-1>', stop_drawing)

    button_frame = tk.Frame(root)
    button_frame.grid(row=1, column=0, columnspan=2, pady=5)

    clear_btn = ttk.Button(button_frame, text="Clear Canvas", command=clear_canvas)
    clear_btn.pack(side=tk.LEFT, padx=5)

    predict_btn = ttk.Button(button_frame, text="Predict Now", command=predict_from_canvas)
    predict_btn.pack(side=tk.LEFT, padx=5)

    result_label = ttk.Label(root, text="Draw a digit (0-9)", font=('Arial', 16))
    result_label.grid(row=2, column=0, columnspan=2, pady=10)

    prob_frame = tk.Frame(root)
    prob_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

    prob_label = ttk.Label(prob_frame, text="Probabilities:", font=('Arial', 12, 'bold'))
    prob_label.pack()

    prob_text = tk.Text(prob_frame, height=12, width=25, font=('Courier', 10))
    prob_text.pack()
    prob_text.config(state='disabled')

    instructions = ("Instructions:\n"
                    "• Draw a digit on the black canvas\n"
                    "• Prediction updates automatically\n"
                    "• Click 'Clear' to start over")
    instr_label = ttk.Label(root, text=instructions, font=('Arial', 9), justify=tk.LEFT)
    instr_label.grid(row=4, column=0, columnspan=2, pady=5)

    return root


def test_model():
    model_path = 'model_checkpoints/best_model_epoch_18_valacc_0.9653.pth'

    try:
        print("="*60)
        print("MNIST Handwritten Digit Recognition - Test Mode")
        print("="*60)

        print(f"\nLoading model from: {model_path}")
        model = load_model(model_path)

        print("\nModel loaded successfully!")
        print("Starting interactive drawing interface...")
        print("="*60)

        root = create_drawing_window(model)
        root.mainloop()

    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("\nAvailable model checkpoints:")
        checkpoint_dir = 'model_checkpoints'
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            for ckpt in sorted(checkpoints):
                print(f"  - {ckpt}")
        else:
            print("  No checkpoints found!")
        print("\nPlease train the model first or check the model path.")

    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model()
