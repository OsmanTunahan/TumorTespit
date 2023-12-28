import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TumorDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tümör Tespit Uygulaması")
        self.image_path = None
        self.model = None
        self.create_widgets()

    def create_widgets(self):
        self.select_button = tk.Button(self.root, text="Görüntü Seç", command=self.select_image)
        self.select_button.pack(pady=10)

        self.detect_button = tk.Button(self.root, text="Tümörü Tespit Et", command=self.detect_tumor)
        self.detect_button.pack(pady=10)

        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack(pady=10)

        self.probability_label = tk.Label(self.root, text="")
        self.probability_label.pack(pady=5)

        self.fig_frame = tk.Frame(self.root)
        self.fig_frame.pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Bir Görüntü Seçin")
        self.image_path = file_path
        self.show_image()

    def show_image(self):
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()

        if self.image_path:
            img = cv2.imread(self.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img)
            ax.axis('off')
            self.canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        else:
            self.result_label.config(text="Lütfen bir görüntü seçin.")
            self.probability_label.config(text="")

    def detect_tumor(self):
        if not self.model:
            try:
                self.model = load_model('brain_model.keras')
            except:
                self.result_label.config(text="Model bulunamadı.\nYeni bir model oluşturmak için: train.py dosyasını kullanın!")
                self.probability_label.config(text="")
                return

        if self.image_path:
            user_image = self.preprocess_image(self.image_path)
            prediction = self.predict_tumor(user_image)
            result, probability = self.evaluate_prediction(prediction)

            if prediction > 0.5:
                self.show_tumor_image(self.image_path)

            self.result_label.config(text=result, fg="red" if prediction > 0.5 else "green")
            self.probability_label.config(text=f"Tümör Olma Olasılığı: {probability[0]*100:.2f}%")
        else:
            self.result_label.config(text="Lütfen bir görüntü seçin.")
            self.probability_label.config(text="")

    def preprocess_image(self, image_path, target_size=(128, 128)):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, target_size)
        img = np.reshape(img, (1, target_size[0], target_size[1], 1))
        return img

    def predict_tumor(self, image):
        prediction = self.model.predict(image)
        return prediction

    def evaluate_prediction(self, prediction):
        if prediction > 0.5:
            return "Tümör tespit edildi.", prediction[0]
        else:
            return "Tümör tespit edilmedi.", prediction[0]

    def show_tumor_image(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result_img[thresh == 255, 0] = 0
        result_img[thresh == 255, 1] = 255
        result_img[thresh == 255, 2] = 0

        self.canvas.get_tk_widget().destroy()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(result_img)
        ax.axis('off')
        self.canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

if __name__ == "__main__":
    root = tk.Tk()
    app = TumorDetectionApp(root)
    plt.show()
    root.mainloop()
