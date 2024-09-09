import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import tkinter as tk
from tkinter import Label, Button, Frame, Radiobutton, StringVar
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import threading
import gzip
import struct
from collections import deque
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for models and processing
model = None
model_type = None
processing = False
cap = None

# Variables to store corrected data for retraining
corrected_images = []
corrected_labels = []

# Initialize a deque to store bounding boxes for stability
bbox_buffer = deque(maxlen=10)
correction_mode = False

# Function to handle the change in model type
def on_model_type_change():
    global model_type
    model_type = model_var.get()
    logging.debug(f"Model type changed to: {model_type}")
    status_label.config(text=f"Model Type: {model_type}", fg="blue")

# Function to load the model in a separate thread to prevent freezing
def load_selected_model():
    threading.Thread(target=load_model_thread).start()
    stop_button.config(text="Stop", command=stop_processing, state=tk.NORMAL)

def load_model_thread():
    global model
    try:
        logging.debug("Opening file dialog to load model...")
        selected_file = filedialog.askopenfilename(title="Select Model File", filetypes=[("Keras Model Files", "*.keras")])
        if selected_file:
            logging.debug(f"Selected model file: {selected_file}")
            model = load_model(selected_file)
            logging.debug(f"Model loaded successfully: {selected_file}")
            status_label.config(text="Model loaded!", fg="green")
            start_button.config(state=tk.NORMAL)
            stop_button.config(state=tk.NORMAL)
            if correction_mode:
                show_correction_mode()

        else:
            logging.debug("No model selected.")
            status_label.config(text="No model selected.", fg="red")
            model = None
            start_button.config(state=tk.DISABLED)
            stop_button.config(state=tk.DISABLED)
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        status_label.config(text=f"Error loading model: {str(e)}", fg="red")
        model = None
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.DISABLED)

# Function to find the index card in the video frame
def find_index_card(frame):
    logging.debug("Finding index card in the frame...")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        logging.debug("Contours found in the frame.")
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 2.0 and w > 50 and h > 50:
                logging.debug(f"Index card found: x={x}, y={y}, w={w}, h={h}")
                return x, y, w, h, frame[y:y+h, x:x+w]
    logging.debug("No valid index card found in the frame.")
    return None, None, None, None, frame

# Function to preprocess the frame for prediction
def preprocess_frame(frame):
    logging.debug("Preprocessing frame for prediction...")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    if np.mean(normalized) > 0.5:
        normalized = 1.0 - normalized
    reshaped = normalized.reshape(1, 28, 28, 1)
    logging.debug(f"Preprocessed frame shape: {reshaped.shape}")
    return reshaped, resized

# Function to predict digit based on model type and update graph
def predict_digit(frame):
    global model
    logging.debug("Running prediction on the frame...")
    processed_frame, display_image = preprocess_frame(frame)

    if model_type == 'DNN/DBN':
        logging.debug("Reshaping frame for DNN/DBN model.")
        processed_frame = processed_frame.reshape(1, 784)

    prediction = model.predict(processed_frame)
    digit = np.argmax(prediction)

    logging.debug(f"Prediction: {prediction}")
    logging.debug(f"Predicted Digit: {digit}")
    
    update_graph(prediction[0])

    return digit, display_image

# Function to update the bar graph
def update_graph(prediction):
    ax.clear()
    bars = ax.bar(np.arange(10), prediction, color='blue')
    ax.set_ylim([0, 1])
    ax.set_xticks(np.arange(10))

    # Add the value labels above each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')

    canvas.draw()

# Function to toggle Correction Mode
def toggle_correction_mode():
    global correction_mode
    correction_mode = not correction_mode
    if correction_mode:
        logging.debug("Correction Mode activated.")
        show_correction_mode()
    else:
        logging.debug("Correction Mode deactivated.")
        hide_correction_mode()

# Show the correction mode interface
def show_correction_mode():
    correction_label.grid(row=6, column=0, pady=5, sticky='W')
    for i, rb in enumerate(digit_radio_buttons):
        rb.grid(row=7, column=i, padx=5, pady=5)
    confirm_correction_button.grid(row=8, column=0, columnspan=2, pady=10)

# Hide the correction mode interface
def hide_correction_mode():
    correction_label.grid_forget()
    for rb in digit_radio_buttons:
        rb.grid_forget()
    confirm_correction_button.grid_forget()

# Function to correct a misprediction
def correct_prediction():
    correct_digit = correct_digit_var.get()
    if digit_area is not None:
        corrected_images.append(cv2.cvtColor(digit_area, cv2.COLOR_BGR2GRAY))
        corrected_labels.append(int(correct_digit))
        logging.debug(f"User corrected prediction to: {correct_digit}")
        status_label.config(text=f"Corrected Digit: {correct_digit}", fg="blue")
    else:
        status_label.config(text="No digit area detected for correction.", fg="red")
    toggle_correction_mode()

# Update video feed and process frames
def update_frame():
    ret, frame = cap.read()
    if ret:
        display_frame = frame.copy()
        if processing:
            logging.debug("Processing is active. Running prediction...")
            x, y, w, h, card_frame = find_index_card(display_frame)
            if x is not None and card_frame.size > 0:
                margin = 0.15
                digit_x1 = max(0, x + int(margin * w))
                digit_y1 = max(0, y + int(margin * h))
                digit_x2 = min(display_frame.shape[1], x + w - int(margin * w))
                digit_y2 = min(display_frame.shape[0], y + h - int(margin * h))

                global digit_area
                digit_area = display_frame[digit_y1:digit_y2, digit_x1:digit_x2]
                digit, display_image = predict_digit(digit_area)
                if digit is not None:
                    digit_label.config(text=f"Predicted Digit: {digit}")

                # Display preprocessed image
                display_image_resized = cv2.resize(display_image, (100, 100), interpolation=cv2.INTER_NEAREST)
                display_image_rgb = cv2.cvtColor(display_image_resized, cv2.COLOR_GRAY2RGB)
                imgtk_display = ImageTk.PhotoImage(image=Image.fromarray(display_image_rgb))
                real_time_image_label.imgtk = imgtk_display
                real_time_image_label.configure(image=imgtk_display)

                # First bounding box around the entire card
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Second bounding box around the detected digit area
                cv2.rectangle(display_frame, (digit_x1, digit_y1), (digit_x2, digit_y2), (0, 255, 0), 2)

        # Convert the frame with bounding boxes to RGB (as Tkinter expects RGB images)
        cv2image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        video_frame.imgtk = imgtk
        video_frame.configure(image=imgtk)
        
    window.after(10, update_frame)

# Start the computer vision processing
def start_processing():
    global processing
    processing = True
    logging.debug("Computer vision started.")
    status_label.config(text="Computer vision started...", fg="green")
    stop_button.config(text="Stop", command=stop_processing, state=tk.NORMAL)

# Stop the computer vision processing
def stop_processing():
    global processing
    processing = False
    logging.debug("Computer vision stopped.")
    status_label.config(text="Computer vision stopped.", fg="red")
    stop_button.config(text="Exit", command=on_exit)

# Function to handle exiting the application
def on_exit():
    global cap
    logging.debug("Exiting the application...")
    cap.release()
    export_corrected_data()
    window.destroy()
    cv2.destroyAllWindows()

# Export data as two separate .gz files for images and labels
def export_corrected_data():
    desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
    images_export_dir = os.path.join(desktop_dir, "corrected_images.gz")
    labels_export_dir = os.path.join(desktop_dir, "corrected_labels.gz")

    # Export images
    with gzip.open(images_export_dir, 'wb') as f:
        f.write(struct.pack('>IIII', 0x00000803, len(corrected_images), 28, 28))
        for image in corrected_images:
            f.write(image.flatten().astype(np.uint8).tobytes())

    # Export labels
    with gzip.open(labels_export_dir, 'wb') as f:
        f.write(struct.pack('>II', 0x00000801, len(corrected_labels)))
        for label in corrected_labels:
            f.write(struct.pack('B', label))

    logging.debug("Corrected data exported to Desktop as corrected_images.gz and corrected_labels.gz.")
    status_label.config(text="Data exported to Desktop as .gz files!", fg="green")

# Function to get available cameras
def get_available_cameras(max_cameras=10):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

# Asynchronous frame update to prevent freezing
def start_update_thread():
    update_thread = threading.Thread(target=update_frame)
    update_thread.daemon = True
    update_thread.start()

# Initialize Tkinter window
window = tk.Tk()
window.title("Real-Time Digit Recognition")
window.geometry("1400x800")

# Create frames for side-by-side layout
main_frame = Frame(window)
main_frame.pack()

left_frame = Frame(main_frame)
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

right_frame = Frame(main_frame)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Title Label
title_label = Label(left_frame, text="Real-Time Digit Recognition", font=("Helvetica", 24))
title_label.pack(pady=20)

# Video Frame
video_frame = Label(left_frame, text="Real-Time Video Feed")
video_frame.pack()

# Predicted Digit Label
digit_label = Label(left_frame, text="Predicted Digit: ", font=("Helvetica", 24))
digit_label.pack(pady=20)

# Real-Time Image Display Label
real_time_image_label = Label(right_frame, text="Real-Time Preprocessed Image")
real_time_image_label.grid(row=0, column=0, padx=20, pady=20)

# Create a frame for buttons
button_frame = Frame(right_frame)
button_frame.grid(row=1, column=0, pady=10)

# Camera Selection (Radio Buttons removed based on user preference)
available_cameras = get_available_cameras()
selected_camera = StringVar(window)
selected_camera.set(available_cameras[0])

# Model Selection Radio Buttons
model_var = StringVar(window)
model_var.set("CNN")
cnn_radio = Radiobutton(button_frame, text="CNN", variable=model_var, value="CNN", command=on_model_type_change)
dnn_radio = Radiobutton(button_frame, text="DNN/DBN", variable=model_var, value="DNN/DBN", command=on_model_type_change)
cnn_radio.grid(row=0, column=0, padx=5)
dnn_radio.grid(row=0, column=1, padx=5)

# Load Model Button
load_button = Button(button_frame, text="Load Model", command=load_selected_model, font=("Helvetica", 16))
load_button.grid(row=1, column=0, columnspan=2, pady=10)

# Start Processing Button (Initially disabled)
start_button = Button(button_frame, text="Start", command=start_processing, font=("Helvetica", 16), state=tk.DISABLED)
start_button.grid(row=2, column=0, columnspan=2, pady=10)

# Stop Processing Button (Initially disabled)
stop_button = Button(button_frame, text="Stop", command=stop_processing, font=("Helvetica", 16), state=tk.DISABLED)
stop_button.grid(row=3, column=0, columnspan=2, pady=10)

# Status Label
status_label = Label(button_frame, text="Status: ", font=("Helvetica", 16))
status_label.grid(row=4, column=0, columnspan=2, pady=10)

# Correction Mode Toggle Button
toggle_correction_button = Button(button_frame, text="Correction Mode", command=toggle_correction_mode, font=("Helvetica", 16))
toggle_correction_button.grid(row=5, column=0, columnspan=2, pady=10)

# Correction Label and Radio Buttons (Initially hidden)
correct_digit_var = StringVar(window)
correct_digit_var.set("0")
correction_label = Label(button_frame, text="Correct Digit:")
digit_radio_buttons = [Radiobutton(button_frame, text=str(i), variable=correct_digit_var, value=str(i)) for i in range(10)]

# Confirm Correction Button
confirm_correction_button = Button(button_frame, text="Confirm Correction", command=correct_prediction, font=("Helvetica", 16))

# Export Corrections Button
export_button = Button(button_frame, text="Export Corrections", command=export_corrected_data, font=("Helvetica", 16))
export_button.grid(row=6, column=0, columnspan=2, pady=10)

# Create a figure for the bar graph and embed it in Tkinter
fig, ax = plt.subplots(figsize=(6, 5))
canvas = FigureCanvasTkAgg(fig, master=left_frame)
canvas.get_tk_widget().pack(pady=10)

# OpenCV video capture
cap = cv2.VideoCapture(int(selected_camera.get()))

# Start updating frames asynchronously
start_update_thread()

# Run the Tkinter event loop
window.mainloop()
