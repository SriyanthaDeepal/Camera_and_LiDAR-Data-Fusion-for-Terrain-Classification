import serial
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import tkinter as tk
from tkinter import ttk
import re
import os
from collections import deque
import threading
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Define Global variables
data_buffer = {'distance': deque(maxlen=30), 'strength': deque(maxlen=30), 'temperature': deque(maxlen=30)}
processed_data = None
svm_probabilities = None

# Weights for weighted averaging
cnn_weight = 0.4
svm_weight = 0.6

model_path = 'E:\\Files of UOSJ\\Fourth Year\\Seventh Semester\\Research\\RestNet50\\Latest model\\SVM_LiDAR.joblib'
scaler_path = 'E:\\Files of UOSJ\\Fourth Year\\Seventh Semester\\Research\\RestNet50\\Latest model\\SVMScaler.joblib'
feature_names_path = 'E:\\Files of UOSJ\\Fourth Year\\Seventh Semester\\Research\\RestNet50\\Latest model\\SVM_Feature_Names.joblib'
cnn_model_path = 'E:\\Files of UOSJ\\Fourth Year\\Seventh Semester\\Research\\RestNet50\\Latest model\\CNN_Camera.keras'
label_binarizer_path = 'E:\\Files of UOSJ\\Fourth Year\\Seventh Semester\\Research\\RestNet50\\Latest model\\Label_Binarizer.joblib'


# Load the pre-trained models, scaler, and feature names
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(feature_names_path)
cnn_model = load_model(cnn_model_path)
label_binarizer = joblib.load(label_binarizer_path)

# Determine if a timestamp is during the day or night
def day_or_night(timestamp):
    time = datetime.strptime(timestamp, '%I:%M:%S %p').time()
    if time >= datetime.strptime('6:00:00 AM', '%I:%M:%S %p').time() and time < datetime.strptime('5:00:00 PM', '%I:%M:%S %p').time():
        return 'day'
    else:
        return 'night'

# Process Lidar data
def process_data(distance, strength, temperature):
    data_buffer['distance'].append(distance)
    data_buffer['strength'].append(strength)
    data_buffer['temperature'].append(temperature)

    if len(data_buffer['distance']) == 30:
        
        moving_average_strength_30 = np.mean(data_buffer['strength'])
        wtc_strength_at_scale_5 = data_buffer['strength'][-5]  # Example feature (modify as needed)
        distance_x_roc_strength = np.mean(np.diff(data_buffer['distance'])) * np.mean(data_buffer['strength'])  # Example feature (modify as needed)
        
        timestamp = datetime.now().strftime('%I:%M:%S %p')
        day_night = day_or_night(timestamp)
        
        # Create a DataFrame for the single observation
        df_features = pd.DataFrame([[moving_average_strength_30, wtc_strength_at_scale_5, distance_x_roc_strength, day_night]],
                                   columns=['Moving Average Strength 30', 'WTC Strength at scale 5', 'Distance x ROC Strength', 'Day/Night'])
        
        # Convert categorical data to numerical format
        df_features = pd.get_dummies(df_features, drop_first=True)
        
        # Align the DataFrame with the training data format
        df_features = df_features.reindex(columns=feature_names, fill_value=0)
        
        # Scale the features
        scaled_features = scaler.transform(df_features)
        
        return scaled_features, moving_average_strength_30, wtc_strength_at_scale_5, distance_x_roc_strength
    else:
        return None, None, None, None

# Read data from the COM port
def read_from_com_port():
    global svm_probabilities
    ser = serial.Serial('COM14', 9600, timeout=1)
    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                # Process the incoming data line
                match = re.search(r'dist\s*=\s*(\d+)\s+strength\s*=\s*(\d+)\s+Chip Temprature\s*=\s*([\d.-]+)\s+celcius degree', line)
                if match:
                    distance = float(match.group(1))
                    strength = float(match.group(2))
                    temperature = float(match.group(3))

                    processed_data, moving_avg_strength, wtc_strength, distance_x_roc = process_data(distance, strength, temperature)
                    if processed_data is not None:
                        svm_probabilities = classify_data(processed_data)
                        update_plots(moving_avg_strength, wtc_strength, distance_x_roc)
                else:
                    print("Invalid data format:", line)
    except serial.SerialException as e:
        print(f"Serial port error: {e}")

# Classify and display results from SVM
def classify_data(processed_data):
    probabilities = model.predict_proba(processed_data)[0]
    return probabilities

# Classify images using CNN
def classify_image(frame):
    image = cv2.resize(frame, (64, 64))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0
    probabilities = cnn_model.predict(image)[0]
    return probabilities

# Start the COM port reading in a separate thread
def start_reading_thread():
    thread = threading.Thread(target=read_from_com_port)
    thread.daemon = True  # Daemonize the thread so it automatically closes when the main program exits
    thread.start()

# Update the plots
def update_plots(moving_avg_strength, wtc_strength, distance_x_roc):
    ma_values.append(moving_avg_strength)
    strength_values.append(wtc_strength)
    roc_values.append(distance_x_roc)

    if len(ma_values) > 100:
        ma_values.pop(0)
        strength_values.pop(0)
        roc_values.pop(0)

    line1.set_ydata(ma_values)
    line1.set_xdata(range(len(ma_values)))

    line2.set_ydata(strength_values)
    line2.set_xdata(range(len(strength_values)))

    line3.set_ydata(roc_values)
    line3.set_xdata(range(len(roc_values)))

    ax1.relim()
    ax1.autoscale_view()

    ax2.relim()
    ax2.autoscale_view()

    ax3.relim()
    ax3.autoscale_view()

# Process video frames from the stream
def process_video():
    global svm_probabilities
    cap = cv2.VideoCapture("http://192.168.43.108:81/stream")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            cnn_probabilities = classify_image(frame)
            if svm_probabilities is not None:
                combined_prob_predictions = cnn_weight * cnn_probabilities + svm_weight * svm_probabilities
                combined_class_idx = np.argmax(combined_prob_predictions)
                combined_class_label = label_binarizer.classes_[combined_class_idx].replace('_', ' ')
                root.after(0, result_label.config, {'text': f"Combined Prediction: {combined_class_label}"})
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (400, 300))
            img = cv2.imencode('.png', frame)[1].tobytes()
            photo = tk.PhotoImage(data=img)
            root.after(0, video_label.config, {'image': photo})
            root.after(0, setattr, video_label, 'image', photo)

# Create the GUI
root = tk.Tk()
root.title("Lidar and Image Classification")

# Apply a theme
style = ttk.Style()
style.theme_use('clam')

# Customize the style
style.configure('TFrame', background='#f0f0f0')
style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 25, 'bold'))
style.configure('TLabel.result.TLabel', font=('Helvetica', 20, 'bold'), foreground='blue')
style.configure('TButton', font=('Helvetica', 12), padding=6)

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0, sticky='nw')

title_label = ttk.Label(frame, text="Camera & LiDAR Data Fusion Classification")
title_label.grid(row=0, column=0, columnspan=2, pady=70, sticky='w')

result_label = ttk.Label(frame, text="Combined Prediction: ", style='result.TLabel')
result_label.grid(row=1, column=0, columnspan=2, pady=10, sticky='w')

video_label = ttk.Label(frame)
video_label.grid(row=4, column=0, columnspan=3, padx=100, pady=100, sticky='sw')

# Set up matplotlib figures
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), facecolor='#f0f0f0')

ax1.set_title('Moving Average Strength')
ax2.set_title('WTC Strength at scale')
ax3.set_title('Distance x ROC Strength')

ma_values = []
strength_values = []
roc_values = []

line1, = ax1.plot(ma_values, label='Moving Average Strength')
line2, = ax2.plot(strength_values, label='WTC Strength at scale')
line3, = ax3.plot(roc_values, label='Distance x ROC Strength')

plt.tight_layout()
ani = animation.FuncAnimation(fig, lambda i: None, interval=1000, cache_frame_data=False)

# Embed the plot into Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().grid(row=0, column=2, rowspan=8, padx=20, pady=20, sticky='ne')

# Start the COM port reading thread
start_reading_thread()

# Start the video processing thread
stop_event = threading.Event()
video_thread = threading.Thread(target=process_video)
video_thread.start()

root.protocol("WM_DELETE_WINDOW", lambda: (stop_event.set(), root.destroy()))

root.mainloop()