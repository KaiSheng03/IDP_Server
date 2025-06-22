from flask import Flask, request, jsonify, send_file
import mysql.connector
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import threading
import cv2
from ultralytics import YOLO
import io
import os
from inference_sdk import InferenceHTTPClient
import base64
import torch
import timm
from torchvision import transforms
import bcrypt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# MySQL Database Configuration
db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='159357',
    database='idp'
)
cursor = db.cursor(dictionary=True)

# TensorFlow Lite Model Loading
model_lock = threading.Lock()  # Lock for thread safety

yolo_model = YOLO("weights (11).pt")  # Adjust path as needed
# Load TensorFlow Lite model and interpreter
# segmentor = tf.lite.Interpreter(model_path="unet_model.tflite")
# classifier = tf.lite.Interpreter(model_path="new_skin_lesion_classifier_ver1.tflite")

# segmentor.allocate_tensors()
# classifier.allocate_tensors()

# # Get input and output details for inference
# segmentor_input_details = segmentor.get_input_details()
# segmentor_output_details = segmentor.get_output_details()

# classifier_input_details = classifier.get_input_details()
# classifier_output_details = classifier.get_output_details()

# Initialize the Inference HTTP Client (Roboflow)
# CLIENT = InferenceHTTPClient(
#     api_url="https://classify.roboflow.com",
#     api_key="R5CuqaiclawRdypI0g2W"
# )

# Pytorch ViT model
vit_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1)
vit_model.load_state_dict(torch.load('best_base_model (5).pth', map_location='cpu'))  # or 'cuda' if using GPU
vit_model.eval()

# Function to preprocess image for UNET/ResNet
def preprocess_image(image, target_size):
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

# def run_unet(image):
    input_data = preprocess_image(image, (224, 224))  
    segmentor.set_tensor(segmentor_input_details[0]['index'], input_data)
    segmentor.invoke()
    output_data = segmentor.get_tensor(segmentor_output_details[0]['index'])
    mask = (output_data[0] > 0.5).astype(np.uint8) * 255  # Binary mask
    return mask

# Function to run ResNet classification
# def classify_image(image):
    input_data = preprocess_image(image, (224, 224))  # Assuming ResNet uses 224x224
    classifier.set_tensor(classifier_input_details[0]['index'], input_data)
    classifier.invoke()
    output_data = classifier.get_tensor(classifier_output_details[0]['index'])
    result = output_data.tolist()
    return result
    class_idx = np.argmax(output_data)
    confidence = np.max(output_data)
    label = "Malignant" if class_idx == 1 else "Benign"
    return label, float(confidence)

# Endpoint to fetch data from MySQL
@app.route('/data', methods=['GET'])
def get_data():
    cursor.execute('SELECT * FROM user')
    results = cursor.fetchall()
    return jsonify(results)

# Endpoint to add a user
@app.route('/addUser', methods=['POST'])
def add_user():
    data = request.json
    
    # Hash the password using bcrypt
    password_bytes = data['password'].encode('utf-8')  # convert to bytes
    hashed_password = bcrypt.hashpw(password_bytes, bcrypt.gensalt())

    query = "INSERT INTO user (name, firstName, lastName, password, email) VALUES (%s, %s, %s, %s, %s)"
    cursor.execute(query, (data['name'], data['first_name'], data['last_name'], hashed_password.decode('utf-8'), data['email']))
    db.commit()
    return jsonify({"message": "User added successfully", "userId": cursor.lastrowid})

# Endpoint for user login
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    name = data['name']
    password = data['password']

    # Fetch the user by name
    query = "SELECT * FROM user WHERE name = %s"
    cursor.execute(query, (name,))
    user = cursor.fetchone()

    if user:
        stored_hashed_password = user['password']  # Assuming you're using a dict cursor
        if bcrypt.checkpw(password.encode('utf-8'), stored_hashed_password.encode('utf-8')):
            return jsonify({"message": "Login successful", "user": user})
    
    return jsonify({"message": "Invalid username or password"}), 401

# Endpoint to make predictions
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     try:
#         image = Image.open(file)
#         processed_image = preprocess_image(image, (224,224))

#         with model_lock:  # Ensure thread-safe prediction
#             classifier.set_tensor(classifier_input_details[0]['index'], processed_image)
#             classifier.invoke()  # Run inference
#             output_data = classifier.get_tensor(classifier[0]['index'])

#         result = output_data.tolist()  # Convert to a list for JSON serialization
#         print(result)
#         return jsonify({'prediction': result})

#     except Exception as e:
#         return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image_rgb = Image.open(file).convert("RGB")      # Original image
    print(image_rgb.size)
    image_rgb_resized = image_rgb.resize((224, 224), Image.Resampling.LANCZOS)
    image_gray_resized = ImageOps.grayscale(image_rgb_resized)       # Grayscale for YOLO
    print(image_gray_resized.size)
    # ---------- YOLO DETECTION ----------
    results = yolo_model.predict(image_gray_resized, conf=0.5)[0]

    if results.masks is None or len(results.masks.data) == 0:
        return jsonify({"error": "No lesions detected"}), 400

    # Get mask from YOLO (already 224x224)
    mask = results.masks.data[0].cpu().numpy()       # Shape: (224, 224)
    mask = (mask * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Convert RGB image to NumPy
    img_np = np.array(image_rgb_resized)

    # Apply mask to all 3 channels
    masked_img = cv2.bitwise_and(img_np, img_np, mask=mask)

    # Convert back to PIL for classification
    pil_img = Image.fromarray(masked_img)

    # Base64 encode for response
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    img_bytes = buf.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    pil_img.show()

    # 2. Define image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Imagenet stats
                            std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(pil_img).unsqueeze(0)

    # Output from model
    logit = vit_model(input_tensor)  # shape: [1, 1]

    # Apply sigmoid to get probability
    prob = torch.sigmoid(logit)

    # Convert to label (threshold at 0.5)
    predicted_class = (prob >= 0.5).int().item()

    # Map to class name
    class_names = ['benign', 'malignant']
    predicted_label = class_names[predicted_class]
    if(predicted_label == "benign"):
        confidence = 1-prob.item()
    else:
        confidence = prob.item()


    # ---------- ROBOFLOW CLASSIFICATION ----------
    # result = CLIENT.infer(pil_img, model_id="skin-classification-2/12")
    # predicted_label = result['top']
    # confidence = result['confidence']

    # ---------- RETURN RESULT ----------
    output_results = {
        "lesion_id": 1,
        "result": predicted_label,
        "confidence": confidence,
        "masked_image": ""
    }

    return jsonify({"results": output_results}), 200

@app.route('/get-patient', methods=['GET'])
def getPatient():
    doctor_id = request.args.get('doctor_id')  # Get doctor_id from query params
    if doctor_id is None:
        return jsonify({'message': 'Doctor ID is required'}), 400
    
    try:
        query = 'SELECT * FROM patient WHERE doctor = %s'
        cursor.execute(query, (doctor_id,))
        patients = cursor.fetchall()

        if not patients:
            return jsonify({'message': 'No patients found'}), 404
        
        for patient in patients:
            patient['ic'] = int(patient['ic'])
            patient['age'] = int(patient['age'])
        print(patients)
        return jsonify(patients)

    except Exception as e:
        print(str(e))
        return jsonify({'message': str(e)}), 500

@app.route('/add-patient', methods=['POST'])
def addPatient():
    data = request.json

    try:
        # Validate required fields
        required_fields = ['name', 'ic', 'age', 'gender', 'doctor']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f"Missing or empty field: {field}"}), 400

        name = data['name']
        patient_ic = data['ic']
        age = data['age']
        gender = data['gender']
        phone = data.get('phone')  # optional
        email = data.get('email')  # optional
        doctor = data['doctor']

        query = '''
            INSERT INTO patient (name, age, ic, email, phone_number, doctor, gender)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        '''
        cursor.execute(query, (name, age, patient_ic, email, phone, doctor, gender))
        db.commit()

        return jsonify({'message': 'Patient added successfully'}), 201

    except Exception as e:
        print("Error:", e)  # Good for debugging
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, threaded=True)  # Enable threading
