# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup, redirect, url_for
from flask_admin.contrib.sqla import ModelView
from flask_admin.form import ImageUploadField
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
import requests
from bs4 import BeautifulSoup
import io
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import config
import pickle
from utils.model import ResNet9
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import sqlite3
from flask import send_file
# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

crop_recommendations = {
'भुट्टा': 'सोयाबीन या अल्फाल्फा के साथ परिवर्तन करें',
'सोयाबीन': 'मक्का या गेहूं के साथ परिवर्तन करें',
'गेहूं': 'अल्फाल्फा या मक्का के साथ परिवर्तन करें',
'अल्फाल्फा': 'दाल या गेहूं के साथ परिवर्तन करें',
'चावल': 'लैग्यूम्स या घास के साथ परिवर्तन करें',
'जौ': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'जई': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'आलू': 'अनाज या लैग्यूम्स के साथ परिवर्तन करें',
'टमाटर': 'बीन्स या अनाज के साथ परिवर्तन करें',
'शिमला मिर्च': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'पत्ता गोभी': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'गाजर': 'अनाज या लैग्यूम्स के साथ परिवर्तन करें',
'पालक': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'लेट्यूस': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'कपास': 'लैग्यूम्स या घास के साथ परिवर्तन करें',
'गन्ना': 'लैग्यूम्स या घास के साथ परिवर्तन करें',
'सूरजमुखी': 'लैग्यूम्स या घास के साथ परिवर्तन करें',
'मूंगफली': 'अनाज या लैग्यूम्स के साथ परिवर्तन करें',
'राजमा': 'अनाज या लैग्यूम्स के साथ परिवर्तन करें',
'मटर': 'अनाज या लैग्यूम्स के साथ परिवर्तन करें',
'राई': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'सेसम': 'लैग्यूम्स या घास के साथ परिवर्तन करें',
'ज्वार': 'लैग्यूम्स या घास के साथ परिवर्तन करें',
'बाजरा': 'लैग्यूम्स या घास के साथ परिवर्तन करें',
'कुणुआ': 'लैग्यूम्स या घास के साथ परिवर्तन करें',
'कुट्टू': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'रामदानी': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'कमल ककड़ी': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'मेथी': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'सेम': 'लैग्यूम्स या घास के साथ परिवर्तन करें',
'ग्वार': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'तिल': 'लैग्यूम्स या घास के साथ परिवर्तन करें',
'मक्का': 'सोयाबीन या अल्फाल्फा के साथ परिवर्तन करें',
'भिंडी': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'टोटा': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'ब्रोकोली': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'आम': 'अनाज या लैग्यूम्स के साथ परिवर्तन करें',
'नाशपाती': 'लैग्यूम्स या घास के साथ परिवर्तन करें',
'सेंदरी': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'अमरूद': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'चिकू': 'लैग्यूम्स या घास के साथ परिवर्तन करें',
'नारियल': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'जामुन': 'लैग्यूम्स या घास के साथ परिवर्तन करें',
'पपीता': 'लैग्यूम्स या घास के साथ परिवर्तन करें',
'नींबू': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'अंगूर': 'लैग्यूम्स या घास के साथ परिवर्तन करें',
'खजूर': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'अखरोट': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'बादाम': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'पिस्ता': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'अनार': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'मौसंबी': 'लैग्यूम्स या घास के साथ परिवर्तन करें',
'नाशपाती': 'लैग्यूम्स या घास के साथ परिवर्तन करें',
'सेंदरी': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'अमरूद': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',
'चिकू': 'लैग्यूम्स या घास के साथ परिवर्तन करें',
'नारियल': 'रूट क्रॉप्स या लैग्यूम्स के साथ परिवर्तन करें',

}


disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crops.db'
db = SQLAlchemy(app) 

# Define Crop model
class Crop(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    price = db.Column(db.Float, nullable=False)
    image = db.Column(db.String(100), nullable=False)
    seller_name = db.Column(db.String(100), nullable=False)

# Route to display all crops
@app.route('/buy')
def buy():
    title = 'KisanKonnect - BuyCrop'
    crops = Crop.query.all()
    return render_template('buy.html', crops=crops,title=title)


# Route to display order details
@app.route('/order/<int:crop_id>')
def order(crop_id):
    title = 'KisanKonnect - OrderDetails'
    crop = Crop.query.get_or_404(crop_id)
    return render_template('orders.html', crop=crop,title=title)

# Route to upload a crop
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    title = 'KisanKonnect - SellCrop'
    if request.method == 'POST':
        # Upload the crop and get its ID
        name = request.form['name']
        description = request.form['description']
        price = request.form['price']
        image = request.files['image']
        seller_name = request.form['seller_name']
        
        # Save image file
        image_path = 'static/upload/' + image.filename
        image.save(image_path)
        
        # Create new crop object
        new_crop = Crop(name=name, description=description, price=price, image=image_path, seller_name=seller_name)
        db.session.add(new_crop)
        db.session.commit()
        
        # Redirect to the order page for the newly uploaded crop
        return redirect(url_for('buy', crop_id=new_crop.id))  # Pass the crop_id parameter
    return render_template('upload.html',title=title)

@app.route('/confirm_order', methods=['POST'])
def confirm_order():
    title = 'KisanKonnect - ConfirmOrder'
    crop_id = request.form['crop_id']
    crop = Crop.query.get_or_404(crop_id)
    # Calculate the actual crop price (if needed)
    actual_price = crop.price  # Assuming the price is already set correctly
    return render_template('QR.html',title=title)

@app.route('/delete', methods=['POST'])
def delete_crop():
    crop_id = request.form['crop_id']
    crop = Crop.query.get(crop_id)
    if crop:
        db.session.delete(crop)
        db.session.commit()
    return redirect(url_for('buy'))



# render home page


@app.route('/')
def home():
    title = 'KisanKonnect - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@app.route('/crop-recommend')
def crop_recommend():
    title = 'KisanKonnect - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'KisanKonnect - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page

# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'KisanKonnect - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        city = request.form.get("city_select")  # Corrected variable name

        if city is not None:  # Check if city is not None
            if weather_fetch(city) is not None:
                temperature, humidity = weather_fetch(city)
                data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                my_prediction = crop_recommendation_model.predict(data)
                final_prediction = my_prediction[0]

                return render_template('crop-result.html', prediction=final_prediction, title=title)
            else:
                return render_template('try_again.html', title=title)
        else:
            return render_template('select_city.html', title=title)  # Render a page with a message prompting the user to select a city

# render fertilizer recommendation result page


@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'KisanKonnect - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['potassium'])  # Fixed variable name

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'KisanKonnect - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


@app.route('/rotation')
def rotation():
    title = 'KisanKonnect - Crop Rotation'
    return render_template('rotation.html', title=title, crop_recommendations=crop_recommendations)


@app.route('/recommendation', methods=['POST'])
def recommendation():
    title = 'KisanKonnect - Rotation Recommendation'
    if request.method == 'POST':
        # Get the selected crop from the form
        selected_crop = request.form['crop']
        # Fetch recommendation from the dictionary
        recommendation = crop_recommendations.get(selected_crop, 'No recommendation found')
        return render_template('recommendation.html',title=title, crop=selected_crop, recommendation=recommendation, crop_recommendations=crop_recommendations)


@app.route('/Yojna.html')
def kisan_yojna():
    title = 'KisanKonnect - Kisan Yojna'
    return render_template('Yojna.html', title=title)

@app.route('/testimonials.html')
def testimonials():
    title = 'KisanKonnect - Testimonials'
    return render_template('testimonials.html', title=title)

@app.route('/about-us.html')
def aboutus():
    title = 'KisanKonnect - AboutUs'
    return render_template('about-us.html', title=title)

@app.route('/soil-testing.html')
def soil_testing():
    title = 'KisanKonnect - Soil-Testing'
    return render_template('soil-testing.html', title=title)

# Initialize Database
def init_db():
    conn = sqlite3.connect('contacts.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS contacts
                      (id INTEGER PRIMARY KEY, name TEXT, phone TEXT, email TEXT, location TEXT)''')
    conn.commit()
    conn.close()

# Route to show contact form
@app.route("/contacts")
def contacts():
    return render_template("contacts.html")

# Route to handle form submission
@app.route("/add_contact", methods=["POST"])
def add_contact():
    name = request.form["name"]
    phone = request.form["phone"]
    email = request.form["email"]
    location = request.form["location"]

    conn = sqlite3.connect('contacts.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO contacts (name, phone, email, location) VALUES (?, ?, ?, ?)", 
                   (name, phone, email, location))
    conn.commit()
    conn.close()

    return redirect(url_for("contacts"))

@app.route("/export")
def export():
    conn = sqlite3.connect('contacts.db')
    df = pd.read_sql_query("SELECT * FROM contacts", conn)
    file_path = "contacts.xlsx"
    df.to_excel(file_path, index=True)
    conn.close()
    
    return send_file(file_path, as_attachment=True)
# ===============================================================================================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
