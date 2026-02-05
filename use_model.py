import io
import numpy as np
import requests
from PIL import Image
from bs4 import BeautifulSoup
from urllib.parse import urljoin
 
 
import pickle
 
# load model from file
with open('modelo.pkl', 'rb') as f:
   model = pickle.load(f)
 
session = requests.Session()
 
TARGET_IP = "127.0.0.1:8000"
BASE_URL = f"http://{TARGET_IP}/captcha.php"
 
def get_images():
   image_urls = []
   page = session.get(BASE_URL)
   soup = BeautifulSoup(page.text, 'html.parser')
   images_container = soup.find('div', class_='captcha-images')
   if images_container:
       images = images_container.find_all('img')
       for img in images:
           image_url = urljoin(BASE_URL, img['src'])
           image_urls.append(image_url)
           print(image_url)
   return image_urls
 
def make_prediction(image_url):
   # Load the image from the URL
   response = session.get(image_url, stream=True)
   response.raw.decode_content = True
   img = Image.open(response.raw)
   np_frame = np.array(img)
 
   # Match the input shape for the model
   image = np.array([np_frame / 255])
 
   # Make the prediction
   predictions = model.predict(image)
   prediction = np.argmax(predictions[0])
   return prediction
 
# First we get the image URLs from the web application
image_urls = get_images()
 
captcha_guess = []
 
# Process each image to get the predicted value
for image_url in image_urls:
   print(image_url)
   guess = make_prediction(image_url)
   print(guess)
   captcha_guess.append(str(guess))
 
# Construct a request with our predictions
captcha_string = "".join(captcha_guess)
print(captcha_string)
 
params = {
   'captcha': captcha_string,
   'submit': 'Verificar+Captcha'
}
 
# Send the request to the form
form_url = f"http://{TARGET_IP}/captcha.php"  # Update the URL if necessary
response = session.post(form_url, data=params)