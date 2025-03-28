import firebase_admin
from firebase_admin import credentials, db
import datetime

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {"databaseURL":"https://plant-ff4f7-default-rtdb.asia-southeast1.firebasedatabase.app/"})

def store_prediction(image_name, result):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {"image_name": image_name, "result": result, "timestamp": timestamp}
    db.reference("Plant_Disease_Detection").push(data)
    print("✅ Prediction stored in Firebase!")
