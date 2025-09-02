import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing import image
import cv2
import numpy as np
from tensorflow.keras.models import load_model
##from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request
import sqlite3
from keras.models import load_model
import shutil

from skimage.filters import laplace, sobel
from skimage import exposure, img_as_float



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/userlog.html')
def userlogg():
    return render_template('userlog.html')

@app.route('/developer.html')
def developer():
    return render_template('developer.html')

@app.route('/graph.html', methods=['GET', 'POST'])
def graph():
    
    images = ['http://127.0.0.1:5000/static/acc_graph.png',
              'http://127.0.0.1:5000/static/loss_graph.png',
              'http://127.0.0.1:5000/static/conf_mat.png']
    content=['Accuracy Graph',
             'Loss Graph(Error Message)',
            'Confusion Matrix']

            
    
        
    return render_template('graph.html',images=images,content=content)
    


@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
 
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        

        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 1. RGB to Grayscale conversion (Luminosity method)
        luminosity_gray = 0.21 * image[:, :, 2] + 0.72 * image[:, :, 1] + 0.07 * image[:, :, 0]
        cv2.imwrite('static/luminosity_gray.jpg', luminosity_gray)
        # 2. Noise removal methods


        # Gaussian filter
        gaussian_filtered = cv2.GaussianBlur(luminosity_gray, (5, 5), 1)
        cv2.imwrite('static/gaussian_filtered.jpg', gaussian_filtered)

        # Sobel filter for edge detection and sharpening
        sobel_x = sobel(gaussian_filtered)
        cv2.imwrite('static/sobel_edge.jpg', sobel_x * 255)

        # 4. Thresholding

        # Gaussian thresholding
        _, gaussian_threshold = cv2.threshold(sobel_x, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite('static/gaussian_thresholded.jpg', gaussian_threshold)

        # 5. Segmentation

        # Edge-based segmentation (Canny edge detection)
        edges = cv2.Canny(gaussian_threshold.astype("uint8"), 100, 200)
        cv2.imwrite('static/edge_based_segmentation.jpg', edges)


        # White areas represent tumor/lymph nodes/metastasis regions
        image = cv2.imread("test/"+fileName, cv2.IMREAD_GRAYSCALE)



       # Apply Gaussian Blur and thresholding
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, binary_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Collect contour areas for dynamic thresholding
        areas = [cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) > 0]

        # Calculate dynamic thresholds based on percentiles
        tumor_threshold = np.percentile(areas, 75) if areas else 0
        lymph_node_threshold = np.percentile(areas, 50) if areas else 0
        metastasis_threshold = np.percentile(areas, 25) if areas else 0

        # Initialize counters and grading scores
        tumor_count = 0
        lymph_node_count = 0
        metastasis_count = 0
        grade_score = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            
            circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
            solidity = area / hull_area if hull_area > 0 else 0

            # Determine TNM category based on dynamic thresholds
            if area > tumor_threshold and circularity > 0.7:
                tumor_count += 1
                label = "Tumor"
            elif lymph_node_threshold < area <= tumor_threshold and circularity > 0.5:
                lymph_node_count += 1
                label = "Lymph Node"
            elif area <= metastasis_threshold:
                metastasis_count += 1
                label = "Metastasis"
            else:
                continue

            # Grade scoring based on solidity and circularity
            if solidity > 0.8 and circularity > 0.7:
                grade = "Low Grade (Well-Organized)"
            elif 0.5 < solidity <= 0.8 and 0.5 < circularity <= 0.7:
                grade = "Intermediate Grade (Moderately Organized)"
            else:
                grade = "High Grade (Disorganized/Solid Growth)"
            grade_score += (1 if "Low Grade" in grade else 2 if "Intermediate" in grade else 3)

            # Draw and label contours
            x, y, w, h = cv2.boundingRect(contour)
            #cv2.putText(image, f"{label}: {grade}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        # Calculate the overall grade score average
        total_cells = tumor_count + lymph_node_count + metastasis_count
        average_grade_score = grade_score / total_cells if total_cells > 0 else 0

        # Print the counts and grading score
        print("Tumor Count:", tumor_count)
        print("Lymph Node Count:", lymph_node_count)
        print("Metastasis Count:", metastasis_count)
        print("Average Grade Score:", grade)


        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        status1=''
        model=load_model('Convolutional_Neural_Network.h5')
        path='static/images/'+fileName
        # Load the trained CNN model
        cnn_model_path = "Convolutional_Neural_Network.h5"
        cnn_model = load_model(cnn_model_path)

        # Define the classes
        class_labels = os.listdir("D:\\ENDOMETRIUM_UPDATED\\test")

        # Function to prepare the image for prediction
        def prepare_test_image(path):
            img = load_img(path, target_size=(128, 128), grayscale=True)
            x = img_to_array(img)
            x = x / 255.0
            return np.expand_dims(x, axis=0)

        # Function to predict and display the result
##        def predict_and_display_image(model, img_path, class_labels):
        result = model.predict(prepare_test_image(path))
        img = cv2.imread(path)

        print("Prediction Result:", result[0])

        class_result = np.argmax(result, axis=1)
        print("Class number:", class_result[0])
        print("Predicted class:", class_labels[class_result[0]])


        
        result=list(result[0])
        if class_result[0]== 0:
            str_label = "Endometrial Adenocarcinoma"
            status1="stage1 cancer"
            print("The predicted image of the Endometrial Adenocarcinoma is with a accuracy of {} %".format(result[class_result[0]]*100))
            accuracy="The predicted image of the Endometrial Adenocarcinoma is with a accuracy of {}%".format(result[class_result[0]]*100)
           
           
            
        elif class_result[0]== 1:
            str_label  = "Endometrial hyper plasia"
            status1="stage2 cancer"
            print("The predicted image of the Endometrial hyper plasia is with a accuracy of {} %".format(result[class_result[0]]*100))
            accuracy="The predicted image of the Endometrial hyper plasia is with a accuracy of {}%".format(result[class_result[0]]*100)
           
            

        elif class_result == 2:
            str_label  = "Endometria polyp"
            status1="stage3 cancer"
            print("The predicted image of the Endometria polyp is with a accuracy of {} %".format(result[class_result[0]]*100))
            accuracy="The predicted image of the Endometria polyp is with a accuracy of {}%".format(result[class_result[0]]*100)
            
           

        elif class_result[0] == 3:
            str_label  = "Normal Endometrium"
            print("The predicted image of the Normal Endometrium is with a accuracy of {} %".format(result[class_result[0]]*100))
            accuracy="The predicted image of the Normal Endometrium is with a accuracy of {}%".format(result[class_result[0]]*100)
        A=float(result[0])
        B=float(result[1])
        C=float(result[2])
        D=float(result[3])
        
        
        dic={'EA':A,'EH':B,'EP':C,'NE':D}
        algm = list(dic.keys()) 
        accu = list(dic.values()) 
        fig = plt.figure(figsize = (5, 5))  
        plt.bar(algm, accu, color ='maroon', width = 0.3)  
        plt.xlabel("Comparision") 
        plt.ylabel("Accuracy Level") 
        plt.title("Accuracy Comparision between Endometrium Cancer....")
        plt.savefig('static/matrix.png')


        print("Tumor Count:", tumor_count)
        print("Lymph Node Count:", lymph_node_count)
        print("Metastasis Count:", metastasis_count)
        print("Cancer Grade:", grade)

        cell_count = [tumor_count, lymph_node_count, metastasis_count,grade]

        ImageDisplay=["http://127.0.0.1:5000/static/images/"+fileName,
        "http://127.0.0.1:5000/static/luminosity_gray.jpg",
        "http://127.0.0.1:5000/static/gaussian_filtered.jpg",
        "http://127.0.0.1:5000/static/sobel_edge.jpg",
        "http://127.0.0.1:5000/static/gaussian_thresholded.jpg",
        "http://127.0.0.1:5000/static/edge_based_segmentation.jpg",
        "http://127.0.0.1:5000/static/matrix.png"]

        labels = ['Original', 'Gray Image',
        'Gaussian Filter', 'Sobel',
        'Gaussian Threshold',
        'Edge Based (Canny)',  'graph']
        return render_template('results.html',labels=labels, status=str_label,accuracy=accuracy,cell_count=cell_count,grade=grade,ImageDisplay=ImageDisplay,n=len(ImageDisplay))


    return render_template('userlog.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
