# Endometrium Cancer Detection (EC)

A Flask-based web application to detect and classify endometrium cancer from images using a convolutional neural network (CNN).

## Features

- User registration and login with SQLite backend.
- Image upload and preprocessing including grayscale conversion, filtering, segmentation.
- Cancer classification into Endometrial Adenocarcinoma, Hyperplasia, Polyp, and Normal Endometrium.
- Visualization of detection results and accuracy graph.
- Cancer grade and stage prediction based on image analysis.

## Installation

1. Clone the repository:
2. Create a virtual environment and activate it:
3. Install dependencies:
4. Run the app:

5. Access the app in your browser at:


## Usage

- Register or log in with user credentials.
- Upload test images for cancer prediction and analysis.
- View images, results, cancer grade, and stage on the results page.

## Project Structure

EC/
├── app.py
├── requirements.txt
├── README.md
├── static/
│ ├── images/
│ └── ...
├── templates/
│ ├── index.html
│ ├── results.html
│ └── ...
└── test/


## Notes

- The CNN model (`Convolutional_Neural_Network.h5`) should be placed in the project root folder.
- Image processing uses OpenCV and TensorFlow/Keras.
- Designed for educational and experimental purposes; not for clinical use.

