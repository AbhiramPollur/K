#Activity Recognition Web Application

##Overview

The Activity Recognition Web Application is a powerful tool that leverages machine learning to classify and understand human activities based on accelerometer data. This application was developed as part of a project to demonstrate the potential of machine learning in recognizing a wide range of physical activities. It utilizes the k-Nearest Neighbors Classifier (k-NN) with GridSearch for hyperparameter tuning to achieve high accuracy in activity recognition.

##Features

-**Flexible Data Input**: Users can upload accelerometer data in CSV format or manually input a single data row through an intuitive web form.
 -**Robust Data Preprocessing**: The application preprocesses uploaded data, ensuring it adheres to the required format, and prepares it for classification.

 -**Hyperparameter Tuning**: Hyperparameters for the k-NN model are optimized using GridSearch to achieve the best possible accuracy.

 -**Responsive Design**: The web application is designed to be responsive and accessible across various devices and screen sizes.

 -**User-Friendly**: Error handling and user feedback mechanisms guide users through the process, providing meaningful feedback in case of errors.

 ##Dataset

 The application is built on a dataset containing accelerometer data collected from 22 participants. Each participant's recordings are stored in seperate CSV files and include timestamp information, triaxial accelerometer readings from both back and thigh sensors, and annotated activity codes.

 ##Usage

 1.Clone the repository to your local machine.
 2.Install the required dependencies ny running 'pip install -r requirements.txt'.
 3.Run the web application by executing 'python main.py', and after the model is 
   trained execute 'python app.py'.
 4.Access the application through a web browser by navigating to a local host.
 5.Choose your preferred data input method (CSV file upload or manual data entry) and 
  follow on-screen instructions.

 ##Results and Conclusion

 For detailed information on the project's results and conclusion, please refer to 
 the 'ProjectK_documentation.pdf' file.
