# Whatsapp Notification On Face Recognition
### Description :
The project is made with real-life implementation perspective. The basic idea behind this project is to keep an eye on the children i.e at what time the come house and what time they leave the house. So, as soon as the face is recognised a whatsapp notification is sent with a message as <br/>
`Last seen: NAME` <br/>
`@ Date Month Year HH:MM AM/PM` <br/>

#### Prerequisites :
  ###### Required 
  - Python Programming Language 
  - Machine Learning algorithms
  - Convolutional Neural Network 
  - [Twilio Account](https://www.twilio.com/)

  ###### Not compulsory(given below), but pior knowledge would be beneficial 
  - Caffe-Based Deep Learning models 
  - OpenCV for Python

#### Installations :
> - [Install Python](https://www.python.org/downloads/)<br/>
> - [Install OpenCV for windows](https://docs.opencv.org/master/d5/de5/tutorial_py_setup_in_windows.html) <br/>
> - [Install OpenCV for Ubuntu](https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html) <br/>
> - pip install twilio <br/>
> - pip install numpy <br/>
> - pip install -U scikit-learn <br/>


#### Procedure
###### 3 steps -
- Step-1, Use OpenCV face detector model to detect face from each input image and use OpenCV embedder model to extract the 128-d face embeddings for each image
- Step-2, We pass these 128-d embeddings of each iamge through Linear SVM classifier to train it and use it as a recognizer model in step-3
- Step-3, We take the frames of face detected, extract 128-d embeddings and then pass this embeddings through our recognizer model
	- Based on the recognizer predictions we recognise whose face it was.
	- It recognise anyone from our dataset(as it was trained on those images) it saves the frames and sends a whatsapp notification to a verified number
	- If unknwon face is detected we just ignore it.   

#### Setup :
1. Make two different empty folders with name `my_dataset` and `motions_caught` inside `face-recognition-whatsapp-notification` directory
2. Inside `my_dataset` directory create folders with name of the each individual and place thier images(atleast 35-40 each person of whom the recognition is to be performed) inside them
3. Now inside `my_dataset` directory create one more folder with name as `unknown` and inside this folder place different 35-40 images of random people (celebrities/friends/images who are not to be recognised) 
4. Once folders created, in python file `faceRecognition_notification.py` check the `details_json` instance variable of `class Activities`. Replace the `person1_name` json oject with the name of the person(the name should be same as done in step 2 and also it is case-sensitive)
5. Consider you want to person detection for two people named as `amitabh` and `yash` 
6. So inside constructure we would replace `person1_name` and `person2_name` with the following:
 ```
 "details_json" :{
		"amitabh": {
			"timestamp": 0,
			"captures": 0,
			"total_capture": 0,
			"image_sent": false
		},
		"yash": {
			"timestamp": 0,
			"captures": 0,
			"total_capture": 0,
			"image_sent": false
		}
	}
 
 ```
- Also, create an empty folder named `output`. This file will store all the embeddings and our trained model

###### NOTE :
- Before running the python code you need to set two environement variable. To do so run (always re-run it when you shutdown your local-machine and log in next time) the two lines given below in terminal/console one by one:
- For Linux run code given below: (for other OS[check here])(https://www.twilio.com/blog/2017/01/how-to-set-environment-variables.html):
```
export TWILIO_ACCOUNT_SID='COPY_YOUR_TWILIO_ACCOUNT_SID'
export TWILIO_AUTH_TOKEN='COPY_TWILIO_AUTH_TOKEN'
```


- This will make sure twilio whatsapp API to work properly on our local-machine. For TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN refer [Twilio Console](https://www.twilio.com/console)(Create twilio account if not created already)
- Also change the `TO_NUMBER` instance variable value to the phone number you wish to recieve msg on.
- You first need to connect that number to twilio via [Twilio Testing Sandbox](https://www.twilio.com/console/sms/whatsapp/learn)(see: Set Up Your Testing Sandbox)

- Once all the setup is done run the following in sequence:
```
python face_extract_embedding.py
python train_model.py
python faceRecognition_notification.py
````

#### Author :
- [Chaitanya Sonavane](https://www.linkedin.com/in/chaitanya-sonavane-3766521a0/) [July 2020] 

#### Acknowledments :
- [Twilio Developers](https://www.twilio.com/) 
- [Adrian Rosebrock](https://www.pyimagesearch.com/)
