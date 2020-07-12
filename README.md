# Whatsapp Notification On Face Recognition
### Description
The project is made with real-life implementation perspective. The basic idea behind this project is to keep an eye on the children i.e at what time the come house and what time they leave the house. So, as soon as the face is recognised a whatsapp notification is sent with a message as <br/>
`Last seen: NAME` <br/>
`@ Date Month Year HH:MM AM/PM` <br/>

#### Prerequisites
  ###### Required
  - Python Programming Language 
  - JavaScript Object Notation(JSON)
  - Machine Learning algorithms
  - Convolutional Neural Network 
  - [Twilio Account](https://www.twilio.com/)

  ###### Not compulsory(given below), but pior knowledge would be beneficial
  - Caffe-Based Deep Learning models 
  - OpenCV 

#### Installations
> [Install Python](https://www.python.org/downloads/)<br/>
> [Install OpenCV for windows](https://docs.opencv.org/master/d5/de5/tutorial_py_setup_in_windows.html) <br/>
> [Install OpenCV for Ubuntu](https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html) <br/>

> pip install twilio <br/>
> pip install imutils <br/>
> pip install numpy <br/>
> pip install -U scikit-learn <br/>

#### Directory structure 
![Directory structure](https://github.com/techycs18/whats-notification-on-face_recognition/blob/master/directory_structure.png) <br/>
For proper execution of this project keep the files in proper given structure

#### Setup
- Collect atleast 45 photos of a each person of whom the recognition are to be performed. Inside `my_dataset` directory create folders with name of the each individual and place thier images(atleast 45 each) inside them
- Now inside `my_dataset` directory create one more folder with name as `unknown` and inside this folder place different 15-20 images of random people (celebrities/friends/images who are not to be recognised) 
- Once folders created, inside `conf.json` file replace the `xyz_person` with person name. The person name and folder name where the images are located should exactly be same(case-sensitive).
- If you want to perfrom multiple face-recognistion, then add more people.For example we have three people of whom face is to be detected and their names are `xyz1, xyz2 and xyz3`<br/>
So inside`conf.json` do:
 ```
 "details_json" :{
		"xyz1": {
			"timestamp": 0,
			"captures": 0,
			"total_capture": 0,
			"image_sent": false
		},
		"xyz2": {
			"timestamp": 0,
			"captures": 0,
			"total_capture": 0,
			"image_sent": false
		},
    	"xyz3": {
			"timestamp": 0,
			"captures": 0,
			"total_capture": 0,
			"image_sent": false
		}
	}
  
 ```
###### NOTE
- Before running the python code you need to set two environement variable. To do so run (always re-run it when you shutdown your local-machine and log in next time) the two lines given below in terminal/console one by one:
```
export TWILIO_ACCOUNT_SID='COPY_YOUR_TWILIO_ACCOUNT_SID'
export TWILIO_AUTH_TOKEN='COPY_TWILIO_AUTH_TOKEN'
```
This is needed inorder for TWILIO whatsapp API to work properly <br/>
For TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN refer [Twilio Console](https://www.twilio.com/console)(Create twilio account if not created already)

- Once all the setup is done run the following in sequence:
```
python face_extract_embedding.py
python train_model.py
python faceRecognition_notification.py
````

#### Author
- [Chaitanya Sonavane](https://www.linkedin.com/in/chaitanya-sonavane-3766521a0/) [July 2020] 

#### Acknowledments
- [Twilio Developers](https://www.twilio.com/) 
- [Adrian Rosebrock](https://www.pyimagesearch.com/)
