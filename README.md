Smart Home System (IoT + AI)

Overview

This project implements a smart home system that integrates IoT and AI to provide intelligent control, automation, and security for home devices. The system allows users to control devices through a web application, use voice commands for natural interaction, and apply face recognition for door access control. All device states are synchronized in real time using a cloud database.

This project was developed as a university assignment at Ho Chi Minh City University of Technology and Education.

Features

The system supports real-time control of devices such as lights, fans, and doors using ESP32 microcontrollers connected to Firebase Realtime Database. Users can turn devices on or off, adjust brightness, change colors of NeoPixel LEDs, and control fan speed.

A web application is developed to manage devices across multiple rooms. The interface allows users to monitor device states and interact with them using buttons, sliders, and color pickers. The system also supports predefined scene modes such as Relax, Movie, and Night, which allow multiple devices to change states simultaneously.

Voice control is implemented using the Whisper speech recognition model. Users can give commands in Vietnamese, which are converted into text and processed to control devices. The system can interpret commands such as turning devices on or off, adjusting brightness, or switching scenes.

Face recognition is implemented using MediaPipe and MobileNetV2. The system detects and identifies registered users and automatically opens the door if the face is recognized.

All components are synchronized using Firebase Realtime Database. The system separates the desired state (commands from users or AI) and the reported state (actual device status from ESP32), ensuring consistent and reliable operation across all components.

System Architecture

The system consists of four main components: a web application for user interaction, Firebase Realtime Database for data synchronization, ESP32 devices for hardware control, and a Flask server for AI processing. The web application and Flask server both interact with Firebase, while ESP32 devices continuously read and update device states.

Technologies Used

The hardware includes ESP32 as the main controller, NeoPixel LED strips, servo motors for door control, optional DHT11 sensors for temperature and humidity, and audio modules such as JQ6500 with speakers.

The software stack includes HTML, CSS, and JavaScript for the web interface, Flask for backend processing, and Firebase Realtime Database for real-time data synchronization.

AI components include Whisper for speech recognition, MediaPipe for face detection, and MobileNetV2 for face recognition.

How It Works

Users interact with the system through the web interface, voice commands, or camera input. Voice and image data are sent to the Flask server for processing. The processed commands are then written to Firebase as desired states. ESP32 devices monitor Firebase for changes, execute the corresponding actions on hardware, and update the reported state back to Firebase. The web interface reflects these changes in real time.

Results

The system is able to control devices reliably in real time, execute voice commands with acceptable accuracy, and perform face recognition for door access control. Data synchronization between all components is stable and responsive.

Limitations

The system is implemented as a small-scale prototype and does not represent a full real-world smart home. Voice recognition performance depends on environmental noise and speaking clarity. Face recognition works best under stable lighting conditions. The system does not yet include advanced security features such as authentication or data encryption. The interface is currently web-based and not optimized for mobile applications.

Future Improvements

Future development can include building a mobile application, improving natural language processing for more flexible voice commands, deploying AI models on edge devices, enhancing system security, and expanding support for more types of smart home devices.

Authors

Tran Hai Nam
Nguyen Van Quoc Tai

Instructor: Pham Ngoc Son

License

This project is for educational purposes only
