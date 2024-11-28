🔥 Advanced AI-Powered Screen Detection System with YOLOv8x 🔥

Optimized for performance, analytics, and real-time interaction.

🚀 About the Project

This project leverages YOLOv8x for real-time object detection with advanced tracking, analytics, and mouse control. The script integrates high-performance CPU utilization, memory monitoring, and interactive features for a cutting-edge experience. It’s a resource-heavy script designed to maximize performance and system utilization.

🎯 Features

	•	State-of-the-art YOLOv8x Integration for maximum accuracy and performance.
	•	Advanced Mouse Control with real-time prediction and smoothing.
	•	Interactive Analytics Dashboard generated via SQLite and Matplotlib.
	•	Multithreaded Detection with CPU-intensive operations.
	•	Battery Drain Mode to push system limits for testing.
	•	Memory Monitoring and Garbage Collection for optimal resource usage.

🛠️ Tech Stack

Category	                                Libraries
Core                                      Frameworks	torch, numpy, cv2, mss, ultralytics
Multithreading & Queues	                  threading, queue, multiprocessing
System Monitoring	                        psutil, tracemalloc, gc
Data Handling	                            sqlite3, pandas, json, pickle, h5py
Visualization	                            matplotlib
Utility Libraries	                        logging, functools, hashlib, random, datetime



🖥️ System Requirements

	•	Python Version: 3.8 or higher
	•	Hardware: Multi-core CPU, at least 16GB RAM
	•	Operating System: macOS, Linux, or Windows (with CoreGraphics alternative)

📦 Installation


	2.	Install dependencies: pip install -r requirements.txt

pip install -r requirements.txt


	3.	(Optional) Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



📋 Configuration

The script is fully configurable via config.json. Key settings include:
	•	Detection Confidence: Adjust YOLO confidence threshold.
	•	FPS Limit: Set frames-per-second limit.
	•	Click Delay Range: Customize mouse click intervals.
	•	Target Prediction: Enable or disable advanced target prediction.


🤝 Contributing
Contributions are welcome! Please:
	1.	Fork the repository.
	2.	Create a new branch.
	3.	Submit a pull request with a detailed explanation.



**Made with the help of ChatGPT**
