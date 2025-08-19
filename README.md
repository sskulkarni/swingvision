# 🤖 Tennis Match Player & Ball Analytics using Computer Vision

This project uses computer vision techniques to extract and analyze player and ball behavior from tennis match videos. It provides visual and statistical feedback such as player movement patterns, speed analysis, and heatmaps — aiding both players and coaches in performance improvement.


## 📽️ Demo
*Insert GIF or image of simulation here*

---

## 🛠️ Project Description

This project focuses on analyzing tennis player performance and match dynamics using computer vision techniques. Leveraging modern object detection models like YOLOv8 and deep learning-based tracking tools, the system processes raw tennis match videos to extract meaningful insights for players and coaches.

The solution tracks and analyzes three key elements on the court: players, the tennis ball, and the racket. From this data, the system computes movement trajectories, player speed, court coverage, and racket pose, and generates visualizations like heatmaps, speed-over-time graphs, and annotated videos.

### Key Features

- 🎯 Player Detection & Tracking using YOLOv8 + Centroid Tracker
- 🎾 Ball Tracking using YOLOv8 + TrackNet
- 📈 Speed Estimation per frame (average and max speeds)
- 🌡️ Player Heatmap generation to show court coverage
- 📊 Speed Distribution Histogram and Speed-over-Time Graph
- 📹 Overlayed Video Generation with bounding boxes and labels
- 📍 Court Detection and positional zone classification
- 🏸 Racket Pose Estimation using body keypoints and angle calculations

---

## 🧠 Technologies Used

- **OpenCV**
- **ultralytics (YOLOv8)**
- **matplotlib**
- **numpy**
- **filterpy (for Kalman filter in tracking)**


## How It Works



- 1. Frame Extraction: Each frame is processed from the input video.
- 2. Detection: YOLOv8 detects players, racket, and ball.
- 3. Tracking: SORT + Centroid tracking used for player trajectories.
- 4. Analysis:
		Movement plotted over time
		Speed computed per frame
		Heatmap generated for court coverage
- 5. Visualization: Results saved as plots and annotated videos.



---

## 📁 Project Structure

```plaintext
.
├── Player_Tracking/
│   ├── player_tracking.py         # Main tracking script
│   ├── sort.py                    # SORT tracker implementation
│   ├── yolo_detect.py             # YOLOv8 detection wrapper
│   └── utils.py                   # Helper functions
├── CSVOutput/
│   └── player_tracking.csv        # Tracked player coordinates and stats
├── Graphs/
│   ├── heatmap.png
│   ├── speed_over_time.png
│   └── speed_distribution.png
├── overlay_video/
│   └── output.avi                 # Output video with annotations
└── README.md

```

---

## 🚀 Getting Started

### Prerequisites

Before running the simulation, install the following:

- Python 3.7+  
- Required Python libraries:

```bash
pip install numpy

```


### Running the Simulation

Clone the Repository
Open a terminal and run:
```bash
git clone https://github.com/sskulkarni/swingvision.git
cd swingvision
```

#### Install the depedencies
```bash
pip install -r requirements.txt
```


#### Run a Video

```bash
python3 Player_Tracking/Player_Movement.py
```
Please make sure to set input variable video_in
The output will be stored in the directory "VideoOutput"
CSV will be stored in directory "CSVOutput"

#### Create a Heatmap

```bash
python3 python3 Player_Tracking/Heatmap.py
```
Please make sure to set input variable "csv_path"
It will use the out put csv generated while Vidoe analysis

#### Create a Player Movement Graph

```bash
python3 python3 Player_Tracking/Player_Movement.py
```
Please make sure to set input variable "csv_path"
It will use the out put csv generated while Vidoe analysis





## 🔍 Future Improvements

- Create a Web App, So end user can work on it.
- Train the model for shot classification.

## 🔍 Results

- Accurate detection of player positions per frame
- Visual insights into player strategy (e.g., left vs. right court usage)
- Graphs highlight moments of peak effort or stillness
- Easy-to-understand overlays and CSV reports