# Real-Time Age and Gender Detection

This project demonstrates real-time age and gender detection using computer vision techniques. It utilizes pre-trained deep learning models to detect faces, estimate age, and classify gender from webcam input.

## Getting Started

### Prerequisites

- Python 3.x
- OpenCV (cv2)
- Pre-trained models:
  - Face detection model (deploy.prototxt.txt and res10_300x300_ssd_iter_140000.caffemodel)
  - Gender classification model (gender_deploy.prototxt and gender_net.caffemodel)
  - Age estimation model (age_deploy.prototxt and age_net.caffemodel)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/real-time-age-gender-detection.git
   ```

2. Download the pre-trained models and place them in the project directory.

### Usage

1. Run the script:

   ```bash
   python age_gender_detection.py
   ```

2. The webcam will open, and the real-time age and gender detection will start. Press 'q' to exit.

## Acknowledgments

- The face detection model is based on the [SSD framework](https://arxiv.org/abs/1512.02325).
- The gender classification and age estimation models are based on the Caffe framework.
