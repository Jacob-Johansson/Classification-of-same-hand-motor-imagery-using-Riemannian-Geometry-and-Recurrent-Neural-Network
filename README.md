# Classification-of-same-hand-motor-imagery-using-Riemannian-Geometry-and-Recurrent-Neural-Network
This project presents a novel hybrid approach for Electroencephalography (EEG)-based Motor Imagery (MI) classification focusing on distinguishing same-hand movements (e.g., left-hand clench vs. left-hand wrist extension). The method leverages Riemannian Geometry to robustly extract spatial features (covariance matrices) from the EEG signals, addressing the non-Euclidean nature of brain signal data. These robust features are then fed into a Recurrent Neural Network (RNN), specifically LSTM (Long short-term memory), to effectively model the temporal dynamics inherent in the MI task. The primary objective is to enhance classification accuracy and robustness, particularly for the challenging task of discriminating subtle variations within a single limb's motor intent.


## 🛠️ Dependency Setup

To run this project locally, you will need Python 3.12.9+ and pip.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Jacob-Johansson/Classification-of-same-hand-motor-imagery-using-Riemannian-Geometry-and-Recurrent-Neural-Network.git)
    cd your-repo
    ```
2.  **Install Dependencies**
    Using a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    pip install -r requirements.txt
    ```


![Screenshot of the app running](Classifier_comparison/.png)
