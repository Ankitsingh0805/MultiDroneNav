# Multi-Agent Drone Navigation using QMIX

This project implements **Multi-Agent Reinforcement Learning (MARL)** using the **QMIX algorithm** for collaborative drone control in a 3D environment. The agents are trained in a simulated environment to coordinate and complete a navigation task.

---

## 📹 Demo
For the demo visualization you can check render.mp4 in the root folder .


---

## 📁 Project Structure

```
HACKATHON/
├── config/
│   └── params.py               # Hyperparameters and configs
│
├── environment/
│   ├── drone_env.py            # Multi-agent drone environment logic
│   └── rendering.py            # 3D rendering using Panda3D
│
├── marl/
│   ├── qmix.py                 # QMIX algorithm implementation
│   └── replay_buffer.py        # Experience replay logic
│
├── models/
│   └── qmix_final.pt           # Trained model weights
│
├── render/                     # (Optional) Rendered output
│
├── utils/
│   ├── helpers.py              # Utility functions
│   ├── metrics.py              # Evaluation metrics
│
├── main.py                     # Entry point
├── train.py                    # Training script
├── visualize.py                # Visualize trained agents
└── requirements.txt            # Project dependencies
```

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Ankitsingh0805/MultiDroneNav.git
cd your-repo-name
pip install -r requirements.txt
```

### requirements.txt

```
numpy==1.26.4
gymnasium==0.29.1
pettingzoo==1.24.3
torch==2.2.1
matplotlib==3.8.2
tqdm==4.66.1
panda3d==1.10.15
```

---

## ▶️ Usage

### To train the model:

```bash
python train.py
```

This will train the QMIX agents using the custom drone environment and save the model to `models/qmix_final.pt`.

---

### To visualize trained agents:

```bash
python visualize.py
```

This uses the trained model to run and render the simulation.

---

## ⚙️ How It Works

- **Environment (`drone_env.py`)**: A custom Gymnasium-compatible environment that simulates multiple drones navigating to goal points.
- **Rendering (`rendering.py`)**: Uses Panda3D for 3D simulation rendering.
- **QMIX (`qmix.py`)**: Implements the QMIX algorithm, a value decomposition method for cooperative MARL.
- **Replay Buffer**: Used to store and sample transitions for stable learning.
- **Training (`train.py`)**: Handles the full training loop including logging, reward collection, and model saving.
- **Visualization (`visualize.py`)**: Loads the trained model and visualizes agent behavior.

---

## 🚀 Future Work: Real Drone Deployment

To implement this on **real drones**, the following improvements will be made:

- Replace simulated sensors with real drone sensors (GPS, IMU, Lidar).
- Interface with drone flight stack (PX4/ArduPilot) via MAVSDK/DroneKit.
- Deploy trained models on embedded systems like **Raspberry Pi** or **NVIDIA Jetson**.
- Add obstacle detection and collision avoidance logic.
- Real-time communication between drones for coordination.

---

## 🤝 Contributing

Interested in expanding this to real-world drone swarms, enhancing training, or improving the rendering? PRs and discussions are welcome!

---

## 📄 License

This project is licensed under the MIT License.

