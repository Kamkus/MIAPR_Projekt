# Planowanie ruchu w 2D z wykorzystaniem A* i gradientu z sieci neuronowej

## Autorzy

Artur Matuszak  
Kamil Markowski

## Wymagania

ROS 2 distro: humble

## Instalacja wymaganych pakietów:

# Instalacja paczki ros-humble-nav2-map-server
```bash
sudo apt-get install ros-humble-nav2-map-server ros-humble-nav2-lifecycle-manager
```

# Instalacja wymaganych bibliotek python
```bash
pip install torch
```

## Uruchomienie
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```
skopiować folder mapr_5_student do katalogu src
```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
ros2 launch mapr_5_student astar_launch.py
```

## Sprawdzenie gradientu
Mapę gradientu można obejrzeć po odpaleniu pliku model/test.py

wszystkie biblioteki potrzebne do uruchomienia testu znajduja się w pliku model/requirements.txt



