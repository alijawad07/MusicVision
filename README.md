# Scene-Inspired Playlist Generator

Generate music playlists inspired by the content of images using object detection and AI-driven music recommendation.

## Description
The project uses YOLOv8 by Ultralytics for object detection to analyze and understand the content of an image. Upon identifying key objects and their placements within the scene, it leverages the Llama-2 model by Meta AI, accessed through Hugging Face's interface, to generate music playlists that complement the mood and theme of the scene. Although initially considered, integrating Spotify's API for playlist creation was set aside for future development.

## Features
- ***Object Detection***: Uses the efficient YOLOv8 model to detect and categorize objects within the images.
- ***Music Recommendation***: Utilizes the powerful Llama-2 model to generate a relevant and mood-matching music playlist.
- ***Image Processing***: Capable of processing individual images or entire directories.

## Installation and Setup

1. Clone the repository:
```
git clone https://github.com/alijawad07/MusicVision.git
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Download the necessary yolo model weights and place them in the weights/ directory.

4. Run the main script:

```
python main.py
```
### Please note that you need to have huggingface api token to use the model

## Future Work
- ***Integration with Spotify***: There's potential to integrate Spotify's API directly to create dynamic playlists on the platform.
- ***User Interface***: Developing a simple GUI or web application for easier user interaction.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
