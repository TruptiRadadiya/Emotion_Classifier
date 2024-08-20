# Project_4
sad and happy face auto detection

# Kaggle API
To use the dataset directly in a Google Colab notebook without downloading it manually, you can use Kaggle’s API to download the dataset directly to your Colab environment. Here’s how you can set it up:

1. Set Up Kaggle API Key
* Go to your Kaggle account settings by clicking on your profile picture in the top right, then clicking “Account.”
* Scroll down to the API section and click “Create New API Token.” This will download a kaggle.json file containing your API credentials.

2. Upload the Kaggle API Key to Colab
* In your Google Colab notebook, run the following code to upload your kaggle.json file:

> from google.colab import files
>
> files.upload()  # Upload the kaggle.json file you just downloaded

3. Set Up Kaggle API in Colab
* After uploading the kaggle.json file, you need to move it to the correct directory and set the necessary permissions:

> !mkdir -p ~/.kaggle
> 
> !mv kaggle.json ~/.kaggle/
> 
> !chmod 600 ~/.kaggle/kaggle.json

4. Download the Dataset Using Kaggle API
* Now, you can download the dataset directly into your Colab environment using the Kaggle API:

> !kaggle datasets download -d alirezaatashnejad/sad-and-happy-face-detection

5. Unzip the Dataset
* Unzip the downloaded dataset:

> !unzip sad-and-happy-face-detection.zip -d ./data

6. Load the Dataset
* After unzipping, you can load the dataset into your notebook using the appropriate libraries (e.g., OpenCV, PIL for images).

> import os
>
> from PIL import Image
> 
> #Example of loading an image
> image_path = './data/happy/your_image.jpg'  # Replace with the actual path
> 
> image = Image.open(image_path)
> 
> image.show()

This setup allows you to access and use the Kaggle dataset directly in Google Colab without needing to download it manually from the Kaggle website.


