import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Emotion Detection from Uploaded Images",
    layout="wide",
    initial_sidebar_state="auto",
)

# Sidebar 
with st.sidebar:
    selected = option_menu(
        None,
        ["Home", "Detection"],
        icons=["house", "tag"],
        default_index=0,
        orientation="vertical",
        styles={"nav-link-selected": {"background-color": "#008000"}}
    )

# Home Page
if selected == "Home":
    st.title(':green[Emotion Detection from Uploaded Images]')
    st.subheader(':blue[Overview:] Develop a Streamlit application that allows users to upload images and detects emotions using a CNN model. The project aims to build a simple interface and use machine learning for accurate emotion detection.')
    st.subheader(':blue[Skills Take Away:] Python, Convolutional Neural Networks (CNNs), Model Evaluation, Streamlit')

# Detection Page
if selected == "Detection":
    st.title("Emotion Detection from Uploaded Images")

    # Data Paths
    Train_data = 'C:/Users/Priya/OneDrive/Desktop/Emotion Detection/train'
    Test_data = 'C:/Users/Priya/OneDrive/Desktop/Emotion Detection/test'

    # Data Transformations
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])

    # Load Datasets
    train_dataset = datasets.ImageFolder(root=Train_data, transform=transform)
    test_dataset = datasets.ImageFolder(root=Test_data, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define the CNN Model
    class EmotionCNN(nn.Module):
        def __init__(self):
            super(EmotionCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(64 * 12 * 12, 256)
            self.fc2 = nn.Linear(256, len(train_dataset.classes))

        def forward(self, x):
            x = self.pool(nn.functional.relu(self.conv1(x)))
            x = self.pool(nn.functional.relu(self.conv2(x)))
            x = x.view(-1, 64 * 12 * 12)
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Load Model for Inference
    @st.cache_resource
    def load_model():
        model = EmotionCNN()
        model.load_state_dict(torch.load('cnn_emotion_model.pth', weights_only=True))
        model.eval()
        return model

    model = load_model()

    # Predict Emotion Function
    def predict_emotion(image):
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
            return train_dataset.classes[predicted.item()]
        except Exception as e:
            st.error(f"Error in predicting emotion: {e}")
            return None

    # Image Upload Section
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display Uploaded Image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)
        st.write("Detecting the Emotion...")

        # Predict Emotion
        predicted_emotion = predict_emotion(image)
        if predicted_emotion:
            st.markdown(f"### Predicted Emotion: :green[{predicted_emotion}]")
        else:
            st.write("Unable to predict the emotion. Please try again.")

