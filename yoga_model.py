import io
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms
import requests
from io import BytesIO

device = 'cpu'
model_dir = 'models/model.pt'
Yoga_dir = 'models/model_classes.txt'

img_path = 'https://github.com/SalvatoreRa/Yoga_position/blob/main/DALL%C2%B7E%202022-11-15%2015.55.47%20-%20digital%20art%20of%20a%20humanoide%20android%20doing%20yoga%20in%20a%20park,%20high%20quality,%204k.png?raw=true'

capt = 'An android playing yoga in a park. Image created by the author with DALL-E'

def load_test_image():
    uploaded_file = st.file_uploader(label='Upload an image for test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def load_model(model_path):
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    return model


def load_yoga_positions(labels_file):
    with open(labels_file, "r") as f:
        yoga_poses = [s.strip() for s in f.readlines()]
        return yoga_poses


def predict(model, categories, image):
    st.write('loaded image in the model')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image=image.convert('RGB')
    input_tensor = preprocess(image)
    
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch).cpu()

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    probs, yoga_pos = torch.topk(probabilities, len(categories))
    probs, yoga_pos = probs[:5], yoga_pos[:5]
    for i in range(probs.size(0)):
        st.write(categories[yoga_pos[i]], probs[i].item())


def main():
    st.title('YOGA position prediction')
    st.subheader('Upload an image and run the model: it will return most probable yoga position')
    response = requests.get(img_path)
    img_screen = Image.open(BytesIO(response.content))
    st.image(img_screen, caption=capt, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.markdown('This webapp is powered by deeplearning. A convolutional neural network has been trained on a dataset of YOGA images')
    st.markdown('[you can find the coda here](https://github.com/SalvatoreRa/Yoga_position)')
    st.markdown('[here you can find the companion tutorial](https://medium.com/mlearning-ai/make-an-app-with-streamlit-in-minutes-bec48ee19d67)')
    model = load_model(model_dir)
    categories = load_yoga_positions(Yoga_dir)
    image = load_test_image()
    predictions = st.button('Predict the Yoga pose')
    if predictions:
        st.write('Making prediction: please wait')
        predict(model, categories, image)


if __name__ == '__main__':
    main()
