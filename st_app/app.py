import streamlit as st

st.set_page_config(
    page_title= 'SIH 1733'
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@300;400;700&family=Roboto+Slab:wght@100;400;900&family=Space+Grotesk:wght@300;400;700&display=swap');

    body {
        font-family: 'Quicksand', sans-serif; /* Apply Quicksand font */
    }

    h1, h5, p {
        font-family: 'Roboto Slab', serif; /* Apply Roboto Slab font for headings and paragraphs */
    }

    .custom-space {
        height: 10px; /* Custom height for empty space */
        width: 100%; /* Full width, adjust as needed */
        margin-bottom: 0em; /* Space below the div, adjust as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: left;'><b>SAR Image Colorization for Comprehensive Insight using Deep Learning Model (h)</b></h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: left;'>Problem Statement no.:1733 </h5>", unsafe_allow_html=True)

st.markdown(
    "<hr style='width: 50%; margin-bottom: 1em; margin-top: 0.5em;'>", 
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<p style='text-align: left;'> SAR is a type of active data collection where a sensor produces its own energy and then records the amount of that energy reflected back after interacting with the Earth. Due to the speckle noise caused by the SAR imaging principle, it is difficult for people to distinguish the ground objects from complex background without professional knowledge. We have made a CNN using UNET architecture to convert the images into a LAB colour format tensor. </p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html= True)

st.markdown("<p style='text-align: left;'> Since the dataset is extensive, we don't have the time to train and test it as it would take us a couple of hours at minimum. Added with the dimensionality issues, it makes it near impossible to do the same.We initially planned to use the ADAMW(Adam with weight decay) optimizer and tuning the learning rate as well as weight decay upon training the data; paired with the Huber Loss loss function.  </p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<p style='text-align: left;'>Because we plan to take this up as a long term project, and after extensive research we have figured out a solution is to exploit Generative Adversarial Networks (GAN) to translate SAR images to optical images which is able to clearly present ground objects with rich color information, i.e., SAR-to-optical image translation.</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<p style='text-align: left;'>The loss functions that we have obtained from a little research that involve the usage of GANs in their architecture are displayed below.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html= True)
st.markdown(
    "<hr style='width: 50%; margin-bottom: 1.5em; margin-top: 1em;'>", 
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html= True)

st.markdown("<h3 style='text-align: left;'><b>An example from the dataset</b></h3>", unsafe_allow_html=True)
st.markdown("<div class='custom-space'></div>", unsafe_allow_html=True)

col1, col2 = st.columns([1,1])
with col1:
    st.image("st_images\project_images\example1_greyscale.png", width = 325)
    st.markdown("<p style='text-align: center; font-size: 18px;'>SAR Image</p>", unsafe_allow_html=True)

with col2:
    st.image("st_images/project_images/example1_colored.png", width = 325)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Optical Image</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


st.markdown("<h3 style='text-align: left;'><b>Loss Function Graph of architectures using GAN</b></h3>", unsafe_allow_html=True)
st.markdown("<div class='custom-space'></div>", unsafe_allow_html=True)

st.image("st_images/project_images/graphs.jpg")

st.markdown("<div class='custom-space'></div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image("st_images/project_images/loss_funcs.jpg")
    st.markdown("<p style='text-align: center; font-size: 18px;'>More Loss Function Graphs</p>", unsafe_allow_html=True)