import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import time

# --- Configuration and Setup ---
# Set the page configuration for a wider layout
st.set_page_config(layout="wide", page_title="K-Means Color Quantizer")

def color_quantization(img, k):
    """
    Performs color quantization on an image using K-Means clustering.

    Args:
        img (PIL.Image): The input image.y
        k (int): The number of colors (clusters) to reduce the image to.

    Returns:
        PIL.Image: The color-quantized image.
    """
    st.info(f"Starting K-Means clustering to reduce to {k} colors. This may take a moment...")
    
    # 1. Convert the PIL Image to a NumPy array
    img_array = np.array(img)
    
    # Get original dimensions (Height, Width, Color Channels)
    h, w, c = img_array.shape
    
    # 2. Reshape the 3D array (H x W x C) to a 2D array (H*W x C)
    # Each row is a pixel, and columns are the R, G, B values.
    data_flat = img_array.reshape(h * w, c)
    
    # 3. Initialize and run K-Means
    # n_init='auto' is used for modern scikit-learn compatibility.
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300)
    
    start_time = time.time()
    
    # Fit the model and predict the cluster label for each pixel
    kmeans.fit(data_flat)
    
    end_time = time.time()
    st.success(f"Clustering complete in {end_time - start_time:.2f} seconds.")
    
    # 4. Get the cluster centroids (the new palette colors) and labels
    # Centroids are the average color of each cluster (K x C array)
    centroids = kmeans.cluster_centers_.astype(np.uint8)
    
    # Labels (H*W array) map each pixel to its cluster index (0 to K-1)
    labels = kmeans.labels_
    
    # 5. Replace each pixel with its corresponding centroid color
    # This creates the quantized 2D data (H*W x C)
    quantized_data_flat = centroids[labels]
    
    # 6. Reshape the data back into the original 3D image dimensions
    quantized_img_array = quantized_data_flat.reshape(h, w, c)
    
    # 7. Convert the NumPy array back to a PIL Image
    quantized_img = Image.fromarray(quantized_img_array)
    
    return quantized_img

# --- Streamlit Application Layout ---

st.title("🎨 K-Means Image Color Quantizer")
st.markdown("""
Upload an image and use the slider to select the desired number of colors (K).
The K-Means algorithm will find the optimal K colors and display the reduced image.
""")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    
    # Input for the number of colors (K)
    k_value = st.slider(
        "Select the number of colors (K):", 
        min_value=2, 
        max_value=64, 
        value=8, 
        step=2,
        help="A smaller K results in a more dramatic color reduction."
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image (JPG, PNG)", 
        type=["jpg", "jpeg", "png"]
    )

# Main content area
if uploaded_file is not None:
    
    # Try to open the uploaded image
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
        
        st.header("Results")
        col1, col2 = st.columns(2)
        
        # Display the original image
        with col1:
            st.subheader("Original Image")
            st.image(original_image, use_column_width=True)
            st.caption(f"Dimensions: {original_image.width}x{original_image.height}")
            
        # Process and display the quantized image
        with col2:
            st.subheader(f"Quantized Image (K={k_value})")
            
            # Use st.spinner to show processing feedback
            with st.spinner("Processing image..."):
                quantized_image = color_quantization(original_image, k_value)
            
            st.image(quantized_image, use_column_width=True)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        st.exception(e)

else:
    # Placeholder when no file is uploaded
    st.info("Please upload an image file using the sidebar to begin color quantization.")
    
st.markdown("---")
st.markdown("Developed using Streamlit, scikit-learn, and PIL.")
