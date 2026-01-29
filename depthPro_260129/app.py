
import gradio as gr
import numpy as np
import PIL.Image
import torch
from matplotlib import pyplot as plt

import gradio_client.utils

# Monkeypatch to fix compatibility issue between Gradio and Pydantic v2
# where schemas can be boolean (True/False) instead of dicts.
original_json_schema_to_python_type = gradio_client.utils._json_schema_to_python_type
def safe_json_schema_to_python_type(schema, defs):
    if isinstance(schema, bool):
        return "Any"
    return original_json_schema_to_python_type(schema, defs)
gradio_client.utils._json_schema_to_python_type = safe_json_schema_to_python_type

from depth_pro import create_model_and_transforms, load_rgb

# Load model and preprocessing transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, transform = create_model_and_transforms(
    device=device,
    precision=torch.half,
)
model.eval()

def predict(input_image, background, min_dist, max_dist):
    """
    Function to predict depth and filter background.
    """
    image_orig = input_image
    # The Gradio Image component returns a numpy array. We need to convert it to a PIL image
    # if it's not already one. However, `load_rgb` expects a file path. 
    # Let's save the numpy array as a temporary image file.
    temp_image_path = "/tmp/temp_image.png"
    PIL.Image.fromarray(input_image).save(temp_image_path)
    
    image_orig, _, f_px = load_rgb(temp_image_path)
    image = transform(image_orig)

    # Run inference.
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m].

    # Create depth image with distance scale
    inverse_depth = 1 / depth.detach().cpu().numpy().squeeze()
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
        max_invdepth_vizu - min_invdepth_vizu
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(inverse_depth_normalized, cmap="turbo")
    fig.colorbar(im, ax=ax, label="Inverse Depth")
    ax.set_title("Depth Map")
    
    fig.canvas.draw()
    depth_img_with_scale = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    depth_img_with_scale = depth_img_with_scale.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)

    # Filter original image
    out_of_range = ~((depth < min_dist) | (depth > max_dist))
    mask_np = out_of_range.detach().cpu().numpy().squeeze()
    
    # The mask might be a different size than the image, so we need to resize it.
    mask_pil = PIL.Image.fromarray(mask_np.astype(np.uint8) * 255)
    mask_pil = mask_pil.resize(image_orig.shape[:2][::-1], PIL.Image.NEAREST)
    mask_np = np.array(mask_pil) / 255.0


    image_filtered = image_orig.copy()
    if background == 'Black':
        image_filtered = image_filtered * mask_np[:, :, np.newaxis]
    elif background == 'White':
        bg_mask = (mask_np == 0)
        image_filtered[bg_mask] = 255
    
    image_filtered = image_filtered.astype(np.uint8)
    
    return depth_img_with_scale, PIL.Image.fromarray(image_filtered)

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# DepthPro Hugging Face Demo")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Input Image")
            background = gr.Radio(["Black", "White"], label="Background Color", value="Black")
            min_dist = gr.Number(label="Min Distance (m)", value=0.0)
            max_dist = gr.Number(label="Max Distance (m)", value=1.8)
            submit_btn = gr.Button("Submit")
        with gr.Column():
            output_depth = gr.Image(label="Depth Map with Distance Scale")
            output_filtered = gr.Image(label="Filtered Original Image")

    submit_btn.click(
        fn=predict,
        inputs=[input_image, background, min_dist, max_dist],
        outputs=[output_depth, output_filtered],
    )
    
    gr.Examples(
        examples=[
            ["IMG_7047.jpg", "Black", 0.0, 1.8],
            ["data/example.jpg", "White", 0.5, 2.0],
        ],
        inputs=[input_image, background, min_dist, max_dist],
        outputs=[output_depth, output_filtered],
        fn=predict,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False, server_port=7861)
