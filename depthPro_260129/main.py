"""
subinjoo depthPro simple project
260129

https://github.com/apple/ml-depth-pro
conda activate depth-pro
"""

from PIL import Image
import depth_pro
import numpy as np
import torch

# Load model and preprocessing transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, transform = depth_pro.create_model_and_transforms(device=device, precision=torch.half)
model.eval()

# Load and preprocess an image.
image_orig, _, f_px = depth_pro.load_rgb('IMG_7047.jpg')
image = transform(image_orig)

# Run inference.
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.

depth_img = depth.detach().cpu().numpy()
result_image = Image.fromarray(depth_img)
result_image.show()


# 2. 거리 필터 설정
min_dist = 0.0 
max_dist = 1.8
backGround = 'black' # 'black' or 'white'

# 조건: 거리 범위를 벗어나는 영역 찾기
out_of_range = ~( (depth < min_dist) | (depth > max_dist) )
mask_np = out_of_range.detach().cpu().numpy()

if(backGround == 'black'): 
    image_orig *= mask_np[:, :, np.newaxis]
elif(backGround == 'white'):
    # 1. 마스크가 0인 곳(배경)을 True로 만듭니다.
    bg_mask = (mask_np == 0)
    # 2. 배경 영역에 해당하는 픽셀들을 흰색(255)으로 설정합니다.
    image_orig[bg_mask] = 255
    image_orig = image_orig.astype(np.uint8)

# 3. 이제 PIL 이미지로 변환이 가능합니다.
result_image = Image.fromarray(image_orig)
result_image.show()
