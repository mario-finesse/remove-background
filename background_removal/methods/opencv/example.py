import cv2
import numpy as np
from PIL import Image

# load image
image_path = "demo_datasets/gai_images/sdxl-human/000.png"
output_image_path = "output/OpenCV"

image_name = image_path.split("/")[-1]
img = cv2.imread(image_path)
background_image_path = "demo_datasets/gai_images/MODEL-ECOMM-TEMPLATE.jpg"
background_image = Image.open(background_image_path)
raw_image = Image.open(image_path)
background_image = background_image.resize(raw_image.size)


# convert to graky
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold input image as mask
mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]

# negate mask
mask = 255 - mask

# apply morphology to remove isolated extraneous noise
# use borderconstant of black since foreground touches the edges
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# anti-alias the mask -- blur then stretch
mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)
mask_image = Image.fromarray(mask)

segmented_image = Image.composite(raw_image, background_image, mask_image)
im_path = f"{output_image_path}/{image_name}"
segmented_image.save(im_path)

# # put mask into alpha channel
# result = img.copy()
# result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
# result[:, :, 3] = mask

# # save resulting masked image
# cv2.imwrite('person_transp_bckgrnd.png', result)

# # display result, though it won't show transparency
# cv2.imshow("INPUT", img)
# cv2.imshow("GRAY", gray)
# cv2.imshow("MASK", mask)
# cv2.imshow("RESULT", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()