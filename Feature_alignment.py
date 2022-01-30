#This is based around the OpenCV course CV1 where we match features in an image-
#-channels aren't aligned, hence creating distorted images.
# 1. **Step 1**: Read Image
# 2. **Step 2**: Detect Features
# 3. **Step 3**: Match Features
# 4. **Step 4**: Calculate Homography
# 5. **Step 5**: Warping Image
# 6. **Step 6**: Merge Channels
#

import cv2
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')


import matplotlib
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Read 8-bit color image.
# This is an image in which the three channels are
# concatenated vertically.
im =  cv2.imread("emir.jpg", cv2.IMREAD_GRAYSCALE)

# Find the width and height of the color image
sz = im.shape
print(sz)

height = int(sz[0]/3);
width = sz[1]

# Extract the three channels from the gray scale image
# and merge the three channels into one color image
im_color = np.zeros((height,width,3), dtype=np.uint8 )
for i in range(0,3) :
    im_color[:,:,i] = im[ i * height:(i+1) * height,:]


blue = im_color[:,:,0]
green = im_color[:,:,1]
red = im_color[:,:,2]

plt.figure(figsize=(20,12))
plt.subplot(1,3,1)
plt.imshow(blue)
plt.subplot(1,3,2)
plt.imshow(green)
plt.subplot(1,3,3)
plt.imshow(red)
plt.show()

#
# We will align Blue and Red frame to the Green frame.
#
# If you align blue and red channels, it might not give very good results since they are visually very different. You may have to do a lot parameter tuning ( MAX_FEATURES, GOOD_MATCH_PERCENT, etc) to get them aligned. On the other hand, Blue and Green channels are reasonably similar. Thus, taking green as the base channel will produce best results.
#
# We detect ORB features in the 3 frames. Although we need only 4 features to compute the homography, typically hundreds of features are detected in the two images. We control the number of features using the parameter `MAX_FEATURES` in the Python code.

###
### YOUR CODE HERE
MAX_FEATURES = 100000
GOOD_MATCH_PERCENT = 0.00489
###
###
### YOUR CODE HERE
orb = cv2.ORB_create(MAX_FEATURES)
keypointsBlue, descriptorsBlue = orb.detectAndCompute(blue, None)
keypointsRed, descriptorsRed = orb.detectAndCompute(red, None)
keypointsGreen, descriptorsGreen = orb.detectAndCompute(green, None)
###

plt.figure(figsize=[20,10])
img2 = cv2.drawKeypoints(blue, keypointsBlue, None, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.subplot(131);plt.imshow(img2[...,::-1])

img2 = cv2.drawKeypoints(green, keypointsGreen, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.subplot(132);plt.imshow(img2[...,::-1])

img2 = cv2.drawKeypoints(red, keypointsRed, None, color=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.subplot(133);plt.imshow(img2[...,::-1])

# You need to find the matching features in the Green channel and blue/red channel, sort them by goodness of match and keep only a small percentage of original matches. We finally display the good matches on the images and write the file to disk for visual inspection. Use the hamming distance as a measure of similarity between two feature descriptors.
#
# Let's first match features between blue and Green channels.
# Match features.
###
### YOUR CODE HERE
matches = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matchesBlueGreen = matches.match(descriptorsBlue, descriptorsGreen, None)

###
# Match features between blue and Green channels
###
### YOUR CODE HERE
###

# Sort matches by score
matchesBlueGreen.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matchesBlueGreen) * GOOD_MATCH_PERCENT)
matchesBlueGreen = matchesBlueGreen[:numGoodMatches]

# Draw top matches
imMatchesBlueGreen = cv2.drawMatches(blue, keypointsBlue, green, keypointsGreen, matchesBlueGreen, None)

plt.figure(figsize=(12,12))
plt.imshow(imMatchesBlueGreen[:,:,::-1])
plt.show()


# We will repeat the same process for Red and Green channels this time.
# Match features.
###
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matchesRedGreen = matcher.match(descriptorsRed,descriptorsGreen, None)
###

# Match features between Red and Green channels
# Sort matches by score
matchesRedGreen.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matchesRedGreen) * GOOD_MATCH_PERCENT)
matchesRedGreen = matchesRedGreen[:numGoodMatches]

# Draw top matches
imMatchesRedGreen = cv2.drawMatches(red, keypointsRed,green, keypointsGreen, matchesRedGreen, None)

plt.figure(figsize=(12,12))
plt.imshow(imMatchesRedGreen[:,:,::-1])
plt.show()

# Extract location of good matches
points1 = np.zeros((len(matchesBlueGreen), 2), dtype=np.float32)
points2 = np.zeros((len(matchesBlueGreen), 2), dtype=np.float32)

for i, match in enumerate(matchesBlueGreen):
    points1[i, :] = keypointsBlue[match.queryIdx].pt
    points2[i, :] = keypointsGreen[match.trainIdx].pt

# Find homography
hBlueGreen, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
# Extract location of good matches
points3 = np.zeros((len(matchesRedGreen), 2), dtype=np.float32)
points4 = np.zeros((len(matchesRedGreen), 2), dtype=np.float32)

for j, matc in enumerate(matchesRedGreen):
    points3[j, :] = keypointsRed[matc.queryIdx].pt
    points4[j, :] = keypointsGreen[matc.trainIdx].pt

# Find homography
hRedGreen, mask1 = cv2.findHomography(points3, points4, cv2.RANSAC)

# Use homography to find blueWarped and RedWarped images
blueWarped = cv2.warpPerspective(blue, hBlueGreen, (width, height))
redWarped = cv2.warpPerspective(red, hRedGreen, (width, height))

plt.figure(figsize=(10,10))
plt.subplot(121);plt.imshow(blueWarped);plt.title("Blue channel aligned w.r.t green channel")
#plt.subplot(121);plt.imshow(green);plt.title("green channel")
plt.subplot(122);plt.imshow(redWarped);plt.title("Red channel aligned w.r.t green channel")

colorImage = cv2.merge((blueWarped,green,redWarped))

originalImage = cv2.merge((blue,green,red))

plt.figure(figsize=(20,10))
plt.subplot(121);plt.imshow(originalImage[:,:,::-1]);plt.title("Original Mis-aligned Image")
plt.subplot(122);plt.imshow(colorImage[:,:,::-1]);plt.title("Aligned Image")

