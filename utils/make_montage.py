#
# Creates an .eps montage of a set of .png files. Title of each image is taken
# from the file name.
#

IMG_WSPACE = 0.00       # space between montage columns
IMG_HSPACE = 0.50       # space between montage rows
IMG_WIDTH = 4         # width of individual images
MONTAGE_DPI = 180       # montage DPI
FONT_SIZE = 4           # image title font size

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys, os, re


if (len(sys.argv) < 3):
    print("Usage: python " + sys.argv[0] + " [image folder] [output file name] "
          "[columns (optional)] [font size (optinal)] [row info (optional)")
    exit()

image_dir = sys.argv[1]  # Directory containing images

images = []
for fname in os.listdir(image_dir):
    if fname.endswith('.png'):
        images.append(os.path.join(image_dir, fname))  # Use full path

if len(images) == 0:
    print("No .png images found in the folder " + image_dir + ".")
    exit()

print(images)

# Sort files numerically based on the number in the filename before .png
images = sorted(images, key=lambda s: int(os.path.basename(s).split(".")[0]))

# Read the first image to determine image aspect ratio.
img = mpimg.imread(images[0])  # Now using full path
aspectRatio = img.shape[0] / img.shape[1]


# Find suitable number of columns and rows, or number of rows if columns given.
rows = np.floor( np.sqrt(len(images)) )
cols = np.ceil( len(images) / rows )
if (len(sys.argv) > 3):
    cols = int(sys.argv[3])
    rows = np.ceil(len(images) / int(cols))

# Custom font size, if given.
if (len(sys.argv) > 4):
    FONT_SIZE = int(sys.argv[4])

# Row descriptions, if given.
row_info = []
if (len(sys.argv) > 5):
    f = open(sys.argv[5])
    row_info = f.read().splitlines()

print(row_info)

# Calculate appropriate figure dimensions.
width = cols*IMG_WIDTH + (cols-1)*IMG_WSPACE
height = rows * aspectRatio * IMG_WIDTH + (rows-1)*IMG_HSPACE
# plt.rcParams['figure.facecolor'] = 'black'
fig = plt.figure(figsize = (width, height))
matplotlib.rc('axes', linewidth=0.0)    # hide image borders

print("Figure height: %lf" %height)
print("Figure width: %lf" %width)
print("Input image aspect ratio: %lf" %aspectRatio)

row = 0 # row counter
for i in range(0, len(images)):
    ax = fig.add_subplot(int(rows), int(cols), i+1)
    img = mpimg.imread(images[i])
    ax.imshow(img, interpolation='none')
    # plt.axis('off')
    plt.tick_params(axis='both', which='both', left=False, right=False, bottom=False, labelbottom=False, labelleft=False)
    s = images[i].strip('.png')
    plt.xlabel(s, fontsize=FONT_SIZE)
    
    # Add optional per-row text: 
    if (len(row_info) >= rows and i%cols == 0):
        pad = max([len(f) for f in row_info]) + 2*FONT_SIZE
        # Split line by every other whitespace:
        label = re.sub("( [^ ]*) ", r"\1\n", row_info[row])
        h = plt.ylabel(label, fontsize=FONT_SIZE, labelpad=pad)
        print(row_info[row])
        h.set_rotation(0)
        row += 1

plt.tight_layout()      # shrink white margins around the montage
fig.savefig(sys.argv[2], dpi=MONTAGE_DPI)
