
# Imagesize
IMAGESIZE_X = 1024
IMAGESIZE_Y = IMAGESIZE_X

BACKGROUNDCOLOR=240 # 138 roughly backgroundcolor of the LDM images
EMPTYIMAGECOLOR=0   # 0=black

PREFIX = 'Os7-S1 Camera'

# mini
DBPATH = './data/output/mini/particlewise.feather'
CROPPEDIMAGEPATH = './data/output/mini/cropped_images'
IMAGEPATH = './data/cache/mini/images'
PATHORDERED = './data/Good_Particle_Ordered.csv'


# FOR FEATURE CALCULATION
threshold_constant = 0.6313  #160/255 #to make it binary for pixelcount

# feather
# DBPATH = '/home/auth/Documents/Projekte/aid_icaps/output/texus/particlewise.feather'
# CROPPEDIMAGEPATH = '/home/auth/Documents/Projekte/aid_icaps/output/texus/cropped_images'
# IMAGEPATH = '/home/auth/Documents/Projekte/aid_icaps/cache/texus/images'

# FOR ORDERING
weight_size = 0.8  # [0...1]
weight_sharpness = 0.5  # [0...1]

# INTERACTIVE FEATURESPACE
INPUTFILE = r'./data/output/mini/particlewise.feather'
SCALE=1
DATAPATH = r'./data/output/mini/cropped_images'
