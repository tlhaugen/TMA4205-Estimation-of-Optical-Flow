import numpy as np
import matplotlib.colors as mcolors

def mycomputeColor(u, v):
    '''
    Construct an rgb image representing the flow field
    
    Input:
    u - first component of the flow field
    v - second component of the flow field
    
    Output: 
    img - rgb image representing the flow field

    saturation and value of the depiction are given by the size of the flow
    field; sizes are scaled to values between 0 and 1.
    '''

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    saturation = np.sqrt(u**2 + v**2)
    saturation_max = np.max(saturation)
    
    if saturation_max > 0:
        saturation_scaled = saturation / saturation_max
    else:
        saturation_scaled = np.zeros_like(saturation)

    # hue is given by the direction of the flow field. The components of the
    # flow field are interpreted as complex numbers (u + iv). As a first step,
    # we compute their (principal) square root us + i vs.
    
    magnitude_uv = np.sqrt(u**2 + v**2)

    us = np.sqrt((u + magnitude_uv) / 2)
    vs = np.sign(v) * np.sqrt((-u + magnitude_uv) / 2)

    # Now we define the hue as the argument of us + i vs, scaled to values
    # between 0 and 1
    with np.errstate(divide='ignore', invalid='ignore'):
        hue = (np.pi / 2 - np.arctan(us / vs)) / np.pi

    # Handle special values for hue
    hue[hue == np.inf] = 0
    hue[hue == -np.inf] = 1
    hue[np.isnan(hue)] = 0.5 # Handles 0/0 case where us/vs -> NaN

    # Set up the flow field as hsv image
    value = 1.0 - (saturation_scaled * (1.0 - saturation_scaled))**2
    img_hsv = np.stack([hue, saturation_scaled, value], axis=-1)

    # Convert hsv to rgb
    img_rgb = mcolors.hsv_to_rgb(img_hsv)

    return img_rgb
                       
def mycolorwheel(n):
    '''
    Generate a color wheel
    '''
    x = np.arange(-n,n+1)/n
    y = np.arange(-n,n+1)/n
    XX,YY = np.meshgrid(x,y)
    circle = XX**2+YY**2 <= 1
    UU = XX*circle
    VV = YY*circle
    img = mycomputeColor(UU,VV)
    return img


def generate_test_image(n, testcase=1):
    '''
    Generates test images
    testcase = 1: One Gaussian moving to the lower right
    testcase = 2: Two Gaussians circling around the center
    '''
    x = list(range(1,n+1))   
    Y, X = np.meshgrid(x,x)

    gauss = lambda x, y, sigma: 255*np.exp(-(x**2+y**2)/(2*sigma**2))

    if testcase == 1:
        # One Gaussian moving to the lower right
        pos_x, pos_y = 0.48, 0.49
        sigma = 0.15
        dx, dy = 0.04, 0.02

        I1 = gauss(X-n*pos_x, Y-n*pos_y, n*sigma)
        I2 = gauss(X-n*(pos_x+dx), Y-n*(pos_y+dy), n*sigma)
        
    elif testcase == 2:
        # Two Gaussians circling around the center
        pos_x, pos_y = 0.5, 0.3
        sigma = 0.05
        dx, dy = 0.05, 0.05

        I1_1 = gauss(X-n*pos_x, Y-n*pos_y, n*sigma)
        I2_1 = gauss(X-n*(pos_x+dx), Y-n*(pos_y+dy), n*sigma)

        pos_x, pos_y = 0.5, 0.7
        sigma = 0.1
        dx, dy = -0.05, -0.05

        I1_2 = gauss(X-n*pos_x, Y-n*pos_y, n*sigma)
        I2_2 = gauss(X-n*(pos_x+dx), Y-n*(pos_y+dy), n*sigma)

        I1 = np.maximum(I1_1,I1_2)
        I2 = np.maximum(I2_1,I2_2)
    else:
        raise ValueError(f"Testcase {testcase} is not defined.")
        
    return I1, I2