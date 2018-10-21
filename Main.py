import glob
import os
from PIL import Image
import pickle
import PIL
from tqdm import tqdm
from random import shuffle
import math

def open_image(path):
  newImage = Image.open(path)
  return newImage

# Save Image
def save_image(image, path):
  image.save(path, 'png')

# Create a new image with the given size
def create_image(i, j):
  image = Image.new("RGB", (i, j), "white")
  return image

# Get the pixel from the given image
def get_pixel(image, i, j):
    # Inside image bounds?
    width, height = image.size
    if i > width or j > height:
      return None

    # Get Pixel
    pixel = image.getpixel((i, j))
    return pixel


# Create a Grayscale version of the image
def convert_grayscale(image):
  # Get size
    width, height = image.size

  # Create new Image and a Pixel Map
    pixels = [ [] for x in range(width)]
    for i in range(len(pixels)):
        for k in range(height):
            pixels[i].append(0)

  # Transform to grayscale
    for i in range(width):
        for j in range(height):
      # Get Pixel
            pixel = get_pixel(image, i, j)

            # Get R, G, B values (This are int from 0 to 255)
            red =   pixel[0]
            green = pixel[1]
            blue =  pixel[2]

            # Transform to grayscale
            gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)

            # Set Pixel in new image
            pixels[i][j] = int(gray)

    # Return new image
    return pixels

def convert_color(image):
  # Get size
    width, height = image.size

  # Create new Image and a Pixel Map
    pixels = [ [] for x in range(width)]
    for i in range(len(pixels)):
        for k in range(height):
            pixels[i].append(0)

  # Transform to grayscale
    for i in range(width):
        for j in range(height):
      # Get Pixel
            pixel = get_pixel(image, i, j)

            # Get R, G, B values (This are int from 0 to 255)
            red =   pixel[0]
            green = pixel[1]
            blue =  pixel[2]

            # Set Pixel in new image
            pixels[i][j] = (red, green, blue)

    # Return new image
    return pixels


def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + abs(a[1]-b[1])**2)


class Pixel:

    def __init__(self, R, V, G):

        self.R = R
        self.V = V
        self.G = G

class PixelTransition:

    def __init__(self, ind, pixel1, pixel2, color = False):
        self.ind = ind
        self.color = color

        if self.color:
            (r1, v1, b1) = pixel1
            (r2, v2, b2) = pixel2

            self.R = r1 - r2
            self.V = v1 - v2
            self.B = b1 - b2
        else:
            self.NB = pixel1 - pixel2


class Transition:

    def __init__(self, A, B, color = False):

        self.M = []
        self.color = color
        self.getTransition(A, B)

    def pixelDistance(self, val1, val2):

        if self.color:
            (r1, v1, b1) = val1
            (r2, v2, b2) = val2

            return math.sqrt((r1-r2)**2 + (r1-r2)**2 + (r1-r2)**2)

        else:
            return abs(val1 - val2)




    def getTransition(self, B, A):

        R = [ [] for x in range(len(A))]
        for i in range(len(R)):
            for k in range(len(A[0])):
                R[i].append(((0, 0), 0))

        for k in range(len(A)):
            for k2 in range(len(A[k])):

                temp = self.pixelDistance(A[k][k2], B[0][0])
                temp_i = (0, 0)

                for i in range(max(0, k - int(math.log2(len(B))) ), min(len(B)-1, k + int(math.log2(len(B))) )):
                    for i2 in range(max(0, k2 - int(math.log2(len(B[i]))) ), min(len(B[i])-1, k2 + int(math.log2(len(B[i]))) )):

                        if self.pixelDistance(A[k][k2], B[i][i2]) < temp:
                            temp = self.pixelDistance(A[k][k2], B[i][i2])
                            temp_i = (i, i2)
                        elif self.pixelDistance(A[k][k2], B[i][i2]) == temp and distance(temp_i, (k,k2)) > distance((i, i2), (k,k2) ) :
                            temp = self.pixelDistance(A[k][k2], B[i][i2])
                            temp_i = (i, i2)

                R[k][k2] = PixelTransition(temp_i, B[temp_i[0]][temp_i[1]], A[k][k2], self.color)

        self.M = R

    def __repr__(self):
        return str(self.M)

    def __str__(self):
        return str(self.M)

    def getElem(self, a, b):

        return self.M[a][b]

    def makeTransitionScale(self, Image, x , y, scale):

        if self.color:
            i = x // scale
            i2 = y // scale
            j1 = x % scale
            j2 = y % scale

            (k1, k2) = self.getElem(i, i2).ind
            (r0, r1, r2) = Image[k1*scale + j1][k2*scale + j2]
            r = r0 - self.getElem(i, i2).R
            v = r1 - self.getElem(i, i2).V
            b = r2 - self.getElem(i, i2).B

            return (r,v,b)
        else:
            (k1, k2) = self.getElem(i, i2).ind

            return Image[k1*scale + j1][k2*scale + j2] - self.getElem(i, i2).NB


    def constructImageScale(self, Image, scale, X):



        R = [ [] for x in range(len(Image))]
        for i in range(len(R)):
            for k in range(len(Image[0])):
                R[i].append(Image[i][k])


        for (i,j) in X:
            R[i][j] = self.makeTransitionScale(Image, i, j, scale)


        return R

    def getT(self):

        return self.M




class TransitionsMatrix:

    def __init__(self, S, color = False):

        self. M = []
        self.color = color
        self.getTransitionsMatrix(S)

    def getTransitionsMatrix(self, S):

        for i in tqdm(range(1, len(S))):
            T = Transition(S[i-1], S[i], self.color)
            self.M.append(T)


    def constructVideoScale(self, Image, scale, timeStrech):

        R = [Image]

        X = [(x,y) for x in range(len(Image)) for y in range(len(Image[0]))]

        for i in tqdm(range(len(self.M))):
            shuffle(X)
            for t in range(timeStrech):
                R.append(self.M[i].constructImageScale(R[(i*timeStrech)+t], scale, X[t*len(X)//timeStrech:(t+1)*len(X)//timeStrech]))

        return R

    def __str__(self):

        return str(self.M)




def listToImageGreyScale(L, i, OUT):

    im = Image.new("RGB", (len(L), len(L[0])))
    pix = im.load()
    for x in range(len(L)):
        for y in range(len(L[x])):
            pix[x,y] = (L[x][y], L[x][y], L[x][y])

    i = '0'*(4-len(str(i))) + str(i)

    im.save(OUT+"/"+str(i)+".png", "PNG")

def listToImageColor(L, i, OUT):

    im = Image.new("RGB", (len(L), len(L[0])))
    pix = im.load()
    for x in range(len(L)):
        for y in range(len(L[x])):
            pix[x,y] = L[x][y]

    i = '0'*(4-len(str(i))) + str(i)

    im.save(OUT+"/"+str(i)+".png", "PNG")

def train(VideoName, scale, color = False):

    name = os.path.splitext(VideoName)[0]
    Folder = "Images/"+os.path.splitext(VideoName)[0]+"/IN"
    os.system("mkdir "+"Images/"+os.path.splitext(VideoName)[0])
    os.system("mkdir "+Folder)

    os.system("ffmpeg -i Videos/"+VideoName+" -vf scale="+str(scale)+":-1 "+Folder+"/%04d.png -hide_banner")


    S = []
    print("Converting ...")
    for file in sorted(glob.glob(Folder+'/*.png')):
        print(file+":")
        if (color):
            S.append(convert_color(Image.open(file)))

        else:
            S.append(convert_grayscale(Image.open(file)))

    print("Learning ...")
    T = TransitionsMatrix(S, color = color)
    pickle.dump((color,T), open( "TRAINED/"+name+".train", "wb" ) )
    print("DONE")


def genVideo(image, model, scale = 1, timeStrech = 1):

    IN = "Images/"+model+"/IN"
    OUT = "Images/"+model+"/OUT"
    TEMP = "Images/"+model+"/TEMP"

    im = Image.open(IN+'/0001.png')
    width, height = im.size

    os.system("mkdir "+OUT)
    os.system("mkdir "+TEMP)

    im2 = Image.open(image)
    im2 = im2.resize((width*scale, height*scale), PIL.Image.ANTIALIAS)
    im2.save(TEMP+image)

    (color,T) = pickle.load( open( "TRAINED/"+model+".train", "rb" ) )

    if color:
        R = T.constructVideoScale(convert_color(Image.open(TEMP+image)), scale, timeStrech)

    else:
        R = T.constructVideoScale(convert_grayscale(Image.open(TEMP+image)), scale, timeStrech)

    if color:
        for i in tqdm(range(len(R))):
            listToImageColor(R[i], i, OUT)

    else:
        for i in tqdm(range(len(R))):
            listToImageGreyScale(R[i], i, OUT)


    os.system("ffmpeg -i '"+OUT+"/%04d.png' -c:v libx264 -preset veryslow -crf 0 OUT/"+str(model)+".mp4 ")
    os.system("open OUT/"+str(model)+".mp4")    



train("2.mp4", 500, color = True)

genVideo('IN.jpg', "2", scale = 2, timeStrech = 20)





