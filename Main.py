import glob
import os
from PIL import Image
import pickle
import PIL
from tqdm import tqdm
from random import shuffle

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


def distance(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


class Transition:

    def __init__(self, A, B):

        self.M = []
        self.getTransition(A, B)


    def getTransition(self, B, A):

        R = [ [] for x in range(len(A))]
        for i in range(len(R)):
            for k in range(len(A[0])):
                R[i].append(((0, 0), 0))

        for k in range(len(A)):
            for k2 in range(len(A[k])):

                temp = abs(A[k][k2]-B[0][0])
                temp_i = (0, 0)

                for i in range(max(0, k - len(B)//20), min(len(B)-1, k + len(B)//20)):
                    for i2 in range(max(0, k2 - len(B)//20), min(len(B[i])-1, k2 + len(B[i])//20)):

                        if abs(A[k][k2]-B[i][i2]) < temp:
                            temp = abs(A[k][k2]-B[i][i2])
                            temp_i = (i, i2)
                        elif abs(A[k][k2]-B[i][i2]) == temp and distance(temp_i, (k,k2)) > distance((i, i2), (k,k2) ) :
                            temp = abs(A[k][k2]-B[i][i2])
                            temp_i = (i, i2)

                R[k][k2] = (temp_i, B[temp_i[0]][temp_i[1]] - A[k][k2])

        self.M = R

    def __repr__(self):
        return str(self.M)

    def __str__(self):
        return str(self.M)

    def getElem(self, a, b):

        return self.M[a][b]

    def makeTransition(self, Image, i , i2):
        k1 = self.getElem(i, i2)[0][0]
        k2 = self.getElem(i, i2)[0][1]

        return Image[k1][k2] - self.getElem(i, i2)[1]

    def makeTransitionScale(self, Image, i , i2, j1, j2, scale):
        k1 = self.getElem(i, i2)[0][0]
        k2 = self.getElem(i, i2)[0][1]

        return Image[k1*scale + j1][k2*scale + j2] - self.getElem(i, i2)[1]

    def constructImage(self, Image):

        R = [ [] for x in range(len(Image))]
        for i in range(len(R)):
            for k in range(len(Image[0])):
                R[i].append(0)

        for i in range(len(Image)):
            for i2 in range(len(Image[i])):

                R[i][i2] = self.makeTransition(Image, i, i2)

        return R

    def constructImageScale(self, Image, scale, X):



        R = [ [] for x in range(len(Image))]
        for i in range(len(R)):
            for k in range(len(Image[0])):
                R[i].append(Image[i][k])

        if scale == 1:

            for (i,j) in X:
                R[i][j] = self.makeTransitionScale(Image, i, j, 0, 0, scale)

        else:

            for i in range(len(Image)//scale):
                for i2 in range(len(Image[i])//scale):
                    for j in range(scale):
                        for j2 in range(scale):

                            if (i*scale+j, i2*scale+j2) in X:

                                R[i*scale+j][i2*scale+j2] = self.makeTransitionScale(Image, i, i2, j, j2, scale)

        return R

    def getT(self):

        return self.M




class TransitionsMatrix:

    def __init__(self, S):

        self. M = []
        self.getTransitionsMatrix(S)

    def getTransitionsMatrix(self, S):

        for i in tqdm(range(1, len(S))):
            T = Transition(S[i-1], S[i])
            self.M.append(T)

    def constructVideo(self, Image):

        R = [Image]

        for i in range(len(self.M)):
            R.append(self.M[i].constructImage(R[i]))

        return R

    def constructVideoScale(self, Image, scale, timeStrech):

        R = [Image]

        X = [(x,y) for x in range(len(Image)) for y in range(len(Image[0]))]

        for i in tqdm(range(len(self.M))):
            shuffle(X)
            for t in range(timeStrech):
                R.append(self.M[i].constructImageScale(R[i], scale, X[t*len(X)//timeStrech:(t+1)*len(X)//timeStrech]))

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

def train(VideoName, scale):

    name = os.path.splitext(VideoName)[0]
    Folder = "Images/"+os.path.splitext(VideoName)[0]+"/IN"
    os.system("mkdir "+"Images/"+os.path.splitext(VideoName)[0])
    os.system("mkdir "+Folder)

    os.system("ffmpeg -i Videos/"+VideoName+" -vf scale="+str(scale)+":-1 "+Folder+"/%04d.png -hide_banner")


    S = []
    print("Converting ...")
    for file in sorted(glob.glob(Folder+'/*.png')):
        print(file+":")
        S.append(convert_grayscale(Image.open(file)))

    print("Learning ...")
    T = TransitionsMatrix(S)
    pickle.dump(T, open( "TRAINED/"+name+".train", "wb" ) )
    print("DONE")

def genVideo(image, model):

    IN = "Images/"+model+"/IN"
    OUT = "Images/"+model+"/OUT"
    TEMP = "Images/"+model+"/TEMP"

    im = Image.open(IN+'/0001.png')
    width, height = im.size

    os.system("mkdir "+OUT)
    os.system("mkdir "+TEMP)

    im2 = Image.open(image)
    im2 = im2.resize((width, height), PIL.Image.ANTIALIAS)
    im2.save(TEMP+image)

    T = pickle.load( open( "TRAINED/"+model+".train", "rb" ) )

    R = T.constructVideo(convert_grayscale(Image.open(TEMP+image)))

    for i in range(len(R)):
        listToImageGreyScale(R[i], i, OUT)

    os.system("ffmpeg -i '"+OUT+"/%04d.png' -c:v libx264 -preset veryslow -crf 0 OUT/"+str(model)+".mp4 ")
    os.system("open OUT/"+str(model)+".mp4")


def genVideoScale(image, model, scale, timeStrech):

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

    T = pickle.load( open( "TRAINED/"+model+".train", "rb" ) )

    R = T.constructVideoScale(convert_grayscale(Image.open(TEMP+image)), scale, timeStrech)

    for i in range(len(R)):
        listToImageGreyScale(R[i], i, OUT)

    os.system("ffmpeg -i '"+OUT+"/%04d.png' -c:v libx264 -preset veryslow -crf 0 OUT/"+str(model)+".mp4 ")
    os.system("open OUT/"+str(model)+".mp4")    



#train("TEST.mp4", 100)

genVideoScale('IN.jpg', "MOVE", 1, 1)





