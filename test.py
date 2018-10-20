L = [[1,16, 15, 14, 13], [2, 17, 24, 23, 12],[3, 18, 25, 22, 11], [4, 19, 20, 21, 10], [5, 6, 7, 8, 9] ]


(k1, k2) = 2, 2
rayon = 2



x=0
y=0
lh=0 #ligne haut
lb=0 #ligne bas
hd=0 #hauteur droite
hl=0 #hauteau gauche
sens='d'
#size = (min(2*rayon, 1000), min(2*rayon, 1000))
size = (len(L), len(L[0]))

for k in range(size[0]*size[1]):
    if(size[0]-x-hd>1 and sens=='d') :
        print(L[x][y])
        x+=1
    elif(sens=='b' and y<size[1]-lb) :
        print(L[x][y])
        y+=1
    elif(sens=='g' and x>=hl):
        print(L[x][y])
        x-=1
    elif(sens=='h' and y>=lh):
        print(L[x][y])
        y-=1
    else :
        if(sens=='d'):
            sens='b'
            lh+=1
            print(L[x][y])
            y+=1
        elif(sens=='b') :
            sens='g'
            hd+=1
            print(L[x-1][y-1])
            y=size[1]-lb-1
            x=size[0]-hd-2
        elif(sens=='g'):
            sens='h'
            lb+=1
            y=size[1]-lb-1
            x=hl
            print(L[x][y])
            y-=1
        elif(sens=='h'):
            sens='d'
            hl+=1
            print(L[x+1][lh])
            x+=2
            y=lh





