import numpy as np
import cv2 as cv
import os
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib



################################################################################
#EN: Needed only for function FindLatestFileInDir
#
#CS: Potřebné pouze pro funkci FindLatestFileInDir
import glob

###
#EN: Following functions are used for image manipulation
#
#CS: Následujcí funkce jsou využívany pro manipulaci s obrazovými daty
################################################################################



def rgb2gs(image):
    image = np.sum(image, axis=2) / 3
    image = np.uint8(image)
    return image


#s and l can be changed based on results, same as thresh
def gs2bi(image, thresh, s = 5,  l = 5): 
    maxv = 255
    kernel_o = np.ones((s, s), np.uint8)
    kernel_c = np.ones((l, l), np.uint8)
    first = np.sum(image, axis=0)
    if (first[0]/np.size(image, 0)) < 100:
        image[:, :] = 255 - image[:, :]
    image = np.uint8((np.array(image) < thresh) * maxv)
    image = cv.dilate(image, kernel_o, iterations=1)
    image = cv.erode(image, kernel_o, iterations=2)
    image = cv.dilate(image, kernel_c, iterations=3)
    image = np.uint8(image)
    return image



def rgb_wb(image, image_bi):
    image[:, :, 0] = np.where(image_bi[:, :] != 0, image[:, :, 0], 255)
    image[:, :, 1] = np.where(image_bi[:, :] != 0, image[:, :, 1], 255)
    image[:, :, 2] = np.where(image_bi[:, :] != 0, image[:, :, 2], 255)
    image = np.uint8(image)
    return image



def find_and_cut(image, image_bi):
    number = 0
    contur = cv.findContours(image_bi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contur = contur[0] if len(contur) == 2 else contur[1]
    objects = []
    for c in contur:
        x, y, w, h = cv.boundingRect(c)
        image_crop = np.uint8(image[y:y + h, x:x + w])
        objects.append(image_crop)
        number += 1
    print(f'Found {number} objects')
    return objects



def save_list(objects, path, file_format="jpg", name="", starting_number=1):
    number = starting_number
    for number in range(len(objects)):
        image = objects[number]
        file = f'{name}{number}.{file_format}'
        write = os.path.join(path, file)
        print(write)
        cv.imwrite(write, image)
        number += 1
    print(f'Saved {number} files in {path}')



def same_size(objects, size=0):
    for number in range(len(objects)):
        image = objects[number]
        if size < image.shape[0] or size < image.shape[1]:
            if size < image.shape[0]:
                size = image.shape[0]
            if size < image.shape[1]:
                size = image.shape[1]
        number += 1
    for number in range(len(objects)):
        image = objects[number]
        wanted_height = (size - image.shape[0]) / 2
        width = image.shape[1]
        height = image.shape[0]
        if (height % 2 == 1 and size % 2 == 0) or (height % 2 == 0 and size % 2 == 1):
            x = 255 * np.ones((1, int(width), 3), dtype=np.uint8)
            image = np.concatenate((image, x))
        x = 255 * np.ones((int(wanted_height), int(width), 3), dtype=np.uint8)
        image = np.concatenate((x, image, x))
        height = image.shape[0]
        wanted_width = (size - image.shape[1]) / 2
        if (width % 2 == 1 and size % 2 == 0) or (width % 2 == 0 and size % 2 == 1):
            y = 255 * np.ones((int(height), 1, 3), dtype=np.uint8)
            image = np.concatenate((image, y), axis=1)
        y = 255 * np.ones((int(height), int(wanted_width), 3), dtype=np.uint8)
        image = np.concatenate((y, image, y), axis=1)
        objects[number] = image
    return objects



def outline(image, image_bi, path):
    contur = cv.findContours(image_bi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contur = contur[0] if len(contur) == 2 else contur[1]
    image = np.copy(image)
    file_name = "outline.jpg"
    file_path = os.path.join(path, file_name)
    for c in contur:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(f'Saved {file_name} in {path}')
    cv.imwrite(file_path, image)



def find_latest_file_in_dir(directory):
    file = max(glob.glob(f'{directory}/*.jpg'), key=os.path.getctime)
    print(f'Last file found: {file}')
    return file



################################################################################
#EN: Following functions are used for image classifications
#
#CS: Následujcí funkce jsou využívany pro klasifikaci
################################################################################



def convert_word_to_label(word):
    if word == '1p':
        return 0
    if word == '1z':
        return 1
    if word == '2p':
        return 2
    if word == '2z':
        return 3
    if word == '5p':
        return 4
    if word == '5z':
        return 5
    if word == '10p':
        return 6
    if word == '10z':
        return 7
    if word == '20p':
        return 8
    if word == '20z':
        return 9
    if word == '50p':
        return 10
    if word == '50z':
        return 11
    else:
        return word
    


def convert_label_to_value(label):
    if label == 0:
        return 1
    if label == 1:
        return 1
    if label == 2:
        return 2
    if label == 3:
        return 2
    if label == 4:
        return 5
    if label == 5:
        return 5
    if label == 6:
        return 10
    if label == 7:
        return 10
    if label == 8:
        return 20
    if label == 9:
        return 20
    if label == 10:
        return 50
    if label == 11:
        return 50



def classification_value(data, path_model):
    value = 0
    for i in range(len(data)):
        img = cv.resize(data[i], (50,50))
        img = np.sum(img, axis=2) / 3
        img = np.uint8(img)
        data[i] = hog(img)
    model = joblib.load(path_model)
    pred = model.predict(data)
    for i in range(len(pred)):
        value += convert_label_to_value(pred[i])
        pred[i] = convert_label_to_value(pred[i])
    print(f'Value of coin: {value}\nHodnota mincí: {value}')
    return pred



def classification(data, path_model):
    for i in range(len(data)):
        img = cv.resize(data[i], (50,50))
        img = np.sum(img, axis=2) / 3
        img = np.uint8(img)
        data[i] = hog(img)
    model = joblib.load(path_model)
    pred = model.predict(data)
    print(pred)
    return pred



def creating_model(directory, save_path, number_of_neighbors = 2, split=7):
    t_data = []
    v_data = []
    t_label = []
    v_label = []
    number = 1
    for dirs in os.listdir(directory):
        dir_path = os.path.join(directory, dirs)
        for file in os.listdir(dir_path):
            path = os.path.join(dir_path, file)
            img = cv.imread(path)
            img = cv.resize(img, (50,50))
            img = np.sum(img, axis=2) / 3
            img = np.uint8(img)
            img = hog(img)
            
            #Nutné jen v případě že se jedná o mince, jinak lze odstranit, funkčnost to ale nezmění.
            label = convert_word_to_label(dirs)

            if number <= 7:
                t_data.append(img)
                t_label.append(label)
            else:
                v_data.append(img)
                v_label.append(label)
            if number == 10:
                number = 1
            else:
                number += 1
    print(f'Added {len(t_data)} data to training and {len(v_data)} to testing\nPřidáno {len(t_data)} na učení a {len(v_data)} na testování\n\n')
    create_model(number_of_neighbors, save_path, t_data, t_label, v_data, v_label)



def create_model(number, save_path, t_data, t_label, v_data = [], v_label = []):
    knn = KNeighborsClassifier(n_neighbors=number)
    print(f'{np.shape(t_data[1])}a {len(t_data)} a {t_label[1]}')
    knn.fit(t_data, t_label)
    joblib.dump(knn, save_path)
    if len(v_data) != 0:
        pred = knn.predict(v_data)
        acc = accuracy_score(v_label, pred)
        print(acc)

################################################################################
# EN: Code below is process for manipulation with image data
#
# CS: Kód níže je proces pro manipulaci s obrazovými daty
################################################################################

def image_manipulation(load_path, starting_image, thresh, save_path = None, name = "", size = 0):
    image_load = os.path.join(load_path, starting_image)
    image = cv.imread(image_load)
    image_gs = rgb2gs(image)
    image_bi = gs2bi(image_gs, thresh, )
    image_wb = rgb_wb(image, image_bi)
    #outline(image, image_bi, save_path)    #Pokud chceme uložit i outline je možné jí odkomentovat
    objects = find_and_cut(image_wb, image_bi)
    #objects = same_size(objects, size)           #Pokud chceme změnit velikosti obrázků na stejnou velikost je možné odkomentovat
    if save_path != None:
        save_list(objects, save_path, "jpg", name)
    return objects

################################################################################
# EN: Code below is process for classification, after = you must add your values
# in case of '' fill it like 'this'
#
# CS: Kód níže je proces pro klasifikaci, za = doplnit hodnoty
# v případe '' vyplnit 'takto'
################################################################################
def process_clas():
    load_path = ''
    starting_image = ''
    thresh = 150
    path_model = ''
    
    
    data = image_manipulation(load_path, starting_image, thresh, load_path)
    print(len(data))
    pred = classification(data, path_model)
    print(pred)



################################################################################
# EN: Code below is process for creating a model from folder
#
# CS: Kód níže je proces pro vytvoření modelu ze složky
################################################################################
def process_create_model():
    load_path = ''
    save_path = ''
    creating_model(load_path, save_path)

if __name__ == '__main__':
    #process_clas() / process_create_model()