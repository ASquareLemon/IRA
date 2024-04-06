import os
import numpy
import shutil 
import random
import skimage

def load_images(loading_size=0,
                img_length = 250, 
                subfolders = ["Cats","Dogs"],
                input_dir = "/main/Data/training_data/", 
                temp_dir = "/main/Data/temp/"):

    main_path = os.getcwd().replace("\\","/")
    input_dir = main_path + input_dir
    temp_dir = main_path + temp_dir
    dump_path = main_path + "/dump/"
    
    data = []
    labels =[]

    for folder in subfolders:

        if loading_size + 10 > len(os.listdir(input_dir+ folder)):
            print("Reloading all images")
            unload_images()
            
    for category_idx, folder in enumerate(subfolders):
        image_list = []

        if os.path.exists(temp_dir) != True:
            os.mkdir(temp_dir)

        if os.path.exists(temp_dir+folder) != True:
            os.mkdir(temp_dir + folder)

        for file in os.listdir(input_dir+folder):
            image_list.append(file)
        
        random.shuffle(image_list)
        
        counter = 0
        for image in image_list:
            img_path = os.path.join(input_dir,folder,image).replace("\\","/")
            temp_path = os.path.join(temp_dir,folder,image).replace("\\","/")

            try: 
                img = skimage.io.imread(img_path)
         
                img = skimage.transform.resize(img,(img_length, img_length))

                if img.shape != (img_length,img_length,3) or img.size != img_length **2 * 3:

                    raise ValueError ("The image failed to properly set ")
            except(ValueError, EOFError, OSError): 
                if os.path.exists(dump_path) != True:
                    os.mkdir(dump_path)
                    

                if os.path.exists(dump_path + folder) != True:
                    os.mkdir(dump_path + folder)


                dump_dir = str(os.path.join(dump_path, folder, file).replace("\\","/"))


                shutil.move(img_path, dump_dir)

                print(f"The file {file} has been moved to the directory {dump_path}")
            
                continue
                

            data.append(img.flatten())
            labels.append(category_idx)

            shutil.move(img_path, temp_path)

            counter +=1
            if counter == loading_size:
                break

    data = numpy.asarray(data)
    labels = numpy.asanyarray(labels)

    return [data, labels]

def unload_images(subfolders = ["Cats", "Dogs"],
                input_dir = "/main/Data/training_data/", 
                temp_dir = "/main/Data/temp/"):
    
    main_path = os.getcwd().replace("\\","/")

    input_dir = main_path + input_dir
    temp_dir = main_path + temp_dir
    
    for folder in subfolders:
        if os.path.exists(temp_dir) != True:
            os.mkdir(temp_dir)
        if os.path.exists(temp_dir + folder) != True:
            os.mkdir(temp_dir + folder)

        for file in os.listdir(temp_dir + folder):
            new_path = str(os.path.join(input_dir,folder, file)).replace("\\", "/")
            old_path = str(os.path.join(temp_dir, folder, file)).replace("\\","/")

            shutil.move(old_path, new_path)

