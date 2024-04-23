import truck
import joblib
import sklearn.linear_model
import sklearn.model_selection
import sklearn.metrics


def teacher(batches = 1, batch_size = 10, img_size = 250, clf = "", 
         alt_group = ["Cats","Dogs"], alt_source = "/main/Data/Training_data/", 
         alt_temp = "/main/Data/temp/" ):

    truck.unload_images()
     
    clf_format = truck.os.path.splitext(clf)

    if clf_format[1] != ".pkl":
        print("no classifier found")
        clf = sklearn.linear_model.SGDClassifier()
        continue_clf = False
    else:
        print(f"loading {clf}")
        clf = joblib.load(clf)
        continue_clf= True
    
    h_score = 0
    h_batch = 0
    avg = []

    joblib_path = truck.os.getcwd().replace("\\","/") + "/Joblib_jar"

    if truck.os.path.exists(joblib_path) != True:
        truck.os.mkdir(joblib_path)

    starting_num = len(truck.os.listdir(joblib_path))

    print(f"continuing from {starting_num}")


    for i in range(starting_num+1, batches + starting_num + 1 ):
        data = []

        data = truck.load_images(batch_size, img_size, alt_group, alt_source, alt_temp)

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data[0],data[1],test_size = 0.2,
                                                                                    shuffle = True,stratify = data[1])
        
        if continue_clf == False:
            print("The training has begun")
            clf.fit(x_train,y_train)

        elif i == 1:
            if continue_clf == True:
                print("Continuing training")
                clf.partial_fit(x_train,y_train)

        else:
            clf.partial_fit(x_train,y_train)

        predict = clf.predict(x_test)

        score = sklearn.metrics.accuracy_score(predict, y_test)*100

        if score >= h_score:
                h_score = score
                h_batch = i

        
        avg.append(score)
        

        
        print(f"The test number {i} is complete, The score is {score}%")


        joblib.dump(clf, joblib_path + f"/IRA_v{i}.pkl")

        

        if i == batches:
            print(f"the last file was saved as /IRA_v{i}.pkl")   
            break
    
    print("The Image recognition AI finished training")
    print(f"The HIGHEST score IRA got ever was {h_score}% at batch {h_batch}!")

    if batches >= 10:
        break_point = 10
    else:
        break_point = batches
        
    avg_num = 0
    c = 0
    for i in avg:
        avg_num  += i
        c += 1
        if c == break_point:
            break
    print(f"The average score over the last 10 batches was {avg_num/break_point}%")

    if batches >= 100:
        break_point = 100
    else:
        break_point = batches
        
    avg_num= 0
    c = 0
    for i in avg: 
        avg_num += i
        c+=1
        if c == break_point:
            break
        
    print(f"The average score over the last 100 batches was {avg_num/break_point}%")

    avg_num = 0
    for i in avg:
        avg_num += i

    print(f"overall average score was {avg_num/batches}% ")
    
    truck.unload_images()
    print(f"AI learning complete")



source = truck.os.getcwd().replace("\\","/") + "/Joblib_jar"

if truck.os.path.exists(source) != True:
        truck.os.mkdir(source)
        
num = len(truck.os.listdir(source))


teacher(20,10,250, f"{source}/IRA_V{num}.pkl" )

