import truck 
import joblib
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics



def test_IRA(test_size = 100, loop = 10, img_size = 250, clf_name = "", 
         alt_group = ["Cats","Dogs"],
         alt_source = "/main/Data/Testing_data/", 
         alt_temp = "/main/Data/temp/",
          ):
    

    sum = 0
    
    clf_format = truck.os.path.splitext(clf_name)
    print(clf_format)
        
    if clf_format[1] != ".pkl":
        print("The file isn't pickled this cannot be run")
        
    else:
        clf = joblib.load(clf_name)

        for i in range(1,loop+1):
            truck.unload_images(alt_group, alt_source, alt_temp)


            data =[]
            data = truck.load_images(test_size, img_size, alt_group, alt_source, alt_temp)

            test = data[0]
            answers = data[1]

            predict = clf.predict(test)

            score = sklearn.metrics.accuracy_score(predict, answers)

            sum += score
            print(f"This was the score for test {i} was {score * 100}%")

        truck.unload_images(alt_group, alt_source, alt_temp)
        sum = sum/(loop)

        print(f"The average result for {clf_name} = {sum*100}%")   




test_IRA(100, 15, 250, "D:/AI/joblib_jar/IRA_v1590.pkl", ["Cats", "Dogs"], "/main/test/")








