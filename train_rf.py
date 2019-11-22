from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import Silence_combined

def get_data():
    matlab_files = ['./NEW/sushant/converted/log_2.mat', './NEW/soham/converted/log_3.mat', './NEW/vin/converted/log_1.mat']

    # # Export sushant's data.
    # sushant_features = []
    # for i in range(1,6):
    #     Silence_combined.load_mat_file('./NEW/sushant/converted/log_'+ str(i) + '.mat')
    #     features = Silence_combined.extract_features(Silence_combined.silence_removal(Silence_combined.butterworth()))
    #     sushant_features.extend(features)
    # Silence_combined.export_sushant_data(sushant_features)

    # # Export soham's data.
    # soham_features = []
    # for i in range(1,6):
    #     Silence_combined.load_mat_file('./NEW/soham/converted/log_'+ str(i) + '.mat')
    #     features = Silence_combined.extract_features(Silence_combined.silence_removal(Silence_combined.butterworth()))
    #     soham_features.extend(features)
    # Silence_combined.export_soham_data(soham_features)

    # # Export vintony's data.
    # vintony_features = []
    # for i in range(1,6):
    #     Silence_combined.load_mat_file('./NEW/vin/converted/log_'+ str(i) + '.mat')
    #     features = Silence_combined.extract_features(Silence_combined.silence_removal(Silence_combined.butterworth()))
    #     vintony_features.extend(features)
    # Silence_combined.export_vintony_data(vintony_features)


    # Export sushant's data.
    Silence_combined.load_mat_file(matlab_files[0])
    features = Silence_combined.extract_features(Silence_combined.silence_removal(Silence_combined.butterworth()))
    Silence_combined.export_sushant_data(features)

    # Export soham's data.
    Silence_combined.load_mat_file(matlab_files[1])
    features = Silence_combined.extract_features(Silence_combined.silence_removal(Silence_combined.butterworth()))
    Silence_combined.export_soham_data(features)

    # Export vintony's data.
    Silence_combined.load_mat_file(matlab_files[2])
    features = Silence_combined.extract_features(Silence_combined.silence_removal(Silence_combined.butterworth()))
    Silence_combined.export_vintony_data(features)


    return Silence_combined.get_data()

if __name__ == "__main__":

    print("Conducting CSI computation and loading variables..")
    X_train, X_test, y_train, y_test = get_data()

    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)

    print("Training classifer...")
    #Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)

    # print(y_test)
    # print(y_pred)

    print("Classfier training complete.")

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    csi_data ={
        'feature_names': [
            'mean',
            'max_val',
            'min_val',
            'skewness',
            'kurtosis_val',
            'variance'
        ],
    }
    feature_imp = pd.Series(clf.feature_importances_,index=csi_data["feature_names"]).sort_values(ascending=False)
    print(feature_imp)

    from joblib import dump
    # save the model to disk
    dump(clf, "CSI_MODEL_RF.joblib")