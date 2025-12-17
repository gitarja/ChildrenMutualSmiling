import os
DATA_PATH = "F://users//prasetia//data//Children//children_sync//data//"

SUMMARY_PKL_FILLIN_PATH = os.path.join(DATA_PATH, "data_fillin_summary.pkl")
SUMMARY_PKL_REMOVENAN_PATH = os.path.join(DATA_PATH, "data_removenan_summary.pkl")
SUMMARY_PKL_REMOVENAN_PATH_02 = os.path.join(DATA_PATH, "data_removenan_summary_02.pkl") # duration 0.2 sec
SUMMARY_PKL_REMOVENAN_PATH_05 = os.path.join(DATA_PATH, "data_removenan_summary_05.pkl") # duration 0.5 sec
SUMMARY_PKL_REMOVENAN_PATH_10 = os.path.join(DATA_PATH, "data_removenan_summary_10.pkl") # duration 1.0 sec
SUMMARY_PKL_REMOVENAN_PATH_20 = os.path.join(DATA_PATH, "data_removenan_summary_20.pkl") # duration 2.0 sec
SUMMARY_PKL_REMOVENAN_PATH_30 = os.path.join(DATA_PATH, "data_removenan_summary_30.pkl") # duration 3.0 sec
SUMMARY_PKL_REMOVENAN_PATH_40 = os.path.join(DATA_PATH, "data_removenan_summary_40.pkl") # duration 4.0 sec

EVENT_PATH = os.path.join(DATA_PATH, "events")

FACE_DETECTION_PATH = os.path.join(DATA_PATH, "face_detection")

SMILE_DETECTION_PATH = os.path.join(DATA_PATH, "smile_detection")

DYADIC_SCORE_PATH = os.path.join(DATA_PATH, "nomination_score", "Dyad_score.csv")
GROUP_SCORE_PATH = os.path.join(DATA_PATH, "nomination_score", "Group_score.csv")
PERSONAL_INFO_SCORE_PATH = os.path.join(DATA_PATH, "nomination_score", "Personal_information.csv")
VALIDATION_INFO_SCORE_PATH = os.path.join(DATA_PATH, "nomination_score", "Validation_information.csv")

IMIGRATION_INFO_SCORE_PATH = os.path.join(DATA_PATH, "nomination_score", "Imigration_information.csv")
LD_INFO_SCORE_PATH = os.path.join(DATA_PATH, "nomination_score", "Learning_disability.csv")

FPS = int(29.97)


# please refer to Effects of temporal dynamics on perceived authenticity of smiles
# the duration of smiles ranges from 3.5 to 4 sec
TAU_MAX = 2 #