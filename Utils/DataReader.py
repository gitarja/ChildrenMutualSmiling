import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Utils.Conf import FACE_DETECTION_PATH, EVENT_PATH, SMILE_DETECTION_PATH, FPS, DYADIC_SCORE_PATH, GROUP_SCORE_PATH, \
    PERSONAL_INFO_SCORE_PATH, DATA_PATH, VALIDATION_INFO_SCORE_PATH, IMIGRATION_INFO_SCORE_PATH, LD_INFO_SCORE_PATH
from sklearn.impute import KNNImputer


class DataReader:

    def __init__(self, results_path=""):
        self.results_path = results_path

    def episodeToStream(self, file_face, file_event, event_stream):
        '''
        :param file:
        :return:
        accept=1
        reject=2
        '''
        face_stream = pd.read_pickle(file_face)
        df = pd.read_pickle(file_event)
        df = df.loc[df["status"] == 1]
        stream = np.zeros(len(face_stream))
        for index, row in df.iterrows():
            start_idx = row["start"]
            stop_idx = row["stop"]

            stream[start_idx:stop_idx] = 1

        return df, stream * event_stream

    def getGroupScore(self, fill_in=True):
        df = pd.read_csv(GROUP_SCORE_PATH)
        after_story_df = df[df["time_point"] == "after_story"]
        pre_story_df = df[df["time_point"] == "pre"]
        post_story_df = df[df["time_point"] == "post"]

        # sort the dfs
        after_story_df = after_story_df.sort_values("id")
        pre_story_df = pre_story_df.sort_values("id")
        post_story_df = post_story_df.sort_values("id")

        group = [word.replace('KG', '') for word in pre_story_df["kg"]]
        subject = [word.split("_")[-1] for word in pre_story_df["id"]]

        # pre
        pre_ios_group = pre_story_df["IOS_group"].values
        populatiry = pre_story_df["popularity_score"].values
        leadership = pre_story_df["leadership_score"].values
        # after story
        after_ios_group = after_story_df["IOS_group"].values





        new_df = pd.DataFrame({
            "group": group, "subject": subject,
            "pre_ios_group": pre_ios_group,
            "popularity": populatiry,
            "leadership": leadership,
            "after_ios_group": after_ios_group,


        })
        if fill_in:
            imputer = KNNImputer(n_neighbors=3)
            new_df[["pre_ios_group", "popularity", "leadership", "after_ios_group"]] = imputer.fit_transform(
                new_df[["pre_ios_group", "popularity", "leadership", "after_ios_group"]])
        return new_df

    def getValidationInfo(self):
        df = pd.read_csv(VALIDATION_INFO_SCORE_PATH)
        group = [word.replace('KG', '').rjust(2, "0") for word in df["Group"]]
        validator_indx, validator_unique_ids = pd.factorize(df["Validator"])
        new_df = pd.DataFrame({
            "group": group,
            "validator": validator_indx,

        })
        return new_df

    def getIndividualInfo(self, fill_in=True):
        df = pd.read_csv(PERSONAL_INFO_SCORE_PATH)
        group = [word.replace('KG', '') for word in df["kg"]]
        subject = [word.split("_")[-1] for word in df["id"]]

        subject_code = df["code"]

        group_count = df["group_count"].values
        age = df["age"].values
        gender = df["gender"].values
        bullying = df["mobb"].values
        prosociality = df["prosociality"].values
        extraversion = df["extraversion"].values
        feeling_safe = df["feeling_safe_3"].values
        feeling_happy = df["feeling_happy_3"].values

        feeling_stressed = df["feeling_stressed_3"].values
        collaboration_climate = df["collaboration"].values

        group_climate_pos = 0.5 * (feeling_safe + feeling_happy)
        group_climate_neg = feeling_stressed

        new_df = pd.DataFrame({
            "group": group, "subject": subject,
            "subject_code": subject_code,
            "group_count": group_count, "age": age,
            "gender": gender,
            "bullying": bullying,
            "prosociality": prosociality,
            "extraversion": extraversion,
            "feeling_safe": feeling_safe,
            "feeling_happy": feeling_happy,
            "feeling_stressed": feeling_stressed,
            "group_collaboration": collaboration_climate,
            "group_climate_pos": group_climate_pos,
            "group_climate_neg": group_climate_neg

        })
        new_df['gender'].replace(['male', 'female'],
                                 [1, 0], inplace=True)
        if fill_in:
            imputer = KNNImputer(n_neighbors=3)
            new_df[["age"]] = imputer.fit_transform(new_df[["age"]])

        return new_df

    def getImigrationInfo(self):
        '''
        :return:
        0: pure german
        1: has imigration background
        2: an imigrant
        '''
        df = pd.read_csv(IMIGRATION_INFO_SCORE_PATH)

        columns = df.columns

        subject_code = df["Geheimcode"]
        german_nationality = df["Nationalitaet_0"]

        other_nationality = np.nansum(df[columns[2:-1]].values, axis=-1)

        nationality = np.zeros(len(german_nationality))

        imig_b = np.argwhere((german_nationality == 1) & (other_nationality >= 1))
        imig = np.argwhere((german_nationality == 0) & (other_nationality >= 1))

        nationality[imig_b] = 1
        nationality[imig] = 2

        new_df = pd.DataFrame({"subject_code": subject_code, "nationality": nationality})

        return new_df

    def getLearningDisbInfo(self):
        df = pd.read_csv(IMIGRATION_INFO_SCORE_PATH)

    def getDyadicScore(self, fill_in=True):

        df = pd.read_csv(DYADIC_SCORE_PATH)
        pre_df = df[df["time_point"] == "pre"]
        after_df = df[df["time_point"] == "after_story"]
        post_df = df[df["time_point"] == "post"]

        group_list = [word.replace('KG', '') for word in pre_df["kg"]]
        subject_01 = [word.split("_")[-1] for word in pre_df["child1"]]
        subject_02 = [word.split("_")[-1] for word in pre_df["child2"]]

        # pre score
        # friendship score
        pre_friendship_12 = pre_df["friendship_1_2"].values
        pre_friendship_21 = pre_df["friendship_2_1"].values
        # ios score
        pre_ios_12 = pre_df["ios_1_2"].values.astype(float)
        pre_ios_21 = pre_df["ios_2_1"].values.astype(float)
        # nice score
        pre_nice_12 = pre_df["nice_1_2"].values.astype(float)
        pre_nice_21 = pre_df["nice_2_1"].values.astype(float)
        # interesting score
        pre_interesting_12 = pre_df["interesting_1_2"].values.astype(float)
        pre_interesting_21 = pre_df["interesting_2_1"].values.astype(float)
        # boring score
        pre_boring_12 = pre_df["boring_1_2"].values.astype(float)
        pre_boring_21 = pre_df["boring_2_1"].values.astype(float)
        # likability score
        pre_likability_12 = pre_df["likability_score_1_2"].values.astype(float)
        pre_likability_21 = pre_df["likability_score_2_1"].values.astype(float)
        # print(df)
        # convert to score format

        # helpful
        pre_helpful_12 = pre_df["helpful_1_2"].values.astype(float)
        pre_helpful_21 = pre_df["helpful_2_1"].values.astype(float)

        # after score
        after_ios_12 = after_df["ios_1_2"].values.astype(float)
        after_ios_21 = after_df["ios_2_1"].values.astype(float)

        # after helpful
        after_helpful_12 = after_df["helpful_1_2"].values.astype(float)
        after_helpful_21 = after_df["helpful_2_1"].values.astype(float)

        # after helpful
        after_trust_12 = after_df["trust_1_2"].values.astype(float)
        after_trust_21 = after_df["trust_2_1"].values.astype(float)

        # post score

        group = np.concatenate([group_list, group_list])
        subject1 = np.concatenate([subject_01, subject_02])
        subject2 = np.concatenate([subject_02, subject_01])
        pre_friendship_score = np.concatenate([pre_friendship_12, pre_friendship_21])
        pre_friendship_score[pre_friendship_score == 0] = "no"
        pre_ios_score = np.concatenate([pre_ios_12, pre_ios_21])

        pre_nice_score = np.concatenate([pre_nice_12, pre_nice_21])
        pre_interesting_score = np.concatenate([pre_interesting_12, pre_interesting_21])
        pre_boring_score = np.concatenate([pre_boring_12, pre_boring_21])
        pre_likability_score = np.concatenate([pre_likability_12, pre_likability_21])
        pre_helpful_score = np.concatenate([pre_helpful_12, pre_helpful_21])

        after_ios_score = np.concatenate([after_ios_12, after_ios_21])
        after_helpful_score = np.concatenate([after_helpful_12, after_helpful_21])
        after_trust_score = np.concatenate([after_trust_12, after_trust_21])

        # post_nice_score = np.concatenate([post_nice_12, post_nice_21])
        # post_interesting_score = np.concatenate([post_interesting_12, post_interesting_21])
        # post_boring_score = np.concatenate([post_boring_12, post_boring_21])
        # post_likability_score = np.concatenate([post_likability_12, post_likability_21])

        new_df = pd.DataFrame({"group": group, "subject1": subject1, "subject2": subject2,
                               "friendship_score": pre_friendship_score,
                               "pre_ios_score": pre_ios_score,
                               "pre_nice_score": pre_nice_score,
                               "pre_interesting_score": pre_interesting_score,
                               "pre_boring_score": pre_boring_score,
                               "pre_likability_score": pre_likability_score,
                               "pre_helpful_score": pre_helpful_score,
                               "after_ios_score": after_ios_score,
                               "after_helpful_score": after_helpful_score,
                               "after_trust_score": after_trust_score,
                               })

        new_df['friendship_score'].replace(['yes', 'no'],
                                           [1, 0], inplace=True)

        if fill_in:
            new_df = new_df.fillna(np.nan)

            imputer = KNNImputer(n_neighbors=3)
            new_df[["friendship_score", "pre_ios_score", "pre_nice_score", "pre_interesting_score", "pre_boring_score",
                    "pre_likability_score", "pre_helpful_score", "after_ios_score",
                    "after_helpful_score"]] = imputer.fit_transform(
                new_df[
                    ["friendship_score", "pre_ios_score", "pre_nice_score", "pre_interesting_score", "pre_boring_score",
                     "pre_likability_score", "pre_helpful_score", "after_ios_score", "after_helpful_score"]])
        return new_df

    def eventStream(self, file_face, file_event):
        face_stream = pd.read_pickle(file_face)
        df = pd.read_csv(file_event)
        stream = np.zeros(len(face_stream))
        for index, row in df.iterrows():
            start_min_idx = row["Event Start(m)"]
            stop_min_idx = row["Event End(m)"]
            start_sec_idx = row["Event Start(s)"]
            stop_sec_idx = row["Event End(s)"]

            start_idx = int((FPS * start_min_idx * 60) + start_sec_idx)
            stop_idx = int((FPS * stop_min_idx * 60) + stop_sec_idx)

            stream[start_idx:stop_idx] = 1

        # get story and discussion episode
        story = df[df["marker"] == "story"]

        story_start = int((FPS * story["Event Start(m)"].values * 60) + story["Event Start(s)"].values)
        story_end = int((FPS * story["Event End(m)"].values * 60) + story["Event End(s)"].values)

        discussion = df[df["marker"].str.contains("discussion", regex=False)]
        discussion_start = int((FPS * discussion.iloc[0]["Event Start(m)"] * 60) + discussion.iloc[0]["Event Start(s)"])
        discussion_end = int((FPS * discussion.iloc[-1]["Event End(m)"] * 60) + discussion.iloc[-1]["Event End(s)"])

        # get smile indices in story
        # df.loc[np.array((FPS * df["Event End(m)"].values * 60) + df["Event End(s)"].values) <= story_end]
        # get discussion indices in story

        return stream, [story_start, story_end], [discussion_start, discussion_end]

    def getData(self, group_num):

        events_file = EVENT_PATH + "\\" + "Group_" + group_num + "_eventstream.csv"

        files_smile = glob.glob(SMILE_DETECTION_PATH + "\\" + "Group_" + group_num + "*.pkl")
        files_face = glob.glob(FACE_DETECTION_PATH + "\\" + "Group_" + group_num + "*.pkl")

        event_stream, story_idx, discussion_idx = self.eventStream(files_face[0], events_file)
        streams = []
        subject_ids = []
        smile_story = []
        smile_discussion = []
        for ff, fe in zip(files_face, files_smile):
            # subject id
            subject_id = ff.split("\\")[-1].split("_")[2]
            smile_indices, smile_stream = self.episodeToStream(ff, fe, event_stream)
            streams.append(smile_stream)
            smile_story.append(smile_indices[smile_indices["stop"].values <= story_idx[-1]])
            smile_discussion.append(smile_indices[(smile_indices["start"].values >= discussion_idx[0]) & (
                    smile_indices["stop"].values <= discussion_idx[-1])])
            subject_ids.append(subject_id)

        return streams, subject_ids, story_idx, discussion_idx, smile_story, smile_discussion
