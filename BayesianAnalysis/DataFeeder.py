import math
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from Utils.DataReader import DataReader
import pandas as pd
from pyunicorn.eventseries import EventSeries
from Utils.Conf import FPS
import glob
import os
from Utils.Conf import EVENT_PATH, SUMMARY_PKL_FILLIN_PATH, SUMMARY_PKL_REMOVENAN_PATH, SUMMARY_PKL_REMOVENAN_PATH_02, SUMMARY_PKL_REMOVENAN_PATH_05, SUMMARY_PKL_REMOVENAN_PATH_10, SUMMARY_PKL_REMOVENAN_PATH_20, SUMMARY_PKL_REMOVENAN_PATH_30, SUMMARY_PKL_REMOVENAN_PATH_40
from Utils.SyncMetrics import groupPoints


class DataFeeder:

    def __init__(self, results_path):
        self.results_path = results_path

    def fetchData(self, only_read=True, fill_in=True, extracted_columns=[], file_path=SUMMARY_PKL_REMOVENAN_PATH, TAU_MAX=2.0):

        def genderSIM(em_gender, re_gender):
            if (em_gender == 1) & (re_gender == 1):
                return 0
            elif (em_gender == 1) & (re_gender == 0):
                return 1
            elif (em_gender == 0) & (re_gender == 1):
                return 2
            else:
                return 3

        def imigSim(em_im, re_im):
            if (em_im == 0) & (re_im == 0):
                return 0
            elif (em_im == 0) & (re_im != 0):
                return 1
            elif (em_im != 0) & (re_im == 0):
                return 2
            else:
                return 3

        if not only_read:
            reader = DataReader(results_path=self.results_path)

            imigration_info = reader.getImigrationInfo()

            dyadic_score = reader.getDyadicScore(fill_in)

            group_score = reader.getGroupScore(fill_in)

            individual_info = reader.getIndividualInfo(fill_in)

            validation_info = reader.getValidationInfo()

            # rates and groups
            trigger_rate_list = []
            experiment_segment_list = []
            experiment_segment_label = ["story", "discussion"]

            # pre score (re: receiver, em:emitter)
            # emitter
            em1_friendship_score = []

            em1_nice_score = []
            em1_interesting_score = []
            em1_boring_score = []
            em1_likability_score = []
            em1_helpful_score = []

            em1_prosociality = []

            em1_popularity_score = []
            em1_leadership_score = []

            # receiver
            re1_ios_score = []
            re1_ios_group_score = []
            re1_bullying_score = []
            re1_extraversion = []

            # after score
            # receiver
            em2_helpful_score = []
            re2_ios_score = []
            re2_ios_group_score = []
            re2_trust_score = []

            # post score
            re3_group_climate_all = []
            em3_group_climate_all = []

            # individual info
            emitter_age = []
            receiver_age = []
            emitter_gender = []
            receiver_gender = []
            receiver_background = []
            emitter_background = []

            emitter_id = []
            receiver_id = []
            group_id = []

            seats_id = []

            # affiliate
            emre_gender = []
            emre_imigration = []

            # group info
            group_count = []
            validator_group = []

            # smile occurence
            emitter_smile_occurence = []
            receiver_smile_occurence = []

            list_of_groups = glob.glob(os.path.join(EVENT_PATH, "*_eventstream.csv"))

            for group in list_of_groups:
                group_num = group.split("\\")[-1].split("_")[1]
                print(group_num)
                streams, subject_ids, story_idx, discussion_idx, smile_story, smile_discussion = reader.getData(
                    group_num)
                indices = [story_idx, discussion_idx]
                for i_idx in range(len(indices)):
                    idx = indices[i_idx]
                    for i in range(len(streams) - 1):
                        for j in range(i + 1, len(streams)):

                            # dyadic score
                            group_name = group_num
                            condition_group12 = dyadic_score[
                                (dyadic_score["group"] == group_name) & (dyadic_score["subject1"] == subject_ids[i]) & (
                                        dyadic_score["subject2"] == subject_ids[j])]
                            condition_group21 = dyadic_score[
                                (dyadic_score["group"] == group_name) & (dyadic_score["subject1"] == subject_ids[j]) & (
                                        dyadic_score["subject2"] == subject_ids[i])]

                            # group score
                            subject_group1 = group_score[
                                (group_score["group"] == group_name) & (group_score["subject"] == subject_ids[i])]
                            subject_group2 = group_score[
                                (group_score["group"] == group_name) & (group_score["subject"] == subject_ids[j])]

                            # individual info
                            subject1 = individual_info[
                                (individual_info["group"] == group_name) & (
                                        individual_info["subject"] == subject_ids[i])]
                            subject2 = individual_info[
                                (individual_info["group"] == group_name) & (
                                        individual_info["subject"] == subject_ids[j])]

                            # imigration info
                            im_subject1 = imigration_info[
                                imigration_info["subject_code"] == subject1["subject_code"].values[0]]
                            im_subject2 = imigration_info[
                                imigration_info["subject_code"] == subject2["subject_code"].values[0]]

                            # validator score
                            validator_score = validation_info[(validation_info["group"] == group_name)]

                            sX = streams[i][idx[0]:idx[1]]
                            sY = streams[j][idx[0]:idx[1]]
                            # print(len(sX))
                            # print(len(sY))
                            series = np.vstack([sX, sY]).T
                            if (len(np.unique(sX)) == 2) & (len(np.unique(sY)) == 2):
                                ev = EventSeries(series, taumax=int(TAU_MAX * FPS))
                                # Y_trigger_X= Y->X: Trigger coincidence rate of X (receiver) by Y (emitter)
                                # X_trigger_Y= X->Y: Trigger coincidence rate of Y (receiver) by X (emitter)

                                _, Y_trigger_X, _, X_trigger_Y = ev.event_coincidence_analysis(*series.T,
                                                                                               taumax=int(
                                                                                                   TAU_MAX * FPS))

                                if (Y_trigger_X >= 0.5) | (X_trigger_Y >= 0.5):
                                    print(Y_trigger_X)
                                    print(X_trigger_Y)
                                    import matplotlib.pyplot as plt

                                    plt.plot(sX)
                                    plt.plot(sY)
                                    plt.show()

                                trigger_rate_list.append(X_trigger_Y)  # emitter X and receiver Y
                                trigger_rate_list.append(Y_trigger_X)  # emitter Y and receiver X

                                # add experiment list
                                experiment_segment_list.extend(experiment_segment_label[i_idx] for k in range(2))

                                # compute smile occurence for both person
                                x_smile_occurence = len(groupPoints(np.nonzero(sX == 1)[0]))
                                y_smile_occurence = len(groupPoints(np.nonzero(sY == 1)[0]))
                                emitter_smile_occurence.extend([x_smile_occurence, y_smile_occurence])
                                receiver_smile_occurence.extend([y_smile_occurence, x_smile_occurence])

                                # condition21 person 2 assess person 1
                                # condition12 person 1 assess person 2

                                # add pre score
                                re1_ios_score.extend(
                                    [condition_group21["pre_ios_score"].values[0],
                                     condition_group12["pre_ios_score"].values[0]])

                                re1_bullying_score.extend(
                                    [subject2["bullying"].values[0], subject1["bullying"].values[0]])
                                em1_prosociality.extend(
                                    [subject1["prosociality"].values[0], subject2["prosociality"].values[0]])
                                re1_extraversion.extend(
                                    [subject2["extraversion"].values[0], subject1["extraversion"].values[0]])

                                em1_friendship_score.extend([condition_group21["friendship_score"].values[0],
                                                             condition_group12["friendship_score"].values[0]])
                                em1_nice_score.extend([condition_group21["pre_nice_score"].values[0],
                                                       condition_group12["pre_nice_score"].values[0]])
                                em1_interesting_score.extend([condition_group21["pre_interesting_score"].values[0],
                                                              condition_group12["pre_interesting_score"].values[0]])
                                em1_boring_score.extend([condition_group21["pre_boring_score"].values[0],
                                                         condition_group12["pre_boring_score"].values[0]])
                                em1_likability_score.extend([condition_group21["pre_likability_score"].values[0],
                                                             condition_group12["pre_likability_score"].values[0]])

                                em1_helpful_score.extend([condition_group21["pre_helpful_score"].values[0],
                                                          condition_group12["pre_helpful_score"].values[0]])

                                # group
                                re1_ios_group_score.extend(
                                    [subject_group2["pre_ios_group"].values[0],
                                     subject_group1["pre_ios_group"].values[0]])
                                em1_popularity_score.extend(
                                    [subject_group1["popularity"].values[0] / subject1["group_count"].values[0],
                                     subject_group2["popularity"].values[0] / subject2["group_count"].values[0]])
                                em1_leadership_score.extend(
                                    [subject_group1["leadership"].values[0] / subject1["group_count"].values[0],
                                     subject_group2["leadership"].values[0] / subject2["group_count"].values[0]])

                                # after
                                re2_ios_score.extend([condition_group21["after_ios_score"].values[0],
                                                      condition_group12["after_ios_score"].values[0]])
                                re2_ios_group_score.extend([subject_group2["after_ios_group"].values[0],
                                                            subject_group1["after_ios_group"].values[0]])
                                em2_helpful_score.extend([condition_group21["after_helpful_score"].values[0],
                                                          condition_group12["after_helpful_score"].values[0]])

                                re2_trust_score.extend([condition_group21["after_trust_score"].values[0],
                                                        condition_group12["after_trust_score"].values[0]])

                                # post
                                re3_group_climate_all.extend([subject2["group_climate"].values[0],
                                                              subject1["group_climate"].values[0]])

                                em3_group_climate_all.extend([subject1["group_climate"].values[0], subject2["group_climate"].values[0]])

                                # individual
                                emitter_age.extend([subject1["age"].values[0], subject2["age"].values[0]])
                                emitter_gender.extend([subject1["gender"].values[0], subject2["gender"].values[0]])

                                receiver_age.extend([subject2["age"].values[0], subject1["age"].values[0]])
                                receiver_gender.extend([subject2["gender"].values[0], subject1["gender"].values[0]])
                                receiver_background.extend(
                                    [im_subject2["nationality"].values[0], im_subject1["nationality"].values[0]])
                                emitter_background.extend([im_subject1["nationality"].values[0], im_subject2["nationality"].values[0]])
                                seats_id.extend([subject1["seat_id"].values[0] + subject2["seat_id"].values[0],
                                                 subject2["seat_id"].values[0] + subject1["seat_id"].values[0]])

                                # emre
                                emre_gender.extend(
                                    [genderSIM(subject1["gender"].values[0], subject2["gender"].values[0]),
                                     genderSIM(subject2["gender"].values[0], subject1["gender"].values[0])])
                                emre_imigration.extend([
                                    imigSim(im_subject1["nationality"].values[0], im_subject2["nationality"].values[0]),
                                    imigSim(im_subject2["nationality"].values[0], im_subject1["nationality"].values[0])
                                ])

                                # group id
                                group_id.extend([subject1["group"].values[0], subject2["group"].values[0]])
                                emitter_id.extend([subject1["group"].values[0] + "_" + subject1["subject"].values[0],
                                                   subject2["group"].values[0] + "_" + subject2["subject"].values[0]])
                                receiver_id.extend([subject2["group"].values[0] + "_" + subject2["subject"].values[0],
                                                    subject1["group"].values[0] + "_" + subject1["subject"].values[0]])
                                group_count.extend(
                                    [subject1["group_count"].values[0], subject2["group_count"].values[0]])
                                validator_group.extend(
                                    [validator_score["validator"].values[0], validator_score["validator"].values[0]])

            # normalize smile occurence
            emitter_smile_occurence = np.array(emitter_smile_occurence) / np.max(emitter_smile_occurence)
            receiver_smile_occurence = np.array(receiver_smile_occurence) / np.max(receiver_smile_occurence)
            summary = {"trigger_rate": trigger_rate_list,
                       "em1_friendship": np.array(em1_friendship_score).astype(int),
                       "em1_nice": np.array(em1_nice_score),

                       "em1_interesting": np.array(em1_interesting_score),
                       "em1_boring": np.array(em1_boring_score),
                       "em1_likability": np.array(em1_likability_score),
                       "re1_ios": np.array(re1_ios_score),

                       "re1_ios_group": np.array(re1_ios_group_score),
                       "em1_popularity": np.array(em1_popularity_score),
                       "em1_leadership": np.array(em1_leadership_score),
                       "em1_helpful": np.array(em1_helpful_score),

                       "re1_bullying": np.array(re1_bullying_score),
                       "em1_prosociality": np.array(em1_prosociality),
                       "re1_extraversion": np.array(re1_extraversion),

                       "em2_helpful": np.array(em2_helpful_score),
                       "re2_ios": np.array(re2_ios_score),
                       "re2_ios_group": np.array(re2_ios_group_score),
                       "re2_trust_score": np.array(re2_trust_score),

                       "re3_group_climate_all": np.array(re3_group_climate_all),
                       "em3_group_climate_all": np.array(em3_group_climate_all),

                       "em_age": np.array(emitter_age),
                       "em_gender": np.array(emitter_gender),

                       "re_age": np.array(receiver_age),
                       "re_gender": np.array(receiver_gender),
                       "re_imigration": np.array(receiver_background),
                       "em_imigration": np.array(emitter_background),
                       "emre_gender": np.array(emre_gender),
                       "emre_ethnics": np.array(emre_imigration),

                       # id
                       "group_id": np.array(group_id),
                       "em_id": np.array(emitter_id),
                       "re_id": np.array(receiver_id),
                       "seat_id": np.array(seats_id),

                       # group and validatior
                       "group_count": np.array(group_count),
                       "validator_group": np.array(validator_group),

                       # smile occurence
                       "emitter_smile_occurence": emitter_smile_occurence,
                       "receiver_smile_occurence": receiver_smile_occurence,

                       "experiment_segment": experiment_segment_list}
            df = pd.DataFrame(summary)
            # if fill_in:
            #     df.to_pickle(SUMMARY_PKL_FILLIN_PATH)
            # else:
            #     df.to_pickle(file_path)
        else:
            if fill_in:
                df = pd.read_pickle(SUMMARY_PKL_FILLIN_PATH)
            else:
                df = pd.read_pickle(file_path)

        # remove nan
        if len(extracted_columns) != 0:
            df = df[extracted_columns]
        df = df.dropna()
        return df

    def fetchPostData(self, fill_in=True, extracted_columns=[], file_path=SUMMARY_PKL_REMOVENAN_PATH):
        if fill_in:
            df = pd.read_pickle(SUMMARY_PKL_FILLIN_PATH)
        else:
            df = pd.read_pickle(file_path)


        group_df = df.groupby(["group_id", "re_id"])

        # predictor
        trigger_rate_story_mean = []
        trigger_rate_discussion_mean = []
        trigger_rate_story_std = []
        trigger_rate_discussion_std = []
        # response
        re3_group_climate_all = []
        em3_group_climate_all = []

        # control
        receiver_age = []
        receiver_gender = []
        receiver_id = []
        receiver_background = []
        receiver_smile_occurence = []

        # emitter control
        emitter_age = []
        emitter_gender = []
        emitter_id = []
        emitter_background = []
        emitter_smile_occurence = []

        group_count = []
        validator_group = []
        group_id = []
        for _, g in group_df:
            if np.sum(~np.isnan(g[g["experiment_segment"] == "story"]["trigger_rate"].values)) == 0:
                tr_story_mean = np.nan
                tr_story_std = np.nan
            else:
                tr_story_mean = np.nanmean(g[g["experiment_segment"] == "story"]["trigger_rate"].values)
                tr_story_std = np.nanstd(g[g["experiment_segment"] == "story"]["trigger_rate"].values)

            if np.sum(~ np.isnan(g[g["experiment_segment"] == "discussion"]["trigger_rate"].values)) == 0:
                tr_discussion_mean = np.nan
                tr_discussion_std = np.nan
            else:
                tr_discussion_mean = np.nanmean(g[g["experiment_segment"] == "discussion"]["trigger_rate"].values)
                tr_discussion_std = np.nanstd(g[g["experiment_segment"] == "discussion"]["trigger_rate"].values)

            # if np.isnan(tr_discussion):
            #     print("error")

            # predictor
            trigger_rate_story_mean.append(tr_story_mean)
            trigger_rate_discussion_mean.append(tr_discussion_mean)
            trigger_rate_story_std.append(tr_story_std)
            trigger_rate_discussion_std.append(tr_discussion_std)

            # response
            re3_group_climate_all.append(g["re3_group_climate_all"].values[0])
            em3_group_climate_all.append(g["em3_group_climate_all"].values[0])

            # control
            receiver_age.append(g["re_age"].values[0])
            receiver_gender.append(g["re_gender"].values[0])
            receiver_id.append(g["re_id"].values[0])
            receiver_background.append(g["re_imigration"].values[0])
            receiver_smile_occurence.append(g["receiver_smile_occurence"].values[0])

            # emitter
            emitter_age.append(g["em_age"].values[0])
            emitter_gender.append(g["em_gender"].values[0])
            emitter_id.append(g["em_id"].values[0])
            emitter_background.append(g["em_imigration"].values[0])
            emitter_smile_occurence.append(g["emitter_smile_occurence"].values[0])

            group_count.append(g["group_count"].values[0])
            group_id.append(g["group_id"].values[0])
            validator_group.append(g["validator_group"].values[0])

        summary = {
            "trigger_story_std": trigger_rate_story_std,
            "trigger_discussion_std": trigger_rate_discussion_std,
            "trigger_story_mean": trigger_rate_story_mean,
            "trigger_discussion_mean": trigger_rate_discussion_mean,
            "re3_group_climate_all": re3_group_climate_all,
            "em3_group_climate_all": em3_group_climate_all,


            "re_age": receiver_age,
            "re_gender": receiver_gender,
            "re_background": receiver_background,
            "re_id": receiver_id,
            "receiver_smile_occurence": receiver_smile_occurence,

            "em_age": emitter_age,
            "em_gender": emitter_gender,
            "em_background": emitter_background,
            "em_id": emitter_id,
            "emitter_smile_occurence": emitter_smile_occurence,

            "group_count": group_count,
            "group_id": np.array(group_id),
            "validator_group": validator_group,
        }
        new_df = pd.DataFrame(summary)
        new_df = new_df[extracted_columns]
        new_df = new_df.dropna()
        return new_df


if __name__ == '__main__':
    results_path = "F:\\users\\prasetia\\data\\Children\\children_sync\\data\\"
    feeder = DataFeeder(results_path)
    feeder.fetchData(fill_in=False, only_read=False, file_path=SUMMARY_PKL_REMOVENAN_PATH_40, TAU_MAX=2.)
    # feeder.fetchPostData()
