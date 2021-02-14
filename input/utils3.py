# utility functions
# based on "eye_blink_detection_3_ml_model_part1" and "eye_blink_detection_3_ml_model_part2"

# import libraries
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, classification_report, roc_auc_score

############################################################################################################

# define read_annotations_v2 (for annotations of EyeBlink8)
def read_annotations_v2(input_file, len_video):
    # Read .tag file using readlines() 
    file1 = open(input_file) 
    Lines = file1.readlines() 

    # find "#start" line 
    start_line = 1
    for line in Lines: 
        clean_line=line.strip()
        if clean_line=="#start":
            break
        start_line += 1

    # length of annotations
    len_annot = len(Lines[start_line : -1]) # -1 since last line will be"#end"

    blink_list = [0] * len_video
    closeness_list = [0] * len_video

    # convert tag file to readable format and build "closeness_list" and "blink_list"
    for i in range(len_annot): 
        annotation=Lines[start_line+i].split(':')

        if int(annotation[1]) > 0:
            # it means a new blink
            blink_frame = int(annotation[0])
            blink_list[blink_frame] = 1

        # if current annotation consist fully closed eyes, append it also to "closeness_list" 
        if annotation[3] == "C" and annotation[5] == "C":
            closed_frame = int(annotation[0])
            closeness_list[closed_frame] = 1

        file1.close()

    result_df = pd.DataFrame(list(zip(closeness_list, blink_list)), columns=['closeness_annot', 'blink_annot'])
    return result_df

############################################################################################################

# Merge relevant frame_info_df with annot file
def merge_pickles(directory):
    annots=[]
    frame_infos=[]
    video_infos=[]

    files = os.listdir(directory)
    for file in files:
        clean_name = os.path.splitext(file)[0]
        if clean_name.endswith('annotations'):
            annots.append(file)
        if clean_name.endswith('video_info_df'):
            video_infos.append(file)
        if clean_name.endswith('frame_info_df'):
            frame_infos.append(file)

    for file in annots:
        clean_name = os.path.splitext(file)[0]
        first_part = clean_name[:-12]

        for file2 in frame_infos:
            clean_name2 = os.path.splitext(file2)[0]
            first_part2 = clean_name2[:-14]
            if first_part == first_part2:
                frame_info_df = pd.read_pickle(directory+'/'+file2)
                annotation = pd.read_pickle(directory+'/'+file)
                if len(frame_info_df) !=len(annotation):
                    os.mkdir(directory+'/fix/')
                    os.rename(directory+file, directory+'/fix/'+file)
                    os.rename(directory+file2, directory+'/fix/'+file2)
                    print(file2, len(frame_info_df))
                    print(file, len(annotation))
                else: 
                    result=pd.concat([frame_info_df,annotation], axis=1)
                    result.to_pickle(directory+'/'+first_part+'_merged_df.pkl')

############################################################################################################ 
                    
# append all of pickles ending particular string (i.e. "merged_df") in a directory
def concat_pickles(directory, ending, output_name):
    pickles = os.listdir(directory)
    pickle_list=[]

    for pickle_file in pickles:
        clean_name = os.path.splitext(pickle_file)[0]
        if clean_name.endswith(ending):
            pickle = pd.read_pickle(directory+'/'+pickle_file)
            pickle_list.append(pickle)

    result = pd.concat(pickle_list)
    result.reset_index(inplace=True, drop=True)
    result.to_pickle(directory+'/'+ output_name + '.pkl')
    
############################################################################################################ 

# display test scores and return result string and indexes of false samples
def display_test_scores_v2(test, pred):
    str_out = ""
    str_out += ("TEST SCORES\n")
    str_out += ("\n")

    #print accuracy
    accuracy = accuracy_score(test, pred)
    str_out += ("ACCURACY: {:.4f}\n".format(accuracy))
    str_out += ("\n")

    #print AUC score
    auc = roc_auc_score(test, pred)
    str_out += ("AUC: {:.4f}\n".format(auc))
    str_out += ("\n")

    #print confusion matrix
    str_out += ("CONFUSION MATRIX:\n")
    conf_mat = confusion_matrix(test, pred)
    str_out += ("{}".format(conf_mat))
    str_out += ("\n")
    str_out += ("\n")

    #print FP, FN
    str_out += ("FALSE POSITIVES:\n")
    fp = conf_mat[1][0]
    pos_labels = conf_mat[1][0]+conf_mat[1][1]
    str_out += ("{} out of {} positive labels ({:.4f}%)\n".format(fp, pos_labels,fp/pos_labels))
    str_out += ("\n")

    str_out += ("FALSE NEGATIVES:\n")
    fn = conf_mat[0][1]
    neg_labels = conf_mat[0][1]+conf_mat[0][0]
    str_out += ("{} out of {} negative labels ({:.4f}%)\n".format(fn, neg_labels, fn/neg_labels))
    str_out += ("\n")

    #print classification report
    str_out += ("PRECISION, RECALL, F1 scores:\n")
    str_out += ("{}".format(classification_report(test, pred)))
    
    false_indexes = np.where(test != pred)
    return str_out, false_indexes

############################################################################################################ 

# define decision function for rbf svc class of sklearn
def rbf_decision_function(sup_vecs, dual_coefs, gamma, intercept,Xtest):
    diff = sup_vecs - Xtest
    norm2 = np.array([np.linalg.norm(diff[n, :]) for n in range(np.shape(sup_vecs)[0])])
    dec_func_vec = (dual_coefs.dot(np.exp(-gamma*(norm2**2))) + intercept)
    return dec_func_vec

############################################################################################################ 

# define predict function for rbf svc class of sklearn 
def rbf_predict(sup_vecs, dual_coefs, gamma, intercept,Xtest):
    list_predictions=[]
    for row in Xtest:
        dec = rbf_decision_function(sup_vecs, dual_coefs, gamma, intercept,row)
        list_predictions.append(1 if dec[0]>0 else 0)
    return np.array(list_predictions)

############################################################################################################ 
