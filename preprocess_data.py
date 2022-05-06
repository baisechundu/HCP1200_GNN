import os

from imports.utils import read_label, get_timeseries, subject_connectivity, save_atrr, trans_adj

cur_path = os.getcwd()
print(f"当前路径是{cur_path}")

BRAIN_CUT_DATA_PATH = os.path.join(cur_path, "辅助数据文件/dataset/cut_brain_data")
data_path = os.path.join(cur_path, '辅助数据文件/dataset/resting_HCP')
if not os.path.exists(data_path):
    os.mkdir(data_path)

left_file = 'Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii'
right_file = 'Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii'


def data_prep():
    label_path = os.path.join(cur_path, "辅助数据文件")

    Left_BRAIN_INDEX_LISTS = read_label(os.path.join(label_path, left_file))

    Right_BRAIN_INDEX_LISTS = read_label(os.path.join(label_path, right_file))

    time_series, subject_IDs = get_timeseries(Left_BRAIN_INDEX_LISTS, Right_BRAIN_INDEX_LISTS, data_path)

    subject_connectivity(time_series, subject_IDs, kind='correlation')
    subject_connectivity(time_series, subject_IDs, kind='partial correlation')

    task_list = ["emotion", "motor", "relational", "social", "wm"]
    task_label = dict(zip(task_list, range(len(task_list))))

    save_atrr(BRAIN_CUT_DATA_PATH, subject_IDs, task_label, kind='correlation')

    trans_adj(subject_IDs, save_path=os.path.join(label_path, 'dataset'))


if __name__ == '__main__':
    data_prep()
