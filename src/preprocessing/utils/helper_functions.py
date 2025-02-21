import os
import shutil


def helper_reset_files(move_from: str = 'retired', move_to: str = 'downloads'):
    for folder_type in ['edf', 'rml']:
        root = os.path.join('data', move_from, folder_type)
        for folder in os.listdir(root):
            for file in os.listdir(os.path.join(root, folder)):
                shutil.move(
                    os.path.join(root, folder, file),
                    os.path.join('data', move_to, folder_type))

    edf_patient_ids = [str(filename.split('-')[0]) for filename in os.listdir(os.path.join('data', 'downloads', 'edf'))]
    rml_patient_ids = [str(filename.split('-')[0]) for filename in os.listdir(os.path.join('data', 'downloads', 'rml'))]
    if set(edf_patient_ids) != set(rml_patient_ids):
        print('Patient IDs do not match')
    print("EDF patient IDs:", set(edf_patient_ids))
    print("RML patient IDs:", set(rml_patient_ids))
