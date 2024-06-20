import unittest
from unittest.mock import MagicMock
import os
from _experiments.preprocess import Preprocessor
from src.utils.globals import get_file_urls


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.edf_urls, self.rml_urls = get_file_urls()
        self.preprocessor = Preprocessor('fake_project/download_dir',
                                         'fake_project/processed_dir',
                                         edf_urls=self.edf_urls,
                                         rml_urls=self.rml_urls)
        self.preprocessor._make_directories()

    def test_folder_structure(self):
        """
        This function tests if the correct number of folders were created in the project directory.
        :return: None
        """
        expected_results = [2, 2, 2, 2, 3]
        top_level_folders = os.listdir('fake_project')  # this should be 2
        download_subfolders = os.listdir(os.path.join('fake_project', 'download_dir'))  # this should be 2
        processed_subfolders = os.listdir(os.path.join('fake_project', 'processed_dir'))  # this should be 2
        audio_subfolders = os.listdir(os.path.join('fake_project', 'processed_dir', 'audio'))  # this should be 2
        spect_subfolders = os.listdir(os.path.join('fake_project', 'processed_dir', 'spectrogram'))  # this should be 3
        results = [
            len(top_level_folders),
            len(download_subfolders),
            len(processed_subfolders),
            len(audio_subfolders),
            len(spect_subfolders)
        ]

        self.assertEqual(results, expected_results, msg="Folder numbers don't match up.")

    def test_edf_downloads(self):
        """
        This function tests if the number of files that were supposed to be downloaded actually ended up
        in the folders.
        :return: None
        """
        number_edf_urls = len(self.edf_urls)
        number_edf_files = len(os.listdir(self.preprocessor.edf_path))

        self.assertEqual(number_edf_files, number_edf_urls, msg="Number of edf files does not match number of urls.")

    def test_rml_downloads(self):
        """
        This function tests if the number of files that were supposed to be downloaded actually ended up
        in the folders.
        :return: None
        """
        number_rml_urls = len(self.rml_urls)
        number_rml_files = len(os.listdir(self.preprocessor.rml_path))

        self.assertEqual(number_rml_files, number_rml_urls, msg="Number of rml files does not match number of urls.")

    def test_compare_edf_and_rml_count(self):
        """
        This function compares if the number of edf and rml files is the same.
        :return:
        """
        edf_folder_content = os.listdir(os.path.join(self.preprocessor.download_dir, 'edfs'))
        rml_folder_content = os.listdir(os.path.join(self.preprocessor.download_dir, 'rmls'))
        edf_folder_user_ids = set([file_name.split('-')[0] for file_name in edf_folder_content])
        edf_user_id_counts = len(edf_folder_user_ids)
        rml_file_counts = len(rml_folder_content)

        self.assertEqual(edf_user_id_counts, rml_file_counts, msg="User_ids don't match label files.")

    def test_edf_file_extension(self):
        """
        This function tests if the files in the folders correspond to the file extension that the folder
        is supposed to represent.
        :return: None
        """
        edf_folder_content = os.listdir(self.preprocessor.edf_path)
        edf_files_count = len(edf_folder_content)
        edf_extension_count = len([file for file in edf_folder_content if file[-4:] == '.edf'])

        self.assertEqual(edf_files_count, edf_extension_count, msg="All files in edf folder have .edf extension.")

    def test_rml_file_extension(self):
        """
        This function tests if the files in the folders correspond to the file extension that the folder
        is supposed to represent.
        :return: None
        """
        rml_folder_content = os.listdir(self.preprocessor.rml_path)
        rml_files_count = len(rml_folder_content)
        rml_extension_count = len([file for file in rml_folder_content if file[-4:] == '.rml'])

        self.assertEqual(rml_files_count, rml_extension_count, msg="All files in rml folder have .rml extension.")

    def test_if_files_sorted(self):
        """
        This function tests if the file dictionary is filled correctly
        :return:
        """
        self.preprocessor._read_edf_data()

        results = []
        user_ids = [user_id for user_id in self.preprocessor.files_dictionary.keys()]  # this should be 1

        for user_id in user_ids:
            list_of_files = self.preprocessor.files_dictionary[user_id]
            sorted_list_of_files = sorted(list_of_files)
            if list_of_files == sorted_list_of_files:
                results.append(True)
            else:
                results.append(False)

        self.assertNotIn(False, results, msg="Files in file dictionary don't appear to be sorted.")

    def test_save_spectrogram(self):
        self.preprocessor._save_spectrogram = MagicMock()
        audio_path = os.path.join(self.preprocessor.processed_dir, 'audio')
        save_path = os.path.join(self.preprocessor.processed_dir, 'save')
        self.preprocessor._save_spectrogram(audio_path, save_path)
        self.preprocessor._save_spectrogram.assert_called_with(audio_path, save_path)

