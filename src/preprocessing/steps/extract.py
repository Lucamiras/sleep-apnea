import os
import gc
import pickle
import shutil
import pyedflib
import numpy as np
import xml.dom.minidom
from scipy.signal import resample
from typing import List, Union, Any, Dict
from src.preprocessing.steps.config import Config


class Extractor:
    """
    A class responsible for extracting signal data from EDF files using labels from RML files .

    Attributes:
        config (Config): An instance of the Config class containing download settings and file paths.

    Methods:
        extract():
            Starts the extraction process and returns a segment dictionary saved as npz.

    """
    def __init__(self, config:Config):
        self.config = config
        self.label_dictionary = {}
        self.segments_dictionary = {}

    def extract(self):
        print("EXTRACTOR -- Starting extraction ...")
        # check if files are in download folder
        if self._check_folder_contains_files(self.config.paths.edf_download_path, self.config.paths.rml_download_path):
            self._move_selected_downloads_to_preprocessing()

        # check if preprocess folder is empty
        if not self._check_folder_contains_files(self.config.paths.edf_preprocess_path, self.config.paths.rml_preprocess_path):
            raise Exception("No EDF or RML files found in preprocess folder.")

        # for each patient_id in preprocessing:
        if not self._match_ids(self.config.paths.edf_preprocess_path, self.config.paths.rml_preprocess_path):
            raise Exception("Not every EDF files has an RML file match.")

        print("EXTRACTOR -- All tests passed.")

        patient_ids_to_process = self.config.audio.ids_to_process if self.config.audio.ids_to_process is not None \
            else os.listdir(self.config.paths.rml_preprocess_path)

        print("EXTRACTOR -- Starting extraction for the following IDs:")
        print(patient_ids_to_process)

        for patient_id in patient_ids_to_process:
            ## get label dictionary
            self._create_dominant_label_dictionary(patient_id)
            ## create segments
            self._get_edf_segments_from_labels(patient_id)
            ## saves as pickle
            self._pickle_segment(patient_id)

    def _move_selected_downloads_to_preprocessing(self):
        """
        Moves files into folders by patient id.
        :returns:
        """
        edf_folder_contents = [file for file in os.listdir(self.config.paths.edf_download_path) if file.endswith('.edf')]
        rml_folder_contents = [file for file in os.listdir(self.config.paths.rml_download_path) if file.endswith('.rml')]

        if self.config.audio.ids_to_process is not None:
            edf_folder_contents = [file for file in edf_folder_contents if file.split('-')[0] in self.config.audio.ids_to_process]
            rml_folder_contents = [file for file in rml_folder_contents if file.split('-')[0] in self.config.audio.ids_to_process]

        unique_edf_file_ids = set([file.split('-')[0] for file in edf_folder_contents])
        unique_rml_file_ids = set([file.split('-')[0] for file in rml_folder_contents])

        assert unique_edf_file_ids == unique_rml_file_ids, ("Some EDF or RML files don't have matching pairs. "
                                                            "Preprocessing will not be possible:"
                                                            f"{unique_edf_file_ids}, {unique_rml_file_ids}")

        print(f"Preprocessing the following IDs: {unique_edf_file_ids}")

        for unique_edf_file_id in unique_edf_file_ids:
            os.makedirs(os.path.join(self.config.paths.edf_preprocess_path, unique_edf_file_id), exist_ok=True)

        for unique_rml_file_id in unique_rml_file_ids:
            os.makedirs(os.path.join(self.config.paths.rml_preprocess_path, unique_rml_file_id), exist_ok=True)

        for edf_file in edf_folder_contents:
            src_path = os.path.join(self.config.paths.edf_download_path, edf_file)
            dst_path = os.path.join(self.config.paths.edf_preprocess_path, edf_file.split('-')[0], edf_file)
            shutil.move(src=src_path, dst=dst_path)

        for rml_file in rml_folder_contents:
            src_path = os.path.join(self.config.paths.rml_download_path, rml_file)
            dst_path = os.path.join(self.config.paths.rml_preprocess_path, rml_file.split('-')[0], rml_file)
            shutil.move(src=src_path, dst=dst_path)

    def _get_edf_segments_from_labels(self, edf_folder) -> None:
        """
        Load edf files and create segments by timestamps.
        :return:
        """
        sample_rate = min(self.config.audio.sample_rate, self.config.audio.new_sample_rate) \
            if self.config.audio.new_sample_rate else self.config.audio.sample_rate

        print(f"Starting to create segments for patient {edf_folder}")
        edf_folder_path = os.path.join(self.config.paths.edf_preprocess_path, edf_folder)
        edf_readout = self._read_out_single_edf_file(edf_folder_path)

        for apnea_event in self.label_dictionary[edf_folder]['events']:
            start_idx = int(apnea_event['start'] * sample_rate)
            end_idx = int(apnea_event['end'] * sample_rate)
            segment = edf_readout[start_idx:end_idx]
            if len(segment) > 0:
                if self._no_change(segment):
                    break
                else:
                    apnea_event['signal'] = segment

        del edf_readout, segment
        gc.collect()

    def _downsample(self, edf_readout:np.array):
        length = int(len(edf_readout) / self.config.audio.sample_rate * self.config.audio.new_sample_rate)
        edf_readout = resample(edf_readout, length)
        return edf_readout

    def _read_out_single_edf_file(self, edf_folder) -> np.array:
        """
        Reads out all files from a single edf directory and concatenates the channel information
        in a numpy array.
        :returns: None
        """
        edf_files: list = sorted([os.path.join(edf_folder, file) for file in os.listdir(edf_folder)])

        full_readout: np.ndarray = np.array([])

        for edf_file in edf_files:
            print(f'Starting readout for file {edf_file}')
            f = pyedflib.EdfReader(edf_file)
            n = f.signals_in_file
            signal_labels = f.getSignalLabels()
            sound_data = None
            for i in np.arange(n):
                if signal_labels[i] == self.config.audio.data_channels:
                    sound_data = f.readSignal(i)
                    break
            f._close()

            if sound_data is None:
                raise ValueError(f"Channel '{self.config.audio.data_channels}' not found in EDF file")
            full_readout = np.append(full_readout, sound_data)
            print(len(full_readout)/self.config.audio.sample_rate)
            del f, n, sound_data

        full_readout = full_readout.astype(np.float16)

        if self.config.audio.new_sample_rate:
            full_readout = self._downsample(full_readout)

        return full_readout

    @staticmethod
    def calculate_overlap_duration(ann_start:float, ann_end:float, seg_start:float, seg_end:float) -> float:
        """
        Calculate the duration of overlap between two time intervals
        """
        overlap_start = max(ann_start, seg_start)
        overlap_end = min(ann_end, seg_end)
        return max(0.0, overlap_end - overlap_start)

    def find_dominant_label(self, segment_start:float, segment_end:float, annotations: List[dict]) -> str:
        """
        Get the dominant label for a segment based on the annotation
        """
        max_overlap = 0.0
        min_overlap = 1.0
        dominant_label = 'normal'

        for ann in annotations:
            overlap_duration = self.calculate_overlap_duration(ann['start'], ann['end'], segment_start, segment_end)
            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                dominant_label = ann['type']
            if max_overlap < min_overlap:
                dominant_label = 'normal'

        return dominant_label

    def _create_annotations(self, events:Any) -> List[Dict[str, Union[float, Any]]]:
        annotations = []
        for event in events:
            if event.getAttribute('Type') in self.config.data.targets:
                event_start = float(event.getAttribute('Start'))
                event_type = event.getAttribute('Type')
                annotations.append(
                    {
                        'start': event_start,
                        'end': event_start + float(self.config.audio.clip_length),
                        'type': event_type
                    }
                )
        return annotations

    @staticmethod
    def _create_segment_list(total_duration:float, segment_duration:float, clip_overlap:float) -> Union[List[Dict[str, float]]]:
        segments = []
        current_time = 0.0
        while current_time + segment_duration <= total_duration:
            segments.append({
                'start': current_time,
                'end': current_time + segment_duration,
            })
            current_time += clip_overlap
        return segments

    def _create_dominant_label_dictionary(self, rml_folder:str) -> None:
        file = os.listdir(os.path.join(self.config.paths.rml_preprocess_path, rml_folder))[0]
        label_path = os.path.join(self.config.paths.rml_preprocess_path, rml_folder, file)
        domtree = xml.dom.minidom.parse(label_path)
        group = domtree.documentElement
        events = group.getElementsByTagName('Event')
        total_duration = float(group.getElementsByTagName('Duration')[0].firstChild.nodeValue)
        gender = group.getElementsByTagName('Gender')[0].firstChild.nodeValue
        segments = self._create_segment_list(total_duration=total_duration,
                                                  segment_duration=self.config.audio.clip_length,
                                                  clip_overlap=self.config.audio.clip_overlap)
        annotations = self._create_annotations(events=events)
        apnea_events = {
            "acq_number": str(rml_folder),
            "gender": gender,
            "events": []
        }
        for i, segment in enumerate(segments):
            dominant_label = self.find_dominant_label(segment['start'], segment['end'], annotations)
            apnea_events['events'].append({
                "start": segment['start'],
                "end": segment['end'],
                "label": dominant_label,
                "signal": []
            })
        print(f"Found {len(apnea_events['events'])} events and non-events for {rml_folder}.")
        self.label_dictionary[rml_folder] = apnea_events

    def _create_sequential_label_dictionary(self, rml_folder:str) -> None:
        """
        This function goes through the EDF file in chunks of a determined size, i.e. 30 seconds,
        and labels each chunk according to the provided RML file.
        """
        file = os.listdir(os.path.join(self.config.paths.rml_preprocess_path, rml_folder))[0]
        label_path = os.path.join(self.config.paths.rml_preprocess_path, rml_folder, file)
        domtree = xml.dom.minidom.parse(label_path)
        group = domtree.documentElement
        events = group.getElementsByTagName('Event')
        gender = group.getElementsByTagName('Gender')[0].firstChild.nodeValue
        last_event_timestamp = int(float(events[-1].getAttribute('Start')))
        segment_duration = self.config.audio.clip_length
        events_timestamps = [
            (float(event.getAttribute('Start')),
             float(event.getAttribute('Duration')),
             event.getAttribute('Type')) for event in events
            if event.getAttribute('Type') in self.config.data.targets
        ]
        all_events = []
        apnea_events = {
            "acq_number": str(rml_folder),
            "gender": gender,
            "events": []
        }
        for segment_start in range(0, last_event_timestamp, segment_duration):
            segment_end = segment_start + segment_duration
            label = 'NoApnea'
            for timestamp in events_timestamps:
                start = timestamp[0]
                end = start + timestamp[1]
                event_type = timestamp[2]
                if start > segment_end:
                    break
                if self._simple_detect(segment_start, segment_end, start, end):
                    label = event_type
                    break
            event = {
                "start": float(segment_start),
                "end": float(segment_end),
                "label": str(label),
                "signal": []
            }
            apnea_events['events'].append(event)
        print(f"Found {len(all_events)} events and non-events for {rml_folder}.")
        self.label_dictionary[rml_folder] = apnea_events

    def _pickle_segment(self, patient_id) -> None:
        assert len(self.label_dictionary) > 0, "No segments available to save."
        data = self.label_dictionary[patient_id]
        file_path = f"{os.path.join(self.config.paths.pickle_path, patient_id)}.pickle"
        with open(file_path, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _overlaps(segment_start, segment_end, label_start, label_end) -> bool:
        """This function checks if at last 50% of a labelled event is covered by
        :return: True / False"""
        mid = (label_start + label_end) / 2
        return segment_start < mid < segment_end

    @staticmethod
    def _simple_detect(segment_start, segment_end, label_start, label_end) -> bool:
        """This function checks if any apnea is present at all in the segment.
        returns True or False"""
        return (
                ((segment_start <= label_start < segment_end) or (segment_start < label_end <= segment_end))
                or
                ((label_start < segment_start) and (label_end > segment_end))
        )

    @staticmethod
    def _no_change(array):
        return (array.sum() / len(array)) == array[0]

    @staticmethod
    def _check_folder_contains_files(edf_folder, rml_folder):
        edf_files_in_folder = os.listdir(edf_folder)
        rml_files_in_folder = os.listdir(rml_folder)
        not_empty = (len(edf_files_in_folder) > 0) and (len(rml_files_in_folder) > 0)
        return not_empty

    @staticmethod
    def _match_ids(edf_folder, rml_folder):
        edf_ids = sorted(os.listdir(edf_folder))
        rml_ids = sorted(os.listdir(rml_folder))
        return edf_ids == rml_ids