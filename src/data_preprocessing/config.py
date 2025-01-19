import os


class Config:
    """
    Configuration class for managing all settings and paths required by the data processing pipeline.

    Attributes:
        project_dir (str): Root directory where all preprocessing will take place.
        classes (dict): Dictionary with class names and integer values. Example: {"NoApnea":0, "ObstructiveApnea":1}.
        download_files (bool): Specify if the preprocessor should include the downloading step.
        extract_signals (bool): Specify if the preprocessor should include the extraction step.
        process_signals (bool): Specify if the preprocessor should include the processing step.
        serialize_signals (bool): Specify if the preprocessor should include the serialization step.
        overrides (dict): Overrides of any other parameters can be changed via the overrides method by passing a
        dictionary with attribute names as keys and parameters as values.
        edf_urls (list): A list with EDF download urls. Default: None
        rml_urls (list): A list with RML download urls. Default: None
        data_channels (str): The data channel to extract from the EDF file. Default: 'Mic'
        edf_step_size (int): Amount of signals to process during EDF extraction. This is necessary as to not overfill
        memory. Default: 10_000_000
        sample_rate (int): The appropriate sample rate for the signal. Default: 48_000
        clip_length (int): Length of extracted samples that will be used for training. Default: 30
        ids_to_process (list): List of IDs in case not all files from download / preprocess folders should be considered.
        Default: None
        train_size (float): How much of the total samples should be considered for training. Default: 0.8
        edf_download_path (str): Path in root folder.
        rml_download_path (str): Path in root folder.
        edf_preprocess_path (str): Path in root folder.
        rml_preprocess_path (str): Path in root folder.
        npz_path (str): Path in root folder.
        audio_path (str): Path in root folder.
        spectrogram_path (str): Path in root folder.
        signals_path (str): Path in root folder.
        retired_path (str): Path in root folder.

    Methods:
        _create_directory_structure():
            Creates the folders in the root directory specified in the project_dir attribute.
    """

    def __init__(self,
                 classes: dict,
                 project_dir: str = 'data',
                 download_files: bool = False,
                 extract_signals: bool = True,
                 process_signals: bool = True,
                 serialize_signals: bool = True,
                 overrides: dict = None):

        # Basic inputs
        self.project_dir = project_dir
        self.classes = classes

        # Download config
        self.download_files = download_files
        self.catalog_filepath = os.path.join(self.project_dir, 'file_catalog.txt')
        self.edf_urls = None
        self.rml_urls = None

        # Extract signals
        self.extract_signals = extract_signals
        self.data_channels = 'Mic'
        self.edf_step_size = 10_000_000
        self.sample_rate = 48_000
        self.new_sample_rate = None
        self.clip_length = 30
        self.n_mels = 128

        # Process signals
        self.process_signals = process_signals
        self.ids_to_process = None
        self.augment_ratio = None
        self.image_size = (224, 224)

        # Serialize signals
        self.serialize_signals = serialize_signals
        self.dataset_file_name = 'dataset.pickle'
        self.train_size = 0.8

        # Paths
        self.edf_download_path = os.path.join(self.project_dir, 'downloads', 'edf')
        self.rml_download_path = os.path.join(self.project_dir, 'downloads', 'rml')
        self.ambient_noise_path = os.path.join(self.project_dir, 'downloads', 'ambient')
        self.edf_preprocess_path = os.path.join(self.project_dir, 'preprocess', 'edf')
        self.rml_preprocess_path = os.path.join(self.project_dir, 'preprocess', 'rml')
        self.npz_path = os.path.join(self.project_dir, 'preprocess', 'npz')
        self.audio_path = os.path.join(self.project_dir, 'processed', 'audio')
        self.spectrogram_path = os.path.join(self.project_dir, 'processed', 'spectrogram')
        self.signals_path = os.path.join(self.project_dir, 'processed', 'signals')
        self.retired_path = os.path.join(self.project_dir, 'retired')

        if overrides:
            for key, value in overrides.items():
                setattr(self, key, value)

        self._catch_errors()
        self._create_directory_structure()

    def _catch_errors(self) -> None:

        if self.new_sample_rate:
            if not isinstance(self.new_sample_rate, int):
                raise TypeError("new_sample_rate must be of type Integer")
            assert self.new_sample_rate < self.sample_rate, "new_sample_rate must be smaller than sample_rate"

    def _create_directory_structure(self) -> None:
        """
        Creates directory structure necessary to download and process data.
        returns: None
        """
        os.makedirs(self.edf_download_path, exist_ok=True)
        os.makedirs(self.rml_download_path, exist_ok=True)
        os.makedirs(self.edf_preprocess_path, exist_ok=True)
        os.makedirs(self.rml_preprocess_path, exist_ok=True)
        os.makedirs(self.npz_path, exist_ok=True)
        os.makedirs(self.audio_path, exist_ok=True)
        os.makedirs(self.spectrogram_path, exist_ok=True)
        os.makedirs(self.signals_path, exist_ok=True)
        os.makedirs(self.retired_path, exist_ok=True)
