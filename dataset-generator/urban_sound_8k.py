import pandas as pd
import numpy as np
import os

np.random.seed(999)


class UrbanSound8K:
    def __init__(self, base_path, *, val_dataset_size, class_ids=None):
        self.base_path = base_path
        self.val_dataset_size = val_dataset_size
        self.class_ids = class_ids

    def _get_urban_sound_8k_files(self):
        urbansound_metadata = pd.read_csv(
            os.path.join(self.base_path, "metadata", "UrbanSound8K.csv")
        )

        # shuffle
        urbansound_metadata.reindex(np.random.permutation(urbansound_metadata.index))

        return urbansound_metadata

    def get_filenames_with_class_id(self, metadata):
        if self.class_ids is None:
            self.class_ids = np.unique(metadata["classID"].values)
            print("Class IDs:", self.class_ids)

        all_files = []
        file_counter = 0
        for c in self.class_ids:
            class_files = metadata[metadata["classID"] == c][
                ["slice_file_name", "fold"]
            ].values
            class_files = [
                os.path.join(self.base_path, "audio", "fold" + str(file[1]), file[0])
                for file in class_files
            ]
            print("There are", len(class_files), "files in class", str(c))
            file_counter += len(class_files)
            all_files.extend(class_files)

        assert len(all_files) == file_counter
        return all_files

    def get_train_val_files(self):
        urbansound_metadata = self._get_urban_sound_8k_files()

        # Using folders 0-9
        urbansound_train = urbansound_metadata[urbansound_metadata.fold != 10]

        urbansound_train_filenames = self.get_filenames_with_class_id(urbansound_train)
        np.random.shuffle(urbansound_train_filenames)

        # Separate noise files for train/validation
        urbansound_val = urbansound_train_filenames[-self.val_dataset_size :]
        urbansound_train = urbansound_train_filenames[: -self.val_dataset_size]
        print("Training noise:", len(urbansound_train))
        print("Validation noise:", len(urbansound_val))

        return urbansound_train, urbansound_val

    def get_test_filenames(self):
        urbansound_metadata = self._get_urban_sound_8k_files()

        # Folder 10 will be used for testing only
        urbansound_test = urbansound_metadata[urbansound_metadata.fold == 10]

        urbansound_test_filenames = self.get_filenames_with_class_id(urbansound_test)
        np.random.shuffle(urbansound_test_filenames)

        print("# Test noise files:", len(urbansound_test_filenames))
        return urbansound_test_filenames
