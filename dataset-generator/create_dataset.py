from mozilla_voice import MozillaVoiceDataset
from urban_sound_8k import UrbanSound8K
from dataset import Dataset
import warnings
import multiprocessing


def start():
    warnings.filterwarnings(action="ignore")

    mozilla_root = "D:\\PROJECT\\en"
    urbansound_root = "D:\\PROJECT\\UrbanSound8K\\UrbanSound8K"

    mozilla_dataset = MozillaVoiceDataset(mozilla_root, val_dataset_size=1000)
    (
        clean_train_filenames,
        clean_val_filenames,
    ) = mozilla_dataset.get_train_val_files()
    print(clean_train_filenames)

    us8k_dataset = UrbanSound8K(urbansound_root, val_dataset_size=200)
    noisy_train_filenames, noisy_val_filenames = us8k_dataset.get_train_val_files()

    window_size = 256
    settings = {
        "window_size": window_size,
        "overlap": round(0.25 * window_size),
        "sample_rate": 16000,
        "max_audio_length": 0.8,
    }

    val_dataset = Dataset(clean_val_filenames, noisy_val_filenames, **settings)
    val_dataset.create_tf_records(prefix="val", subset_size=2000)

    train_dataset = Dataset(clean_train_filenames, noisy_train_filenames, **settings)
    train_dataset.create_tf_records(prefix="train", subset_size=4000)

    # Create Test Dataset
    clean_test_filenames = mozilla_dataset.get_test_filenames()

    noisy_test_filenames = us8k_dataset.get_test_filenames()

    test_dataset = Dataset(clean_test_filenames, noisy_test_filenames, **settings)
    test_dataset.create_tf_records(prefix="test", subset_size=1000, parallel=False)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    start()
