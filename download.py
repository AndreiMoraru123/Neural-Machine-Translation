# standard imports
import os
import tarfile
import shutil
import subprocess

# third-party imports
import wget  # type: ignore


def download_data(data_folder: str):
    """
    Downloads the training, validation, and test files for WMT '14 en-de translation task.

    Training: Europarl v7, Common Crawl, News Commentary v9
    Validation: newstest2013
    Testing: newstest2014

    The homepage for the WMT '14 translation task, https://www.statmt.org/wmt14/translation-task.html, contains links to
    the datasets.

    :param data_folder: the folder where the files will be downloaded

    """
    train_urls = [
        "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
        "https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
        "http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz",
    ]

    print("\n\nThis may take a while.")

    # Create a folder to store downloaded TAR files
    if not os.path.isdir(os.path.join(data_folder, "tar files")):
        os.mkdir(os.path.join(data_folder, "tar files"))
    # Create a fresh folder to extract downloaded TAR files; previous extractions deleted to prevent tarfile errors
    if os.path.isdir(os.path.join(data_folder, "extracted files")):
        shutil.rmtree(os.path.join(data_folder, "extracted files"))
        os.mkdir(os.path.join(data_folder, "extracted files"))

    # Download and extract training data
    for url in train_urls:
        filename = url.split("/")[-1]
        if not os.path.exists(os.path.join(data_folder, "tar files", filename)):
            print("\nDownloading %s..." % filename)
            wget.download(url, os.path.join(data_folder, "tar files", filename))
        print("\nExtracting %s..." % filename)
        tar = tarfile.open(os.path.join(data_folder, "tar files", filename))
        members = [m for m in tar.getmembers() if "de-en" in m.path]
        tar.extractall(os.path.join(data_folder, "extracted files"), members=members)
        tar.close()

    print("\nExtraction completed.")

    # Download validation and testing data using sacreBLEU since we will be using this library to calculate BLEU scores
    subprocess.run(
        [
            "sacrebleu",
            "-t",
            "wmt13",
            "-l",
            "en-de",
            "--echo",
            "src",
            ">",
            os.path.join(data_folder, "val.en"),
        ],
        shell=True,
    )
    subprocess.run(
        [
            "sacrebleu",
            "-t",
            "wmt13",
            "-l",
            "en-de",
            "--echo",
            "ref",
            ">",
            os.path.join(data_folder, "val.de"),
        ],
        shell=True,
    )
    subprocess.run(
        [
            "sacrebleu",
            "-t",
            "wmt14/full",
            "-l",
            "en-de",
            "--echo",
            "src",
            ">",
            os.path.join(data_folder, "test.en"),
        ],
        shell=True,
    )
    subprocess.run(
        [
            "sacrebleu",
            "-t",
            "wmt14/full",
            "-l",
            "en-de",
            "--echo",
            "ref",
            ">",
            os.path.join(data_folder, "test.de"),
        ],
        shell=True,
    )

    # Move files if they were extracted into a subdirectory
    for dir in [
        d
        for d in os.listdir(os.path.join(data_folder, "extracted files"))
        if os.path.isdir(os.path.join(data_folder, "extracted files", d))
    ]:
        for f in os.listdir(os.path.join(data_folder, "extracted files", dir)):
            shutil.move(
                os.path.join(data_folder, "extracted files", dir, f),
                os.path.join(data_folder, "extracted files"),
            )
        os.rmdir(os.path.join(data_folder, "extracted files", dir))


if __name__ == "__main__":
    download_data("data")
