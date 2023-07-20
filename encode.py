# standard imports
import os
import codecs

# third-party imports
import youtokentome  # type: ignore
from tqdm import tqdm  # type: ignore


def tokenize_and_filter_data(data_folder: str, euro_parl: bool = True, common_crawl: bool = True,
                             news_commentary: bool = True, min_length: int = 3, max_length: int = 100,
                             max_length_ratio: float = 1.5, vocab_size: int = 37000, retain_case: bool = True):
    """
    Filters and prepares the training data, trains a Byte-Pair Encoding (BPE) model.

    :param vocab_size: the total size of the vocabulary
    :param data_folder: the folder where the files were downloaded
    :param euro_parl: include the Europarl v7 dataset in the training data?
    :param common_crawl: include the Common Crawl dataset in the training data?
    :param news_commentary: include theNews Commentary v9 dataset in the training data?
    :param min_length: exclude sequence pairs where one or both are shorter than this minimum BPE length
    :param max_length: exclude sequence pairs where one or both are longer than this maximum BPE length
    :param max_length_ratio: exclude sequence pairs where one is much longer than the other
    :param retain_case: retain case?
    """
    # Read raw files and combine
    german = list()
    english = list()
    files = list()
    assert euro_parl or common_crawl or news_commentary, "Set at least one dataset to True!"
    if euro_parl:
        files.append("europarl-v7.de-en")
    if common_crawl:
        files.append("commoncrawl.de-en")
    if news_commentary:
        files.append("news-commentary-v9.de-en")
    print("\nReading extracted files and combining...")
    for file in files:
        with codecs.open(os.path.join(data_folder, "extracted files", file + ".de"), "r", encoding="utf-8") as f:
            if retain_case:
                german.extend(f.read().split("\n"))
            else:
                german.extend(f.read().lower().split("\n"))
        with codecs.open(os.path.join(data_folder, "extracted files", file + ".en"), "r", encoding="utf-8") as f:
            if retain_case:
                english.extend(f.read().split("\n"))
            else:
                english.extend(f.read().lower().split("\n"))
        assert len(english) == len(german)

    # Write to file so stuff can be freed from memory
    print("\nWriting to single files...")
    with codecs.open(os.path.join(data_folder, "train.en"), "w", encoding="utf-8") as f:
        f.write("\n".join(english))
    with codecs.open(os.path.join(data_folder, "train.de"), "w", encoding="utf-8") as f:
        f.write("\n".join(german))
    with codecs.open(os.path.join(data_folder, "train.ende"), "w", encoding="utf-8") as f:
        f.write("\n".join(english + german))
    del english, german  # free some RAM

    # Perform BPE
    print("\nLearning BPE...")
    youtokentome.BPE.train(data=os.path.join(data_folder, "train.ende"),
                           model=os.path.join(data_folder, "bpe.model"),
                           vocab_size=vocab_size)

    # Load BPE model
    print("\nLoading BPE model...")
    bpe_model = youtokentome.BPE(model=os.path.join(data_folder, "bpe.model"))

    # Re-read English, German
    print("\nRe-reading single files...")
    with codecs.open(os.path.join(data_folder, "train.en"), "r", encoding="utf-8") as f:
        english = f.read().split("\n")
    with codecs.open(os.path.join(data_folder, "train.de"), "r", encoding="utf-8") as f:
        german = f.read().split("\n")

    # Filter
    print("\nFiltering...")
    pairs = list()
    for en, de in tqdm(zip(english, german), total=len(english)):
        en_tok = bpe_model.encode(en, output_type=youtokentome.OutputType.ID)
        de_tok = bpe_model.encode(de, output_type=youtokentome.OutputType.ID)
        len_en_tok = len(en_tok)
        len_de_tok = len(de_tok)
        if min_length < len_en_tok < max_length and \
                min_length < len_de_tok < max_length and \
                1. / max_length_ratio <= len_de_tok / len_en_tok <= max_length_ratio:
            pairs.append((en, de))
        else:
            continue
    print("\nNote: %.2f pct. of en-de pairs were filtered out." % (100. * (len(english) - len(pairs)) / len(english)))

    # Rewrite files
    english, german = zip(*pairs)
    print("\nRe-writing filtered sentences to single files...")
    os.remove(os.path.join(data_folder, "train.en"))
    os.remove(os.path.join(data_folder, "train.de"))
    os.remove(os.path.join(data_folder, "train.ende"))
    with codecs.open(os.path.join(data_folder, "train.en"), "w", encoding="utf-8") as f:
        f.write("\n".join(english))
    with codecs.open(os.path.join(data_folder, "train.de"), "w", encoding="utf-8") as f:
        f.write("\n".join(german))
    del english, german, bpe_model, pairs

    print("\n...DONE!\n")


if __name__ == "__main__":
    tokenize_and_filter_data(data_folder="data", euro_parl=True, common_crawl=False, news_commentary=False,
                             min_length=3, max_length=100, max_length_ratio=2.0,
                             retain_case=True, vocab_size=30000)
