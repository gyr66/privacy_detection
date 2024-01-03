# coding=utf-8
import datasets
import pandas as pd


logger = datasets.logging.get_logger(__name__)


_DESCRIPTION = """\
privacy detection dataset, which includes the following categories of privacy information: [position, name, movie, organization, company, book, address, scene, mobile, email, game, government, QQ, vx].
The dataset consists of 3 columns. The first column is id, the second column is the list of text characters, and the third column is the list of privacy entity annotations. The entity annotation format is such that each entity's leading character is labeled as B-TYPE, the internal characters of the entity are labeled as I-TYPE, and the character is labeled O if it does not belong to any entity.
For more details see: https://www.datafountain.cn/competitions/472.
"""

_URL = "data.csv"


class PrivacyDetectionConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(PrivacyDetectionConfig, self).__init__(**kwargs)


class PrivacyDectection(datasets.GeneratorBasedBuilder):
    """PrivacyDectection dataset."""

    BUILDER_CONFIGS = [
        PrivacyDetectionConfig(
            name="privacy_detection",
            version=datasets.Version("1.0.0"),
            description="privacy detection dataset",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "B-position",
                                "I-position",
                                "B-name",
                                "I-name",
                                "B-movie",
                                "I-movie",
                                "B-organization",
                                "I-organization",
                                "B-company",
                                "I-company",
                                "B-book",
                                "I-book",
                                "B-address",
                                "I-address",
                                "B-scene",
                                "I-scene",
                                "B-mobile",
                                "I-mobile",
                                "B-email",
                                "I-email",
                                "B-game",
                                "I-game",
                                "B-government",
                                "I-government",
                                "B-QQ",
                                "I-QQ",
                                "B-vx",
                                "I-vx",
                            ]
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://www.datafountain.cn/competitions/472",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        downloaded_file = dl_manager.download_and_extract(_URL)
        data_files = {
            "train": downloaded_file,
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}
            )
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        data = pd.read_csv(filepath)
        for _, row in data.iterrows():
            id_ = row["id"]
            tokens = eval(row["tokens"])
            ner_tags = eval(row["ner_tags"])
            yield id_, {
                "id": id_,
                "tokens": tokens,
                "ner_tags": ner_tags,
            }
