import json
import logging
import os

from typing import List, Optional, Union

import torch
import tqdm
from transformers.data import DataProcessor, InputExample, InputFeatures, SingleSentenceClassificationProcessor
from sklearn.metrics import f1_score
from transformers import (
    PreTrainedTokenizer,
    glue_compute_metrics,
    glue_output_modes,
    glue_processors
)
logger = logging.getLogger(__name__)
from torch.utils.data import TensorDataset


class MCInputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None):
        """Constructs an MCInputExample.
        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label

class MCInputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label

def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float]:
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b if example.text_b is not None else '') for example in examples],
        max_length=max_length, pad_to_max_length=True, return_token_type_ids=True
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    # for i, example in enumerate(examples[:5]):
    #     logger.info("*** Example ***")
    #     logger.info("guid: %s" % (example.guid))
    #     logger.info("features: %s" % features[i])

    return features


def convert_mc_examples_to_features(
    examples: List[MCInputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[MCInputFeatures]:
    """
    Loads a data file into a list of `MCInputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="converting MC examples to features"):
        if ex_index % 10000 == 0:
            logger.info(" Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            inputs = tokenizer.encode_plus(
                text_a, text_b, add_special_tokens=True, max_length=max_length, return_token_type_ids=True
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids))

        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("winogrande_id: {}".format(example.example_id))
            logger.info("winogrande_context: {}".format(example.contexts[0]))
            for choice_idx, (input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
                logger.info(f"choice {choice_idx}: {example.endings[choice_idx]}")
                logger.info("input_ids: {}".format(" ".join(map(str, input_ids))))
                logger.info("attention_mask: {}".format(" ".join(map(str, attention_mask))))
                logger.info("token_type_ids: {}".format(" ".join(map(str, token_type_ids))))
                logger.info(f"label: {label == choice_idx}")

        features.append(MCInputFeatures(example_id=example.example_id,
                                        choices_features=choices_features,
                                        label=label,))

    return features

def get_winogrande_tensors(features):
    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    # Convert to Tensors and build dataset
    input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
    input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
    segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
    label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    example_ids = torch.tensor([f.example_id for f in features], dtype=torch.long)

    dataset = TensorDataset(input_ids, input_mask, segment_ids, label_ids, example_ids)
    return dataset

def load_and_cache_examples(args, task, tokenizer, evaluate=False, data_split="train"):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    # if test:
    #     data_split = "test"
    # elif evaluate:
    #     data_split = "dev"
    # else:
    #     data_split = "train"
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            data_split,
            list(filter(None, args.base_model.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        if data_split == 'test':
            examples = processor.get_test_examples(args.data_dir)
        elif data_split == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        elif data_split == 'train':
            examples = processor.get_train_examples(args.data_dir)
        else:
            logger.warning(f"{data_split} is not specified")
            raise NotImplementedError
        if task in ['winogrande']:
            features = convert_mc_examples_to_features(
                examples,
                label_list,
                args.max_seq_length,
                tokenizer,
                pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=tokenizer.pad_token_type_id, )
        else:
            features = convert_examples_to_features(
                examples,
                tokenizer,
                label_list=label_list,
                max_length=args.max_seq_length,
                output_mode=output_mode,
            )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if task == "winogrande":
        return get_winogrande_tensors(features)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


class SentenceDataProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return NotImplementedError

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

# monkey-patch all glue classes to have test examples
# def get_test_examples(self, data_dir):
#     return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
#
# for task in glue_processors:
#     processor = glue_processors[task]
#     processor.get_test_examples = get_test_examples

# Override Cola for test dataset
class ColaProcessor(SentenceDataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class SST2Processor(SentenceDataProcessor):
    def get_labels(self):
        return ["0", "1"]

# Other datasets
class PubMedProcessor(SentenceDataProcessor):
    def get_labels(self):
        labels = ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]
        return labels

class AGNewsProcessor(SentenceDataProcessor):
    def get_labels(self):
        labels = ["World", "Sports", "Business", "Sci/Tech"]
        return labels

class IMDBProcessor(SentenceDataProcessor):
    def get_labels(self):
        labels = ["pos", "neg"]
        return labels
class WinograndeProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "train.jsonl")))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "dev.jsonl")))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "test.jsonl")))

    def get_labels(self):
        return ["1", "2"]

    def _read_jsonl(cls, input_file):
        """Reads a tab separated value file."""
        records = []
        with open(input_file, "r", encoding="utf-8-sig") as f:
            for line in f:
                records.append(json.loads(line))
            return records

    def _build_example_from_named_fields(self, guid, sentence, name1, name2, label):
      conj = "_"
      idx = sentence.index(conj)
      context = sentence[:idx]
      option_str = "_ " + sentence[idx + len(conj):].strip()

      option1 = option_str.replace("_", name1)
      option2 = option_str.replace("_", name2)

      mc_example = MCInputExample(
          example_id=int(guid),
          contexts=[context, context],
          question=conj,
          endings = [option1, option2],
          label=label
      )
      return mc_example

    def _create_examples(self, records):
        examples = []
        for (i, record) in enumerate(records):
            sentence = record['sentence']

            name1 = record['option1']
            name2 = record['option2']
            if not 'answer' in record:
                # This is a dummy label for test prediction.
                # test.jsonl doesn't include the `answer`.
                label = "1"
            else:
                label = record['answer']

            mc_example = self._build_example_from_named_fields(i, sentence, name1, name2, label)
            examples.append(mc_example)

        return examples

class DBPEDIAProcessor(SentenceDataProcessor):
    def get_labels(self):
        """See base class."""
        return ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



processors = glue_processors.copy()
processors.update(
    {"pubmed": PubMedProcessor, "agnews": AGNewsProcessor, "imdb": IMDBProcessor, "winogrande": WinograndeProcessor,
     "cola": ColaProcessor, "sst2": SST2Processor, "qnli":QnliProcessor, "rte": RteProcessor, "dbpedia": DBPEDIAProcessor}
)
output_modes = glue_output_modes
output_modes.update(
    {"pubmed":"classification", "agnews":"classification", "imdb":"classification", "winogrande":"multiple_choice",
     "sst2":"classification", "dbpedia":"classification"}
)

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name in ["pubmed", "agnews", "imdb","sst2","dbpedia"]:
        return {"f1":f1_score(y_true=labels, y_pred=preds, average="micro")}
    elif task_name == "winogrande":
        return glue_compute_metrics("mnli", preds, labels)
    elif task_name in glue_processors:
        return glue_compute_metrics(task_name, preds, labels)
    else:
        raise NotImplementedError

