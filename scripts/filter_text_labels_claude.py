import json
import random
from pathlib import Path

from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from simap.user_config import get_user_config

SIMAP_DIR = Path(get_user_config()["simap_dir"])

EFFORT = "high"  # low | medium | high | xhigh | max
MAX_TOKENS = 2048  # caps thinking plus answer together
THINKING = {"type": "adaptive"}
SENTINEL = "__NONE__"


class PromptGenerator:
    def __init__(
        self,
        text_label_set: set[str] | None = None,
        sample_size: int | None = None,
        seed: int = 0,
    ):
        if text_label_set is None:
            with open(SIMAP_DIR / "metadata.json", "r") as f:
                metadata = json.load(f)
            self.text_label_set = set([m["text_label"] for m in metadata])
        else:
            self.text_label_set = text_label_set
        if sample_size is not None and sample_size < len(self.text_label_set):
            # a seeded random sample, for a small trial batch before the whole set.
            # Random rather than the first n, because sorting puts every label
            # beginning with punctuation or a digit first, which is not representative.
            self.text_label_set = set(
                random.Random(seed).sample(sorted(self.text_label_set), sample_size)
            )
        if len(self.text_label_set) > 100000:
            raise ValueError(
                f"The number of text labels is {len(self.text_label_set)}, "
                f"However, a single batch may include up to 100,000 requests. "
                f"Please split the text labels into multiple batches."
            )

    def generate_prompt(self, text_label: str) -> str:
        prompt = f"""Please filter this instrument/preset name: "{text_label}". 
Your goal is to keep only the entries that describe a musical sound, timbre, or instrument.
Strictly remove:
Generic bank/program info (e.g., "Instr bnk 0", "Instrument 0-0").
Memory addresses or hex codes (e.g., "@0x503e44").
Standalone numbers or generic markers (e.g., "5", "Marker #15").
Trailing variant numbers (e.g., the "2" in "Piano 2").
Non-descriptive file identifiers (e.g., "midi-32").
Unrecognized prefixes (e.g., the "BFB" in "BFB Koto").
Keep:
Actual instrument names (e.g., Piano, Trumpet, TR-808).
Descriptors/Adjectives (e.g., Bright, FM).
Sound effects (e.g., Door Slam, Scream).
Meaningful trailing numbers (e.g., the "16" in "Bourdon 16").
Transform:
Meaningful coded names into full forms (e.g., "SynBa" to "Synth Bass").
Your response:
Only the filtered name on one line, or exactly {SENTINEL} if nothing should be kept.
Do not include any other text."""
        return prompt

    def write_batch_prompts(self, prompt_filepath: str | Path, model_name: str) -> None:
        """
        Write one request per text label, as jsonl.

        Unlike the OpenAI batch file this is not what gets uploaded -- the Anthropic
        batch endpoint takes the requests inline -- but it is still written to disk,
        because a custom_id may only hold [A-Za-z0-9_-] up to 64 characters and a
        text_label holds anything at all. The id is therefore positional and this file
        is what maps it back, so get_mapping needs it alongside the responses.

        Labels are sorted so the same label set always produces the same ids.
        """
        with open(prompt_filepath, "w") as f:
            for i, text_label in enumerate(sorted(self.text_label_set)):
                prompt = self.generate_prompt(text_label)
                request = {
                    "custom_id": f"label-{i:06d}",
                    "text_label": text_label,
                    "params": {
                        "model": model_name,
                        "max_tokens": MAX_TOKENS,
                        "thinking": THINKING,
                        "output_config": {"effort": EFFORT},
                        "messages": [{"role": "user", "content": prompt}],
                    },
                }
                f.write(json.dumps(request) + "\n")


def submit_requests(prompt_filepath: str | Path) -> None:
    client = Anthropic()
    requests = []
    with open(prompt_filepath, "r") as f:
        for line in f:
            entry = json.loads(line)
            requests.append(
                Request(
                    custom_id=entry["custom_id"],
                    params=MessageCreateParamsNonStreaming(**entry["params"]),
                )
            )
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch submitted for {prompt_filepath}. ID: {batch.id}")


def get_responses(batch_id: str, response_filepath: str | Path) -> None:
    client = Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        raise ValueError(
            f"Batch is still {batch.processing_status!r}, so results are not ready. "
            f"Counts so far: {batch.request_counts}"
        )
    with open(response_filepath, "w") as f:
        for result in client.messages.batches.results(batch_id):
            f.write(json.dumps(result.to_dict()) + "\n")


def get_mapping(
    prompt_filepath: str | Path, response_filepath: str | Path
) -> dict[str, str]:
    """
    Join the responses back onto their text labels.

    Results come back in an arbitrary order, so they are keyed by custom_id rather
    than by position, and the prompt file is what turns a custom_id back into the
    label it was written for.
    """
    text_labels = {}
    with open(prompt_filepath, "r") as f:
        for line in f:
            entry = json.loads(line)
            text_labels[entry["custom_id"]] = entry["text_label"]

    mapping, failed, truncated = {}, 0, 0
    with open(response_filepath, "r") as f:
        for line in f:
            result = json.loads(line)
            if result["result"]["type"] != "succeeded":
                failed += 1  # errored, canceled or expired -- resubmit these labels
                continue
            message = result["result"]["message"]
            # MAX_TOKENS caps thinking plus response text together, so a request that
            # thought its way to the ceiling still succeeds while returning a cut-off
            # answer or none at all -- and an empty answer is indistinguishable from
            # the model deliberately removing the label. Drop these instead.
            if message["stop_reason"] == "max_tokens":
                truncated += 1
                continue
            text_label = text_labels[result["custom_id"]]
            # join the text blocks and skip the rest: a thinking block precedes them
            # whenever the model thought, so the first block is not always the answer
            filtered_text_label = "".join(
                block["text"]
                for block in message["content"]
                if block["type"] == "text"
            ).strip()
            if filtered_text_label == SENTINEL:
                filtered_text_label = ""  # the sentinel is how the model says "remove"
            mapping[text_label] = filtered_text_label
    if failed:
        print(f"{failed} requests did not succeed and are missing from the mapping.")
    if truncated:
        print(
            f"{truncated} requests hit MAX_TOKENS ({MAX_TOKENS}) and are missing from "
            f"the mapping. Raise it, or lower EFFORT, and resubmit those labels."
        )
    return mapping


if __name__ == "__main__":

    prompt_filepath = "prompts.jsonl"
    response_filepath = "responses.jsonl"
    model_name = "claude-sonnet-5"

    # The following example code submits requests and gets responses in batches using the Anthropic API.
    # It requires the Anthropic API key to be set in the environment variable ANTHROPIC_API_KEY,
    # or an `ant auth login` profile, which the client picks up on its own.
    # It is not free, though the batch endpoint is half the price of the same calls made one at a time.
    # Use it at your own consideration.

    # Pass sample_size for a small trial batch first -- it costs cents, times the round
    # trip, and shows what the filter does to the labels before the whole set is spent on.
    # prompt_generator = PromptGenerator(sample_size=400)

    # prompt_generator = PromptGenerator()
    # prompt_generator.write_batch_prompts(prompt_filepath, model_name)
    # submit_requests(prompt_filepath)

    # Batch processing is *asynchronous*. Monitor it in the Anthropic Console, or poll
    # client.messages.batches.retrieve(batch_id).processing_status until it is "ended".
    # Most batches finish within an hour; the maximum is 24 hours, and results are kept
    # for 29 days. Read the comments below before running the code.

    # batch_id = None  # Is printed by submit_requests. You can also find it in the Anthropic Console.
    # get_responses(batch_id, response_filepath)  # Wait until the batch is completed to run this.

    # mapping = get_mapping(prompt_filepath, response_filepath)
    # for text_label, filtered_text_label in sorted(mapping.items()):
    #     print(f"{text_label!r:40} -> {filtered_text_label!r}")

    # The mapping goes to the data dir, beside metadata.parquet, because the
    # downstream scripts read it from there by default.
