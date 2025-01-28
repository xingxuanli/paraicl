import json
import torch
import numpy as np
from collections import Counter
from typing import List, Optional
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModel
from transformers import LogitsProcessor, LogitsProcessorList
from transformers.utils import add_start_docstrings


# ---------------------------------------------------------------------
# Custom LogitsProcessor
# ---------------------------------------------------------------------
LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head for each vocabulary token.

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.
"""

class ParallelICLLogitsProcessor(LogitsProcessor):
    """
    A custom LogitsProcessor that combines logits from multiple parallel prompts
    using various methods: 'average', 'majority', or 'weighted_average'.
    """

    def __init__(self, method: str, weights: Optional[List[float]] = None):
        """
        Args:
            method: One of ["average", "majority", "weighted_average"].
            weights: List of float weights to use if method == "weighted_average".
                     Length must match the number of parallel prompts (batch size).
        """
        valid_methods = ["average", "majority", "weighted_average"]
        if method not in valid_methods:
            raise ValueError(
                f"`method` must be one of {valid_methods}, but got '{method}'."
            )

        self.method = method
        self.weights = weights

        # If weighted_average, ensure we have a valid weights list
        if self.method == "weighted_average":
            if not weights or len(weights) == 0:
                raise ValueError(
                    "You must provide a non-empty `weights` list when method='weighted_average'."
                )

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Combine logits from multiple parallel prompts.
        - average: Mean of all prompts' logits
        - majority: Pick the batch row whose argmax is the majority, replicate it
        - weighted_average: Weighted mean of all prompts' logits
        """
        # scores shape: (batch_size, vocab_size)
        if self.method == "average":
            avg_scores = torch.mean(scores, dim=0)           # shape: (vocab_size,)
            scores = avg_scores.unsqueeze(0).repeat(scores.shape[0], 1)

        elif self.method == "majority":
            max_indices = torch.argmax(scores, dim=1)        # shape: (batch_size,)
            index_counts = Counter(max_indices.tolist())
            most_common_index = index_counts.most_common(1)[0][0]
            row_indices = [i for i, idx in enumerate(max_indices) if idx == most_common_index]
            scores = scores[row_indices[0]].unsqueeze(0).repeat(scores.shape[0], 1)

        elif self.method == "weighted_average":
            # Convert weights to a tensor and expand along vocab dimension
            weights_tensor = torch.tensor(self.weights, dtype=scores.dtype, device=scores.device)
            # Weighted sum over batch dimension
            weighted_sum = (scores.transpose(0, 1) * weights_tensor).transpose(0, 1).sum(dim=0)
            # Weighted average
            weighted_avg_scores = weighted_sum / weights_tensor.sum()
            # Replicate final logits across all rows
            scores = weighted_avg_scores.unsqueeze(0).repeat(scores.shape[0], 1)

        return scores
    


# ---------------------------------------------------------------------
# SimCSE Embedding Helpers
# ---------------------------------------------------------------------

# You can switch to another SimCSE model, e.g. "princeton-nlp/sup-simcse-bert-base-uncased"
EMBEDDING_MODEL_NAME = "princeton-nlp/unsup-simcse-bert-base-uncased"

embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model.to(device)

def get_embedding(text: str) -> torch.Tensor:
    """
    Get a normalized sentence embedding for the given text
    using SimCSE from the Princeton-NLP repository.
    """
    inputs = embedding_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        # For SimCSE, we typically use the pooler_output for an embedding
        # if available. If not, you can mean-pool the last hidden states.
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            # fallback to mean-pool if pooler_output not present
            embeddings = outputs.last_hidden_state.mean(dim=1)

    # Normalize to unit length for cosine similarity
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    return embeddings[0]  # 1D vector


def compute_batch_weights(question: str, training_data: List[List[Dict[str, Any]]]) -> List[float]:
    """
    Given a test question and M batches of training examples,
    compute a similarity score for each batch, then normalize.

    training_data: list of M batches, each batch is list of N examples
    (each example is a dict with keys "question", "answer").

    Return: a list of M weights in [0, 1] that sum to 1.
    """
    question_emb = get_embedding(question)

    cos_sims = []
    for batch_examples in training_data:
        # For each batch, we sum (or average) the similarity to each training example's question.
        # You could also combine Q + A if you prefer. Here we just use the question text.
        batch_sim_sum = 0.0
        for ex in batch_examples:
            ex_emb = get_embedding(ex["question"])
            cos_sim = float(torch.dot(question_emb, ex_emb))
            # clamp negative to 0 if you want
            batch_sim_sum += max(cos_sim, 0.0)

        cos_sims.append(batch_sim_sum)

    total = sum(cos_sims) if sum(cos_sims) != 0 else 1e-9
    normalized_weights = [x / total for x in cos_sims]
    return normalized_weights



# ---------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------
def build_prompt(batch_examples: List[Dict[str, Any]], test_question: str) -> str:
    """
    Given N training examples + a final question, build a single prompt.
    """
    # Example structure
    # "Instruction: ...\n\n"
    # for each training example: Q / A
    # then final question + "Answer: "
    prompt_str = "Instruction: Please answer the question below. Follow the example answer format, only give the answer.\n\n"
    for ex in batch_examples:
        prompt_str += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
    prompt_str += f"Question: {test_question}\nAnswer: "
    return prompt_str



# ---------------------------------------------------------------------
# Main Script
# ---------------------------------------------------------------------
def main():
    # Example usage: We have M=3 batches, each with N=2 training examples, etc.
    # For demonstration, let's define or load it from somewhere:
    training_data = [
        [
            {"question": "When was Apple founded?", "answer": "1976."},
            {"question": "Who is the CEO of Apple?", "answer": "Tim Cook."},
        ],
        [
            {"question": "What is the capital of France?", "answer": "Paris."},
            {"question": "Who wrote Les Misérables?", "answer": "Victor Hugo."},
        ],
        [
            {"question": "What is the chemical symbol for water?", "answer": "H2O."},
            {"question": "Who proposed the theory of relativity?", "answer": "Albert Einstein."},
        ],
    ]
    # This is just an example structure. Adjust or load your own MxN training examples.

    # A file of test data, each with a question
    test_data_path = "datasets/hotpotqa.json"
    output_path = "results/hotpotqa_mxN_simcse_weighted.json"

    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    # Load Llama model
    llama_model_name = "meta-llama/Llama-2-7b-chat-hf"
    print("Loading LLaMA tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(llama_model_name, padding_side="left")
    model = LlamaForCausalLM.from_pretrained(llama_model_name).to(device)

    # Define PAD Token = BOS Token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    model.config.pad_token_id = model.config.bos_token_id

    # We'll do Weighted Average
    method = "weighted_average"

    print("Generating answers...")
    all_results = []
    for i, sample in enumerate(test_data):
        print(f"Processing item {i}...")
        question = sample["question"]

        # 1) Compute the weights for M batches based on the test question.
        weights = compute_batch_weights(question, training_data)

        # 2) Build each of the M parallel prompts
        parallel_prompts = [
            build_prompt(batch_examples, question)
            for batch_examples in training_data
        ]

        # 3) Create a LogitsProcessorList with Weighted Average
        logits_processor_list = LogitsProcessorList([
            ParallelICLLogitsProcessor(method=method, weights=weights),
        ])

        # 4) Tokenize and run generation. We have M parallel prompts in one batch.
        inputs = tokenizer(parallel_prompts, return_tensors="pt", padding=True).to(device)

        output_sequences = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            logits_processor=logits_processor_list
        )

        # 5) Decode. By design, the weighted ensemble means all M outputs
        # converge step by step, but the model returns M sequences.
        # We can just take the first one or any one as they should be identical
        # after weighting. If you want to confirm, you could compare them.

        decoded = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        # Because each of the M outputs was the same length of prefix, we can do:
        prefix_len = len(parallel_prompts[0])
        generated_answer = decoded[0][prefix_len:].strip()

        sample["generated_results"] = generated_answer
        all_results.append(sample)

    print(f"Writing results to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print("Done.")


if __name__ == "__main__":
    main()