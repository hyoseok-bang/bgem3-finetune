import torch
from transformers import Trainer


class ContrastiveTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        if kwargs.get("train_dataset") is not None:
            print("train_dataset keys: ", kwargs["train_dataset"][0].keys())
        if kwargs.get("eval_dataset") is not None:
            print("eval_dataset keys: ", kwargs["eval_dataset"][0].keys())
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False
    ):
        print(f"compute_loss inputs 키: {inputs.keys()}")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        batch_size = input_ids.size(0)

        # input_ids와 attention_mask는 (batch_size, seq_length) 형태일 것입니다.
        # 이진 분류를 위해 두 문장을 하나로 합쳐서 토큰화했을 가능성을 고려하여 처리합니다.
        # 또는 데이터셋 생성 시 이미 [text1, text2] 형태로 되어 있다면 아래와 같이 분리할 수 있습니다.
        # (batch_size, 2, seq_length) 형태라고 가정하고 분리합니다.
        if len(input_ids.shape) > 2 and input_ids.shape[1] == 2:
            input_ids_1 = input_ids[:, 0, :]
            attention_mask_1 = attention_mask[:, 0, :]
            input_ids_2 = input_ids[:, 1, :]
            attention_mask_2 = attention_mask[:, 1, :]
        else:
            # 두 문장이 합쳐져서 토큰화된 경우, 적절한 분리 로직이 필요합니다.
            # 이 예시에서는 간단하게 처리하거나, 데이터셋 생성 방식을 수정하는 것을 고려합니다.
            raise ValueError(
                "입력 형태가 예상과 다릅니다. 데이터셋 생성 및 토큰화 과정을 확인하세요."
            )

        # 두 개의 텍스트를 각각 인코딩
        try:
            outputs1 = model(input_ids_1, attention_mask=attention_mask_1)
            embeddings1 = outputs1.last_hidden_state.mean(dim=1)  # 또는 다른 풀링 방식

            outputs2 = model(input_ids_2, attention_mask=attention_mask_2)
            embeddings2 = outputs2.last_hidden_state.mean(dim=1)  # 또는 다른 풀링 방식

            # 코사인 유사도 계산
            cos_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)

            # 목표: 같은 의미면 유사도 높이기 (label 1), 다른 의미면 유사도 낮추기 (label 0)
            # 간단한 이진 분류 손실 함수 (조정 필요)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            logits = cos_sim  # 코사인 유사도를 로짓으로 사용 (적절한 스케일링 필요할 수 있음)
            loss = loss_fn(logits.unsqueeze(1), labels.float().unsqueeze(1))

            print(f"배치 손실: {loss.item()}")
            return loss
        except Exception as e:
            print(f"compute_loss 오류 발생: {e}")
            print(f"입력 input_ids 형태: {input_ids.shape}")
            print(f"입력 attention_mask 형태: {attention_mask.shape}")
            print(f"입력 labels 형태: {labels.shape}")
            raise


import torch
import torch.nn.functional as F
from typing import Tuple


# --- Helper Function to Calculate Pairwise Loss ---
def calculate_pairwise_loss(
    emb_k: torch.Tensor,
    emb_e: torch.Tensor,
    emb_ktoe: torch.Tensor,
    emb_etok: torch.Tensor,
    loss_type: str = "cosine",  # 'cosine' or 'mse'
) -> torch.Tensor:
    """
    Calculates the sum of pairwise losses (1-cosine_sim or mse)
    between all unique pairs in the quadruplet.

    Args:
        emb_k, emb_e, emb_ktoe, emb_etok: Tensors of shape (batch_size, embedding_dim)
        loss_type: 'cosine' for 1 - cosine_similarity, 'mse' for squared L2 distance.

    Returns:
        Tensor of shape (batch_size,) containing the sum of pairwise losses for each item.
    """
    embeddings = [emb_k, emb_e, emb_ktoe, emb_etok]
    batch_size = emb_k.shape[0]
    total_pairwise_loss = torch.zeros(batch_size, device=emb_k.device)
    num_pairs = 0

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            emb_i = embeddings[i]
            emb_j = embeddings[j]

            if loss_type == "cosine":
                # 1 - cosine similarity
                sim = F.cosine_similarity(emb_i, emb_j, dim=1)
                loss = 1.0 - sim
            elif loss_type == "mse":
                # Squared L2 distance
                loss = torch.sum((emb_i - emb_j) ** 2, dim=1)
            else:
                raise ValueError("loss_type must be 'cosine' or 'mse'")

            total_pairwise_loss += loss
            num_pairs += 1

    # Optional: Average over the number of pairs if desired,
    # but summing works fine as it's consistent.
    # return total_pairwise_loss / num_pairs
    return total_pairwise_loss


# --- Helper Function for Scale Regularization ---
def calculate_scale_regularization(
    emb_k: torch.Tensor,
    emb_e: torch.Tensor,
    emb_ktoe: torch.Tensor,
    emb_etok: torch.Tensor,
    target_norm: float,
) -> torch.Tensor:
    """
    Calculates the scale regularization loss.

    Args:
        emb_k, emb_e, emb_ktoe, emb_etok: Tensors of shape (batch_size, embedding_dim)
        target_norm: The target L2 norm scalar value.

    Returns:
        Tensor of shape (batch_size,) containing the scale regularization loss for each item.
    """
    embeddings = [emb_k, emb_e, emb_ktoe, emb_etok]
    total_scale_loss = torch.zeros(emb_k.shape[0], device=emb_k.device)

    for emb in embeddings:
        norm = torch.norm(emb, p=2, dim=1)
        scale_loss = (norm - target_norm) ** 2
        total_scale_loss += scale_loss

    # Average over the 4 embedding types
    return total_scale_loss / len(embeddings)


# --- Loss Function Implementations ---


def compute_cosine_loss_with_scale_reg(
    emb_k: torch.Tensor,
    emb_e: torch.Tensor,
    emb_ktoe: torch.Tensor,
    emb_etok: torch.Tensor,
    target_norm: float,
    lambda_scale: float = 0.01,  # Hyperparameter to balance terms
) -> torch.Tensor:
    """
    Computes Loss = (Pairwise 1-CosineSim) + lambda * (Scale Regularization)

    Args:
        emb_k, emb_e, emb_ktoe, emb_etok: Tensors of shape (batch_size, embedding_dim)
        target_norm: The target L2 norm.
        lambda_scale: Weight for the scale regularization term.

    Returns:
        Scalar loss tensor averaged over the batch.
    """
    # Pairwise Cosine Similarity Loss (mean over batch)
    pairwise_loss = calculate_pairwise_loss(
        emb_k, emb_e, emb_ktoe, emb_etok, loss_type="cosine"
    )
    loss_sim = torch.mean(pairwise_loss)

    # Scale Regularization Loss (mean over batch)
    scale_loss = calculate_scale_regularization(
        emb_k, emb_e, emb_ktoe, emb_etok, target_norm
    )
    loss_scale = torch.mean(scale_loss)

    # Total Loss
    total_loss = loss_sim + lambda_scale * loss_scale
    return total_loss


def compute_mse_loss_with_scale_reg(
    emb_k: torch.Tensor,
    emb_e: torch.Tensor,
    emb_ktoe: torch.Tensor,
    emb_etok: torch.Tensor,
    target_norm: float,
    lambda_scale: float = 0.01,  # Hyperparameter to balance terms
) -> torch.Tensor:
    """
    Computes Loss = (Pairwise MSE) + lambda * (Scale Regularization)

    Args:
        emb_k, emb_e, emb_ktoe, emb_etok: Tensors of shape (batch_size, embedding_dim)
        target_norm: The target L2 norm.
        lambda_scale: Weight for the scale regularization term.

    Returns:
        Scalar loss tensor averaged over the batch.
    """
    # Pairwise MSE Loss (mean over batch)
    pairwise_loss = calculate_pairwise_loss(
        emb_k, emb_e, emb_ktoe, emb_etok, loss_type="mse"
    )
    loss_dist = torch.mean(pairwise_loss)

    # Scale Regularization Loss (mean over batch)
    scale_loss = calculate_scale_regularization(
        emb_k, emb_e, emb_ktoe, emb_etok, target_norm
    )
    loss_scale = torch.mean(scale_loss)

    # Total Loss
    total_loss = loss_dist + lambda_scale * loss_scale
    return total_loss


# --- Placeholder for Scale-Regularized Contrastive Loss (InfoNCE style) ---
# This requires a more complex setup with negative sampling, often done via in-batch negatives.


def compute_infonce_loss_with_scale_reg(
    emb_k: torch.Tensor,
    emb_e: torch.Tensor,
    emb_ktoe: torch.Tensor,
    emb_etok: torch.Tensor,
    target_norm: float,
    lambda_scale: float = 0.01,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Computes InfoNCE-style Contrastive Loss + Scale Regularization.
    Assumes in-batch negative sampling.

    Args:
        emb_k, emb_e, emb_ktoe, emb_etok: Tensors of shape (batch_size, embedding_dim)
        target_norm: The target L2 norm.
        lambda_scale: Weight for the scale regularization term.
        temperature: Temperature scaling factor for InfoNCE.

    Returns:
        Scalar loss tensor averaged over the batch.
    """
    device = emb_k.device
    batch_size = emb_k.shape[0]
    embedding_dim = emb_k.shape[1]

    # Concatenate all embeddings for easier similarity matrix calculation
    # Shape: (batch_size * 4, embedding_dim)
    all_embeddings = torch.cat([emb_k, emb_e, emb_ktoe, emb_etok], dim=0)

    # Calculate pairwise cosine similarities (logits)
    # Shape: (batch_size * 4, batch_size * 4)
    logits = F.cosine_similarity(
        all_embeddings.unsqueeze(1), all_embeddings.unsqueeze(0), dim=2
    )
    logits /= temperature  # Apply temperature scaling

    # Create labels and masks for InfoNCE
    # Labels indicate the index of the positive counterpart within the large batch.
    # For InfoNCE, typically anchor vs one positive, others negative.
    # Here, we have multiple positives per anchor within the quadruplet.

    # Simplified approach: Treat each K as anchor, E/KtoE/EtoK as positives. Repeat for other anchors.
    total_contrastive_loss = torch.tensor(0.0, device=device)
    anchors = [emb_k, emb_e, emb_ktoe, emb_etok]
    anchor_indices = [
        torch.arange(batch_size) + i * batch_size for i in range(4)
    ]  # Indices for K, E, KtoE, EtoK in all_embeddings

    for i in range(4):  # Iterate through K, E, KtoE, EtoK as anchors
        current_anchor_indices = anchor_indices[i]
        # Logits for the current anchors vs all embeddings
        anchor_logits = logits[
            current_anchor_indices
        ]  # Shape: (batch_size, batch_size * 4)

        # Create target labels: indicate positive pairs
        # Positive keys are the other embeddings from the *same* original item.
        labels = torch.zeros_like(anchor_logits, dtype=torch.bool)
        for j in range(4):
            if i == j:
                continue  # Don't mark anchor as its own positive
            positive_indices = anchor_indices[j]
            # Mark True where the column index corresponds to a positive pair for the row anchor
            # Need element-wise check. A bit tricky with broadcasted indices.
            # Let's usearange for batch items:
            batch_indices = torch.arange(batch_size, device=device)
            labels[batch_indices, positive_indices] = True

        # InfoNCE Loss calculation: log_softmax over all pairs, gather positives.
        log_prob = F.log_softmax(anchor_logits, dim=1)

        # Sum log probabilities of *all* positive pairs for the anchor
        # Avoid division by zero if no positives (shouldn't happen here)
        mean_log_prob_pos = (log_prob * labels).sum(1) / labels.sum(1).clamp(min=1)

        # Contrastive loss is the negative of the mean log probability of positives
        contrastive_loss = -mean_log_prob_pos
        total_contrastive_loss += contrastive_loss.mean()  # Average over batch

    # Average contrastive loss over the 4 anchor types
    loss_contrastive = total_contrastive_loss / 4.0

    # Scale Regularization Loss (mean over batch)
    scale_loss = calculate_scale_regularization(
        emb_k, emb_e, emb_ktoe, emb_etok, target_norm
    )
    loss_scale = torch.mean(scale_loss)

    # Total Loss
    total_loss = loss_contrastive + lambda_scale * loss_scale
    return total_loss


class CustomTrainer(Trainer):
    def __init__(
        self, *args, target_norm=1.0, lambda_scale=0.01, loss_type="cosine", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.target_norm = target_norm
        self.lambda_scale = lambda_scale
        self.loss_type = loss_type  # 'cosine', 'mse', or 'infonce'

        # --- Pre-calculate Target Norm (Recommended) ---
        # It's best to calculate this once before training starts.
        # You might need a separate script or do it here based on the original model
        # and a sample of your training data.
        # self.target_norm = self._calculate_target_norm(self.model, self.train_dataset)
        # print(f"Using target norm: {self.target_norm}")
        # Let's assume it's pre-set for this example
        print(
            f"Initialized CustomTrainer with target_norm={self.target_norm}, lambda_scale={self.lambda_scale}, loss_type='{self.loss_type}'"
        )

    # --- Placeholder function for target norm calculation ---
    # def _calculate_target_norm(self, model, dataset):
    #     # Logic to iterate through dataset, get original embeddings, compute avg norm
    #     # Remember to set model to eval mode and use torch.no_grad()
    #     print("Calculating target norm...")
    #     # ... implementation details ...
    #     avg_norm = 1.0 # Replace with actual calculation
    #     return avg_norm

    def compute_loss(
        self, model, inputs, num_items_in_batch=None, return_outputs=False
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # --- 1. Extract Inputs ---
        # Assumes your dataset provides inputs like:
        # 'input_ids_k', 'attention_mask_k',
        # 'input_ids_e', 'attention_mask_e',
        # 'input_ids_ktoe', 'attention_mask_ktoe',
        # 'input_ids_etok', 'attention_mask_etok'
        # Adjust keys based on your actual Dataset implementation

        print(f"compute_loss inputs 키: {inputs.keys()}")
        inputs_k = {
            "input_ids": inputs.pop("Korean"),
            "attention_mask": inputs.pop("attention_mask_k"),
        }
        inputs_e = {
            "input_ids": inputs.pop("input_ids_e"),
            "attention_mask": inputs.pop("attention_mask_e"),
        }
        inputs_ktoe = {
            "input_ids": inputs.pop("input_ids_ktoe"),
            "attention_mask": inputs.pop("attention_mask_ktoe"),
        }
        inputs_etok = {
            "input_ids": inputs.pop("input_ids_etok"),
            "attention_mask": inputs.pop("attention_mask_etok"),
        }
        # Any remaining inputs might be labels, etc., which we might ignore here

        # --- 2. Get Embeddings from Model ---
        # This depends heavily on your model architecture.
        # Assuming a standard sentence embedding model where you pass inputs and get [CLS] token or mean pooling.
        # Make sure the model's forward pass returns embeddings in a predictable way.
        # Example: assuming model outputs have an 'embedding' key or are the embeddings directly.
        # Use model.training to ensure correct mode (dropout etc.)

        # IMPORTANT: Ensure model is in training mode if needed (Trainer usually handles this)
        # Make sure gradients are enabled (default within compute_loss)

        # You might need to define how your specific model returns embeddings.
        # Let's assume a simple function `get_embedding` exists on the model.
        # If your model's forward pass directly returns embeddings based on input keys, adapt accordingly.

        # Example: If model's forward returns a dict {'last_hidden_state': ..., 'pooler_output': ...}
        # outputs_k = model(**inputs_k)
        # emb_k = outputs_k.pooler_output # Or mean pooling of last_hidden_state

        # Placeholder - **Replace with your actual model embedding extraction**
        def get_embedding_from_model(model, **kwargs):
            # This is highly dependent on your model structure!
            # Example for sentence-transformers style or similar pooler output:
            outputs = model(**kwargs)
            # Option 1: Pooler output
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            # Option 2: Mean pooling of last hidden state
            elif hasattr(outputs, "last_hidden_state"):
                last_hidden = outputs.last_hidden_state
                attention_mask = kwargs["attention_mask"]
                mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                )
                sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
                sum_mask = mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                return sum_embeddings / sum_mask
            else:
                # Fallback or error
                raise NotImplementedError(
                    "Model output format not recognized for embedding extraction"
                )

        emb_k = get_embedding_from_model(model, **inputs_k)
        emb_e = get_embedding_from_model(model, **inputs_e)
        emb_ktoe = get_embedding_from_model(model, **inputs_ktoe)
        emb_etok = get_embedding_from_model(model, **inputs_etok)

        # --- 3. Calculate Loss ---
        if self.loss_type == "cosine":
            loss = compute_cosine_loss_with_scale_reg(
                emb_k, emb_e, emb_ktoe, emb_etok, self.target_norm, self.lambda_scale
            )
        elif self.loss_type == "mse":
            loss = compute_mse_loss_with_scale_reg(
                emb_k, emb_e, emb_ktoe, emb_etok, self.target_norm, self.lambda_scale
            )
        elif self.loss_type == "infonce":
            loss = compute_infonce_loss_with_scale_reg(
                emb_k,
                emb_e,
                emb_ktoe,
                emb_etok,
                self.target_norm,
                self.lambda_scale,  # Add temperature if needed
            )
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # The Trainer expects the loss as the first element if return_outputs=True
        # If return_outputs=False, it just expects the loss tensor.
        # We don't have model 'outputs' in the traditional sense (like logits for classification),
        # so we just return the computed loss.
        return (
            (loss, {"embeddings": [emb_k, emb_e, emb_ktoe, emb_etok]})
            if return_outputs
            else loss
        )


# --- Example Usage ---
# model = YourMultilingualEmbeddingModel.from_pretrained(...)
# train_dataset = YourQuadrupletDataset(...)
# eval_dataset = YourQuadrupletDataset(...)

# training_args = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=3,
#     per_device_train_batch_size=16,
#     learning_rate=2e-5,
#     # ... other arguments
# )

# # --- Determine target_norm ---
# # Ideally, calculate this beforehand based on the initial state of 'model'
# pre_calculated_target_norm = 15.0 # Example value - CALCULATE THIS!

# trainer = CustomTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     # Pass custom parameters
#     target_norm=pre_calculated_target_norm,
#     lambda_scale=0.02, # Tune this
#     loss_type='cosine'   # Choose 'cosine', 'mse', or 'infonce'
# )

# trainer.train()
