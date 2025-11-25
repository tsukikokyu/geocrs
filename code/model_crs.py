import torch
from torch import nn

class PromptCRSModel(nn.Module):
    def __init__(self, language_model, prompt_encoder, text_encoder):
        super().__init__()
        self.language_model = language_model
        self.prompt_encoder = prompt_encoder
        self.text_encoder = text_encoder
        self.language_model.requires_grad_(False)
        self.text_encoder.requires_grad_(False)


    def _calculate_rec_loss(self, last_hidden_state, labels, entity_embeds, input_ids, pad_token_id, num_virtual_tokens):
        batch_size = last_hidden_state.size(0)
        input_lengths = torch.ne(input_ids, pad_token_id).sum(dim=1)
        sequence_lengths = num_virtual_tokens + input_lengths - 1
        last_token_hidden_state = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        rec_logits = last_token_hidden_state @ entity_embeds.T
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(rec_logits, labels)
        return loss, rec_logits

    def forward(self, context, prompt, address=None, lonlat=None, metadata=None):
        device = next(self.parameters()).device
        for key in context:
            if isinstance(context[key], torch.Tensor):
                context[key] = context[key].to(device)
        for key in prompt:
            if isinstance(prompt[key], torch.Tensor):
                prompt[key] = prompt[key].to(device)
        token_embeds = self.text_encoder(**prompt).last_hidden_state
        prompt_vectors, entity_embeds = self.prompt_encoder(
            token_embeds=token_embeds,
            address=address,
            lonlat=lonlat,
            metadata=metadata,
        )
        input_ids = context['input_ids']
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        final_inputs_embeds = torch.cat([prompt_vectors, inputs_embeds], dim=1)
        prompt_attention_mask = torch.ones(prompt_vectors.shape[:2], dtype=torch.long, device=prompt_vectors.device)
        final_attention_mask = torch.cat([prompt_attention_mask, context['attention_mask']], dim=1)
        outputs = self.language_model(
            inputs_embeds=final_inputs_embeds,
            attention_mask=final_attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = outputs.hidden_states[-1]
        loss, rec_logits = self._calculate_rec_loss(
            last_hidden_state=last_hidden_state,
            labels=context['rec_labels'],
            entity_embeds=entity_embeds,
            input_ids=input_ids,
            pad_token_id=self.language_model.config.pad_token_id,
            num_virtual_tokens=prompt_vectors.size(1)
        )
        return {"loss": loss, "rec_logits": rec_logits}