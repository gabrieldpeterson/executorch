# Step 4: Download and Export Llama 3.2 1B Model

## Download model with llama-stack
===

 * In a Chromium-based browser, navigate to [Llama Downloads](https://www.llama.com/llama-downloads/) (This website doesn't work in Firefox)
 * Enter your name, birthdate, email, country, organization, and job title into the required fields
 * Check the box for **Llama 3.2: 1B & 3B** Lightweight models
 * Click **Next**
 * Scroll to the bottom of the Terms and Conditions, then check **I accept Llama 3.2 terms and conditions**
 * Click **Accept and continue**
 * Don't close this tab, you'll need the unique URL on this page

Back in your terminal
Install required packages
```bash,run
pip install llama-stack httpx
```

Type "URL=' followed by the unique URL meta gave you, and close it off with a single quotation **'**
```bash,
URL='
```

Your input should look similar to this
```bash
# URL='https://llama3-2-lightweight.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiNnIxZ20xMmI0ZGlkNml4aTEyam41cHAwIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvbGxhbWEzLTItbGlnaHR3ZWlnaHQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc1MTQ5MDA5OX19fV19&Signature=tu%7Ev0QPcOEZp6tKC1k2Zwk5ZBp1hpdq2noRMiIRUXsaueESLaQRLowE485qv68hH4-QFboRJr-yOvoudd6XepYO5qB2HRaNNb3S3xxbNs%7EwTJKXuRI26rNXA9qj-0BprK%7EMYDIE-tfXSSFmATHG50g9cGZA%7EJCE-6bbxPxpiVLk6IOYwrdS9Q032qWyRvSrLcqYuV6QbA3%7EjyP6Q-j6DYHWzrHvjvqlmPplzTJnxegJHwKI3zN3iaNpiw5pu6yjMjaUtbquqW2R0X2j0cRpf8BdAAyhI-9qJnOvz1dUAX3l7uTajYEPI0SsFrrwObX0BKe46AAf6IiJYB4i8KNf0dw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1257688409279718'
```

Download the files
```bash,run
echo "$URL" | python3 llama-download.py model download --source meta --model-id Llama3.2-1B-Instruct
```

## Verify files
===
```bash,run
ls $HOME/.llama/checkpoints/Llama3.2-1B-Instruct
```
The output should show: checklist.chk consolidated.00.pth  params.json  tokenizer.model

## Export the model to ExecuTorch format
===
```bash,run
python3 -m examples.models.llama.export_llama \
--checkpoint $HOME/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth \
--params $HOME/.llama/checkpoints/Llama3.2-1B-Instruct/params.json \
-kv --use_sdpa_with_kv_cache -X --xnnpack-extended-ops -qmode 8da4w \
--group_size 64 -d fp32 \
--metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001, 128006, 128007]}' \
--embedding-quantize 4,32 \
--output_name="llama3_1B_kv_sdpa_xnn_qe_4_64_1024_embedding_4bit.pte" \
--max_seq_length 1024 \
--max_context_length 1024
```

**Why**: This exports the PyTorch model to ExecuTorch's PTE format with:
- KV cache support for efficient inference
- SDPA (Scaled Dot Product Attention) optimization
- XNNPACK backend with extended operations
- 8-bit dynamic quantization with 4-bit weights
- Embedding quantization to reduce model size
- 1024 token context window
