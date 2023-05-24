import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from rich.console import Console

console = Console()

with console.status("[bold green]Loading model..."):
    device = torch.cuda.current_device() if torch.cuda.is_available() else -1

    checkpoint = "./easyword_model/"
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, src_lang="eng_Latn", tgt_lang="kor_Hang"
    )
    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang="eng_Latn",
        tgt_lang="kor_Hang",
        device=device,
    )

# Get user input until ctrl-D
while True:
    console.print(">", end=" ", style="bold cyan")
    try:
        jargon = console.input()
    except EOFError:
        break
    target_seq = translator(jargon, max_length=128)
    console.print(target_seq[0]["translation_text"], style="blue")
print()

# anonymous function
# computable
# target pursuit
# randomized Turing machine
# learnable class
# loop unrolling
# fixpoint semantics
# structural operational semantics
# primitive type
# structural induction
